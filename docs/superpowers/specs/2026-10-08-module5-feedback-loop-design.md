# Module 5 — Verification Feedback Loop Design Spec

## Overview

Module 5 is the ground-truth feedback loop the top-level design spec names as
the project's differentiator: a ranger confirms or rejects a model bleaching
flag in the field, that judgment is recorded with provenance, and it feeds the
next retraining run as a label that outranks AIMS. Existing tools (NOAA Coral
Reef Watch, Allen Coral Atlas) stop at coarse risk; nothing closes the loop
with in-field verification tied to a specific cell and date.

This spec covers the ranger interaction, the data model for a confirmation,
how confirmations become a third label tier in the existing weighting scheme,
the minimal write path on top of the current read-only app, and the honesty
and leakage guarantees the loop must not break. It does not change the model
architecture, the training label mechanism, or the prediction contract. Those
are already built (Build 1) and the top-level spec commits to extending them
"same weighting mechanism, one more tier, no retraining architecture change."

Nothing here is validated against field use. There is no ranger, no real
confirmation, and no retraining run driven by one yet. This is a design for
work not done.

## What exists today (the ground this builds on)

Reading the shipped code, not the aspiration:

- `labels.build_training_frame` filters to eligible rows, then sets
  `label = confirmed_label` where an AIMS survey matched, else the DHW weak
  label. `sample_weight` is `CONFIRMED_WEIGHT = 1.0` for AIMS-confirmed rows
  and `WEAK_LABEL_WEIGHT = 0.5` otherwise. Two tiers, one weight column.
- `train.train_model` fits LightGBM on `feature_matrix(df)` with
  `sample_weight=df.sample_weight`. It reads the weight column; it does not
  care how many tiers produced it.
- `train.leave_one_survey_date_out_splits` holds out one `survey_date` at a
  time, tests only AIMS-confirmed eligible rows, and purges training rows that
  share a held-out `survey_id`, a held-out `source_scene_id`, or a feature
  interval overlapping any test interval expanded by a 14-day embargo.
- `schema.TRAINING_SCHEMA` is version 2. The confirmed-label provenance triple
  is `confirmed_label`, `survey_date`, `survey_id`, and `validate_training`
  enforces that they are populated together or all null.
- `pipeline.publish_run` stages a run, rechecks keys, writes SHA-256 checksums
  of every file into `manifest.json`, and atomically renames the staging
  directory into place. Runs are immutable; a published run is never replaced.
- `web.py` is read-only. `load_run_geojson` resolves a run directory, verifies
  the manifest checksums for `grid.geojson` and `predictions.parquet`, selects
  one reef-wide date, and merges predictions onto grid features. There is one
  `GET` endpoint and no database, no `POST`, no auth.

Module 5 adds exactly one new input to this chain — a confirmations store —
and one new label tier derived from it. Everything downstream of
`build_training_frame` stays byte-for-byte the same code path.

## 1. Ranger confirm/reject interaction

### What the ranger sees

The ranger opens the existing map against a published run and a selected date,
exactly as today. Each cell already carries `probability`, `predicted_label`,
`confidence_band`, `data_quality`, and `top_contributing_features`. The cell
popup gains two actions and a status line:

- **Confirm bleached** / **Confirm healthy** — the ranger asserts the ground
  state they observed in the field for this cell on (or near) this date.
- **Reject flag** — the model flagged bleached and the ranger saw healthy, or
  vice versa. Reject is not a third class; it is a confirmation of the
  *opposite* observed state, recorded with the model's flag it contradicts so
  disagreement is auditable.
- A status line showing whether this cell/date already has a confirmation,
  who recorded it, and when. A second ranger disagreeing is a new record, not
  an overwrite (see supersession below).

Cells with `data_quality: insufficient` or `record_status: missing` can still
be confirmed. A ranger standing on the reef has better evidence than masked
imagery; the point of the loop is to capture exactly the observations the
model could not make. A confirmation on an insufficient-imagery cell is stored
but, by the training rules below, is only usable for the model once that cell
also has usable imagery for a nearby date — the confirmation does not conjure
features.

Copy is terse and field-specific: "Confirm what you see on this cell." No
celebratory toasts, no "thank you for your contribution." A confirmation
either records or fails with the reason.

### What gets recorded

A ranger confirmation is a first-class observation record with its own
provenance, parallel to how an AIMS `Survey` is a provenance-bearing record
today. It attaches to a `(reef_cell_id, observed_date)` pair — the cell the
ranger inspected and the date they inspected it, not the imagery acquisition
date. The imagery date the ranger was looking at is retained separately so the
UI context is auditable.

Proposed record (SQLite row; see the write path for storage):

| Field | Meaning |
|---|---|
| `confirmation_id` | Stable UUID, generated server-side. The join/identity key, analogous to `survey_id`. |
| `reef_cell_id` | Grid cell the ranger inspected. Must exist in the run's `grid.geojson`. |
| `observed_date` | Date the ranger observed the reef state (YYYY-MM-DD). This is the label's effective date. |
| `observed_label` | `bleached` or `healthy`. The ground state the ranger asserts. Reject is encoded as the opposite label of the flag. |
| `action` | `confirm` or `reject`, retained for audit (did the ranger agree or overturn the model?). Not used as a label value. |
| `ranger_id` | Who recorded it. An opaque authenticated identity, never a free-typed name. |
| `recorded_at` | Server UTC timestamp of when the record was written. |
| `run_name` | The published run the ranger was viewing when they acted. |
| `viewed_date` | The reef-wide imagery date selected in the UI at the time. |
| `model_predicted_label` | The flag shown to the ranger for this cell/date, nullable (insufficient/missing cells have none). |
| `model_probability` | The probability shown, nullable, for disagreement analysis. |
| `app_version` | Build/version string of the app that captured it, for schema provenance. |
| `superseded_by` | `confirmation_id` of a later record that replaces this one, or null. Append-only history. |
| `note` | Optional short free text (field conditions). Never parsed into a label. |

Rules, mirroring the AIMS record's discipline:

- `observed_label`, `observed_date`, `ranger_id`, and `recorded_at` are
  required together. A confirmation with a missing identity or date is
  rejected at write time, the same way `load_aims_labels` rejects a blank
  `survey_id` or an out-of-set label.
- `reef_cell_id` must be a cell in the referenced run's grid. An unknown cell
  is a client bug and fails the write, matching `validate_training`'s
  "unknown cell" discipline.
- Records are append-only. A ranger (or a second ranger) correcting an earlier
  call writes a new record and sets `superseded_by` on the old one. Nothing is
  deleted or mutated in place, so the full confirmation history stays auditable
  — consistent with immutable runs.

## 2. Ranger confirmations as a third, highest-confidence label tier

The top-level spec commits to this being additive: "ranger-confirmed labels
become a third, highest-confidence tier above AIMS — same weighting mechanism,
one more tier, no retraining architecture change." The code makes that literal.

### Weighting

Today `labels.py` has two constants and a two-way `where`:

```python
CONFIRMED_WEIGHT = 1.0   # AIMS confirmed
WEAK_LABEL_WEIGHT = 0.5  # DHW-only weak label
```

Module 5 adds one tier *above* AIMS. Because `train_model` only reads a scalar
`sample_weight` per row, "highest confidence" is expressed as the largest
weight. AIMS is downgraded from the ceiling to the middle tier; ranger
confirmations take the top:

```python
RANGER_WEIGHT = 1.0    # in-field ranger confirmation (new top tier)
AIMS_WEIGHT = 0.75     # AIMS survey (was 1.0)
WEAK_LABEL_WEIGHT = 0.5  # DHW-only weak label (unchanged)
```

The exact `AIMS_WEIGHT` value is an open question (see §5) — the structural
point is that ranger > AIMS > DHW as an ordered weight ladder, and the
mechanism is still a single sample-weight column. The relabeling in
`build_training_frame` extends from a two-way `where` to a three-way priority:
ranger label if present, else AIMS confirmed label, else DHW weak label; and
the weight is chosen by the same priority. No new model, no new loss, no new
training entry point.

### Where the ranger label enters

The clean seam is a new, optional join step before `build_training_frame`,
parallel to the AIMS join already in `build_training_dataset`:

1. The pipeline resolves ranger confirmations for the run's cells and date
   range from the confirmations store (§3), matching a confirmation to a
   `(reef_cell_id, date)` row by the same nearest-within-window rule AIMS uses
   (`find_matching_label`, 14-day window), using `observed_date` in place of
   `survey_date`.
2. A matched confirmation populates a new provenance triple on the training
   row — `ranger_label`, `ranger_observed_date`, `confirmation_id` — mirroring
   the `confirmed_label` / `survey_date` / `survey_id` triple exactly, with the
   same "all three or none" validation.
3. `build_training_frame` picks the label and weight by tier priority.

This keeps schema v2's existing columns untouched and adds the ranger triple
as new nullable fields. Because the top-level spec allows additive schema
growth ("additive, not a breaking change to the module 4 API contract"), this
is a schema v2.1 extension: three new nullable columns, old runs still load.
The decision to bump the schema version vs. extend v2 in place is an open
question (§5); either way the prediction contract (`PREDICTION_SCHEMA`) does
not change, so Module 3/4 and `web.py`'s reader are unaffected.

A confirmation on an insufficient-imagery row cannot become a training row:
`eligible_rows` still gates on usable features first, exactly as it does for
AIMS confirmed labels today ("even a confirmed label cannot make unusable
imagery trainable"). The confirmation is stored and visible, but it only
trains the model when the same cell has usable imagery within the matching
window — e.g. a clear acquisition a few days off the ranger's visit.

## 3. The write path

The current app is explicitly read-only with no database (the POC brief lists
"No write path at all. This is read-only" as out of scope, deferred to
Module 5). This is where that gets built, following the top-level spec's
Module 4 direction: FastAPI + SQLite, "chosen over PostgreSQL/PostGIS for
pilot simplicity — single reef, a few hundred labeled points."

### Storage

A single SQLite database file outside the immutable `runs/` tree — e.g.
`confirmations.db` under a configurable `--confirmations-db` path. It must not
live inside a run directory: runs are immutable and checksum-verified, and a
mutable database inside one would break `manifest.json`'s SHA-256 guarantee
that `web.py` enforces on load. The confirmations store is the one mutable
(append-only) surface in the system; runs stay frozen.

One table, `ranger_confirmations`, with the columns from §1. A UNIQUE index on
`(reef_cell_id, observed_date, ranger_id)` where `superseded_by IS NULL`
enforces one live call per ranger per cell/date; a correction supersedes
rather than violating it. SQLite WAL mode handles the handful of concurrent
field writes a single-reef pilot produces; this is not a high-throughput store.

### Write endpoint

One new authenticated endpoint on the existing FastAPI app:

```
POST /api/runs/{run_name}/confirmations
```

Body: `reef_cell_id`, `observed_date`, `observed_label`, `action`, and the
model context the client already holds (`viewed_date`, `model_predicted_label`,
`model_probability`). The server fills `confirmation_id`, `ranger_id` (from the
authenticated session, never the request body), `recorded_at`, `run_name`, and
`app_version`. It validates the cell against the run's grid and the label
against `{bleached, healthy}`, reusing the same validation posture as the read
path, then inserts (or supersedes-then-inserts) one row.

Auth is required for writes — a confirmation without a trustworthy `ranger_id`
is worthless as provenance. The read path (`GET /api/runs/{run_name}`) can stay
anonymous for the pilot; the write path cannot. The specific auth mechanism
(shared pilot token vs. per-ranger credential) is an open question (§5). The
read path is otherwise unchanged, so the existing map keeps working with no
database present — the DB is only touched when a confirmation is written or
when the UI asks whether a cell already has one.

### Feeding back into a retraining run

Confirmations never mutate a published run. They feed the *next* run:

1. An operator triggers a new batch run (`coral-bleaching run ...`) exactly as
   today. The run optionally receives `--confirmations-db <path>`.
2. During dataset build, confirmations for the run's cells and interval are
   read from the store and joined as the top label tier (§2).
3. The run trains, validates, and publishes a new immutable run directory as
   usual. The new run's `manifest.json` records which confirmations fed it:
   a `ranger_confirmations` block listing the `confirmation_id`s consumed, the
   store's path, and a SHA-256 of the exact confirmation rows used (snapshot
   hash), so the run is reproducible even though the live store keeps growing.
   This matches the existing manifest's provenance discipline — AIMS surveys
   are already hashed in via `survey_sha256`; confirmations get the same
   treatment.
4. The versioned-run/manifest model is otherwise untouched: staging, key
   recheck, checksums, atomic rename. A new run with ranger confirmations is
   just another immutable run with one more provenance source recorded.

So the loop is: ranger writes to the mutable store → operator runs a new batch
→ a new immutable run is published that the map can then serve, now shaped by
field confirmations. There is no online/continuous learning; retraining stays
the deliberate, versioned, offline batch the top-level spec describes.

## 4. Leakage avoidance and honesty guarantees

The loop must not quietly undo the validation discipline Build 1 was corrected
to enforce. Concretely:

- **Confirmations go through the same purge.** The ranger triple
  (`confirmation_id`, `ranger_observed_date`) must participate in
  `leave_one_survey_date_out_splits` the way the AIMS triple does. A held-out
  validation fold that tests a ranger-confirmed cell/date must purge training
  rows that share that confirmation's identity, scene, or an overlapping
  feature interval under the 14-day embargo — across the reef, not per cell.
  Otherwise a confirmation used both to train and to test is leakage. This is a
  direct extension of the existing purge logic to a third identity column.
- **What counts as the held-out test set is an explicit decision, not a
  default.** Today validation tests only AIMS-confirmed rows, deliberately,
  because AIMS is the independent ground truth. If ranger confirmations are
  allowed into the test set, the model is being graded by the same rangers who
  may have been shown its flags — a feedback-contaminated metric. The
  conservative first version keeps **AIMS as the sole validation authority**
  and uses ranger confirmations for training only. Reporting a ranger-graded
  AUC as if it were independent accuracy would be exactly the kind of
  overstatement README.md and build-1-status.md are written to avoid. The
  alternative (a separately reported, clearly-labeled ranger-agreement metric
  that is never conflated with CV AUC) is an open question (§5).
- **Confirmation-vs-model independence.** Because the ranger sees the model's
  flag before confirming, confirmations are not independent of the model. The
  stored `model_predicted_label` and `model_probability` exist precisely so
  this dependence is measurable (agreement rate, systematic over/under-call)
  and never silently treated as blind ground truth.
- **Synthetic stays distinguishable from real.** The `demo` path is synthetic
  and the manifest already carries `{"synthetic": True}`. Confirmations must
  carry the same honesty: a confirmation's `run_name` ties it to a run whose
  manifest states whether it was synthetic, and the confirmations store should
  refuse (or clearly tag) records against synthetic demo runs so a synthetic
  demo can never masquerade as field-verified data. A retraining run fed by
  confirmations records `synthetic: False` only if every input — imagery,
  NOAA, AIMS, and confirmations — is real. Any synthetic input keeps the run
  honestly marked synthetic.
- **No imputation, no footprint inflation.** The AIMS rule "do not enlarge
  survey footprints or invent per-cell labels to get more rows" applies
  verbatim to confirmations. One confirmation labels one cell for its observed
  date within the matching window. It is not spread to neighbors to manufacture
  training rows.

## 5. Proposed stakeholder recommendations (not yet scoped)

Source: email feedback from Sydney's team after their 2026-10-08 meeting
(relayed by the project owner). For traceability, their three suggestions are
recorded against every design doc. **None of the three is a Module 5 feature.**
They land elsewhere:

- Reef farming / restoration attempts as an assessment variable — a model
  covariate; see `docs/superpowers/plans/2026-10-08-real-data-acquisition.md`.
- Invasive biota (e.g. crown-of-thorns starfish) impact — a model covariate;
  same acquisition plan.
- Governing-body view per reef section — a UI/backend overlay; see
  `docs/superpowers/plans/2026-10-08-module3-4-ui-backend-design.md`.

They are noted here only because the first two interact with the feedback
loop's honesty guarantees, and that interaction is in Module 5's scope to state.

### Interaction with ranger confirmations (the only Module 5 touch-point)

If restoration activity and invasive-biota pressure become covariates, they
sharpen a confounder the confirmation loop must respect: a cell can lose coral
cover for reasons that are not thermal bleaching (a COTS outbreak, disturbance
at a restoration site). A ranger confirming "bleached" on such a cell may be
recording real coral loss with the wrong cause — a misattribution, not a
mistake the ranger would notice from the model's flag alone.

Consequences for this design, all consistent with §4:

- The confirmation vocabulary stays `bleached` / `healthy` in v1. Cause
  attribution (bleaching vs. predation vs. restoration disturbance) is **not** a
  ranger field in the first version; adding a cause taxonomy is a possible later
  extension, not a v1 commitment. Recording a miscaused "bleached" would quietly
  teach the detector the wrong signal.
- If a COTS/restoration covariate exists for a cell/date, it is a feature the
  model already sees — it does not change the confirmation record or the label
  tier. The `note` field remains free text and is never parsed into a label or
  a cause.
- This strengthens the case for keeping AIMS as the sole validation authority
  in v1: an independent bleaching-classified survey is less likely to confuse
  predation/restoration loss with bleaching than an in-field ranger reacting to
  a flag.

No data model, weighting, or write-path change follows from the stakeholder
recommendations. This section exists so the misattribution risk is on record,
not to add scope.

## 6. Open questions and non-goals

### Open questions

- **AIMS vs. ranger weight.** Is `AIMS_WEIGHT = 0.75` right, or should AIMS and
  ranger confirmations both sit at 1.0 and be distinguished only by validation
  role? The ordering (ranger ≥ AIMS > DHW) is firmer than the exact value, and
  the value should be set against real agreement data, not guessed.
- **Validation authority.** Should ranger confirmations ever enter the held-out
  test set, and if so how is the model-exposure bias reported honestly? The
  conservative default here is AIMS-only validation plus a separate, clearly
  non-independent ranger-agreement statistic.
- **Auth model.** Shared pilot token vs. per-ranger credential, and how
  `ranger_id` is provisioned and kept stable across devices in the field.
- **Conflict resolution.** When two rangers disagree on the same cell/date,
  the supersession model keeps both but training needs one. Latest-wins,
  seniority, or exclude-when-conflicting (mirroring AIMS'
  `conflicting_equidistant_surveys` → unlabeled) — undecided.
- **Schema versioning.** Add the ranger triple as a schema v2.1 extension in
  place, or bump to v3? Either keeps the prediction contract fixed, but the
  reader-side version checks in `web.py` and `pipeline.read_dataset` must agree.
- **Offline field capture.** Rangers often have no connectivity on the reef.
  Does the first version require live connectivity to `POST`, or is a local
  queue-and-sync needed? Treated as a non-goal below, but it is the most likely
  real-world gap.

### Non-goals (first version)

- No online or continuous learning. Retraining stays a deliberate, versioned,
  offline batch run.
- No in-field offline capture / sync. First version assumes the ranger has
  connectivity when confirming; offline queueing is a later addition.
- No multi-reef confirmation store. Single-reef pilot, same scope boundary as
  the rest of the project.
- No PostGIS. SQLite is the pilot store; PostGIS remains the documented upgrade
  path if the project expands, backend-only.
- No ranger-facing analytics dashboard (agreement trends, per-ranger stats).
  The data to build one is recorded; building it is out of scope here.
- No automatic retraining trigger on write. A human runs the batch; the loop is
  not self-driving in v1.
- No editing or deletion of confirmations. Append-only with supersession is the
  entire mutation model.
