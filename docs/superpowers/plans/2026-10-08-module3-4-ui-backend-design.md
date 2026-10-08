# Module 3/4 — Reef-ranger UI and backend design

Revision: 2026-10-08. Promotes the throwaway proof-of-concept map into a
real ranger-facing map UI (Module 3) and a stable backend API (Module 4).

This is a design and staged implementation plan, not an implementation.
Build 1 (Modules 1/2) is done; see `README.md` and
`docs/build-1-status.md`. The POC map is `coral_bleaching.web` plus
`static/index.html`; the brief that produced it is
`docs/superpowers/plans/2026-09-08-poc-ui-brief.md`. The design spec that
Modules 3/4 must satisfy is
`docs/superpowers/specs/2026-09-04-coral-bleaching-detection-design.md`.

Honesty rule carried forward from Build 1: synthetic runs demonstrate
software behavior, not field performance. The UI must keep saying so, and
must distinguish "no bleaching detected" from "couldn't tell" at every
layer. Nothing here claims a validated bleaching detector.

## 1. Gap analysis — POC vs. spec

### What the POC already does, and should be kept

The POC is small but not naive. It already does several things the real
version must not regress on:

- **Run-integrity verification.** `web.py` reads `manifest.json`, checks
  `schema_version == 2`, and recomputes SHA-256 over `grid.geojson` and
  `predictions.parquet`, comparing against `manifest["sha256"]`. A
  tampered or half-written run is rejected (422), not served. This is the
  trust boundary between an immutable pipeline run and the UI. Keep it.
- **Schema enforcement at read time.** It compares the Parquet schema to
  `PREDICTION_SCHEMA` (metadata stripped) and runs `validate_predictions`
  on the selected slice. The API refuses to serve data that doesn't match
  the Module 2 contract.
- **Honest missing-data rendering.** Three distinct cell states:
  colored prediction, grey-dashed `insufficient` imagery, and dotted
  "no record on this date." The popup text differs per state. This
  directly satisfies the spec's "distinguish no-bleaching from couldn't-tell."
- **Single-date-at-a-time join.** The endpoint selects one reef-wide date
  (latest by default, `?date=` to move) so the map never mixes
  acquisitions. Correct and worth preserving.
- **A domain palette, already.** The POC does *not* have the AI-webapp
  look: teal→muted-teal→amber→coral risk ramp on an ocean-grey canvas, no
  purple gradient, no glassmorphism, no emoji. The real UI extends this
  rather than restyling from scratch.
- **Path-traversal defense and clear 404s** for unknown runs and missing
  files.

### What the POC does not do, that the spec requires of Modules 3/4

- **It is read-only with no database.** The spec's Module 4 is "FastAPI +
  SQLite, with a GeoJSON export," and the differentiator is Module 5's
  ranger confirm/reject feedback loop. The POC has no store and no write
  path, so there is nowhere for a confirmation to go.
- **It renders one run, one date — no trend over time.** The spec asks
  Module 3 to render "reef health, risk flags, **trends**." The POC's date
  menu swaps the whole reef between acquisitions; it cannot show how a
  single cell moved across dates, which is the signal a ranger actually
  reasons about ("this cell has been climbing for three weeks").
- **It serves straight from `runs/<name>/` on disk.** Fine for a demo,
  but there's no concept of "the current published run for this region,"
  no run registry, and no stable region identifier in the API. A ranger
  shouldn't have to know run directory names.
- **It is single-reef by directory convention, not region-agnostic by
  contract.** The spec repeats that the contract must stay region-agnostic
  so a second reef is a config entry, not new code. The POC's API path is
  `/api/runs/{run_name}` with no region in it.
- **No risk-flag surfacing distinct from probability.** The spec separates
  the model's `predicted_label`/`probability` from `dhw_risk_flag` (a weak
  thermal-stress proxy). The POC only shows model probability. A ranger
  needs to see "model says healthy, but DHW is 6.2" — that disagreement is
  informative.
- **No ranger identity, no auth, no audit.** Needed before any write path
  is trustworthy. The spec defers this but Module 5 can't land without it.

The gap is not "rewrite the POC." It is: put a real data store and a
stable, region-keyed API behind the same verification discipline, add a
write path shaped for Module 5, and grow the frontend from a single-run
inspector into a time-aware reef dashboard.

## 2. Backend — Module 4

### Choice, restated

FastAPI + SQLite + GeoJSON export, exactly as the spec specifies for the
pilot. PostGIS is the documented upgrade path and is backend-only: the API
contract below is the real interface boundary, so a Postgres/PostGIS swap
must not change any response shape the frontend depends on.

The API still never touches the model. The pipeline produces an immutable
`runs/<name>/` directory (Build 1 behavior, unchanged). A separate,
explicit **ingest** step loads a verified run into SQLite. The serving API
reads SQLite. This keeps the pipeline fully decoupled from the UI: the
pipeline doesn't know the API exists, and the API never imports Module 2.

```
pipeline run ──> runs/<name>/ (immutable, checksummed)
                      │
                      ▼  ingest (verifies manifest + SHA-256, then loads)
                   SQLite  ◀────── ranger confirmations (write path)
                      │
                      ▼  read
                  FastAPI ──> GeoJSON / JSON ──> map UI
```

### Why a store at all, instead of reading Parquet per request

Three reasons the POC's per-request Parquet read doesn't carry forward:
trends need cheap cross-date queries per cell; the write path (Module 5)
needs somewhere durable to put confirmations that is *not* the immutable
run directory; and the ingest step is the right place to run the
checksum/schema verification once, at load, rather than on every request.
The run directory stays the source of truth and stays immutable; SQLite is
a derived, rebuildable serving + annotation store.

### Data model (SQLite, pilot)

Geometry stays as stored GeoJSON strings plus a bounding box in columns;
no spatial extension needed at single-reef scale (a few hundred cells).
PostGIS later replaces the geometry column with a real geometry type and
the bbox filter with a spatial index — a backend-only change.

- `region(region_id, name, crs, cell_size_m, boundary_geojson)` — one row
  per reef. `heron-island` is the pilot.
- `reef_cell(region_id, reef_cell_id, geometry_geojson, min_lon, min_lat,
  max_lon, max_lat)` — the grid, from `grid.geojson`. Stable IDs.
- `run(run_id, region_id, ingested_at, source_run_dir, manifest_sha256,
  schema_version, status)` — one row per ingested pipeline run. `status`
  is `published` or `superseded`. Only one `published` run per region.
- `prediction(run_id, reef_cell_id, date, probability, predicted_label,
  confidence_band, data_quality, dhw, dhw_risk_flag,
  top_contributing_features_json)` — one row per cell/date, straight from
  `predictions.parquet` joined with the per-cell `dhw`/`dhw_risk_flag` from
  the training dataset. Insufficient rows keep null probability/label and
  empty contributions, matching the Module 2 contract.
- `confirmation(confirmation_id, region_id, reef_cell_id, date, run_id,
  ranger_id, verdict, note, created_at)` — the Module 5 write target.
  `verdict` is `bleached` / `healthy` / `unclear`. `run_id` records which
  run's flag the ranger was looking at, so a confirmation is traceable to
  the prediction it responded to.

`confirmation` lives alongside, not inside, the immutable run. Re-ingesting
or superseding a run never destroys confirmations; they key on
`(region_id, reef_cell_id, date)` which is stable across runs.

### API contract

Region-first and run-aware. The frontend addresses regions and cells, not
run-directory names. All geometry responses are GeoJSON
`FeatureCollection`s; all non-geometry responses are JSON. Dates are
`YYYY-MM-DD`. Prediction fields match `PREDICTION_SCHEMA` names exactly so
the Module 2 contract is the response contract.

Read:

- `GET /api/regions` → list of
  `{region_id, name, published_run_id, available_dates, cell_count}`.
  Lets the UI open without knowing any run name.
- `GET /api/regions/{region_id}/grid` → GeoJSON FeatureCollection of the
  reef-cell geometry only (no predictions). Cached hard; geometry changes
  only when the grid changes.
- `GET /api/regions/{region_id}/predictions?date=YYYY-MM-DD` → GeoJSON
  FeatureCollection: one feature per cell, properties merge the prediction
  for that date onto the cell, plus `dhw`, `dhw_risk_flag`,
  `record_status` (`available` / `missing`), and `confirmation` (the
  ranger verdict for that cell/date if one exists, else null). `date`
  defaults to the latest available. This is the POC's current response,
  extended with risk-flag and confirmation fields, and keyed by region
  instead of run directory. `top_contributing_features` keeps the
  `{feature, value, contribution}` struct shape from Module 2.
  `metadata.available_dates` and `metadata.selected_date` are retained.
- `GET /api/regions/{region_id}/cells/{reef_cell_id}/history` → JSON time
  series for one cell across every date in the published run:
  `[{date, probability, predicted_label, confidence_band, data_quality,
  dhw, dhw_risk_flag}]`, plus any confirmations on that cell. This is the
  endpoint the POC lacks and the trend view needs. It is per-cell, so it
  stays cheap even as date count grows.
- `GET /api/regions/{region_id}/runs` → run history
  (`run_id`, `ingested_at`, `status`, `manifest_sha256`). Lets the UI show
  provenance and lets an operator see what's published.

Write (shaped now so Module 5 builds on it without a contract change):

- `POST /api/regions/{region_id}/confirmations`
  body `{reef_cell_id, date, verdict, note?, run_id?}` →
  `201 {confirmation_id, ...}`. `verdict` ∈ `bleached|healthy|unclear`.
  Server validates that `(reef_cell_id, date)` exists in the published run
  and that `run_id`, if given, is a real run; it records the current
  published `run_id` when omitted. Rejects unknown cells/dates with 422,
  the same discipline the read path already applies.
- `GET /api/regions/{region_id}/confirmations?date=...` → list, for
  rendering the confirmation layer and for the eventual retraining export.
- `DELETE /api/regions/{region_id}/confirmations/{confirmation_id}` →
  correct a mistaken entry. Soft-delete (keep a tombstone) so the audit
  trail survives; a confirmation is survey-adjacent data, not UI state.

Confirmations are the eventual third, highest-confidence label tier the
spec describes for Module 5. The write path deliberately captures
`run_id`, `ranger_id`, and `created_at` now so the retraining export has
provenance later. No retraining logic lives in Module 4; the loop is a
later module reading `confirmation` rows.

Operator-side (not a browser endpoint): an `ingest` CLI command that takes
a `runs/<name>/` directory and a `region_id`, re-runs the POC's
verification (manifest schema check + SHA-256 over `grid.geojson` and
`predictions.parquet`), loads the grid/predictions, and marks the run
`published` (demoting the prior one to `superseded`) in a single
transaction. Verification failure aborts ingest; a bad run never becomes
the published one. This is where the Build 1 checksum discipline moves to.

### Auth

Pilot: a single shared ranger token (header), enough to stamp `ranger_id`
on confirmations and keep the write path from being anonymous. Real
per-ranger accounts are a Module 5 concern and an explicit non-goal here.
State it plainly so no one mistakes the shared token for real auth.

## 3. Frontend — Module 3

A ranger-facing reef dashboard, not a single-run inspector. Grows from the
POC's layout and palette; does not restyle into a generic web app.

### Layout

Keep the POC's two-pane shape: map left, inspection panel right, because it
works and it isn't the AI-webapp hero-and-cards pattern. Changes:

- **Map (left, dominant).** The reef-cell choropleth, same canvas-rendered
  Leaflet layer. Add a small, honest run-provenance strip (region name,
  published run id, ingest date, and the "synthetic — not field
  performance" caveat when the run is synthetic).
- **Date control becomes a timeline, not a dropdown.** A horizontal strip
  of the available acquisition dates with a small per-date reef-wide
  summary (share of cells flagged, share insufficient) so a ranger can see
  which dates matter before clicking. Selecting a date drives the map, as
  the POC's dropdown does now.
- **Inspection panel (right).** When a cell is selected it shows what the
  POC shows — probability, label, confidence band, data quality, signed
  contributions — **plus** the DHW value and `dhw_risk_flag`, and **plus a
  per-cell sparkline/trend** of probability across all dates (from the
  `history` endpoint). This is where "trends over time, not just one run"
  actually lands for the ranger: pick a cell, see its trajectory.
- Below the trend: a confirm/reject control (confirmed/healthy/unclear +
  optional note). Non-functional stub acceptable until Module 5, but the
  layout reserves its place now so the write path has a home.

### Risk color treatment (domain-driven)

Keep and formalize the POC's ramp. The gradient encodes bleaching
probability, not a brand scale:

- Low probability (healthy reef) → **teal/ocean** (`#087f8c`), the color of
  living reef water. Reef is fine here.
- Rising → **muted teal** (`#73bfb8`).
- Mid → **amber** (`#f2b84b`), caution.
- High probability → **coral/warm red** (`#d95d39`), the literal color of a
  stressed/bleaching-adjacent warning. The ramp reads as a risk gradient
  because it moves from reef-water teal to warning-coral, not from one
  brand hue to another.

This is a sequential, colorblind-aware ramp (teal→amber→coral has
luminance separation, not just hue). Pick exact stops deliberately and
comment the choice in CSS, per `AGENTS.md`.

Non-probability states stay outside the ramp so they can't be misread as
low or high risk:

- `data_quality: insufficient` → neutral grey, dashed border ("couldn't
  tell"). Already in the POC.
- No record on the selected date → transparent fill, dotted warm border.
  Already in the POC.
- **DHW risk disagreement** → when the model says healthy but
  `dhw_risk_flag` is true, mark the cell with a thin warning hatch *on top*
  of its teal fill rather than recoloring it. The two signals are
  different claims (confirmed-ish model output vs. weak thermal proxy); the
  UI shows both instead of collapsing them.

### Legibility at reef-cell density

~50–100m cells over a ~26 km² pilot is a few hundred to low thousands of
cells. The POC already renders with `preferCanvas: true`, which carries
forward. Additional rules:

- Cell outlines thin and low-contrast at the choropleth zoom; the fill
  carries the signal. Heavy per-cell borders turn into visual noise at
  density — only the non-probability states (insufficient/missing/DHW
  disagreement) get a distinct border so they pop against the ramp.
- No per-cell labels on the map. Identity and numbers live in the
  inspection panel on click/focus, as in the POC.
- Fill opacity tuned so adjacent cells read as a field, not a grid of
  stickers; the POC's ~0.78 is a reasonable start.
- Keyboard: the map container is focusable (POC does this) and cells are
  reachable; the inspection panel is the accessible readout. Risk is never
  encoded by color alone — the panel states the probability and label in
  text, and the DHW-disagreement state uses a pattern, not just a color.

### Copy

Terse, written like someone who has read survey data. Empty state: "Select
a cell to see its prediction, model reasoning, and trend." Insufficient:
"No prediction — imagery below the quality threshold on this date." Keep
the synthetic-data caveat visible, don't bury it.

### Dependencies

Stay dependency-light, matching the POC: vanilla JS + Leaflet via CDN, no
build step. The sparkline is a small inline SVG, not a charting library.
Default system font stack (the POC already uses it); no Inter/Poppins.

## 4. Staged implementation plan

Ordering keeps the pipeline decoupled throughout and preserves run-integrity
verification at every stage. Each stage is demoable.

**Stage 0 — Fake data contract check (no new code paths).** Confirm the
real API can be built against a hand-written `predictions.parquet` +
`grid.geojson` before touching Modules 1/2, exactly as the spec intends.
This is the regression guard for "swapping real data later doesn't touch
the UI."

**Stage 1 — Ingest + store, read parity with the POC.** Build the SQLite
schema and the `ingest` command. Move the POC's manifest/SHA-256/schema
verification into ingest (verify once at load, abort on failure). Then
reimplement the POC's single-date GeoJSON response from SQLite behind the
new region-keyed endpoints (`/api/regions`, `/grid`, `/predictions`).
Acceptance: ingesting `runs/demo-001` and hitting `/predictions` yields the
same cell states (colored / insufficient / missing) the POC shows today.
No frontend change yet beyond repointing fetches at the region endpoints.

**Stage 2 — Trends.** Add `dhw`/`dhw_risk_flag` into `prediction` at
ingest, add the `cells/{id}/history` endpoint, and add the per-cell trend
sparkline + DHW readout + DHW-disagreement hatch to the UI. This is the
first real capability beyond the POC. Pipeline untouched.

**Stage 3 — Timeline and provenance UI.** Replace the date dropdown with
the per-date summary timeline; add the run-provenance strip and
`/regions/{id}/runs`. Still read-only.

**Stage 4 — Write path (foundation for Module 5).** Add the
`confirmation` table, `POST`/`GET`/`DELETE` confirmation endpoints, the
shared-token stamp, and wire the UI's confirm/reject control to them.
Confirmations render as an overlay and in the inspection panel. Module 5
(retraining) consumes `confirmation` rows later; it is not built here.

**Stage 5 — PostGIS readiness (documentation + seams only).** Keep
geometry access behind a thin query layer so the Postgres/PostGIS swap is a
backend change with no response-shape change. Don't migrate for the pilot;
just don't paint into a corner.

Throughout: the pipeline (`coral_bleaching` Modules 1/2) gains no imports
from the API, and the API gains no imports from Module 2. The run directory
stays the immutable source of truth; SQLite is rebuildable from it (except
confirmations, which are backed up separately as the one piece of
non-derived state).

## 5. Proposed stakeholder recommendations (not yet scoped)

Source: email feedback from Sydney's team after their 2026-10-08 meeting
(relayed by the project owner). Of their three suggestions, two (restoration
activity and invasive biota) are data/model inputs captured in
`docs/superpowers/plans/2026-10-08-real-data-acquisition.md`. The third is a
UI/backend feature and is recorded here. It is proposed, not accepted, and not
scoped into the staged plan above; this section describes where it would fit if
pursued.

### Governing-body view per reef section

The team suggested a feature to view the governing body responsible for each
section of a reef.

- **What it is.** A jurisdiction/management overlay: for a given cell or reef
  section, show which authority governs it (e.g. GBRMPA zone, a specific marine
  park zone, a Traditional Owner sea-country area, or a research-permit holder).
  This is administrative/geographic metadata, not a model input and not a
  bleaching signal — it does not touch Modules 1/2 or the prediction contract.
- **Where it fits in the data model.** It is region/geometry metadata, so it
  attaches to the backend's region layer, not to `prediction` or
  `confirmation`. Two honest options:
  - a `management_zone(region_id, zone_id, name, governing_body, geometry_geojson)`
    table holding the jurisdiction polygons, independent of the reef-cell grid;
    a cell's governing body is resolved by spatial lookup (which zone contains
    the cell), computed at ingest and cached per cell. This keeps zones and the
    model grid decoupled — zone boundaries do not have to align with 75 m cells.
  - or, if zones are coarse and stable, a precomputed `reef_cell.zone_id`
    column populated at ingest. Simpler, but recomputed whenever zones change.

  The spatial-lookup table is the more honest default: management boundaries are
  authoritative polygons in their own right and should not be approximated onto
  the analysis grid, the same discipline applied to the reef boundary itself.
- **API.** Additive, region-first, no change to existing response shapes:
  - `GET /api/regions/{region_id}/zones` → GeoJSON FeatureCollection of the
    management-zone polygons with `governing_body` in properties, for a map
    overlay layer.
  - the per-cell `predictions` and `history` responses gain an optional
    `governing_body` / `zone_id` property resolved at ingest, so a cell popup
    can state its jurisdiction without a second request.
- **UI.** A toggleable overlay layer on the existing map (zone outlines +
  a legend of governing bodies), and a line in the inspection panel naming the
  governing body for the selected cell. It must read as an administrative
  overlay distinct from the risk ramp — outline/label treatment, not a fill
  that competes with the bleaching-probability colors. No new color scale that
  could be confused with risk.
- **Data availability — unverified.** Authoritative zoning polygons likely
  exist (GBRMPA Marine Park zoning is published spatial data), but the exact
  source, format, license, and whether "governing body" means statutory zoning,
  Traditional Owner sea country, or something more granular are open. Treat as
  a data-request item; resolution against the pilot extent is unknown until a
  source is identified.

### Status

Proposed, not accepted. Needs a confirmed authoritative zoning source, a
license, and a decision on what "governing body" means for the pilot before any
work. If pursued, it is additive to the Module 4 region layer and the Module 3
map; it does not alter the pipeline, the prediction contract, or the Module 5
write path.

## 6. Open questions and non-goals

### Non-goals (first real version)

- **Not multi-reef in practice.** The contract is region-agnostic and the
  schema carries `region_id`, but the pilot ships one region
  (`heron-island`). No region-management UI.
- **No real per-ranger auth.** A single shared token stamps confirmations.
  Accounts, roles, and sessions are deferred to Module 5.
- **No retraining loop.** Module 4 captures confirmations with provenance;
  it does not feed them back into the model. That's Module 5.
- **No live/real-time inference.** Unchanged from the spec — the pipeline
  is an offline batch job; the API serves ingested runs.
- **No PostGIS, no containers, no HTTPS/deployment hardening** for the
  pilot. Documented upgrade paths, not v1 work.
- **No claim of field accuracy.** Synthetic runs stay labeled synthetic in
  the UI.

### Open questions

1. **Confirmation granularity vs. the grid.** A ranger confirms a physical
   observation at a point/time, but the schema keys confirmations on
   `(reef_cell_id, date)` where `date` is an *acquisition* date. Does a
   ranger confirm against the nearest acquisition date (mirroring the
   14-day AIMS matching window from Module 1), or against their own
   observation date with the join resolved at retraining time? The latter
   keeps the ranger honest about when they looked; the former is simpler to
   render. This needs a decision before the write path is final.
2. **Trend semantics the ranger should see.** Is the useful signal raw
   probability over time, `index_shift` (the rolling spectral change
   Module 1 already computes), or probability with the DHW overlay? These
   can disagree, and showing all three risks clutter at reef-cell density.
   Which one leads, and which are secondary?
3. **One published run, or diffable runs?** The design publishes one run
   per region and supersedes the prior one. Do rangers need to compare two
   runs (e.g., "did last week's reprocess change this cell's flag"), or is
   a single current view plus per-cell history enough? Run-diff is more
   backend and UI; worth confirming it's wanted before building the run
   registry beyond provenance display.
