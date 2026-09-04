# Coral Bleaching Detection — Design Spec

## Overview

A coral health map webapp for reef rangers that goes beyond existing tools
(NOAA Coral Reef Watch, Allen Coral Atlas) by adding **localized bleaching
confirmation** and a **ground-truth feedback loop**: rangers confirm or
reject model-generated bleaching flags in the UI, and those confirmations
feed back into retraining the detection model. The feedback loop is the
project's differentiator — existing tools show risk at a coarse level but
don't close the loop with in-field verification.

No hard deadline. Optimize for a correct, scalable pipeline over a rushed
demo.

## Non-Goals (v1)

- Global or multi-reef coverage (single reef pilot only)
- Live/real-time inference (pipeline runs as an offline batch job)
- CNN/deep-learning imagery model (gradient-boosted trees on spectral
  indices first; CNN is an explicit future option, not a v1 requirement)

## Pilot Region

**Heron Island Reef** (southern Great Barrier Reef). Chosen for:
- AIMS Long-Term Monitoring Program survey data as an independent
  ground-truth source
- Documented bleaching events in both 2016 and 2020 (positive and
  negative examples in one small area)
- Small enough area for a bounded pilot

## Data Sources

| Source | Purpose |
|---|---|
| Sentinel-2 (10m resolution, via Google Earth Engine Python API) | Spectral reflectance imagery, primary model input |
| NOAA Coral Reef Watch | Degree Heating Weeks (DHW) as a weak-label proxy for thermal stress |
| Allen Coral Atlas | Reef boundary geometry, used to generate the reef-cell grid |
| AIMS Long-Term Monitoring Program survey records | High-confidence ground-truth labels (confirmed bleached/healthy) |
| Published bleaching event records (2016/2020 GBR) | Validation cross-check |

## System Architecture: 5 Modules

1. **Data pipeline** — pulls imagery/NOAA/Atlas/AIMS data, outputs spectral
   indices + DHW + labels per reef-cell/date. Pure Python, no web app.
2. **Detection model** — consumes pipeline output, produces bleaching
   flags with probability, reasoning, and confidence. Pure Python.
3. **Web app / map UI** — renders reef health, risk flags, trends for
   rangers.
4. **Backend API + data store** — stable contract between modules 1/2 and
   module 3.
5. **Verification feedback loop** — rangers confirm/reject flags in the
   UI; confirmations feed back into retraining module 2.

Modules 1/2 are being designed first, since they establish the data
contract that 3/4/5 build against. Because the contract is a versioned
file format (below), module 3/4 can be built and demoed against
hand-written fake data before the real ML pipeline exists, and swapping in
real data later doesn't touch the UI as long as the contract holds.

## Data Pipeline (Module 1)

Runs as an offline batch job producing a versioned dataset, not a live
service.

```
[GEE: Sentinel-2] ──┐
[NOAA DHW API]    ──┼──> fetch & align by (reef-cell, date) ──> spectral index calc ──> label join ──> training_dataset.parquet
[Allen Coral Atlas]──┤                                                                        ↑
[AIMS survey records]┘                                                                  ground-truth join
```

### Reef-cell grid

The reef area is divided into a fixed grid of reef-cells (~50–100m cells)
derived from the Allen Atlas reef boundary polygon. Every dataset row is
one `(reef_cell_id, date)` pair. This grid is the stable geometry shared
by the ML pipeline, the API, and the frontend map.

Grid generation is driven by a region config (reef boundary polygon, date
range) rather than hardcoded — adding a second reef later is a new config
entry, not new pipeline code.

### Row schema (`training_dataset.parquet`)

Partitioned by `reef_cell_id` / `date` so new dates/reefs append without
schema migration.

| Field | Description |
|---|---|
| `reef_cell_id` | Grid cell identifier |
| `date` | Observation date |
| `spectral_indices` | Reflectance-ratio indices computed over a rolling window (captures shift over time, not single-date snapshot) |
| `dhw` | Coincident NOAA Degree Heating Weeks value |
| `cloud_cover_fraction` | Used for weighting/filtering, not automatic row-dropping |
| `data_quality` | `ok` / `insufficient` (e.g. cloud cover too high to compute reliable indices) |
| `confirmed_label` | `bleached` / `healthy` / `null` — set when an AIMS survey exists within ~2 weeks of the cell/date |
| `dhw_risk_flag` | `true`/`false` from DHW ≥ 4 threshold — separate weak-label column |

### Error handling

- Rows with unusable imagery (e.g. persistent cloud cover) are written
  with `data_quality: insufficient`, not dropped or imputed — downstream
  consumers must be able to distinguish "no bleaching detected" from
  "couldn't tell."
- AIMS/DHW join failures (e.g. source API downtime) hard-fail the pipeline
  run rather than proceeding with partial labels. This is a batch job, so
  a failed run simply doesn't produce a new dataset version.

## Detection Model (Module 2)

### Training labels — combined strategy

- Rows with `confirmed_label` (from AIMS) are the high-confidence set —
  used for both training and held-out validation.
- Rows with only `dhw_risk_flag` are weak-labeled — included in training
  but down-weighted (e.g. half sample weight), since DHW is a thermal-
  stress proxy, not a confirmed observation.
- This label-tier/weighting scheme is designed to extend cleanly: when
  Module 5 (feedback loop) comes online, ranger-confirmed labels become a
  third, highest-confidence tier above AIMS — same weighting mechanism,
  one more tier, no retraining architecture change.

### Model

Gradient-boosted trees (LightGBM or XGBoost) on per-cell spectral index
features + DHW + cloud-cover-quality. Binary output: bleaching-flag
probability per `(reef_cell_id, date)`.

Chosen over a CNN for v1 because the grid-cell + spectral-index
representation is already a hand-engineered feature set — a CNN would
need raw imagery patches instead, is more data-hungry, and is harder to
keep interpretable. GBTs give feature-importance output for free, which
directly satisfies the "flags with reasoning" requirement (e.g. "flagged:
NDCI drop of 0.14 over 3 weeks + DHW 6.2").

CNN migration path stays open: the per-cell schema can later gain an
optional raw-imagery-patch reference alongside the existing spectral-index
columns — additive, not a breaking change to the module 4 API contract.

### Output contract (`predictions.parquet`)

| Field | Description |
|---|---|
| `reef_cell_id`, `date` | Join key, matches pipeline schema |
| `probability` | Model output probability of bleaching |
| `predicted_label` | Thresholded bleached/healthy |
| `top_contributing_features` | SHAP or GBT feature importances, for the "reasoning" shown to rangers |
| `confidence_band` | Derived from probability distance from 0.5 (or ensemble variance) |

This contract is region-agnostic — Modules 3/4 don't change as more reefs
are added.

### Validation

Leave-one-survey-date-out cross-validation, not random split. Random
splitting would leak information (adjacent cells on the same date are
highly spatially/temporally correlated) and overstate accuracy. Tracked
as an ongoing metric, not a one-time pass/fail test.

## Backend / Data Store (Module 4)

FastAPI + SQLite, with a GeoJSON export for the frontend map. Chosen over
PostgreSQL/PostGIS for pilot simplicity — single reef, a few hundred
labeled points, no deployment requirements yet. PostGIS is the natural
upgrade path if the project expands to more reefs; that migration is
backend-only and doesn't touch the frontend, since the API contract above
is the real interface boundary.

The API never touches the model directly — it reads `predictions.parquet`
(produced by running Module 2 over the latest pipeline output) and serves
it. This is what allows Module 3/4 to be built and demoed against a
hand-written fake `predictions.parquet` before Module 1/2 are real.

## Testing

- Unit tests on spectral-index math: known reflectance inputs → expected
  index values.
- Unit tests on grid-cell generation: boundary polygon → expected cell
  count/coverage.
- Integration test running the full pipeline against a small fixed date
  range with mocked GEE/NOAA responses, verifying join logic end-to-end.
- Model validation via leave-one-survey-date-out CV (see above), tracked
  as a metric over time.

## Phase 1 Setup Tasks

- Sign up for Google Earth Engine access (no existing access; approval
  can take time, so this should start early and in parallel with other
  setup work).

## Open Items for Later Phases

- Module 3 (web UI) and Module 5 (feedback loop) design are deferred
  until the Module 1/2 data contract is implemented and stable.
- Multi-reef expansion (config-driven grid generation already supports
  this; not scheduled for v1).
