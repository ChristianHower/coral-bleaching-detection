# Coral Bleaching Data Pipeline & Detection Model Implementation Plan

Revision: 2026-09-04 — correctness fixes before Build 1.

This revision replaces the original copy-and-paste implementations, which
contained validation leakage and missing-data errors. The twelve tasks below
are the implementation checklist. Build 1 is now implemented locally; see
README.md and docs/build-1-status.md for commands, validation, and remaining
live-data requirements.

## Scope and architecture

Implement Modules 1 and 2 as an offline Python package under
`src/coral_bleaching/`, with tests under `tests/`. No UI, API, or database.
Sentinel-2 observations, NOAA DHW, and normalized AIMS survey records produce
`training_dataset.parquet`; LightGBM produces `predictions.parquet`.
External clients must be injectable. Unit and integration tests use offline
fixtures, including tests of the actual adapter parsing and selection logic.

Use Python 3.11+, pandas, pyarrow, shapely, pyproj, numpy, LightGBM,
scikit-learn, requests, pystac-client, rasterio, and pytest. Resolve and record
compatible dependency versions during scaffolding. Earth Engine is optional.

Pilot defaults: Heron Island; EPSG:32756; 75 m cells; DHW risk threshold 4;
AIMS matching window 14 days; maximum unusable pixel fraction 0.5;
confirmed weight 1.0 and weak-label weight 0.5. Validate config ranges.
Start with the 2020 pilot. Do not assume 2016 L2A coverage.

## Corrected data contracts (version 2)

Store schema version, config, source identifiers, processing settings, software
versions, and observation interval in a manifest next to each dataset.
Use explicit Arrow schemas so all-null and empty outputs retain their types.
Use Hive partitioning by `reef_cell_id`, with a new immutable run directory
for each publication. Stage and validate before publication; never partially
replace a successful run.

Training rows have these fields:

| Field | Meaning |
|---|---|
| `reef_cell_id`, `date` | Unique cell and actual acquisition date |
| `ndci_like_index` | Nullable finite green/red normalized difference; experimental feature, not a validated bleaching measurement |
| `index_shift` | Nullable difference between consecutive three-observation window means |
| `dhw` | Finite nonnegative DHW for acquisition date |
| `cloud_cover_fraction` | Nullable fraction of unusable pixels in the reef-cell intersection; includes clouds, shadow and nodata |
| `data_quality` | `ok` or `insufficient` |
| `dhw_risk_flag` | DHW >= configured threshold; weak proxy only |
| `confirmed_label` | `bleached`, `healthy`, or null |
| `survey_date` | Matched source survey date, nullable; never inferred from imagery date |
| `survey_id` | Stable source survey-record identity, nullable |
| `source_scene_id` | Scene identity for provenance and overlap checks, nullable for missing imagery |
| `feature_start_date` | Earliest imagery date contributing to this row's features |

`confirmed_label`, `survey_date`, and `survey_id` must be populated together.
No matching survey is valid; a failed or malformed survey import is an error.
Missing expected observations may use the requested date with null scene ID,
`insufficient` quality, and null imagery features. Actual scenes must not be
repeated under different requested dates.

Prediction rows have `reef_cell_id`, `date`, `probability`, `predicted_label`,
`top_contributing_features`, `confidence_band`, and **`data_quality`**.
For insufficient imagery: probability and label are null, contributions are
an empty list, confidence is `unavailable`, and quality is `insufficient`.
For usable imagery: probability is in [0,1], label is thresholded at 0.5,
and confidence uses distance from 0.5 (<0.1 low, <0.3 medium, otherwise high).
These bands are descriptive model scores, not calibrated uncertainty.
Contributions are up to three structs containing feature name, nullable input
value, and signed SHAP contribution in model raw-score units.

## Validation and missing-data rules

Hold out one **source survey date** at a time. Test only AIMS-confirmed,
quality-eligible rows. Keep all records sharing that survey date together,
including observations matched within the 14-day window.

Purge training rows, including weak labels, if they share a held-out survey ID,
a held-out source scene, or a feature interval overlapping any test feature
interval expanded by 14 days in either direction. Apply this across the pilot
reef, not just individual cells. Require source provenance for usable rows.
This conservative split can leave too little training data; report that fact.
Do not silently loosen the split to manufacture validation scores.

Return one structured result per source survey date: date, status, reason,
train/test counts, class counts, and nullable ROC AUC. Status is `scored` or
`skipped`. Empty training/test sets and one-class training/test sets receive
explicit reasons and null AUC. An empty or entirely skipped evaluation must
report `not_evaluable`, never success. Score only confirmed test labels.
Because one survey can label several acquisitions, also report unique survey
counts and weight each survey record equally in evaluation (each of its n
matched rows receives weight 1/n). Report the DHW threshold baseline on the
same held-out confirmed rows to expose proxy-only performance.

Quality filtering precedes label assignment: even a confirmed label cannot
make unusable imagery trainable. Keep excluded rows in the raw dataset and
prediction output. Never fill missing reflectance or index history with zero.
A null shift during valid history warmup may use LightGBM's native missing
value support; missing current reflectance or unknown quality must abstain.
Zero-denominator and nonfinite spectral inputs yield missing, not zero.

## Task 1: Scaffolding

- [x] Create pyproject.toml, package layout, README, and offline test setup.
- [x] Resolve dependencies in an isolated environment and record versions.
- [x] Document test and batch-run commands, input formats, and output schemas.

## Task 2: Region config

- [x] Implement immutable RegionConfig with defaults above, date range,
  boundary path, provider/collection, and max history gap (default 14 days).
- [x] Validate dates, positive cell size, thresholds and overlap fractions.

## Task 3: Reef grid

- [x] Load Polygon/MultiPolygon GeoJSON and reject invalid/empty geometry.
- [x] Generate stable IDs from a fixed region grid origin in projected meters.
- [x] Retain cells with at least 10% overlap; aggregate imagery only over the
  cell's reef intersection. Export this geometry and the grid definition.
- [x] Test exact projected synthetic dimensions, coverage, ID stability and
  clipped coastal cells. Do not assume a WGS84 rectangle is exactly 300 m.
- [x] Mark the supplied test rectangle as synthetic, never an Atlas boundary.

## Task 4: Spectral features

- [x] Implement normalized difference with missing/nonfinite handling.
- [x] Compute shifts from chronological, unique, valid acquisitions only.
  Require six valid acquisitions for two three-observation windows; return
  null if any consecutive gap exceeds configured max history gap.
- [x] Track feature_start_date from contributing observations. Never include
  cloudy observations in history or use future acquisitions for a row.
- [x] Test zero denominator, nonfinite values, warmup, gaps and ordering.

## Task 5: Sentinel-2 imagery adapter

- [x] Define provider-neutral imagery interface accepting reef geometry and
  time range and returning scene IDs, actual timestamps, cell statistics,
  quality information, and feature provenance.
- [x] Implement Earth Search STAC with public HTTPS L2A COG assets as the
  default. Search once per reef/date range, paginate, cache and deduplicate.
- [x] Read only bounding raster windows, apply pixel masks, and aggregate
  over reef intersections. Use green/red and SCL; do not treat scene-wide
  cloud percentage or pixel cloud probability as cell cloud fraction.
- [x] Apply documented raster scale/offset and nodata metadata; align SCL
  masks to reflectance grids with nearest-neighbor resampling. Preserve water
  pixels. Record the exact unusable SCL classes in the run manifest.
- [x] Distinguish empty search/masked imagery from network or parsing errors;
  the latter fail the run with a useful error. Reject requester-pays assets
  in the default anonymous configuration.
- [x] Test actual adapter logic with fake STAC pages and small local rasters:
  pagination, empty search, masks, scaling, deduplication, and partial cells.
- [ ] Optional follow-up (not required for Build 1): Earth Engine adapter
  following the same contract. Public STAC is the implemented provider.

## Task 6: NOAA DHW adapter

- [x] Verified and implemented NOAA CoastWatch `noaacrwdhwDaily` with variable
  `degree_heating_week`; the earlier NOAA_DHW endpoint was not used.
- [x] Inject HTTP transport; parse by column names, not last-column position.
- [x] Match acquisition date and location, record source provenance, reject
  nonfinite/negative values and fail missing required DHW joins.
- [x] Test response errors, missing rows, malformed values and coordinates.

## Task 7: AIMS labels

- [x] Normalize survey records with stable survey_id, survey_date, label,
  and defensible spatial mapping to reef cells. Document upstream fields.
- [x] Match within 14 days and return the full matched record, not just label.
- [x] Reject malformed imports. Handle tied conflicting labels explicitly
  as ambiguous/unlabeled with diagnostic provenance; never choose CSV order.
- [x] Test nearest match, no match, ties, invalid labels and retained identity.

## Task 8: Pipeline

- [x] Join by actual acquisition date; retain source and feature provenance.
- [x] Preserve insufficient imagery rows with nullable features and labels.
- [x] Fail required-source errors; never publish partial output.
- [x] Write versioned Parquet and manifest through a staging directory.
- [x] Test survey provenance, unknown quality, duplicate acquisitions, empty
  outputs, Parquet null types, failure isolation and schema consistency.

## Task 9: Training labels

- [x] Filter quality first; then prefer confirmed labels at weight 1.0,
  otherwise assign DHW weak labels at weight 0.5.
- [x] Use explicit null checks (including pandas null values), validate labels,
  and preserve original raw rows for full-coverage prediction output.
- [x] Test confirmed-but-insufficient rows, nulls, invalid labels and weights.

## Task 10: Training and validation

- [x] Use features: ndci_like_index, index_shift, dhw, cloud_cover_fraction.
  Preserve null shift; reject invalid current features. Set random seed.
- [x] Require two training classes. Persist fitted model, config and metrics.
- [x] Implement source-survey-date splits and purging as specified above.
- [x] Test different imagery dates tied to one survey, multiple surveys on
  one date, nearby weak rows, shared scenes, overlapping history and distant
  eligible training rows. Assert no prohibited train/test overlap.
- [x] Construct scored-fold fixtures with both classes on each survey date,
  sufficiently separated dates, and separate single-class fixtures.
- [x] Assert one result per date, explicit skips, all-skipped not_evaluable,
  survey weighting, confirmed-only scoring, and baseline reporting.

## Task 11: Predictions

- [x] Generate outputs for the full raw frame. Send only eligible rows to
  model prediction and SHAP methods; rejoin results to original row order.
- [x] Abstain on insufficient rows, even with high DHW or confirmed labels.
- [x] Emit contribution magnitudes and signs; preserve explicit Arrow schema.
- [x] Test mixed quality, all-insufficient and empty frames, confidence
  boundaries, null round trips and no model calls for unusable imagery.

## Task 12: Integration and acceptance

- [x] Run full chain offline with fake sources and enough acquisitions for
  rolling features. Include both labels on separated source survey dates.
- [x] Reload raw Parquet, train, validate, and predict over the entire raw
  dataset. Include confirmed cloudy and missing-image observations.
- [x] Assert exact schemas, raw/prediction key equality, null abstentions,
  nonempty scored folds in the balanced fixture, and honest skipped folds.
- [x] Run pytest with network disabled; all tests must pass.
- [x] Separately document optional live-source smoke checks. Passing fake
  tests does not establish live-source compatibility or bleaching accuracy.

## Free imagery access decision and verification

Use Earth Search public HTTPS Sentinel-2 L2A assets for the initial 2020
pilot. Earth Engine authentication is no longer a prerequisite. Local raster
processing replaces Earth Engine compute; download only reef windows.

Live check on 2026-09-04: anonymous Earth Search search for bounding box
[151.90, -23.46, 151.94, -23.42], 2020-01-01 through 2020-05-01 returned
97 matching items. Scene S2A_56KLV_20200429_0_L2A exposed green/red/SCL assets;
a 16-byte HTTP range read of its green COG returned HTTP 206 without login.
This verifies catalog/file access, not clear-water pixels or model suitability.
The same query for 2016 returned zero items in sentinel-2-l2a. Do not silently
mix L1C and L2A to fill that gap. Investigate reprocessed L2A archives or a
separately validated atmospheric-correction workflow if 2016 is essential.

Alternatives: Microsoft Planetary Computer provides free imagery access with
anonymous SAS token signing; Copernicus Data Space provides free Sentinel
products with registration for authenticated downloads. Check actual regional
and historical coverage before switching provider. Switching mirrors solves
access, not clouds or missing acquisitions.

Sources checked 2026-09-04:
- https://github.com/Element84/earth-search/blob/main/README.md
- https://element84.com/earth-search/examples
- https://planetarycomputer.microsoft.com/docs/concepts/sas/
- https://sentinels.copernicus.eu/sentinel-data-access/registration
- https://documentation.dataspace.copernicus.eu/APIs/STAC.html
