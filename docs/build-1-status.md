# Build 1 delivery

Implemented on September 4, 2026. Prepared for review on the `build-1` branch.

## Implemented

- Installable Python package with demo, live batch and saved-model prediction commands.
- Fixed projected reef grid, stable cell IDs, and clipped reef geometry.
- Public Sentinel-2 STAC discovery and windowed raster processing with explicit
  scaling, water-preserving quality masks, acquisition provenance and deduplication.
- NOAA daily DHW adapter, response validation, pixel/date cache and provenance.
- Normalized survey import, coordinate-to-cell matching, retained survey identity,
  nearest-date joins and explicit conflicting-label diagnostics.
- Nullable spectral features, quality-first training labels and weighted LightGBM.
- Original-survey-date validation with temporal/scene purging, per-survey weights,
  explicit skipped folds and independently scored DHW baselines.
- Full-coverage predictions with abstention and signed SHAP contributions.
- Typed Parquet, a reloadable model, grid export, manifest and staged publication.
- Dependency lock, macOS OpenMP setup helper, offline tests and CI configuration.

## Validation performed

38 offline tests passed. Formatting, lint, dependency consistency and git diff
whitespace checks passed. CI configuration is supplied; a remote CI run has not
been triggered. Local validation used macOS and Python 3.12.

The CLI demo produced 72 cell/date rows, including 6 null abstentions, and three
scored synthetic validation folds. A second CLI invocation loaded the saved
model and dataset and generated predictions. Model reload equivalence is also
tested. Synthetic validation scores are not real-world accuracy estimates.

The live public-source script successfully read and processed Sentinel-2 scene
S2A_56KLV_20200429_1_L2A for a small access-test rectangle. It returned unusable
fraction 1.0 and null reflectance; this is a source-compatibility check, not a
clear-reef-image result. NOAA returned DHW 6.89 for the same date. The selected
NOAA endpoint is https://coastwatch.noaa.gov/erddap/griddap/noaacrwdhwDaily.json
with variable degree_heating_week. An application User-Agent is required for
the working Python request used here.

## Remaining study inputs and limitations

Before a real field-evaluated run, obtain suitable bleaching observations
(classifying bleached/healthy state, not just coral cover), their spatial
mapping, and usable imagery across enough independent survey dates. The Heron
Island Reef boundary is now bundled (`data/heron_island_reef_boundary.geojson`,
from an Allen Coral Atlas CC BY 4.0 export; see `data/sources/README.md`). No
real bleaching-survey data is bundled and no live trained bleaching model is
claimed. The 2016 L2A archive gap remains unresolved; the default pilot targets
2020. Confidence bands are uncalibrated descriptive scores. Earth Engine is an
optional future adapter.

Run commands and detailed data contracts are in README.md. The source archive
contains the package, tests, documentation and examples; the synthetic demo is
provided separately.
