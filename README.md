# Coral bleaching detection — Build 1 + map POC

An offline Python pipeline, LightGBM model, and local read-only map for a Heron Island pilot.
It reads public Sentinel-2 L2A imagery, joins NOAA heat stress and normalized
field survey labels, and writes versioned Parquet predictions with signed
feature contributions. The proof-of-concept map reads a completed local run;
it does not add a database or write path.

The software is tested with synthetic data and small raster fixtures. The
synthetic model is a demonstration, not a validated bleaching detector.
Live source access has been exercised separately; real field evaluation still
requires an authoritative reef boundary and suitable independent surveys.

## Install

Use Python 3.11 or newer (verified here with Python 3.12). From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -e '.[dev]'
python scripts/configure_macos_openmp.py
pytest -q
```

The lock file records the versions used for this build. On macOS, the helper
configures LightGBM to find the OpenMP library bundled with the installed
scikit-learn wheel. It only changes the active environment's LightGBM binary.
On Linux the helper does nothing. If the bundled library is unavailable,
install your platform's OpenMP runtime. Do not copy the supplied environment
between machines; recreate it with the commands above.

NumPy is constrained below 2.3 and affine below 3 to avoid compatibility
warnings with the tested pandas/rasterio stack. All automated tests prohibit
Python socket access; raster tests read local GeoTIFFs through injected clients.

## Offline demonstration

```sh
coral-bleaching demo --output runs/demo-001
```

This command requires no account, credentials, or network. It generates
synthetic observations and survey labels across three separated survey dates,
trains a model, evaluates folds, and preserves cloudy/missing rows as null
predictions. Use a new output directory for every run; existing runs are never
overwritten. The synthetic validation scores demonstrate software behavior
and must not be reported as field performance.

Each published run contains:

- `training_dataset.parquet/`: Hive partitions by reef_cell_id, schema v2.
- `predictions.parquet`: one row per raw cell/date, including abstentions.
- `model.txt`: fitted LightGBM model (absent for an entirely insufficient run).
- `validation.json`: every fold, skips and reasons, and a DHW baseline.
- `grid.geojson`: the clipped cells and their stable identifiers.
- `manifest.json`: configuration, source provenance, diagnostics, library
  versions, and SHA-256 checksums of the other files.

## Local prediction map

Generate the synthetic demonstration above, then serve the runs directory:

```sh
coral-bleaching ui --runs-root runs --run demo-001
```

Open `http://127.0.0.1:8000`. The map uses the latest observation for each
reef cell. Click a cell to inspect its bleaching probability, label, confidence
band, data quality, and signed model contributions. Grey dashed cells are
observations where the model abstained because imagery was insufficient.

Leaflet and OpenStreetMap tiles load from public CDNs, so the map background
requires network access. The run data stays on the local machine. The included
`demo-001` values are synthetic and do not represent field performance.

A run is staged, reloaded, checked, and atomically renamed into place only
when complete. A one-class usable training set fails with an explicit error;
an all-insufficient run can publish abstentions without a model. An entirely
skipped evaluation is `not_evaluable`, never a successful validation claim.

## Live pilot

No Earth Engine login is needed. The default adapter searches Earth Search
once per reef/time interval, follows STAC pagination, reads HTTPS COG windows,
and selects the best usable-pixel fraction per cell/acquisition day. Raster
blocks are cached by GDAL during processing; searches are cached in memory.
This is a bounded pilot implementation, not a distributed imagery processor.

1. Obtain an authoritative WGS84 Polygon/MultiPolygon reef boundary. The
   demonstration rectangle is synthetic and is not an Allen Coral Atlas export.
2. Obtain survey observations that actually classify bleaching/healthy state.
   Percent coral cover alone is not a bleaching label. Normalize as below.
3. Copy `examples/heron_island.json`, point its boundary path to the real file,
   and set an acquisition interval supported by the public L2A catalog.
4. Run:

```sh
coral-bleaching run --config examples/heron_island.json \
  --surveys /path/to/normalized-surveys.csv --output runs/heron-001
```

Normalized survey CSV columns are `survey_id,survey_date,label` plus either
`longitude,latitude` or `reef_cell_id`. Dates use YYYY-MM-DD, labels must be
`bleached` or `healthy`, and survey_id must uniquely identify one spatial
observation. Coordinate records must fall unambiguously in a single clipped
cell; alternatively pre-map records to this run's exported grid with a
scientifically justified spatial join. Keep the input file's source citation
and normalization method with the study. Inputs are hashed in the manifest.
Do not enlarge survey footprints or invent per-cell labels to get more rows.

Nearest surveys within 14 days are matched with original date/ID retained.
Equidistant conflicting labels are left unconfirmed and recorded in manifest
diagnostics. Imports with malformed dates/labels or unknown cells fail.

NOAA uses the public `noaacrwdhwDaily` ERDDAP dataset and its
`degree_heating_week` variable. Queries are cached per 0.05-degree pixel/day;
returned dates and grid coordinates are checked. This coarse thermal stress
is a weak label and covariate, not independent localized bleaching evidence.
Missing values, invalid data, and required-source HTTP failures fail the run.

## Inference using a saved model

```sh
coral-bleaching predict --dataset runs/heron-001/training_dataset.parquet \
  --model runs/heron-001/model.txt --output runs/heron-predictions.parquet
```

The model uses LightGBM's text format, not pickle. Feature names/order must
match the v2 contract. Invalid current reflectance or insufficient quality
never enters prediction. Missing rolling history stays null; it is not zero.

## Data quality and evaluation

Green/red reflectance uses each asset's explicit scale/offset. SCL masks are
aligned by nearest neighbor and combined with nodata masks over pixel centers
inside each clipped reef cell. SCL classes 0,1,3,7,8,9,10,11 are unusable; class
6 (water) is retained. `cloud_cover_fraction` is the fraction of all unusable
pixels, including shadow/nodata, not just clouds. No represented pixel centers
or no valid pixels produces insufficient quality. The >0.5 default quality
threshold is configurable and recorded in the run.

The experimental green/red normalized difference and its rolling change need
scientific validation for reef water conditions. They are not named or treated
as an established coral bleaching index. A zero denominator is missing.
Rolling change compares two three-acquisition means, excludes bad imagery,
requires no gap greater than 14 days, and records its earliest input date.

Training first excludes insufficient rows, then weights confirmed labels at
1.0 and DHW-only labels at 0.5. Cross-validation holds out original survey
dates and tests only confirmed labels. Training observations sharing held-out
surveys/scenes or overlapping feature intervals (plus 14-day embargo) are
purged across the reef, including weak labels. Repeated observations of each
survey receive total evaluation weight 1. Each fold reports a reason if AUC
cannot be calculated, and scored folds report the DHW baseline on identical
confirmed test rows. Conservative purging may leave too little data to score.

SHAP contributions are signed model raw-score contributions, with feature
values included. Confidence bands describe distance from 0.5; they are not
calibrated confidence intervals. Insufficient rows have null probability and
label, empty contributions, and confidence `unavailable`.

## Live access check and remaining data work

```sh
python scripts/smoke_public_sources.py --output live-source-check.json
```

This optional network check processes a small access-test rectangle near
Heron Island on 2020-04-29 and fetches NOAA DHW. On this build's check the
selected scene was fully masked in that rectangle, yielding null reflectance
and unusable fraction 1.0; NOAA returned 6.89 Celsius-weeks. That confirms
source/adapter operation and honest missing-data handling, not usable imagery
for the study. The earlier broader catalog search found 97 scenes in
January–April 2020; it did not establish how many contain clear reef pixels.

The same earlier search found no 2016 scenes in the selected L2A collection.
2016 remains contingent on compatible archive coverage; do not substitute
L1C without separately validated atmospheric correction. Alternative mirrors
cannot recover cloud-obscured observations. Establish usable multi-date
coverage, real survey availability/spatial resolution, and a defensible reef
boundary before interpreting live model results.

Sources:
- [Earth Search API and asset conventions](https://github.com/Element84/earth-search/blob/main/README.md)
- [NOAA daily DHW dataset](https://coastwatch.noaa.gov/erddap/griddap/noaacrwdhwDaily.html)
- [NOAA DHW methodology](https://coralreefwatch.noaa.gov/product/5km/methodology.php)

See `docs/superpowers/` for the design and Build 1 acceptance plan.
