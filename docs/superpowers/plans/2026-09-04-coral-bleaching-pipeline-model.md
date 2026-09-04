# Coral Bleaching Data Pipeline & Detection Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **This plan is written for execution by Codex, not Claude.** Every task is
> self-contained: exact file paths, exact code, exact shell commands, exact
> library versions. Do not assume access to any prior conversation — every
> fact you need is either in this document or in the spec it links to.

**Goal:** Implement Module 1 (data pipeline) and Module 2 (detection model)
from the coral bleaching detection design: a batch pipeline that builds a
labeled per-reef-cell/date dataset from Sentinel-2, NOAA DHW, and AIMS
survey data, and a gradient-boosted-tree model that consumes it to produce
interpretable bleaching-risk predictions.

**Architecture:** A pure-Python package (`coral_bleaching`) with no web
server, no database, and no UI. Module 1 writes a partitioned Parquet
dataset; Module 2 reads that dataset and writes a predictions Parquet
file. External data sources (Google Earth Engine, NOAA ERDDAP, AIMS CSV
export) are wrapped behind small interfaces that accept injectable
clients/functions, so every unit test runs offline with fakes — no network
access or credentials required to pass the test suite.

**Tech Stack:** Python 3.11+, pandas, pyarrow, shapely, pyproj, lightgbm,
scikit-learn, requests, earthengine-api, pytest, pytest-mock.

**Spec:** `docs/superpowers/specs/2026-09-04-coral-bleaching-detection-design.md`

## Global Constraints

- Pilot region: Heron Island Reef, southern Great Barrier Reef (UTM zone
  for meter-based grid math: EPSG:32756).
- Grid cell size: 75 meters (spec range was 50-100m; 75m is the concrete
  value used throughout this plan).
- DHW weak-label risk threshold: DHW >= 4.0.
- AIMS confirmed-label matching window: 14 days.
- Cloud-cover hard threshold for `data_quality`: cloud_cover_fraction >
  0.5 is `"insufficient"`.
- Sample weights: AIMS-confirmed label = 1.0, DHW-only weak label = 0.5.
- Module 1 output: `training_dataset.parquet` (Hive-partitioned by
  `reef_cell_id`). Module 2 output: `predictions.parquet`.
- No network calls in tests. Every external source (GEE, NOAA, AIMS) is
  accessed through an injectable client/function so tests use fakes.
- Validation method: leave-one-survey-date-out cross-validation (never a
  random split).
- Python package lives under `src/coral_bleaching/`; tests live under
  `tests/`, mirroring the package structure.

---

## Prerequisites (human setup, cannot be automated)

Before Task 5 (the real Google Earth Engine client) can be exercised
against live data, a human must:

1. Sign up for Google Earth Engine access at
   https://code.earthengine.google.com (approval is not instant).
2. Run `earthengine authenticate` locally once access is granted.

This plan's tasks do not depend on this being done yet — every task's
tests use fake/injected clients. Only running the real pipeline against
live Sentinel-2 data (outside the scope of this plan's tasks) needs it.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/coral_bleaching/__init__.py`
- Create: `src/coral_bleaching/sources/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_package_scaffolding.py`

**Interfaces:**
- Produces: an installable package named `coral_bleaching`, importable as
  `import coral_bleaching`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "coral-bleaching-detection"
version = "0.1.0"
description = "Coral bleaching detection pipeline and model (Modules 1-2)"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2,<3",
    "pyarrow>=15,<16",
    "shapely>=2.0,<3",
    "pyproj>=3.6,<4",
    "lightgbm>=4.3,<5",
    "scikit-learn>=1.4,<2",
    "requests>=2.31,<3",
    "earthengine-api>=0.1.400,<0.2",
]

[project.optional-dependencies]
dev = ["pytest>=8,<9", "pytest-mock>=3.14,<4"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty package `__init__.py` files**

```bash
mkdir -p src/coral_bleaching/sources
touch src/coral_bleaching/__init__.py
touch src/coral_bleaching/sources/__init__.py
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 3: Write the scaffolding smoke test**

`tests/test_package_scaffolding.py`:

```python
def test_package_is_importable():
    import coral_bleaching  # noqa: F401


def test_sources_subpackage_is_importable():
    import coral_bleaching.sources  # noqa: F401
```

- [ ] **Step 4: Create and activate a virtual environment, install the package**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_package_scaffolding.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/coral_bleaching tests/__init__.py tests/test_package_scaffolding.py
git commit -m "chore: scaffold coral_bleaching package"
```

---

### Task 2: Region config

**Files:**
- Create: `src/coral_bleaching/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `RegionConfig` dataclass, used by every later task that needs
  region-specific constants (cell size, thresholds, date range).

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
from datetime import date

from coral_bleaching.config import RegionConfig


def test_region_config_defaults():
    region = RegionConfig(
        reef_id="heron_island",
        boundary_geojson_path="tests/fixtures/heron_island_boundary.geojson",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 3, 1),
    )
    assert region.cell_size_meters == 75.0
    assert region.dhw_risk_threshold == 4.0
    assert region.aims_label_window_days == 14
    assert region.cloud_cover_drop_threshold == 0.5
    assert region.utm_epsg == "EPSG:32756"


def test_region_config_overrides():
    region = RegionConfig(
        reef_id="heron_island",
        boundary_geojson_path="tests/fixtures/heron_island_boundary.geojson",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 3, 1),
        cell_size_meters=100.0,
    )
    assert region.cell_size_meters == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coral_bleaching.config'`

- [ ] **Step 3: Write minimal implementation**

`src/coral_bleaching/config.py`:

```python
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class RegionConfig:
    reef_id: str
    boundary_geojson_path: str
    start_date: date
    end_date: date
    cell_size_meters: float = 75.0
    dhw_risk_threshold: float = 4.0
    aims_label_window_days: int = 14
    cloud_cover_drop_threshold: float = 0.5
    utm_epsg: str = "EPSG:32756"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/coral_bleaching/config.py tests/test_config.py
git commit -m "feat: add RegionConfig"
```

---

### Task 3: Grid cell generation

**Files:**
- Create: `src/coral_bleaching/grid.py`
- Create: `tests/fixtures/heron_island_boundary.geojson`
- Test: `tests/test_grid.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure geometry).
- Produces: `ReefCell` dataclass (`reef_cell_id: str`, `geometry_wgs84:
  shapely.geometry.Polygon`, `centroid_lat: float`, `centroid_lon:
  float`), `load_boundary(path: str) -> shapely.geometry.base.BaseGeometry`,
  `generate_grid(boundary_wgs84, cell_size_meters, reef_id,
  min_overlap_fraction=0.1, utm_epsg="EPSG:32756") -> list[ReefCell]`.
  Later tasks (5, 8) consume `ReefCell.centroid_lon` /
  `ReefCell.centroid_lat` / `ReefCell.reef_cell_id`.

- [ ] **Step 1: Create the boundary fixture**

A ~300m x 300m rectangle near Heron Island Reef (approx. -23.4423,
151.9148), in WGS84 (EPSG:4326):

`tests/fixtures/heron_island_boundary.geojson`:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"name": "heron_island_test_boundary"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [151.9148, -23.4423],
          [151.917739, -23.4423],
          [151.917739, -23.439587],
          [151.9148, -23.439587],
          [151.9148, -23.4423]
        ]]
      }
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

`tests/test_grid.py`:

```python
from coral_bleaching.grid import generate_grid, load_boundary


def test_load_boundary_returns_polygon_with_positive_area():
    boundary = load_boundary("tests/fixtures/heron_island_boundary.geojson")
    assert boundary.area > 0


def test_generate_grid_produces_expected_cell_count():
    boundary = load_boundary("tests/fixtures/heron_island_boundary.geojson")
    cells = generate_grid(boundary, cell_size_meters=100.0, reef_id="heron_island")
    # The fixture is an exact ~300m x 300m rectangle, so a 100m grid
    # should tile it into exactly 9 cells with no partial-overlap remainder.
    assert len(cells) == 9


def test_generate_grid_cell_ids_are_unique():
    boundary = load_boundary("tests/fixtures/heron_island_boundary.geojson")
    cells = generate_grid(boundary, cell_size_meters=100.0, reef_id="heron_island")
    ids = [cell.reef_cell_id for cell in cells]
    assert len(ids) == len(set(ids))


def test_generate_grid_centroids_fall_within_boundary_bounds():
    boundary = load_boundary("tests/fixtures/heron_island_boundary.geojson")
    minx, miny, maxx, maxy = boundary.bounds
    cells = generate_grid(boundary, cell_size_meters=100.0, reef_id="heron_island")
    for cell in cells:
        assert minx - 0.001 <= cell.centroid_lon <= maxx + 0.001
        assert miny - 0.001 <= cell.centroid_lat <= maxy + 0.001
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_grid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coral_bleaching.grid'`

- [ ] **Step 4: Write minimal implementation**

`src/coral_bleaching/grid.py`:

```python
import json
import math
from dataclasses import dataclass

from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform, unary_union


@dataclass(frozen=True)
class ReefCell:
    reef_cell_id: str
    geometry_wgs84: BaseGeometry
    centroid_lat: float
    centroid_lon: float


def load_boundary(path: str) -> BaseGeometry:
    with open(path) as f:
        geojson = json.load(f)
    geoms = [shape(feature["geometry"]) for feature in geojson["features"]]
    return unary_union(geoms)


def _reproject(geom: BaseGeometry, src_epsg: str, dst_epsg: str) -> BaseGeometry:
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return shapely_transform(transformer.transform, geom)


def generate_grid(
    boundary_wgs84: BaseGeometry,
    cell_size_meters: float,
    reef_id: str,
    min_overlap_fraction: float = 0.1,
    utm_epsg: str = "EPSG:32756",
) -> list[ReefCell]:
    boundary_utm = _reproject(boundary_wgs84, "EPSG:4326", utm_epsg)
    minx, miny, maxx, maxy = boundary_utm.bounds

    n_cols = math.ceil((maxx - minx) / cell_size_meters)
    n_rows = math.ceil((maxy - miny) / cell_size_meters)

    cells: list[ReefCell] = []
    for row in range(n_rows):
        for col in range(n_cols):
            cx0 = minx + col * cell_size_meters
            cy0 = miny + row * cell_size_meters
            cell_box = box(cx0, cy0, cx0 + cell_size_meters, cy0 + cell_size_meters)

            intersection = cell_box.intersection(boundary_utm)
            if intersection.is_empty:
                continue
            if (intersection.area / cell_box.area) < min_overlap_fraction:
                continue

            cell_wgs84 = _reproject(cell_box, utm_epsg, "EPSG:4326")
            centroid = cell_wgs84.centroid
            cells.append(
                ReefCell(
                    reef_cell_id=f"{reef_id}_r{row}_c{col}",
                    geometry_wgs84=cell_wgs84,
                    centroid_lat=centroid.y,
                    centroid_lon=centroid.x,
                )
            )
    return cells
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_grid.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/coral_bleaching/grid.py tests/fixtures/heron_island_boundary.geojson tests/test_grid.py
git commit -m "feat: add reef grid-cell generation"
```

---

### Task 4: Spectral index math

**Files:**
- Create: `src/coral_bleaching/spectral.py`
- Test: `tests/test_spectral.py`

**Interfaces:**
- Consumes: nothing (pure math).
- Produces: `normalized_difference(band_a: float, band_b: float) ->
  float`, `rolling_index_shift(values: list[float], window: int) ->
  float`. Task 8 (pipeline orchestration) calls both.

- [ ] **Step 1: Write the failing test**

`tests/test_spectral.py`:

```python
import pytest

from coral_bleaching.spectral import normalized_difference, rolling_index_shift


def test_normalized_difference_basic():
    assert normalized_difference(0.6, 0.2) == pytest.approx(0.5)


def test_normalized_difference_zero_denominator_returns_zero():
    assert normalized_difference(0.0, 0.0) == 0.0


def test_rolling_index_shift_detects_upward_shift():
    # window=2: prior mean of [0.1, 0.1] = 0.1, recent mean of [0.3, 0.3] = 0.3
    values = [0.1, 0.1, 0.3, 0.3]
    assert rolling_index_shift(values, window=2) == pytest.approx(0.2)


def test_rolling_index_shift_raises_when_not_enough_history():
    with pytest.raises(ValueError):
        rolling_index_shift([0.1, 0.2], window=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_spectral.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coral_bleaching.spectral'`

- [ ] **Step 3: Write minimal implementation**

`src/coral_bleaching/spectral.py`:

```python
def normalized_difference(band_a: float, band_b: float) -> float:
    denominator = band_a + band_b
    if denominator == 0:
        return 0.0
    return (band_a - band_b) / denominator


def rolling_index_shift(values: list[float], window: int) -> float:
    if len(values) < window * 2:
        raise ValueError(
            f"Need at least {window * 2} values to compute a rolling shift "
            f"with window={window}, got {len(values)}"
        )
    recent = sum(values[-window:]) / window
    prior = sum(values[-2 * window:-window]) / window
    return recent - prior
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_spectral.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/coral_bleaching/spectral.py tests/test_spectral.py
git commit -m "feat: add spectral index math"
```

---

### Task 5: Sentinel-2 (Google Earth Engine) data source

**Files:**
- Create: `src/coral_bleaching/sources/gee.py`
- Test: `tests/test_sources_gee.py`

**Interfaces:**
- Consumes: nothing from earlier tasks directly (a `ReefCell`'s
  `centroid_lon`/`centroid_lat` from Task 3 are passed in by the caller in
  Task 8, not imported here).
- Produces: `GEEClient` protocol with method `get_reflectance(self, lon:
  float, lat: float, obs_date: datetime.date) -> dict | None` returning
  `{"band_green": float, "band_red": float, "band_rededge": float,
  "cloud_cover_fraction": float}` or `None` if no image exists for that
  date/location. `EarthEngineClient` is the real implementation (requires
  live GEE access; not exercised by tests). Task 8 depends only on the
  `get_reflectance` method signature and return shape, and can be handed
  any fake implementing it.

- [ ] **Step 1: Write the failing test**

`tests/test_sources_gee.py`:

```python
from datetime import date

from coral_bleaching.sources.gee import GEEClient


class FakeGEEClient:
    """Test double satisfying the GEEClient protocol."""

    def __init__(self, response):
        self._response = response

    def get_reflectance(self, lon: float, lat: float, obs_date: date):
        return self._response


def test_fake_client_satisfies_protocol():
    client: GEEClient = FakeGEEClient(
        {
            "band_green": 0.05,
            "band_red": 0.03,
            "band_rededge": 0.04,
            "cloud_cover_fraction": 0.1,
        }
    )
    result = client.get_reflectance(151.9148, -23.4423, date(2020, 2, 1))
    assert result["band_green"] == 0.05
    assert result["cloud_cover_fraction"] == 0.1


def test_fake_client_can_return_none_for_missing_image():
    client: GEEClient = FakeGEEClient(None)
    result = client.get_reflectance(151.9148, -23.4423, date(2020, 2, 1))
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources_gee.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coral_bleaching.sources.gee'`

- [ ] **Step 3: Write minimal implementation**

`src/coral_bleaching/sources/gee.py`:

```python
from datetime import date
from typing import Protocol, TypedDict


class ReflectanceObservation(TypedDict):
    band_green: float
    band_red: float
    band_rededge: float
    cloud_cover_fraction: float


class GEEClient(Protocol):
    def get_reflectance(
        self, lon: float, lat: float, obs_date: date
    ) -> ReflectanceObservation | None: ...


class EarthEngineClient:
    """Real Google Earth Engine-backed implementation.

    Requires `earthengine authenticate` to have been run locally
    (see plan Prerequisites). Not exercised by the test suite.
    """

    def __init__(self):
        import ee

        ee.Initialize()
        self._ee = ee

    def get_reflectance(
        self, lon: float, lat: float, obs_date: date
    ) -> ReflectanceObservation | None:
        ee = self._ee
        point = ee.Geometry.Point([lon, lat])
        start = ee.Date(obs_date.isoformat())
        end = start.advance(5, "day")

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate(start, end)
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        image = collection.first()

        values = (
            image.select(["B3", "B4", "B6", "MSK_CLDPRB"])
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10)
            .getInfo()
        )
        if not values or values.get("B3") is None:
            return None

        return {
            "band_green": values["B3"] / 10000.0,
            "band_red": values["B4"] / 10000.0,
            "band_rededge": values["B6"] / 10000.0,
            "cloud_cover_fraction": (values.get("MSK_CLDPRB") or 0) / 100.0,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources_gee.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/coral_bleaching/sources/gee.py tests/test_sources_gee.py
git commit -m "feat: add GEE reflectance client interface"
```

---

### Task 6: NOAA DHW data source

**Files:**
- Create: `src/coral_bleaching/sources/noaa.py`
- Test: `tests/test_sources_noaa.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `fetch_dhw(lon: float, lat: float, obs_date: datetime.date,
  http_get=requests.get) -> float | None`. Task 8 calls this with an
  injected fake `http_get` in tests and the real `requests.get` default in
  production.

- [ ] **Step 1: Write the failing test**

`tests/test_sources_noaa.py`:

```python
from datetime import date

from coral_bleaching.sources.noaa import fetch_dhw


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_fetch_dhw_returns_parsed_value():
    payload = {"table": {"rows": [["2020-02-01T12:00:00Z", -23.4423, 151.9148, 5.5]]}}

    def fake_get(url, timeout):
        return FakeResponse(payload)

    result = fetch_dhw(151.9148, -23.4423, date(2020, 2, 1), http_get=fake_get)
    assert result == 5.5


def test_fetch_dhw_returns_none_when_no_rows():
    payload = {"table": {"rows": []}}

    def fake_get(url, timeout):
        return FakeResponse(payload)

    result = fetch_dhw(151.9148, -23.4423, date(2020, 2, 1), http_get=fake_get)
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources_noaa.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coral_bleaching.sources.noaa'`

- [ ] **Step 3: Write minimal implementation**

`src/coral_bleaching/sources/noaa.py`:

```python
from datetime import date

import requests

NOAA_DHW_ERDDAP_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.json"


def fetch_dhw(
    lon: float, lat: float, obs_date: date, http_get=requests.get
) -> float | None:
    query = (
        f"{NOAA_DHW_ERDDAP_URL}?"
        f"CRW_DHW%5B({obs_date.isoformat()}T12:00:00Z)%5D"
        f"%5B({lat})%5D%5B({lon})%5D"
    )
    response = http_get(query, timeout=30)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("table", {}).get("rows", [])
    if not rows:
        return None
    dhw_value = rows[0][-1]
    return float(dhw_value) if dhw_value is not None else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources_noaa.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/coral_bleaching/sources/noaa.py tests/test_sources_noaa.py
git commit -m "feat: add NOAA DHW data source"
```

---

### Task 7: AIMS survey label source

**Files:**
- Create: `src/coral_bleaching/sources/aims.py`
- Create: `tests/fixtures/aims_labels_fixture.csv`
- Test: `tests/test_sources_aims.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `load_aims_labels(csv_path: str) -> list[dict]` (each dict has
  keys `reef_cell_id: str`, `survey_date: datetime.date`, `label: str`),
  `find_matching_label(records: list[dict], reef_cell_id: str, obs_date:
  datetime.date, window_days: int = 14) -> str | None`. Task 8 calls
  `find_matching_label` for every row it builds.

- [ ] **Step 1: Create the AIMS fixture CSV**

`tests/fixtures/aims_labels_fixture.csv`:

```csv
reef_cell_id,survey_date,label
heron_island_r0_c0,2020-01-15,bleached
heron_island_r0_c0,2020-02-20,healthy
heron_island_r1_c1,2020-01-20,healthy
```

- [ ] **Step 2: Write the failing test**

`tests/test_sources_aims.py`:

```python
from datetime import date

from coral_bleaching.sources.aims import find_matching_label, load_aims_labels


def test_load_aims_labels_parses_csv():
    records = load_aims_labels("tests/fixtures/aims_labels_fixture.csv")
    assert len(records) == 3
    assert records[0] == {
        "reef_cell_id": "heron_island_r0_c0",
        "survey_date": date(2020, 1, 15),
        "label": "bleached",
    }


def test_find_matching_label_within_window():
    records = load_aims_labels("tests/fixtures/aims_labels_fixture.csv")
    label = find_matching_label(
        records, "heron_island_r0_c0", date(2020, 1, 20), window_days=14
    )
    assert label == "bleached"


def test_find_matching_label_returns_none_outside_window():
    records = load_aims_labels("tests/fixtures/aims_labels_fixture.csv")
    label = find_matching_label(
        records, "heron_island_r0_c0", date(2020, 6, 1), window_days=14
    )
    assert label is None


def test_find_matching_label_picks_closest_survey():
    records = load_aims_labels("tests/fixtures/aims_labels_fixture.csv")
    # 2020-02-05 is 21 days after the 01-15 survey and 15 days before the
    # 02-20 survey -> within a 21-day window, closest is 02-20 (healthy).
    label = find_matching_label(
        records, "heron_island_r0_c0", date(2020, 2, 5), window_days=21
    )
    assert label == "healthy"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_sources_aims.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coral_bleaching.sources.aims'`

- [ ] **Step 4: Write minimal implementation**

`src/coral_bleaching/sources/aims.py`:

```python
import csv
from datetime import date


def load_aims_labels(csv_path: str) -> list[dict]:
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                {
                    "reef_cell_id": row["reef_cell_id"],
                    "survey_date": date.fromisoformat(row["survey_date"]),
                    "label": row["label"],
                }
            )
    return records


def find_matching_label(
    records: list[dict],
    reef_cell_id: str,
    obs_date: date,
    window_days: int = 14,
) -> str | None:
    best_label: str | None = None
    best_delta: int | None = None
    for record in records:
        if record["reef_cell_id"] != reef_cell_id:
            continue
        delta = abs((record["survey_date"] - obs_date).days)
        if delta <= window_days and (best_delta is None or delta < best_delta):
            best_label = record["label"]
            best_delta = delta
    return best_label
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_sources_aims.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/coral_bleaching/sources/aims.py tests/fixtures/aims_labels_fixture.csv tests/test_sources_aims.py
git commit -m "feat: add AIMS survey label source"
```

---

### Task 8: Pipeline orchestration (Module 1 output)

**Files:**
- Create: `src/coral_bleaching/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `RegionConfig` (Task 2), `ReefCell` (Task 3),
  `normalized_difference`/`rolling_index_shift` (Task 4), `GEEClient`
  protocol shape (Task 5), `fetch_dhw` signature (Task 6),
  `find_matching_label` (Task 7).
- Produces: `PipelineError` exception, `build_training_dataset(region:
  RegionConfig, cells: list[ReefCell], dates: list[datetime.date],
  gee_client, fetch_dhw_fn, aims_records: list[dict], rolling_window: int
  = 3) -> pandas.DataFrame` with columns `reef_cell_id, date,
  ndci_like_index, index_shift, dhw, cloud_cover_fraction, data_quality,
  dhw_risk_flag, confirmed_label`, and `write_dataset(df:
  pandas.DataFrame, output_dir: str) -> None`. Task 9 (labels) and Task 11
  (integration test) consume this DataFrame schema directly.

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:

```python
from datetime import date

import pandas as pd
import pytest

from coral_bleaching.config import RegionConfig
from coral_bleaching.grid import ReefCell
from coral_bleaching.pipeline import PipelineError, build_training_dataset, write_dataset
from shapely.geometry import box


def make_region(**overrides):
    defaults = dict(
        reef_id="heron_island",
        boundary_geojson_path="tests/fixtures/heron_island_boundary.geojson",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 15),
    )
    defaults.update(overrides)
    return RegionConfig(**defaults)


def make_cell(cell_id="heron_island_r0_c0"):
    return ReefCell(
        reef_cell_id=cell_id,
        geometry_wgs84=box(151.9148, -23.4423, 151.9155, -23.4416),
        centroid_lat=-23.44195,
        centroid_lon=151.91515,
    )


class FakeGEEClient:
    def __init__(self, responses):
        # responses: dict[(cell_id, date), dict | None]
        self._responses = responses

    def get_reflectance(self, lon, lat, obs_date):
        return self._responses.get(obs_date)


def fake_fetch_dhw_factory(value):
    def fake_fetch_dhw(lon, lat, obs_date):
        return value

    return fake_fetch_dhw


def test_build_training_dataset_happy_path():
    region = make_region()
    cell = make_cell()
    dates = [date(2020, 1, 1), date(2020, 1, 8)]
    responses = {
        date(2020, 1, 1): {
            "band_green": 0.06,
            "band_red": 0.02,
            "band_rededge": 0.03,
            "cloud_cover_fraction": 0.1,
        },
        date(2020, 1, 8): {
            "band_green": 0.05,
            "band_red": 0.03,
            "band_rededge": 0.03,
            "cloud_cover_fraction": 0.2,
        },
    }
    gee_client = FakeGEEClient(responses)
    fetch_dhw_fn = fake_fetch_dhw_factory(5.0)

    df = build_training_dataset(
        region=region,
        cells=[cell],
        dates=dates,
        gee_client=gee_client,
        fetch_dhw_fn=fetch_dhw_fn,
        aims_records=[],
    )

    assert len(df) == 2
    assert set(df.columns) == {
        "reef_cell_id",
        "date",
        "ndci_like_index",
        "index_shift",
        "dhw",
        "cloud_cover_fraction",
        "data_quality",
        "dhw_risk_flag",
        "confirmed_label",
    }
    assert (df["dhw_risk_flag"] == True).all()  # noqa: E712
    assert (df["data_quality"] == "ok").all()


def test_build_training_dataset_flags_high_cloud_cover_as_insufficient():
    region = make_region()
    cell = make_cell()
    dates = [date(2020, 1, 1)]
    responses = {
        date(2020, 1, 1): {
            "band_green": 0.06,
            "band_red": 0.02,
            "band_rededge": 0.03,
            "cloud_cover_fraction": 0.9,
        }
    }
    gee_client = FakeGEEClient(responses)
    fetch_dhw_fn = fake_fetch_dhw_factory(1.0)

    df = build_training_dataset(
        region=region,
        cells=[cell],
        dates=dates,
        gee_client=gee_client,
        fetch_dhw_fn=fetch_dhw_fn,
        aims_records=[],
    )
    assert df.iloc[0]["data_quality"] == "insufficient"


def test_build_training_dataset_marks_missing_image_as_insufficient():
    region = make_region()
    cell = make_cell()
    dates = [date(2020, 1, 1)]
    gee_client = FakeGEEClient({date(2020, 1, 1): None})
    fetch_dhw_fn = fake_fetch_dhw_factory(2.0)

    df = build_training_dataset(
        region=region,
        cells=[cell],
        dates=dates,
        gee_client=gee_client,
        fetch_dhw_fn=fetch_dhw_fn,
        aims_records=[],
    )
    assert df.iloc[0]["data_quality"] == "insufficient"
    assert pd.isna(df.iloc[0]["ndci_like_index"])


def test_build_training_dataset_raises_on_dhw_failure():
    region = make_region()
    cell = make_cell()
    dates = [date(2020, 1, 1)]
    gee_client = FakeGEEClient(
        {
            date(2020, 1, 1): {
                "band_green": 0.06,
                "band_red": 0.02,
                "band_rededge": 0.03,
                "cloud_cover_fraction": 0.1,
            }
        }
    )
    fetch_dhw_fn = fake_fetch_dhw_factory(None)

    with pytest.raises(PipelineError):
        build_training_dataset(
            region=region,
            cells=[cell],
            dates=dates,
            gee_client=gee_client,
            fetch_dhw_fn=fetch_dhw_fn,
            aims_records=[],
        )


def test_build_training_dataset_joins_aims_label():
    region = make_region()
    cell = make_cell("heron_island_r0_c0")
    dates = [date(2020, 1, 10)]
    gee_client = FakeGEEClient(
        {
            date(2020, 1, 10): {
                "band_green": 0.06,
                "band_red": 0.02,
                "band_rededge": 0.03,
                "cloud_cover_fraction": 0.1,
            }
        }
    )
    fetch_dhw_fn = fake_fetch_dhw_factory(1.0)
    aims_records = [
        {
            "reef_cell_id": "heron_island_r0_c0",
            "survey_date": date(2020, 1, 15),
            "label": "bleached",
        }
    ]

    df = build_training_dataset(
        region=region,
        cells=[cell],
        dates=dates,
        gee_client=gee_client,
        fetch_dhw_fn=fetch_dhw_fn,
        aims_records=aims_records,
    )
    assert df.iloc[0]["confirmed_label"] == "bleached"


def test_write_dataset_creates_partitioned_parquet(tmp_path):
    df = pd.DataFrame(
        {
            "reef_cell_id": ["a", "b"],
            "date": ["2020-01-01", "2020-01-01"],
            "ndci_like_index": [0.1, 0.2],
            "index_shift": [0.05, None],
            "dhw": [1.0, 2.0],
            "cloud_cover_fraction": [0.1, 0.1],
            "data_quality": ["ok", "ok"],
            "dhw_risk_flag": [False, False],
            "confirmed_label": [None, "bleached"],
        }
    )
    output_dir = tmp_path / "training_dataset"
    write_dataset(df, str(output_dir))

    read_back = pd.read_parquet(output_dir)
    assert len(read_back) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coral_bleaching.pipeline'`

- [ ] **Step 3: Write minimal implementation**

`src/coral_bleaching/pipeline.py`:

```python
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from coral_bleaching.config import RegionConfig
from coral_bleaching.grid import ReefCell
from coral_bleaching.sources.aims import find_matching_label
from coral_bleaching.spectral import normalized_difference, rolling_index_shift


class PipelineError(Exception):
    """Raised when a required data source fails to return usable data."""


@dataclass
class DatasetRow:
    reef_cell_id: str
    date: str
    ndci_like_index: Optional[float]
    index_shift: Optional[float]
    dhw: float
    cloud_cover_fraction: Optional[float]
    data_quality: str
    dhw_risk_flag: bool
    confirmed_label: Optional[str]


def build_training_dataset(
    region: RegionConfig,
    cells: list[ReefCell],
    dates: list[date],
    gee_client,
    fetch_dhw_fn,
    aims_records: list[dict],
    rolling_window: int = 3,
) -> pd.DataFrame:
    rows: list[DatasetRow] = []
    index_history: dict[str, list[float]] = {}

    for cell in cells:
        index_history[cell.reef_cell_id] = []
        for obs_date in dates:
            dhw = fetch_dhw_fn(cell.centroid_lon, cell.centroid_lat, obs_date)
            if dhw is None:
                raise PipelineError(
                    f"NOAA DHW lookup failed for {cell.reef_cell_id} on {obs_date}"
                )

            confirmed_label = find_matching_label(
                aims_records,
                cell.reef_cell_id,
                obs_date,
                region.aims_label_window_days,
            )

            reflectance = gee_client.get_reflectance(
                cell.centroid_lon, cell.centroid_lat, obs_date
            )

            if reflectance is None:
                rows.append(
                    DatasetRow(
                        reef_cell_id=cell.reef_cell_id,
                        date=obs_date.isoformat(),
                        ndci_like_index=None,
                        index_shift=None,
                        dhw=dhw,
                        cloud_cover_fraction=None,
                        data_quality="insufficient",
                        dhw_risk_flag=dhw >= region.dhw_risk_threshold,
                        confirmed_label=confirmed_label,
                    )
                )
                continue

            index_value = normalized_difference(
                reflectance["band_green"], reflectance["band_red"]
            )
            index_history[cell.reef_cell_id].append(index_value)
            history = index_history[cell.reef_cell_id]
            index_shift = (
                rolling_index_shift(history, rolling_window)
                if len(history) >= rolling_window * 2
                else None
            )

            cloud_fraction = reflectance["cloud_cover_fraction"]
            data_quality = (
                "insufficient"
                if cloud_fraction > region.cloud_cover_drop_threshold
                else "ok"
            )

            rows.append(
                DatasetRow(
                    reef_cell_id=cell.reef_cell_id,
                    date=obs_date.isoformat(),
                    ndci_like_index=index_value,
                    index_shift=index_shift,
                    dhw=dhw,
                    cloud_cover_fraction=cloud_fraction,
                    data_quality=data_quality,
                    dhw_risk_flag=dhw >= region.dhw_risk_threshold,
                    confirmed_label=confirmed_label,
                )
            )

    return pd.DataFrame([row.__dict__ for row in rows])


def write_dataset(df: pd.DataFrame, output_dir: str) -> None:
    df.to_parquet(output_dir, partition_cols=["reef_cell_id"], index=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/coral_bleaching/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestration for Module 1 dataset"
```

---

### Task 9: Label and sample-weight assignment (Module 2 input prep)

**Files:**
- Create: `src/coral_bleaching/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: rows shaped like Task 8's `build_training_dataset` output
  (dict-like access to `confirmed_label`, `data_quality`,
  `dhw_risk_flag`).
- Produces: `CONFIRMED_WEIGHT = 1.0`, `WEAK_LABEL_WEIGHT = 0.5`,
  `assign_label_and_weight(row: dict) -> tuple[str | None, float]`,
  `build_training_frame(df: pandas.DataFrame) -> pandas.DataFrame` (same
  columns as input, plus `label: str` and `sample_weight: float`, with
  unusable rows dropped). Task 10 consumes the `label` and `sample_weight`
  columns.

- [ ] **Step 1: Write the failing test**

`tests/test_labels.py`:

```python
import pandas as pd

from coral_bleaching.labels import assign_label_and_weight, build_training_frame


def test_assign_label_and_weight_uses_confirmed_label():
    row = {"confirmed_label": "bleached", "data_quality": "ok", "dhw_risk_flag": False}
    label, weight = assign_label_and_weight(row)
    assert label == "bleached"
    assert weight == 1.0


def test_assign_label_and_weight_uses_dhw_when_no_confirmed_label():
    row = {"confirmed_label": None, "data_quality": "ok", "dhw_risk_flag": True}
    label, weight = assign_label_and_weight(row)
    assert label == "bleached"
    assert weight == 0.5


def test_assign_label_and_weight_healthy_weak_label():
    row = {"confirmed_label": None, "data_quality": "ok", "dhw_risk_flag": False}
    label, weight = assign_label_and_weight(row)
    assert label == "healthy"
    assert weight == 0.5


def test_assign_label_and_weight_drops_insufficient_quality():
    row = {"confirmed_label": None, "data_quality": "insufficient", "dhw_risk_flag": True}
    label, weight = assign_label_and_weight(row)
    assert label is None
    assert weight == 0.0


def test_build_training_frame_drops_unlabeled_rows():
    df = pd.DataFrame(
        [
            {"confirmed_label": "bleached", "data_quality": "ok", "dhw_risk_flag": False},
            {"confirmed_label": None, "data_quality": "insufficient", "dhw_risk_flag": True},
            {"confirmed_label": None, "data_quality": "ok", "dhw_risk_flag": True},
        ]
    )
    result = build_training_frame(df)
    assert len(result) == 2
    assert list(result["label"]) == ["bleached", "bleached"]
    assert list(result["sample_weight"]) == [1.0, 0.5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coral_bleaching.labels'`

- [ ] **Step 3: Write minimal implementation**

`src/coral_bleaching/labels.py`:

```python
from typing import Optional

import pandas as pd

CONFIRMED_WEIGHT = 1.0
WEAK_LABEL_WEIGHT = 0.5


def assign_label_and_weight(row: dict) -> tuple[Optional[str], float]:
    if row.get("confirmed_label"):
        return row["confirmed_label"], CONFIRMED_WEIGHT
    if row.get("data_quality") == "insufficient":
        return None, 0.0
    weak_label = "bleached" if row["dhw_risk_flag"] else "healthy"
    return weak_label, WEAK_LABEL_WEIGHT


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    labels = []
    weights = []
    for _, row in df.iterrows():
        label, weight = assign_label_and_weight(row)
        labels.append(label)
        weights.append(weight)

    result = df.copy()
    result["label"] = labels
    result["sample_weight"] = weights
    return result[result["label"].notna()].reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_labels.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/coral_bleaching/labels.py tests/test_labels.py
git commit -m "feat: add combined AIMS/DHW label weighting"
```

---

### Task 10: Model training with leave-one-survey-date-out CV

**Files:**
- Create: `src/coral_bleaching/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: DataFrame shaped like Task 9's `build_training_frame` output
  (must have columns `ndci_like_index, index_shift, dhw,
  cloud_cover_fraction, label, sample_weight, date, confirmed_label`).
- Produces: `FEATURE_COLUMNS: list[str]`,
  `leave_one_survey_date_out_splits(df: pandas.DataFrame) ->
  Iterator[tuple[numpy.ndarray, numpy.ndarray]]`, `train_model(df:
  pandas.DataFrame, params: dict | None = None) ->
  lightgbm.LGBMClassifier`, `cross_validate(df: pandas.DataFrame, params:
  dict | None = None) -> list[float]`. Task 11 (predict.py) imports
  `FEATURE_COLUMNS` and consumes the `LGBMClassifier` returned by
  `train_model`.

- [ ] **Step 1: Write the failing test**

`tests/test_train.py`:

```python
import pandas as pd

from coral_bleaching.train import (
    FEATURE_COLUMNS,
    cross_validate,
    leave_one_survey_date_out_splits,
    train_model,
)


def make_training_frame():
    return pd.DataFrame(
        {
            "reef_cell_id": [f"cell_{i}" for i in range(20)],
            "date": (["2020-01-15"] * 5 + ["2020-02-20"] * 5) * 2,
            "ndci_like_index": [0.5, 0.52, 0.48, 0.51, 0.49] * 4,
            "index_shift": [0.1] * 20,
            "dhw": [6.0] * 10 + [1.0] * 10,
            "cloud_cover_fraction": [0.1] * 20,
            "confirmed_label": (["bleached"] * 5 + ["healthy"] * 5) * 2,
            "label": (["bleached"] * 5 + ["healthy"] * 5) * 2,
            "sample_weight": [1.0] * 20,
        }
    )


def test_feature_columns_match_expected_set():
    assert FEATURE_COLUMNS == ["ndci_like_index", "index_shift", "dhw", "cloud_cover_fraction"]


def test_leave_one_survey_date_out_splits_covers_each_survey_date():
    df = make_training_frame()
    splits = list(leave_one_survey_date_out_splits(df))
    held_out_dates = set()
    for _, test_idx in splits:
        held_out_dates.update(df.loc[test_idx, "date"].unique())
    assert held_out_dates == {"2020-01-15", "2020-02-20"}


def test_train_model_returns_fitted_classifier_with_predict_proba():
    df = make_training_frame()
    model = train_model(df)
    probs = model.predict_proba(df[FEATURE_COLUMNS])
    assert probs.shape == (20, 2)


def test_cross_validate_returns_one_score_per_survey_date():
    df = make_training_frame()
    scores = cross_validate(df, params={"n_estimators": 10, "min_child_samples": 1})
    assert len(scores) == 2
    for score in scores:
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coral_bleaching.train'`

- [ ] **Step 3: Write minimal implementation**

`src/coral_bleaching/train.py`:

```python
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

FEATURE_COLUMNS = ["ndci_like_index", "index_shift", "dhw", "cloud_cover_fraction"]


def leave_one_survey_date_out_splits(
    df: pd.DataFrame,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    survey_dates = sorted(df.loc[df["confirmed_label"].notna(), "date"].unique())
    for held_out_date in survey_dates:
        test_idx = df.index[df["date"] == held_out_date].to_numpy()
        train_idx = df.index[df["date"] != held_out_date].to_numpy()
        yield train_idx, test_idx


def train_model(df: pd.DataFrame, params: Optional[dict] = None) -> LGBMClassifier:
    X = df[FEATURE_COLUMNS].fillna(0.0)
    y = (df["label"] == "bleached").astype(int)
    weights = df["sample_weight"]
    model = LGBMClassifier(**(params or {}))
    model.fit(X, y, sample_weight=weights)
    return model


def cross_validate(df: pd.DataFrame, params: Optional[dict] = None) -> list[float]:
    scores = []
    for train_idx, test_idx in leave_one_survey_date_out_splits(df):
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]
        if test_df["label"].nunique() < 2:
            continue
        model = train_model(train_df, params)
        probs = model.predict_proba(test_df[FEATURE_COLUMNS].fillna(0.0))[:, 1]
        y_true = (test_df["label"] == "bleached").astype(int)
        scores.append(roc_auc_score(y_true, probs))
    return scores
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/coral_bleaching/train.py tests/test_train.py
git commit -m "feat: add GBT training with leave-one-survey-date-out CV"
```

---

### Task 11: Prediction generation (Module 2 output contract)

**Files:**
- Create: `src/coral_bleaching/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS` and the `LGBMClassifier` shape from Task 10.
- Produces: `confidence_band(probability: float) -> str`,
  `generate_predictions(df: pandas.DataFrame, model:
  lightgbm.LGBMClassifier) -> pandas.DataFrame` with columns
  `reef_cell_id, date, probability, predicted_label,
  top_contributing_features, confidence_band`, and `write_predictions(df:
  pandas.DataFrame, output_path: str) -> None`. This is the final Module
  2 output contract that Module 4 (not in this plan's scope) will read.

- [ ] **Step 1: Write the failing test**

`tests/test_predict.py`:

```python
import pandas as pd

from coral_bleaching.predict import confidence_band, generate_predictions, write_predictions
from coral_bleaching.train import train_model


def make_training_frame():
    return pd.DataFrame(
        {
            "reef_cell_id": [f"cell_{i}" for i in range(20)],
            "date": (["2020-01-15"] * 5 + ["2020-02-20"] * 5) * 2,
            "ndci_like_index": [0.5, 0.52, 0.48, 0.51, 0.49] * 4,
            "index_shift": [0.1] * 20,
            "dhw": [6.0] * 10 + [1.0] * 10,
            "cloud_cover_fraction": [0.1] * 20,
            "confirmed_label": (["bleached"] * 5 + ["healthy"] * 5) * 2,
            "label": (["bleached"] * 5 + ["healthy"] * 5) * 2,
            "sample_weight": [1.0] * 20,
        }
    )


def test_confidence_band_thresholds():
    assert confidence_band(0.5) == "low"
    assert confidence_band(0.55) == "low"
    assert confidence_band(0.65) == "medium"
    assert confidence_band(0.95) == "high"
    assert confidence_band(0.05) == "high"


def test_generate_predictions_has_expected_columns_and_row_count():
    df = make_training_frame()
    model = train_model(df, params={"n_estimators": 10, "min_child_samples": 1})
    predictions = generate_predictions(df, model)

    assert len(predictions) == len(df)
    assert set(predictions.columns) == {
        "reef_cell_id",
        "date",
        "probability",
        "predicted_label",
        "top_contributing_features",
        "confidence_band",
    }
    assert predictions["predicted_label"].isin(["bleached", "healthy"]).all()
    assert predictions["top_contributing_features"].apply(len).eq(3).all()


def test_write_predictions_round_trips(tmp_path):
    df = make_training_frame()
    model = train_model(df, params={"n_estimators": 10, "min_child_samples": 1})
    predictions = generate_predictions(df, model)

    output_path = tmp_path / "predictions.parquet"
    write_predictions(predictions, str(output_path))

    read_back = pd.read_parquet(output_path)
    assert len(read_back) == len(predictions)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coral_bleaching.predict'`

- [ ] **Step 3: Write minimal implementation**

`src/coral_bleaching/predict.py`:

```python
import pandas as pd
from lightgbm import LGBMClassifier

from coral_bleaching.train import FEATURE_COLUMNS


def confidence_band(probability: float) -> str:
    distance = abs(probability - 0.5)
    if distance < 0.1:
        return "low"
    if distance < 0.3:
        return "medium"
    return "high"


def generate_predictions(df: pd.DataFrame, model: LGBMClassifier) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].fillna(0.0)
    probabilities = model.predict_proba(X)[:, 1]
    contributions = model.booster_.predict(X, pred_contrib=True)

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        contrib_row = contributions[i][:-1]  # last column is the base value
        ranked = sorted(
            zip(FEATURE_COLUMNS, contrib_row),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
        top_features = [name for name, _ in ranked[:3]]
        probability = float(probabilities[i])
        results.append(
            {
                "reef_cell_id": row["reef_cell_id"],
                "date": row["date"],
                "probability": probability,
                "predicted_label": "bleached" if probability >= 0.5 else "healthy",
                "top_contributing_features": top_features,
                "confidence_band": confidence_band(probability),
            }
        )
    return pd.DataFrame(results)


def write_predictions(df: pd.DataFrame, output_path: str) -> None:
    df.to_parquet(output_path, index=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_predict.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/coral_bleaching/predict.py tests/test_predict.py
git commit -m "feat: add prediction generation with reasoning output contract"
```

---

### Task 12: End-to-end integration test

**Files:**
- Test: `tests/test_integration.py`

**Interfaces:**
- Consumes: every public function/class produced by Tasks 1-11.
- Produces: nothing new — this task only verifies the full chain works
  together against fixture data with no network access.

- [ ] **Step 1: Write the end-to-end test**

`tests/test_integration.py`:

```python
from datetime import date, timedelta

import pandas as pd

from coral_bleaching.config import RegionConfig
from coral_bleaching.grid import generate_grid, load_boundary
from coral_bleaching.labels import build_training_frame
from coral_bleaching.pipeline import build_training_dataset, write_dataset
from coral_bleaching.predict import generate_predictions, write_predictions
from coral_bleaching.sources.aims import load_aims_labels
from coral_bleaching.train import train_model


class FakeGEEClient:
    def get_reflectance(self, lon, lat, obs_date):
        # Deterministic fake: index shifts over time so the model has
        # signal to find; cloud cover always acceptable.
        day_offset = (obs_date - date(2020, 1, 1)).days
        green = 0.05 + 0.0005 * day_offset
        red = 0.03
        return {
            "band_green": green,
            "band_red": red,
            "band_rededge": 0.03,
            "cloud_cover_fraction": 0.1,
        }


def fake_fetch_dhw(lon, lat, obs_date):
    day_offset = (obs_date - date(2020, 1, 1)).days
    return 6.0 if day_offset >= 14 else 1.0


def test_full_pipeline_to_predictions(tmp_path):
    region = RegionConfig(
        reef_id="heron_island",
        boundary_geojson_path="tests/fixtures/heron_island_boundary.geojson",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 2, 1),
    )
    boundary = load_boundary(region.boundary_geojson_path)
    cells = generate_grid(boundary, region.cell_size_meters, region.reef_id)
    assert len(cells) > 0

    dates = [
        region.start_date + timedelta(days=7 * i)
        for i in range(5)
    ]

    aims_records = load_aims_labels("tests/fixtures/aims_labels_fixture.csv")
    # Retag one fixture record onto a cell that actually exists in this grid
    # so the integration test has at least one confirmed label to work with.
    aims_records = [
        {**record, "reef_cell_id": cells[0].reef_cell_id} for record in aims_records
    ]

    raw_df = build_training_dataset(
        region=region,
        cells=cells,
        dates=dates,
        gee_client=FakeGEEClient(),
        fetch_dhw_fn=fake_fetch_dhw,
        aims_records=aims_records,
    )
    assert len(raw_df) == len(cells) * len(dates)

    dataset_dir = tmp_path / "training_dataset"
    write_dataset(raw_df, str(dataset_dir))
    reloaded_df = pd.read_parquet(dataset_dir)
    assert len(reloaded_df) == len(raw_df)

    training_frame = build_training_frame(reloaded_df)
    assert len(training_frame) > 0
    assert "label" in training_frame.columns

    model = train_model(training_frame, params={"n_estimators": 10, "min_child_samples": 1})
    predictions = generate_predictions(training_frame, model)

    predictions_path = tmp_path / "predictions.parquet"
    write_predictions(predictions, str(predictions_path))
    reloaded_predictions = pd.read_parquet(predictions_path)

    assert len(reloaded_predictions) == len(training_frame)
    assert reloaded_predictions["probability"].between(0.0, 1.0).all()
    assert reloaded_predictions["confidence_band"].isin(["low", "medium", "high"]).all()
```

- [ ] **Step 2: Run the full test suite**

Run: `pytest -v`
Expected: All tests across every task pass, including this integration
test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end pipeline-to-predictions integration test"
```

---

## Definition of Done

- `pytest -v` passes with zero failures and no network access.
- `training_dataset.parquet` and `predictions.parquet` schemas match the
  contracts documented in Tasks 8 and 11 exactly — these are the
  contracts Module 3 (web UI) and Module 4 (backend API), designed later,
  will depend on.
- No task's implementation references Google Earth Engine, NOAA, or AIMS
  credentials directly in test code — all are injected fakes.
