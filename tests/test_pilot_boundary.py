import importlib.util
from pathlib import Path

import pyproj
from shapely.ops import transform

from coral_bleaching.cli import load_region
from coral_bleaching.grid import generate_grid, load_boundary

BOUNDARY = Path("data/heron_island_reef_boundary.geojson")
TO_UTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=True).transform


def _area_km2(geom):
    return transform(TO_UTM, geom).area / 1e6


def test_committed_boundary_is_heron_reef():
    boundary = load_boundary(BOUNDARY)
    assert boundary.geom_type in ("Polygon", "MultiPolygon")
    # Heron Reef is commonly cited at ~26 km2; guard against a stale file that
    # is empty, the whole Capricornia Cays export (~97 km2), or Wistari instead.
    assert 24 < _area_km2(boundary) < 29
    minx, miny, maxx, maxy = boundary.bounds
    assert 151.88 < minx and maxx < 151.98
    assert -23.48 < miny and maxy < -23.42


def test_boundary_generates_a_plausible_grid():
    cells = generate_grid(load_boundary(BOUNDARY), 75.0, "heron_island")
    assert 4000 < len(cells) < 6000
    ids = [cell.reef_cell_id for cell in cells]
    assert len(ids) == len(set(ids))
    for cell in cells:
        assert 151.88 < cell.centroid_lon < 151.98
        assert -23.48 < cell.centroid_lat < -23.42


def test_example_config_resolves_to_the_committed_boundary():
    region = load_region("examples/heron_island.json")
    assert region.reef_id == "heron_island"
    assert Path(region.boundary_geojson_path) == BOUNDARY.resolve()
    load_boundary(region.boundary_geojson_path)


def _load_build_script():
    spec = importlib.util.spec_from_file_location(
        "build_pilot_boundary", Path("scripts/build_pilot_boundary.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_script_is_reproducible():
    regenerated = _load_build_script().build_boundary()
    committed = load_boundary(BOUNDARY)
    # Symmetric difference should be negligible (float rounding in the written
    # coordinates only), not a structural change.
    assert regenerated.symmetric_difference(committed).area < 1e-9
