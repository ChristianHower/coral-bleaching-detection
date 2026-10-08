from datetime import date
from unittest.mock import Mock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from coral_bleaching.grid import ReefCell, reproject
from coral_bleaching.sources.noaa import NOAAClient
from coral_bleaching.sources.sentinel import EarthSearchClient


def noaa_response(value=6.89, day="2020-04-29", lon=151.925):
    response = Mock()
    response.json.return_value = {
        "table": {
            "columnNames": ["degree_heating_week", "longitude", "time", "latitude"],
            "rows": [[value, lon, f"{day}T12:00:00Z", -23.425]],
        }
    }
    return response


def test_noaa_named_columns_cache_and_query():
    get = Mock(return_value=noaa_response())
    client = NOAAClient(get)
    for _ in range(2):
        assert client.fetch_dhw(151.9148, -23.4423, date(2020, 4, 29)) == pytest.approx(6.89)
    assert get.call_count == 1
    assert "degree_heating_week" in get.call_args.args[0]
    assert len(client.provenance) == 1


@pytest.mark.parametrize(
    "response",
    [
        noaa_response(None),
        noaa_response(-1),
        noaa_response(float("nan")),
        noaa_response(day="2020-04-28"),
        noaa_response(lon=150),
    ],
)
def test_noaa_invalid_data_fails(response):
    with pytest.raises((TypeError, ValueError)):
        NOAAClient(Mock(return_value=response)).fetch_dhw(151.9148, -23.4423, date(2020, 4, 29))


@pytest.fixture
def raster_scene(tmp_path):
    # 40 m square: SCL 20 m raster aligned onto 10 m reflectance.
    paths = {}
    for name, data, resolution in [
        ("green", np.full((4, 4), 600, dtype="uint16"), 10),
        ("red", np.full((4, 4), 200, dtype="uint16"), 10),
        ("scl", np.array([[6, 9], [6, 6]], dtype="uint16"), 20),
    ]:
        path = tmp_path / f"{name}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs="EPSG:32756",
            transform=from_origin(389000, 7407000, resolution, resolution),
            nodata=0,
        ) as ds:
            ds.write(data, 1)
        paths[f"https://fixture.test/{name}.tif"] = path
    geometry = reproject(box(389000, 7406960, 389040, 7407000), "EPSG:32756", "EPSG:4326")
    item = {
        "id": "scene",
        "properties": {"datetime": "2020-01-01T00:00:00Z"},
        "geometry": mapping(geometry),
        "assets": {
            name: {"href": url, "raster:bands": [{"scale": 0.0001, "offset": 0, "nodata": 0}]}
            for name, url in zip(["green", "red", "scl"], paths)
        },
    }
    return item, ReefCell("cell", geometry, geometry.centroid.y, geometry.centroid.x), paths


def test_actual_raster_processing_masks_cloud_and_keeps_water(raster_scene):
    item, cell, paths = raster_scene
    client = EarthSearchClient(raster_open=lambda url: rasterio.open(paths[url]))
    obs = client.observe(item, cell)
    assert obs.band_green == pytest.approx(0.06)
    assert obs.band_red == pytest.approx(0.02)
    assert obs.cloud_cover_fraction == pytest.approx(0.25)
    assert obs.date == date(2020, 1, 1)
    item["assets"]["green"]["raster:bands"][0]["offset"] = -0.01
    assert client.observe(item, cell).band_green == pytest.approx(0.05)


def test_public_asset_enforcement(raster_scene):
    item, cell, paths = raster_scene
    item["assets"]["green"]["href"] = "s3://requester-pays/image.tif"
    with pytest.raises(ValueError, match="HTTPS"):
        EarthSearchClient().observe(item, cell)


def test_search_consumes_pages_deduplicates_and_caches(raster_scene):
    item, cell, paths = raster_scene
    catalog = Mock()
    second = dict(item, id="scene2", properties={"datetime": "2020-01-02T00:00:00Z"})

    def pages():
        yield item
        yield item
        yield second

    catalog.search.return_value.items.side_effect = pages
    client = EarthSearchClient(catalog=catalog)
    result = client.search(cell.geometry_wgs84, date(2020, 1, 1), date(2020, 1, 2))
    assert len(result) == 2
    assert client.search(cell.geometry_wgs84, date(2020, 1, 1), date(2020, 1, 2)) is result
    catalog.search.assert_called_once()
    empty_catalog = Mock()
    empty_catalog.search.return_value.items.return_value = iter([])
    assert (
        EarthSearchClient(catalog=empty_catalog).search(
            cell.geometry_wgs84, date(2020, 1, 1), date(2020, 1, 2)
        )
        == []
    )


def test_reef_intersection_excludes_clouds_outside_cell(raster_scene):
    item, cell, paths = raster_scene
    geometry = reproject(box(389000, 7406960, 389020, 7407000), "EPSG:32756", "EPSG:4326")
    clipped = ReefCell("clipped", geometry, geometry.centroid.y, geometry.centroid.x)
    client = EarthSearchClient(raster_open=lambda url: rasterio.open(paths[url]))
    assert client.observe(item, clipped).cloud_cover_fraction == pytest.approx(0)
    item["assets"]["green"]["raster:bands"][0].pop("scale")
    with pytest.raises(ValueError, match="scaling"):
        client.observe(item, clipped)


def test_scene_is_opened_once_per_band_for_multiple_cells(raster_scene):
    item, _, paths = raster_scene
    left_geometry = reproject(box(389000, 7406960, 389020, 7407000), "EPSG:32756", "EPSG:4326")
    right_geometry = reproject(box(389020, 7406960, 389040, 7407000), "EPSG:32756", "EPSG:4326")
    cells = [
        ReefCell("left", left_geometry, left_geometry.centroid.y, left_geometry.centroid.x),
        ReefCell("right", right_geometry, right_geometry.centroid.y, right_geometry.centroid.x),
    ]
    raster_open = Mock(side_effect=lambda url: rasterio.open(paths[url]))
    observations = EarthSearchClient(raster_open=raster_open).observe_cells(item, cells)

    assert raster_open.call_count == 3
    assert observations["left"].cloud_cover_fraction == pytest.approx(0)
    assert observations["right"].cloud_cover_fraction == pytest.approx(0.5)


def test_noaa_http_errors_missing_rows_and_malformed_columns():
    response = noaa_response()
    response.raise_for_status.side_effect = RuntimeError("HTTP 503")
    with pytest.raises(RuntimeError, match="503"):
        NOAAClient(Mock(return_value=response)).fetch_dhw(151.915, -23.44, date(2020, 4, 29))
    response = noaa_response()
    response.json.return_value["table"]["rows"] = []
    with pytest.raises(ValueError, match="exactly one"):
        NOAAClient(Mock(return_value=response)).fetch_dhw(151.915, -23.44, date(2020, 4, 29))
    response = noaa_response()
    response.json.return_value["table"]["columnNames"][0] = "wrong_variable"
    with pytest.raises(KeyError):
        NOAAClient(Mock(return_value=response)).fetch_dhw(151.915, -23.44, date(2020, 4, 29))
