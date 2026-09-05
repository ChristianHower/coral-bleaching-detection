"""Anonymous STAC search and windowed reef-cell COG processing."""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlparse

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from shapely.geometry import mapping, shape

from coral_bleaching.grid import reproject

STAC_URL = "https://earth-search.aws.element84.com/v1"
# SCL 6 is water and MUST stay valid. Includes unclassified and snow/ice.
UNUSABLE_SCL = (0, 1, 3, 7, 8, 9, 10, 11)


@dataclass(frozen=True)
class Observation:
    date: object
    source_scene_id: str
    band_green: float | None
    band_red: float | None
    cloud_cover_fraction: float | None


class EarthSearchClient:
    def __init__(self, catalog=None, raster_open=None, collection="sentinel-2-l2a"):
        self.catalog = catalog
        self.raster_open = raster_open or rasterio.open
        self.collection = collection
        self._search_cache = {}
        self.provenance = {}

    def search(self, geometry, start_date, end_date):
        key = (geometry.wkb, start_date, end_date)
        if key not in self._search_cache:
            if self.catalog is None:
                from pystac_client import Client

                self.catalog = Client.open(STAC_URL)
            query = self.catalog.search(
                collections=[self.collection],
                intersects=mapping(geometry),
                datetime=f"{start_date.isoformat()}T00:00:00Z/{end_date.isoformat()}T23:59:59Z",
                limit=100,
            )
            # items() follows all STAC pagination links; duplicate IDs are discarded.
            items = {}
            for entry in query.items():
                item = entry.to_dict() if hasattr(entry, "to_dict") else entry
                timestamp = datetime.fromisoformat(
                    item["properties"]["datetime"].replace("Z", "+00:00")
                )
                if timestamp.utcoffset() != timedelta(0):
                    raise ValueError("Scene timestamps must be UTC")
                if start_date <= timestamp.date() <= end_date:
                    items[item["id"]] = item
            self._search_cache[key] = sorted(
                items.values(), key=lambda x: (x["properties"]["datetime"], x["id"])
            )
        return self._search_cache[key]

    @staticmethod
    def _asset(item, name):
        asset = item["assets"][name]
        href = asset["href"]
        if urlparse(href).scheme != "https" or asset.get("storage:requester_pays", False):
            raise ValueError("Anonymous adapter requires public HTTPS assets")
        return asset

    def observe(self, item, cell):
        assets = {name: self._asset(item, name) for name in ("green", "red", "scl")}
        scene_date = datetime.fromisoformat(
            item["properties"]["datetime"].replace("Z", "+00:00")
        ).date()
        empty = Observation(scene_date, item["id"], None, None, 1.0)
        if not shape(item["geometry"]).intersects(cell.geometry_wgs84):
            return empty
        with self.raster_open(assets["green"]["href"]) as green_ds:
            geometry = reproject(cell.geometry_wgs84, "EPSG:4326", green_ds.crs)
            raw_window = from_bounds(*geometry.bounds, transform=green_ds.transform)
            col, row = math.floor(raw_window.col_off), math.floor(raw_window.row_off)
            width = math.ceil(raw_window.col_off + raw_window.width) - col
            height = math.ceil(raw_window.row_off + raw_window.height) - row
            window = rasterio.windows.Window(col, row, width, height)
            transform = green_ds.window_transform(window)
            green = green_ds.read(1, window=window, boundless=True, masked=True)
            inside = geometry_mask(
                [mapping(geometry)], out_shape=green.shape, transform=transform, invert=True
            )
            if not inside.any():
                return empty
            arrays = {"green": green}
            for name in ("red", "scl"):
                with self.raster_open(assets[name]["href"]) as ds:
                    with WarpedVRT(
                        ds,
                        crs=green_ds.crs,
                        transform=transform,
                        width=width,
                        height=height,
                        resampling=Resampling.nearest,
                        nodata=assets[name].get("raster:bands", [{}])[0].get("nodata", ds.nodata),
                    ) as vrt:
                        arrays[name] = vrt.read(1, masked=True)
            valid = inside.copy()
            values = {}
            for name, arr in arrays.items():
                metadata = assets[name].get("raster:bands", [{}])[0]
                raw = np.asarray(arr.data, dtype=float)
                valid &= ~np.ma.getmaskarray(arr) & np.isfinite(raw)
                if "nodata" in metadata:
                    valid &= raw != metadata["nodata"]
                if name != "scl":
                    if "scale" not in metadata or "offset" not in metadata:
                        raise ValueError(f"Missing explicit reflectance scaling for {name}")
                    scale, offset = float(metadata["scale"]), float(metadata["offset"])
                    if not math.isfinite(scale) or not math.isfinite(offset) or scale <= 0:
                        raise ValueError("Invalid raster scaling")
                    values[name] = raw * scale + offset
            valid &= ~np.isin(arrays["scl"].data, UNUSABLE_SCL)
            fraction = 1 - float(valid.sum()) / int(inside.sum())
            self.provenance[item["id"]] = {
                "datetime": item["properties"]["datetime"],
                "collection": self.collection,
                "assets": assets,
                "unusable_scl": list(UNUSABLE_SCL),
            }
            return Observation(
                scene_date,
                item["id"],
                float(values["green"][valid].mean()) if valid.any() else None,
                float(values["red"][valid].mean()) if valid.any() else None,
                fraction,
            )

    def observations(self, cells, boundary, start_date, end_date):
        """One deterministic best-quality acquisition per cell/day; retain cloudy days."""
        items = self.search(boundary, start_date, end_date)
        result = {c.reef_cell_id: [] for c in cells}
        for cell in cells:
            by_day = {}
            for item in items:
                observation = self.observe(item, cell)
                current = by_day.get(observation.date)
                rank = (observation.cloud_cover_fraction, observation.source_scene_id)
                if current is None or rank < (
                    current.cloud_cover_fraction,
                    current.source_scene_id,
                ):
                    by_day[observation.date] = observation
            result[cell.reef_cell_id] = sorted(by_day.values(), key=lambda x: x.date)
        return result
