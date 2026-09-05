import json
import math
from dataclasses import dataclass
from pathlib import Path

from pyproj import CRS, Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform, unary_union


@dataclass(frozen=True)
class ReefCell:
    reef_cell_id: str
    geometry_wgs84: object
    centroid_lat: float
    centroid_lon: float


def reproject(geometry, source, destination):
    return transform(Transformer.from_crs(source, destination, always_xy=True).transform, geometry)


def load_boundary(path):
    data = json.loads(Path(path).read_text())
    if data.get("type") == "FeatureCollection":
        geometry = unary_union([shape(f["geometry"]) for f in data["features"]])
    else:
        geometry = shape(data["geometry"] if data.get("type") == "Feature" else data)
    if (
        geometry.geom_type not in ("Polygon", "MultiPolygon")
        or geometry.is_empty
        or not geometry.is_valid
    ):
        raise ValueError("Boundary must contain valid, nonempty polygons")
    if not (
        -180 <= geometry.bounds[0] <= geometry.bounds[2] <= 180
        and -90 <= geometry.bounds[1] <= geometry.bounds[3] <= 90
    ):
        raise ValueError("Boundary must use WGS84 longitude/latitude")
    return geometry


def generate_grid(
    boundary_wgs84,
    cell_size_meters,
    reef_id,
    min_overlap_fraction=0.1,
    utm_epsg="EPSG:32756",
    grid_origin_x=0.0,
    grid_origin_y=0.0,
):
    crs = CRS(utm_epsg)
    if not crs.is_projected or any(a.unit_name != "metre" for a in crs.axis_info):
        raise ValueError("Grid CRS must be projected in meters")
    if not math.isfinite(cell_size_meters) or cell_size_meters <= 0:
        raise ValueError("Cell size must be positive")
    if not 0 <= min_overlap_fraction <= 1:
        raise ValueError("Overlap fraction must lie in [0, 1]")
    boundary = reproject(boundary_wgs84, "EPSG:4326", crs)
    x0, y0, x1, y1 = boundary.bounds
    size = cell_size_meters
    cells = []
    for row in range(
        math.floor((y0 - grid_origin_y) / size), math.ceil((y1 - grid_origin_y) / size)
    ):
        for col in range(
            math.floor((x0 - grid_origin_x) / size), math.ceil((x1 - grid_origin_x) / size)
        ):
            x, y = grid_origin_x + col * size, grid_origin_y + row * size
            clipped = box(x, y, x + size, y + size).intersection(boundary)
            if clipped.area <= 0 or clipped.area / size**2 + 1e-10 < min_overlap_fraction:
                continue
            geometry = reproject(clipped, crs, "EPSG:4326")
            point = geometry.representative_point()
            cells.append(ReefCell(f"{reef_id}_r{row}_c{col}", geometry, point.y, point.x))
    return cells


def grid_geojson(cells):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"reef_cell_id": c.reef_cell_id},
                "geometry": mapping(c.geometry_wgs84),
            }
            for c in cells
        ],
    }
