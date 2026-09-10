"""Derive the Heron Island Reef pilot boundary from Allen Coral Atlas reef extent.

The Allen Coral Atlas "Reef Extent" export for the area downloaded covers the
whole Capricornia Cays group (Heron, Wistari, Sykes and smaller patches),
~97 km2 of reef. The pilot only needs Heron Island Reef, so this script:

  1. unions the reef-extent polygons,
  2. keeps the single largest contiguous reef body (Heron Reef is fused with
     the reef extending east toward Sykes in the source data),
  3. clips it to a bounding box around Heron Reef proper,
  4. writes the result as a WGS84 GeoJSON the pipeline's load_boundary() reads.

Run from the repo root:

    python scripts/build_pilot_boundary.py

Source: Allen Coral Atlas (2020). Imagery, maps and monitoring of the world's
tropical coral reefs. https://doi.org/10.5281/zenodo.3833242
Reef-extent maps are (c) 2020 Allen Coral Atlas Partnership and Vulcan, Inc.,
licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
"""

import json
from pathlib import Path

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

SOURCE = Path("data/sources/allen_coral_atlas_reef_extent.geojson")
OUTPUT = Path("data/heron_island_reef_boundary.geojson")

# Heron Reef clip, WGS84 lon/lat. The eastern edge (151.975) separates Heron
# Reef from the reef platform continuing toward Sykes Reef; the result is a
# single contiguous body of ~26 km2, matching the commonly cited Heron Reef
# area. Wistari Reef to the southwest is already a separate reef-extent body
# and is excluded by keeping only the largest body before clipping.
HERON_CLIP = box(151.884, -23.478, 151.975, -23.417)

ATTRIBUTION = (
    "Allen Coral Atlas (2020). Imagery, maps and monitoring of the world's "
    "tropical coral reefs. https://doi.org/10.5281/zenodo.3833242 "
    "Reef-extent maps (c) 2020 Allen Coral Atlas Partnership and Vulcan, Inc., "
    "licensed CC BY 4.0. Derived: unioned, largest contiguous body kept, "
    f"clipped to {HERON_CLIP.bounds}."
)


def build_boundary():
    data = json.loads(SOURCE.read_text())
    extent = unary_union([shape(feature["geometry"]) for feature in data["features"]])
    bodies = list(extent.geoms) if extent.geom_type == "MultiPolygon" else [extent]
    heron_platform = max(bodies, key=lambda geom: geom.area)
    boundary = heron_platform.intersection(HERON_CLIP)
    if boundary.is_empty or boundary.geom_type not in ("Polygon", "MultiPolygon"):
        raise SystemExit("Clip produced no usable polygon; check HERON_CLIP")
    return boundary


def main():
    boundary = build_boundary()
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Heron Island Reef",
                    "reef_id": "heron_island",
                    "source": "Allen Coral Atlas reef extent",
                    "attribution": ATTRIBUTION,
                    "license": "CC BY 4.0",
                },
                "geometry": mapping(boundary),
            }
        ],
    }
    OUTPUT.write_text(json.dumps(feature_collection))
    print(f"Wrote {OUTPUT} ({boundary.geom_type}, bounds {[round(v, 5) for v in boundary.bounds]})")


if __name__ == "__main__":
    main()
