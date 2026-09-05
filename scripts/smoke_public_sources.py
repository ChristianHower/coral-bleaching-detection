"""Optional live check. Public imagery + NOAA only; no model accuracy claim."""

import argparse
import json
from datetime import date
from pathlib import Path

from shapely.geometry import box

from coral_bleaching.grid import ReefCell
from coral_bleaching.sources.noaa import NOAAClient
from coral_bleaching.sources.sentinel import EarthSearchClient

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
boundary = box(151.9148, -23.4423, 151.9155, -23.4416)
cell = ReefCell("access_test_rectangle", boundary, boundary.centroid.y, boundary.centroid.x)
client = EarthSearchClient()
items = client.search(boundary, date(2020, 4, 29), date(2020, 4, 29))
if not items:
    raise RuntimeError("No matching scene")
observation = client.observe(items[0], cell)
noaa = NOAAClient()
dhw = noaa.fetch_dhw(cell.centroid_lon, cell.centroid_lat, observation.date)
output = {
    "purpose": "Live source compatibility; rectangle is not a verified reef boundary",
    "observation": {**observation.__dict__, "date": observation.date.isoformat()},
    "dhw": dhw,
    "sentinel": client.provenance,
    "noaa": noaa.provenance,
}
Path(args.output).write_text(json.dumps(output, indent=2))
print(
    json.dumps(
        {
            "scene": observation.source_scene_id,
            "green": observation.band_green,
            "red": observation.band_red,
            "unusable_fraction": observation.cloud_cover_fraction,
            "dhw": dhw,
        }
    )
)
