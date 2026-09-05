import math
from datetime import datetime
from urllib.parse import quote

import requests

NOAA_DHW_ERDDAP_URL = "https://coastwatch.noaa.gov/erddap/griddap/noaacrwdhwDaily.json"


class NOAAClient:
    def __init__(self, http_get=None, endpoint=NOAA_DHW_ERDDAP_URL):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "coral-bleaching-detection/0.1 (+https://github.com/ChristianHower/coral-bleaching-detection)"
        )
        self.http_get = http_get or self.session.get
        self.endpoint = endpoint
        self.cache = {}
        self.provenance = {}

    def fetch_dhw(self, lon, lat, obs_date):
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Invalid NOAA coordinates")
        # Snap to NOAA's 0.05 degree cell centers; reuse across reef cells.
        x = round(math.floor((lon + 180) / 0.05) * 0.05 - 179.975, 3)
        y = round(math.floor((lat + 90) / 0.05) * 0.05 - 89.975, 3)
        key = (x, y, obs_date.isoformat())
        if key in self.cache:
            return self.cache[key]
        query = quote(
            f"degree_heating_week[({obs_date.isoformat()}T12:00:00Z)][({y})][({x})]", safe="(),:-"
        )
        url = self.endpoint + "?" + query
        response = self.http_get(url, timeout=60)
        response.raise_for_status()
        table = response.json()["table"]
        names, rows = table["columnNames"], table["rows"]
        if len(rows) != 1:
            raise ValueError("NOAA must return exactly one daily grid value")
        record = dict(zip(names, rows[0], strict=True))
        value = float(record["degree_heating_week"])
        timestamp = datetime.fromisoformat(record["time"].replace("Z", "+00:00"))
        if timestamp.date() != obs_date:
            raise ValueError("NOAA returned a different date")
        if (
            abs(float(record["latitude"]) - y) > 0.001
            or abs(float(record["longitude"]) - x) > 0.001
        ):
            raise ValueError("NOAA returned a different grid cell")
        if not math.isfinite(value) or value < 0:
            raise ValueError("Missing or invalid NOAA DHW")
        self.cache[key] = value
        self.provenance[str(key)] = {"url": url, "dhw": value, "record": record}
        return value
