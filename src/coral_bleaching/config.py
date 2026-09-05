import math
import re
from dataclasses import asdict, dataclass
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
    min_overlap_fraction: float = 0.1
    max_history_gap_days: int = 14
    rolling_window: int = 3
    grid_origin_x: float = 0.0
    grid_origin_y: float = 0.0
    provider: str = "earth-search"
    collection: str = "sentinel-2-l2a"

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.reef_id):
            raise ValueError("reef_id must be a nonempty path-safe identifier")
        if self.start_date > self.end_date:
            raise ValueError("start_date must not follow end_date")
        for key in ("cell_size_meters", "dhw_risk_threshold", "grid_origin_x", "grid_origin_y"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if self.cell_size_meters <= 0 or self.dhw_risk_threshold < 0:
            raise ValueError("Invalid cell size or DHW threshold")
        for key in ("cloud_cover_drop_threshold", "min_overlap_fraction"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must lie in [0, 1]")
        for key in ("aims_label_window_days", "max_history_gap_days", "rolling_window"):
            value = getattr(self, key)
            if not isinstance(value, int) or value < (0 if key == "aims_label_window_days" else 1):
                raise ValueError(f"Invalid {key}")
        if self.provider != "earth-search":
            raise ValueError("Build 1 currently supports earth-search")

    def to_dict(self):
        return {k: v.isoformat() if isinstance(v, date) else v for k, v in asdict(self).items()}
