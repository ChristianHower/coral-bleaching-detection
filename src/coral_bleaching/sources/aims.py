import csv
from dataclasses import dataclass
from datetime import date

from shapely.geometry import Point


@dataclass(frozen=True)
class Survey:
    survey_id: str
    reef_cell_id: str
    survey_date: date
    label: str


def load_aims_labels(csv_path, cells=None):
    """Read normalized surveys, never infer bleaching from coral cover percentages."""
    records, identities = [], set()
    with open(csv_path, newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"survey_id", "survey_date", "label"} <= set(reader.fieldnames or []):
            raise ValueError("AIMS input needs survey_id, survey_date and label")
        for row in reader:
            sid, label = row["survey_id"].strip(), row["label"].strip()
            if not sid or label not in ("bleached", "healthy"):
                raise ValueError("Invalid survey identity or label")
            cell_id = row.get("reef_cell_id", "").strip()
            if not cell_id:
                point = Point(float(row["longitude"]), float(row["latitude"]))
                matches = [c.reef_cell_id for c in (cells or []) if c.geometry_wgs84.covers(point)]
                if len(matches) != 1:
                    raise ValueError(f"Survey {sid} must map unambiguously to one reef cell")
                cell_id = matches[0]
            if cells is not None and cell_id not in {c.reef_cell_id for c in cells}:
                raise ValueError(f"Unknown cell {cell_id}")
            if sid in identities:
                raise ValueError(f"Duplicate survey_id {sid}; use unique observation IDs")
            identities.add(sid)
            records.append(Survey(sid, cell_id, date.fromisoformat(row["survey_date"]), label))
    return records


def find_matching_label(records, reef_cell_id, obs_date, window_days=14):
    candidates = [
        (abs((r.survey_date - obs_date).days), r)
        for r in records
        if r.reef_cell_id == reef_cell_id and abs((r.survey_date - obs_date).days) <= window_days
    ]
    if not candidates:
        return None, None
    distance = min(d for d, _ in candidates)
    nearest = [r for d, r in candidates if d == distance]
    if len({r.label for r in nearest}) > 1:
        return None, {
            "reason": "conflicting_equidistant_surveys",
            "survey_ids": sorted(r.survey_id for r in nearest),
        }
    return sorted(nearest, key=lambda r: (r.survey_date, r.survey_id))[0], None
