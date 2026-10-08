# Proof-of-Concept Map UI — Build Brief

**Status:** barebones proof of concept. Real UI/visual design input comes
later — get something on screen that proves the shape of the concept, and
follow the style rules in `AGENTS.md` in the meantime so the first pass
isn't obviously AI-generated.

**Goal:** view a Build 1 run's predictions on a map. Click a reef cell, see
its risk and the reasoning behind it.

## Scope

In scope:
- A read-only local web app that visualizes one `runs/<name>/` directory
  at a time (e.g. `runs/demo-001`, produced by `coral-bleaching demo`).
- A map of the reef-cell grid (`grid.geojson`), each cell colored by its
  prediction from `predictions.parquet`.
- Click a cell to see: `probability`, `predicted_label`, `confidence_band`,
  `data_quality`, and `top_contributing_features` (the reasoning).
- Cells with `data_quality: insufficient` are visually distinct (e.g. grey/
  hatched) rather than colored by a null probability.

Out of scope for this pass (do not build yet):
- No database, no auth, no multi-run history, no feedback loop
  (ranger confirm/reject) — that's Module 5, later.
- No write path at all. This is read-only.
- No production deployment concerns (HTTPS, containerization, etc).
- No polished visual design — functional and legible is the bar, not
  refined. Real design direction comes in a later pass.

## Suggested shape (adjust as needed — you have latitude here)

- A small FastAPI (or Flask, your call) app with one endpoint,
  `GET /api/runs/{run_name}`, that:
  - Reads `runs/{run_name}/grid.geojson` and
    `runs/{run_name}/predictions.parquet`.
  - Joins predictions onto grid features by `reef_cell_id` + `date`. The
    endpoint selects one reef-wide date (latest by default) and accepts a
    `date` query parameter so the map never mixes acquisition dates.
  - Returns a single GeoJSON FeatureCollection with the prediction fields
    merged into each feature's properties.
  - 404s clearly if the run directory or its files don't exist.
- A single static HTML page (vanilla JS, Leaflet via CDN — no build step)
  that fetches that endpoint and renders the map, colored cells, and a
  click popup with the reasoning fields.
- Wire up a way to run it locally (a script or a documented command) and
  add that command to README.md the same way existing CLI commands are
  documented there.

## Acceptance

- Running against `runs/demo-001` (generate it with `coral-bleaching demo
  --output runs/demo-001` if it doesn't exist) shows all cells on the map,
  colored by prediction, with working click-to-inspect.
- A cell with `data_quality: insufficient` renders distinctly and its
  popup shows there's no prediction available, rather than showing a
  null/blank probability.
- A cell without a record on the selected date is visually and textually
  distinct from an imagery-quality abstention.
- Document the run command in README.md.
- No new heavyweight dependencies beyond a web framework (FastAPI/Flask)
  and its ASGI/WSGI server — keep the frontend dependency-free (CDN
  Leaflet is fine).

Use your own judgment on file layout and framework choice within the
above — this brief intentionally leaves implementation details open since
the priority right now is a working proof of concept, not a fully
specified plan.
