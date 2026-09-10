# Source data

## `allen_coral_atlas_reef_extent.geojson`

Reef-extent polygons for the Capricornia Cays area (Heron, Wistari, Sykes and
smaller reef patches), exported from the Allen Coral Atlas on 2026-09-08.
WGS84 (CRS84), 31 polygons, all class `Reef`.

Used by `scripts/build_pilot_boundary.py` to derive
`data/heron_island_pilot_extent.geojson`. The source does not contain named
reef boundaries and joins Heron Reef to the sandy shoal extending toward
Sykes Reef. The derived file uses a documented longitude clip and is an
approximate pilot analysis extent, not an authoritative ecological or
management boundary.

**Citation:** Allen Coral Atlas (2020). Imagery, maps and monitoring of the
world's tropical coral reefs. https://doi.org/10.5281/zenodo.3833242

**License:** Reef-extent maps are © 2020 Allen Coral Atlas Partnership and
Vulcan, Inc., licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Not included from the same export (kept out of version control for size, and
not needed for the boundary): the benthic habitat map (~8 MB), the geomorphic
map, and the 2020 Planet Labs satellite mosaic (~32 MB, separate CC BY-NC-SA
4.0 license). Re-download from https://allencoralatlas.org if needed.
