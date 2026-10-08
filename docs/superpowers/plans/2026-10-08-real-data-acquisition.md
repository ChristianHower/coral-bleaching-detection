# Real data acquisition — research and plan

Date: 2026-10-08.

The single biggest blocker to a field-meaningful run is data, not code. Build 1
is a working pipeline validated only on synthetic observations and small raster
fixtures (see `docs/build-1-status.md`). To replace the synthetic demonstration
we need three real things, each of which is currently either missing or
unverified for the Heron Island pilot:

1. Multi-date Sentinel-2 imagery with actual clear reef pixels (not just catalog
   hits) over the pilot extent.
2. Survey records that classify bleached vs. healthy state (not percent coral
   cover) and map onto the normalized-survey CSV contract.
3. A defensible analysis boundary (partly addressed — the bundled extent exists
   but its eastern cutoff is unreviewed; out of scope for this doc).

This document covers (1) and (2) and gives a prioritized acquisition plan with
explicit open questions. It does not claim any of the external data has been
downloaded or validated; every claim about external availability was checked
against the sources listed at the end and is dated 2026-10-08.

## 1. Sentinel-2 clear-imagery coverage over Heron Island

### What the existing checks already established

Two live checks have run against the public Earth Search STAC catalog
(`https://earth-search.aws.element84.com/v1`, collection `sentinel-2-l2a`). Both
are recorded in `docs/build-1-status.md` and the implementation plan:

- A broad catalog search for bbox `[151.90, -23.46, 151.94, -23.42]`,
  2020-01-01 through 2020-05-01, returned **97 matching items**. This counts
  scenes whose footprint intersects the box — it says nothing about cloud, sun
  glint, or water clarity over the reef.
- The repeatable smoke check (`scripts/smoke_public_sources.py`) processes a
  tiny access-test rectangle (`box(151.9148, -23.4423, 151.9155, -23.4416)`) on
  2020-04-29. It selected scene `S2A_56KLV_20200429_1_L2A`, and after quality
  masking the rectangle was **fully masked: null reflectance, unusable fraction
  1.0**. NOAA returned DHW 6.89 Celsius-weeks for the same date. That rectangle
  is explicitly not a reef boundary; the result confirms the adapter and honest
  missing-data handling, not usable imagery.
- The same broad search for 2016 returned **zero items** in `sentinel-2-l2a`.

So the state of knowledge is: catalog access works, the masking works, and we
have exactly one quality result — a fully-masked rectangle. We do **not** know
how many of the 97 scenes yield usable reef pixels over the real pilot extent.

### How the masking actually decides "usable" (the thing we need to count)

The adapter is `src/coral_bleaching/sources/sentinel.py`
(`EarthSearchClient`). Per scene it opens green, red, and SCL, reads one common
reef-bounding window, rasterizes cell IDs over that window, and aggregates per
cell. A pixel center counts as usable only if all of the following hold
(`observe_cells`):

- it falls inside a clipped reef cell (`represented = labels > 0`);
- green, red, and SCL are unmasked and finite, and not equal to each band's
  `nodata`;
- its SCL class is not in `UNUSABLE_SCL = (0, 1, 3, 7, 8, 9, 10, 11)` — i.e.
  nodata, saturated/defective, cloud shadow, clouds (medium/high prob), cirrus,
  snow/ice. SCL class 6 (water) is deliberately retained.

For each cell/scene it emits an `Observation` with mean green/red over usable
pixels and `cloud_cover_fraction = 1 - usable/represented` (fraction of *all*
unusable pixels, including shadow and nodata, not just cloud). If a cell has no
represented pixel centers or zero usable pixels, reflectance is null and the
fraction is 1.0. `observations()` then keeps, per cell per calendar day, the
single acquisition with the lowest unusable fraction, breaking ties by scene ID.

Downstream, a cell/date row is trainable only if `cloud_cover_fraction` is at or
below the configured threshold (default 0.5, i.e. at least half the cell's
reef pixels usable), and the experimental green/red index and its rolling shift
need six valid acquisitions with no gap over 14 days (README "Data quality and
evaluation"; implementation plan Task 4). Usable *coverage* and usable *time
series* are different bars — the second is much harder to clear.

### Concrete method to quantify usable clear reef pixels

The goal is to turn "97 catalog hits" into "N acquisition days with usable reef
pixels, and M cells with a long-enough clear time series," using the real
pilot extent and the exact masking logic already shipped — no new science.

Proposed throwaway coverage script (does not need to live in the package; it
reuses existing code so it stays honest about what "usable" means):

1. Load the real grid, not a rectangle. Build cells from
   `data/heron_island_pilot_extent.geojson` with the pilot config
   (`examples/heron_island.json`: EPSG:32756, 75 m cells, 2020-01-01 to
   2020-06-30), using the same grid code the pipeline uses.
2. Call `EarthSearchClient().observations(cells, boundary, start, end)` over the
   full pilot window. This runs the real search, pagination, windowed COG reads,
   scale/offset, SCL masking, and per-day best-scene selection.
3. For each returned `Observation`, record `(date, reef_cell_id,
   cloud_cover_fraction, usable_pixel_count, represented_pixel_count)`. The
   counts are computable from the same arrays `observe_cells` already builds; if
   the script needs them exposed, extend the throwaway script rather than
   changing the adapter's contract.
4. Aggregate into the numbers that actually matter:
   - **acquisition days with any usable reef pixel** (reef-wide): the real
     denominator the 97-scene figure was standing in for;
   - **per-day usable-cell fraction**: of the grid cells, how many clear the
     0.5 threshold on that day;
   - **per-cell count of usable days** across the window, and how many cells
     reach six usable acquisitions with no >14-day gap — the actual
     feature-eligibility bar for `index_shift`;
   - distribution of `cloud_cover_fraction` across all cell/day observations, to
     see whether 0.5 is doing real work or discarding almost everything.
5. Write the result to a dated JSON/CSV next to `live-source-check.json` and
   report it the way `docs/build-1-status.md` reports the smoke check: counts
   and limitations, no accuracy claim.

This is a read-only diagnostic. It tells us whether 2020 has enough usable
multi-date coverage over Heron to compute the rolling features at all, before
anyone spends time on labels. If almost no cells reach six clean acquisitions,
that is itself the finding, and it reshapes the rest of the plan.

Cost note: this reads only reef-bounding windows per scene (the adapter is
built for that), but it still touches up to ~97 scenes × 3 bands over HTTPS.
Run it once, cache, and record the run.

### The 2016 gap

2016 returned zero `sentinel-2-l2a` items in the earlier search. Heron sits in
MGRS tile 56KLV; Sentinel-2A was operating in 2016, so L1C almost certainly
exists, but the public **L2A** (surface reflectance) archive did not go back
that far in the queried Element 84 collection at check time. Options, in
honesty order:

- Check whether a reprocessed/Collection-1 L2A archive now covers 2016 for this
  tile on another mirror (Copernicus Data Space, Planetary Computer). This is an
  availability check, not a guarantee — verify the tile and date, do not assume.
- Do **not** silently substitute L1C. The adapter reads L2A surface reflectance
  with explicit scale/offset; feeding L1C top-of-atmosphere values through the
  same path would be an undocumented, unvalidated atmospheric-correction change.
  If 2016 is essential, a separately validated Sen2Cor (or equivalent) workflow
  is a real project, not a config tweak.
- Accept 2020-only for the first real run. The design (spec "Pilot Region")
  wanted 2016 and 2020 as positive/negative examples; dropping 2016 narrows the
  label range and should be stated as a limitation, not hidden.

Switching mirrors solves *access*, never clouds or missing acquisitions. If a
date was cloudy over Heron, every archive has the same cloudy scene.

## 2. Real bleaching labels

### The core problem: cover is not a bleaching label

The project is explicit (README live-pilot step 2; spec Module 1): percent
coral cover alone is not a bleaching label. The normalized CSV contract wants a
per-observation `label` of `bleached` or `healthy`. The most obvious "AIMS
data" is the wrong shape for this.

What AIMS actually publishes openly, verified 2026-10-08:

- **AIMS LTMP + MMP benthic survey downloads** (via eAtlas and the AIMS metadata
  catalog): two products — manta-tow reef-perimeter estimates (coral cover, COTS,
  dead coral, from 1994) and fixed-site photo-transect **percent cover** of
  high-level benthos groups (Hard Coral, Soft Coral, Algae, Other). These are
  **percent-cover** time series, not a bleached/healthy state classification.
  Access is via a short request form on the metadata record pages; a readme ships
  with the download describing fields and conditions of use. Source:
  https://eatlas.org.au/gbr/ltmp-data and the LTMP metadata record
  https://apps.aims.gov.au/metadata/view/a17249ab-5316-4396-bb27-29f2d568f727
- **AIMS Data Platform API** (`dataaimsr` R client, REST API with key): exposes
  only the **weather stations** and **sea-water temperature loggers** long-term
  datasets — not the LTMP benthic/bleaching survey records. So the convenient
  programmatic path does not deliver labels; it delivers in-situ temperature,
  which is a covariate at best. Source: https://docs.ropensci.org/dataaimsr/
- The AIMS **Reef Monitoring Dashboard** and annual GBR summaries report
  bleaching qualitatively and regionally (e.g. the 2024–25 summary describes
  bleaching severity by region), but the openly downloadable series behind them
  is still coral cover. Source: AIMS/GCRMN 2024–25 summary,
  https://gcrmn.net/2025/08/06/gbr-monitoring-2024/

So there is a real gap between "AIMS has bleaching knowledge" and "AIMS gives me
a per-site bleached/healthy label I can download in a table." Closing it needs
one of:

- a dataset that records **bleaching observation/severity per site/date**
  (in-water or aerial bleaching surveys), which may be request-based,
  publication-supplementary, or not public at the needed resolution; or
- a defensible, documented derivation of a binary label from a severity or
  cover-change signal — which is a scientific modeling decision, not a data
  download, and must be reviewed before use.

### Candidate real label sources (verify before trusting)

- **AIMS LTMP/MMP fixed-site surveys** — richest spatial detail (named reefs,
  fixed transects, site coordinates), annual cadence, back to the 1990s. But the
  open product is percent cover. Worth requesting to see whether the full survey
  export (behind the form) contains a bleaching field or scorable per-transect
  observations beyond the aggregated cover groups. Unknown until requested.
- **AIMS aerial / in-water bleaching surveys (2016, 2017, 2020, 2022, 2024
  events)** — these produced per-reef bleaching severity categories during mass
  bleaching years. Availability as a downloadable per-site table (vs. maps in
  reports) is **unverified**; this is the most promising shape if obtainable.
- **GBRMPA Reef eLibrary bleaching assessment reports** — contain bleaching
  monitoring results but as documents/maps, not clean tabular labels. Useful for
  cross-checking an event, not as a primary label feed.
- **Allen Coral Atlas bleaching "Reef Threats" layer** — satellite-derived
  bleaching alerts, bi-weekly, from January 2021 onward (source:
  https://allencoralatlas.org/blog/worlds-first-satellite-based-coral-reef-monitoring-system-deploys-globally...).
  It starts in 2021, so it does **not** cover the 2020 pilot window, and it is
  itself remote-sensing-derived — using it as ground truth for a remote-sensing
  model is circular. Reef-extent polygons are CC BY 4.0 (already used for the
  boundary); the Planet mosaic is CC BY-NC-SA 4.0. Treat the bleaching layer as
  a cross-reference, not a label source for 2020.
- **NOAA Coral Reef Watch DHW** — already integrated as a weak thermal-stress
  proxy and covariate, never independent localized bleaching evidence (README).
  It is not a label and must not be promoted to one.

### Likely resolution, format, licensing

Honest expectations, to be confirmed on download:

- **Format**: AIMS LTMP downloads are tabular (CSV/Excel) with a readme. The
  API returns JSON. Expect per-site / per-transect rows with reef name, site,
  lat/lon, survey date, depth, and benthos group percentages.
- **Spatial resolution**: fixed-site transects are point/transect-scale with
  real coordinates — finer than our 75 m cells, which is good: records can be
  mapped to a single clipped cell. Manta-tow is reef-perimeter aggregate — too
  coarse to place in one 75 m cell honestly, and should not be stretched to do
  so (README: "Do not enlarge survey footprints or invent per-cell labels").
- **Temporal match**: LTMP is roughly annual per reef. The pipeline matches
  surveys to acquisitions within 14 days (README; implementation plan). An
  annual survey gives few acquisition matches per year, so the number of
  *confirmed* training rows will be small — expect the conservative CV purging to
  leave little or nothing to score, exactly as the design warns.
- **Licensing**: AIMS data is generally released for reuse with attribution and
  a request form recording intended use; the `dataaimsr` client is MIT, but that
  is the code license, not the data license. Record the actual data license and
  citation from the download readme and keep it with the study (README: "Keep
  the input file's source citation and normalization method with the study").
  Do not assume CC BY without reading the readme.

### Mapping onto the normalized-survey CSV contract

Target contract (README live-pilot; implementation plan Task 7):
`survey_id, survey_date, label` plus either `longitude, latitude` or
`reef_cell_id`. Dates `YYYY-MM-DD`; `label` strictly `bleached` or `healthy`;
`survey_id` uniquely identifies one spatial observation; coordinate records
must fall unambiguously in a single clipped cell.

Mapping steps once a bleaching-classified source is in hand:

1. `survey_id` ← the source's stable record/transect ID (do not synthesize from
   row order; the importer rejects ambiguity and ties).
2. `survey_date` ← actual survey date, never inferred from an imagery date.
3. `label` ← the hard part. Only legitimate if the source records bleaching
   state or a severity the project decides (and documents) to threshold into
   bleached/healthy. If the source is percent cover only, there is **no honest
   direct mapping** and this step blocks until a bleaching-classified source or a
   reviewed derivation rule exists.
4. Coordinates ← transect lat/lon in WGS84. Confirm each falls in exactly one
   75 m clipped cell for this run's grid; drop or pre-map (with a justified
   spatial join) anything ambiguous. Reef-aggregate rows (manta tow) do not get a
   per-cell label.
5. Keep source citation, license, and the exact normalization method with the
   file. Inputs are hashed into the run manifest.

## 3. Prioritized acquisition plan

Ordered so the cheapest disqualifying check runs first. Stop and reassess at any
step that fails.

1. **Quantify 2020 usable imagery coverage** (Section 1 method). Read-only,
   needs no accounts, reuses shipped code. Decision point: are there enough cells
   with six clean acquisitions and no >14-day gap to compute rolling features at
   all? If not, the imagery side is the blocker and labels are premature.
2. **Confirm the pilot window against the real goal.** If 2016 is required for
   positive examples, run the mirror availability checks for L2A over tile 56KLV
   in 2016. Decision point: 2016 L2A exists on a verified mirror → extend window;
   does not exist → document 2020-only and the lost label range.
3. **Request AIMS LTMP/MMP fixed-site data** through the metadata form, stating
   the research use honestly. On receipt, inspect whether anything beyond
   aggregated percent cover is present (per-transect bleaching, severity).
   Decision point: is there a bleaching-state field, or only cover?
4. **If only cover exists, pursue a bleaching-specific source**: AIMS aerial/
   in-water bleaching survey tables for 2020, or published event
   supplementary data. Decision point: can we obtain per-site bleaching severity
   with coordinates and dates, licensed for reuse? If yes, proceed. If no, the
   project either defines and reviews a derivation rule from cover change (a
   scientific decision, logged and reviewed) or honestly cannot produce confirmed
   labels yet.
5. **Normalize to the CSV contract** (Section 2 mapping) once a
   bleaching-classified source exists. Validate against the importer on a small
   subset first; it rejects malformed dates/labels, unknown cells, and ties.
6. **Review the analysis boundary** (`data/heron_island_pilot_extent.geojson`
   eastern cutoff) against a named-reef or management boundary before any field
   interpretation. Tracked separately; it gates interpretation, not the data
   pull.
7. **Dry-run the real pipeline** on whatever real imagery + labels exist, read
   the per-fold validation honestly (expect `skipped` / `not_evaluable` with few
   confirmed labels), and report it like `docs/build-1-status.md` — software
   behavior, not field accuracy.

### Open questions and decision points

- Does any AIMS open product give a **bleached/healthy** (or scorable severity)
  label per site/date, or is everything accessible to us percent cover? This is
  the pivotal unknown; it was not resolvable from public metadata alone and
  needs the actual request/download to answer.
- For 2020, how many Heron grid cells actually reach **six clean acquisitions
  with no >14-day gap**? Unknown until the Section 1 coverage script runs; the
  97-scene figure does not answer it.
- Is 2016 required? If so, is there a verified L2A archive for tile 56KLV in
  2016, and are we willing to fund a validated atmospheric-correction workflow if
  not?
- If no direct bleaching label is obtainable, will the project define and have
  reviewed a derivation from cover change / severity, or hold the real run until
  a proper label source exists? This is a scientific decision, out of scope to
  decide here.
- Even with imagery and labels, the green/red index and its rolling change are
  unvalidated for reef water (README): a successful run still would not be a
  validated bleaching detector.

### What remains unknown / unvalidated

- No real bleaching-survey data has been downloaded, inspected, or normalized.
  All label-source claims above are from public metadata and are marked
  unverified where that is the case.
- Usable 2020 reef-pixel coverage over the real extent is unquantified; only the
  one fully-masked rectangle result exists.
- 2016 L2A availability on alternative mirrors was not re-tested in this doc; the
  only data point is the earlier zero-result search in `sentinel-2-l2a`.
- The pilot boundary's eastern cutoff is still unreviewed.
- The spectral index itself is experimental and unvalidated for reef conditions.

Nothing here establishes field performance. It establishes what to acquire, in
what order, and where the plan is still guessing.

## Sources checked (2026-10-08)

- Sentinel-2 adapter and masking: `src/coral_bleaching/sources/sentinel.py`
  (this repo).
- Earth Search catalog, 97-scene / zero-2016 results, access-test rectangle:
  `docs/build-1-status.md`, `README.md`,
  `docs/superpowers/plans/2026-09-04-coral-bleaching-pipeline-model.md`,
  `scripts/smoke_public_sources.py` (this repo).
- AIMS LTMP/MMP data (percent cover, request-form download):
  https://eatlas.org.au/gbr/ltmp-data and
  https://apps.aims.gov.au/metadata/view/a17249ab-5316-4396-bb27-29f2d568f727
- AIMS Data Platform API scope (weather + temperature loggers only):
  https://docs.ropensci.org/dataaimsr/
- AIMS/GCRMN 2024–25 GBR monitoring summary (bleaching reported qualitatively,
  cover reported quantitatively): https://gcrmn.net/2025/08/06/gbr-monitoring-2024/
- Allen Coral Atlas bleaching monitoring layer (bi-weekly, Jan 2021 onward):
  https://allencoralatlas.org/blog/worlds-first-satellite-based-coral-reef-monitoring-system-deploys-globally-paving-the-way-for-innovation-driven-conservation/
- Allen Coral Atlas reef-extent license / citation (CC BY 4.0; mosaic CC
  BY-NC-SA 4.0): `data/sources/README.md` (this repo).
- NOAA Coral Reef Watch DHW (weak proxy, already integrated):
  https://coralreefwatch.noaa.gov/product/5km/methodology.php
- Copernicus Data Space Sentinel-2 reference (mirror/availability check path):
  https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html
