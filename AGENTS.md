# AGENTS.md

This file applies to the entire repository. Read it before changing code, data, documentation, or tooling. A more deeply nested `AGENTS.md`, if added later, overrides this file for its subtree.

## Start here

Before making changes:

1. Run `git status --short` and preserve unrelated user changes.
2. Read the task-relevant source plus the linked files in the architecture map below.
3. For network-generation behavior, read `DEBUG_GUIDE.md` before changing thresholds.
4. If `graphify-out/graph.json` exists, use `graphify-out/GRAPH_REPORT.md` and the graph as an orientation aid. Treat `INFERRED` graph edges as hypotheses and verify them in source.
5. Establish a baseline with the narrowest useful check; for general work, use `./gradlew build`.

Do not assume the README is more authoritative than the build and source. The Gradle build requests JDK 23. The current planner runs on Ktor; the older JDK `HttpServer` flow is historical.

## Project overview

PublicTransportationPlanner is a small Kotlin/JVM application that serves a single-page Leaflet map and builds proposed metro networks from population-grid points. There is no separate deployed frontend or database.

- Language: Kotlin 2.2.20 on JVM toolchain 23
- Build: Gradle 8.14 wrapper with the `application` plugin
- Entrypoint: `org.lsoffice.MainKt`
- HTTP server: Ktor Netty, with JSON API routes in `Main.kt`
- Frontend: `map.html`, `planner.css`, and `planner.js` using CDN-hosted Leaflet and Leaflet Draw
- Data: classpath CSV/JSON resources, including a roughly 29 MB population grid
- Tests: Kotlin tests under `src/test/kotlin`

## Architecture map

### Backend and domain

- `src/main/kotlin/Main.kt`
  - Starts the local HTTP server on the first free port in `5000..5010`.
  - Serves classpath resources and implements all HTTP handlers.
  - Parses CSV requests and manually serializes response JSON.
  - Owns the shared `haversineMeters(lon1, lat1, lon2, lat2)` helper.
- `src/main/kotlin/MetroBuilder.kt`
  - Contains the domain types: `Zone`, `GridPoint`, `Station`, `Hub`, `Corridor`, `Line`, `LineType`, `BuilderParams`, and fitness-report types.
  - Contains both the older OD/hub/corridor builder and the main natural-network algorithm.
  - Computes journey metrics, network fitness, line frequency, interchange snapping, corridor classification, and station pruning.
- `src/main/kotlin/PtalLookup.kt`
  - Lazily loads `ptal_spatial.csv`.
  - Performs a linear nearest-neighbour lookup within 3 km by default.
  - Converts London PTAI values into demand/activity multipliers; locations outside coverage fall back to neutral weights.

### Frontend

- `src/main/resources/map.html`
  - Is source code, not a generated build artifact.
  - Loads the population grid, supports polygon drawing, calculates area/centroid, draws density points, requests route suggestions, renders routes/interchanges, reverse-geocodes station names, and generates a downloadable schematic tube map.
  - Depends on external CDNs and OpenStreetMap/Nominatim at runtime.

### Data and documentation

- `src/main/resources/gbr_pd_2020_1km_ASCII_XYZ.csv`: primary population/density grid, columns `X,Y,Z` (`lon,lat,value`); large file, about 499,000 data rows.
- `src/main/resources/ptal_spatial.csv`: runtime PTAL lookup data, columns `lon,lat,avgPtai2015,ptalNumeric`.
- `src/main/resources/ptal_lsoa2011.json`: source/metadata-rich PTAL dataset; not loaded by current Kotlin runtime code.
- `scripts/sample.csv`: small input fixture useful for manual work.
- `DEBUG_GUIDE.md`: authoritative guide to the natural-network checkpoints and common threshold failures.
- `README.md`: user-facing overview and quickstart; update it when user-visible behavior changes.
- `.trunk/trunk.yaml`: optional formatting/linting/secret-scanning configuration.

## Runtime flows

The browser's normal route workflow is:

1. `map.html` loads the population-grid CSV.
2. The user draws a polygon; Turf filters points inside it.
3. The browser posts matching rows as `lon,lat,value` CSV to `/suggestions`.
4. The server applies PTAL demand weights where London PTAL data is available.
5. `MetroBuilder.buildNaturalNetworkFromGrid` clusters places, builds demand relationships and candidate chains, classifies/selects corridors, prunes stations, evaluates fitness, snaps nearby cross-line stations into interchanges, and assigns trains per hour.
6. The server returns lines; the browser renders geographic and schematic maps.

Do not confuse that path with `/metro_suggestions`. The latter runs the older staged flow (`buildODMatrix` -> hubs -> stations -> corridors -> `optimizeNetwork`) over the full bundled grid and is not the route-button flow used by `map.html`.

### HTTP surface

| Method | Path                                 | Contract                                                                                                 |
| ------ | ------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `GET`  | `/` and static paths                 | Serves `map.html` and classpath resources.                                                               |
| `GET`  | `/density?lon=...&lat=...&max_m=...` | Returns the nearest population-grid cell or 404. Default radius is 1,000 m.                              |
| `POST` | `/suggestions`                       | Accepts newline-separated `lon,lat,value`; returns `{ "lines": [...] }`. This is the main UI route flow. |
| `GET`  | `/metro_suggestions`                 | Runs the legacy full-grid planning flow and returns stations plus lines.                                 |
| `GET`  | `/proxy?url=...`                     | Proxies an external HTTP(S) GET; used for Nominatim station naming. Security-sensitive.                  |

The `/suggestions` line response is consumed directly by `map.html`. Preserve these fields unless backend and frontend change together:

- Line: `id`, `type`, `isLoop`, `length_m`, `cost`, `trains_per_hour`, `stations`
- Station: `id`, `lon`, `lat`, `value`

`LineType` values are `RADIAL_TRUNK`, `ORBITAL`, `CORE_DISTRIBUTOR`, and `NOT_METRO`. Shared interchange station IDs use the `INT_` prefix.

## Geospatial and model conventions

- Kotlin domain objects and CSV rows use longitude before latitude.
- GeoJSON/Turf coordinates use `[longitude, latitude]`.
- Leaflet display coordinates use `[latitude, longitude]`.
- Distances are meters unless a name explicitly says kilometres; journey times are minutes; line frequency is trains per hour.
- `haversineMeters` is the common geographic distance function. Preserve argument order when calling it.
- The grid `value` is treated as a density/demand proxy. Avoid relabelling it as an exact population without checking the calling flow.
- PTAL enrichment applies only near the bundled London centroids. Neutral fallback behavior outside London is intentional.
- `MetroBuilder(debug = true)` logs checkpoint details by default. Find the first collapsing checkpoint before tuning any threshold, and change one threshold at a time.
- Some legacy station-generation paths use random coordinate jitter, so their exact output is nondeterministic. Do not write exact-coordinate assertions for those paths without injecting or removing the randomness.

## Build, run, and checks

Use the wrapper; do not require a separately installed Gradle.

```bash
./gradlew build
./gradlew test
./gradlew run
```

`./gradlew build` is the required baseline check for Kotlin/resource changes. It currently reports `NO-SOURCE` for tests. The running URL is printed to the terminal and may use ports 5000 through 5010.

Optional repository-wide checks, when the Trunk CLI is installed:

```bash
trunk check
trunk fmt
```

There is no Gradle lint task, JavaScript package manager, browser-test suite, CI workflow, or database migration process in this repository. Do not claim those checks ran.

### Verification by change type

- Kotlin/domain logic: add focused `kotlin.test` tests under `src/test/kotlin` where practical, then run `./gradlew test` and `./gradlew build`.
- Natural-network thresholds or classification: run a representative `/suggestions` request, inspect all debug checkpoints, and compare line types, station counts, lengths, loops, interchanges, and fitness output.
- HTTP contracts: smoke-test the affected endpoint and verify status, content type, error path, and browser-consumed field names.
- `map.html`: run the app and manually exercise drawing, density display, route show/hide, station naming, and tube-map generation as relevant. Remember that CDN/network availability affects the page.
- Resource data: validate headers, row parsing, coordinate order, missing-resource behavior, and representative in/out-of-London lookups. Avoid loading or rewriting the 29 MB population CSV unnecessarily.
- Documentation/tooling only: run the smallest relevant formatter or syntax check; a full build is optional if runtime inputs did not change.

## Coding conventions

- Follow official Kotlin style (`kotlin.code.style=official`), four-space indentation, trailing commas in multiline constructs, and existing package `org.lsoffice`.
- Prefer small named helpers over adding more deeply nested logic to `Main.kt` or `buildNaturalNetworkFromGrid`.
- Preserve immutable data-class transformations (`copy`) where used.
- Keep units in names for new numeric values (`Meters`, `Mins`, `PerYear`, and similar).
- Validate untrusted HTTP and CSV input at the boundary and return explicit 4xx errors for client mistakes.
- Use a JSON serializer for new complex contracts when feasible. If touching current manual serialization, account for escaping and update `map.html` in the same change when the schema changes.
- Keep frontend dependencies pinned to explicit versions. Match the existing plain JavaScript and CSS architecture unless the task explicitly introduces a frontend build system.
- Do not perform broad formatting of `map.html` or the 2,000-line `MetroBuilder.kt` during an unrelated change.
- Comments should explain model assumptions, thresholds, units, or non-obvious trade-offs rather than restating code.

## Guardrails and known hazards

- Preserve large source datasets. Do not regenerate, normalize, sort, or reformat CSV/JSON resources unless the task specifically requires it and provenance/row counts are verified.
- The Ktor dependencies in `build.gradle.kts` are currently unused. Do not implement handlers as though a Ktor application already exists; removing or adopting Ktor should be an intentional change.
- `/proxy` accepts arbitrary HTTP(S) targets and handles `GITHUB_TOKEN`; treat changes there as security-sensitive. Never print tokens, send credentials to untrusted hosts, or broaden proxy behavior. Prefer host allowlisting and host-scoped authorization when addressing it.
- Endpoint handlers repeatedly scan large CSV data. Be alert to latency and memory impact; do not add additional full scans casually.
- Network evaluation assumes station IDs identify graph nodes. Preserve stable IDs and shared interchange identity when modifying line assembly.
- `snapInterchangeStations` deliberately replaces nearby stations on different lines with a shared weighted midpoint. Recompute line lengths after station-coordinate changes.
- `DEBUG_GUIDE.md` documents current checkpoint meaning. Update it alongside changes to checkpoint order, thresholds, classification, pruning, or log wording.
- No license is declared. Do not add third-party data/code without checking its licence and recording attribution.

## Definition of done

A change is complete when it is scoped to the request, preserves unrelated work, keeps backend/frontend/data contracts aligned, passes the applicable checks above, and documents any untested or nondeterministic behavior. Report the commands actually run and any remaining verification gaps.
