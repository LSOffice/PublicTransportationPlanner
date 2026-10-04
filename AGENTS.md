# Repository guidance

Read this file and run `git status --short` before editing. Preserve unrelated work. Read the source being changed, `DEBUG_GUIDE.md` for generation thresholds, and `graphify-out/GRAPH_REPORT.md` if present (verify inferred edges in source). Use `./gradlew build` as the general baseline.

## Architecture

- Kotlin 2.2.20, JVM 23, Gradle 8.14 wrapper, Ktor 3.6 Netty. `Main.kt` owns HTTP routes, response handling, static resources, and `haversineMeters`. The entrypoint is `org.lsoffice.MainKt`.
- `PlannerModels.kt` defines the versioned planner JSON contract. `PlannerService.kt` validates study areas, loads Greater London grid cells, builds analysis sessions, and bridges generation. `PrimitiveRoutingEngine.kt` implements Dijkstra/A*.
- `SparseParetoGenerator.kt` clusters points with a spatial hash, builds a JTS Delaunay graph, blends regional NUMBAT evidence when available, discovers bounded corridors, searches five objective profiles, and returns plans. `GenerationJobs.kt` retains four jobs for 30 minutes and emits typed SSE. `MetroBuilder.kt` remains for legacy generation and is also used to consolidate stations and estimate build technology for planner output.
- `NumbatDemand.kt` loads the compact derived 2024 regional OD resource, falling back to raw files when it is unavailable. `NumbatPreaggregate.kt` and `NumbatCalibration.kt` are offline preparation tasks. Never parse the raw NUMBAT files on the request path when the compact resource exists.
- `map.html` and `planner.css` are committed resources. `frontend/planner.ts` is the TypeScript browser entrypoint; esbuild produces `build/generated-resources/planner.js`, which Gradle adds to classpath resources. Do not commit the compiled bundle. Leaflet and Leaflet Draw remain pinned CDN dependencies. Every launch shows the project chooser before opening a local project. Projects are stored in IndexedDB with schema v2; Review drafts remain outside project JSON and commit once through Lock in routes after evaluation. Tablet widths 768–1099px use a bottom task sheet; `frontend/planner-state.ts` migrates v1 projects and imports.
- `src/main/resources/gbr_pd_2020_1km_ASCII_XYZ.csv` is a large source grid. Preserve it and all raw data files. See `DATA_SOURCES.md` for provenance and checksum details.

## HTTP surface

- `POST /api/v1/generation/jobs` creates an asynchronous job (`202`). `GET /api/v1/generation/jobs/{id}/events` is SSE with monotonic event IDs and replay. `GET` on the job path recovers status/result. `DELETE` cancels. `POST /api/v1/generation/jobs/{id}/evaluate` evaluates selected candidate IDs.
- `POST /api/v1/networks/generate` is the synchronous balanced-plan compatibility wrapper. `/api/v1/analysis/sessions` and its demand, issues, journey, and simulation subroutes support Analyse. `/api/v1/coverage`, `/density`, and `/proxy` are also in `Main.kt`.
- The old `/suggestions` and `/metro_suggestions` routes are compatibility surfaces. Do not confuse them with the staged planner workflow.

## Conventions and checks

- Coordinates are longitude, latitude in Kotlin and GeoJSON; Leaflet displays latitude, longitude. Distances are metres, journey times minutes, frequency trains per hour. Grid values are density/demand proxies, not exact populations. PTAL outside London falls back to neutral weight.
- Keep station IDs stable and shared at interchanges. Recompute line lengths after moving stations. New untrusted HTTP/CSV input gets boundary validation and explicit 4xx responses. Treat `/proxy` and token handling as security sensitive.
- Kotlin logic changes: focused `kotlin.test` tests, `./gradlew test`, and `./gradlew build`. Generator changes: run the London job test and inspect candidate counts, types, lengths, station consolidation, costs, certification/gaps, and SSE events. UI changes: `npm test`, `npm run build`, and a desktop render check. Full gate: `./gradlew clean build`.
- Generated graph and Impeccable review artifacts are ignored. Do not reformat or regenerate large source datasets. Update `README.md`, `PLANNER_MODEL.md`, and `DEBUG_GUIDE.md` when the public model or checkpoints change.
