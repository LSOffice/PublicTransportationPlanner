# London Network Planner

A local desktop and tablet workspace for proposing and testing hypothetical metro networks around Greater London. The map, network editor, synthetic demand analysis, journey routing, and disruption simulation run from one Kotlin application. Results are planning estimates, not live TfL conditions or engineering feasibility claims.

## Workflow

1. **Create:** draw a supported study area, choose NUMBAT blend or simulated gravity demand, set guidance strength, and start a streamed search. The timeline and map show candidate discovery; review the balanced recommendation and four alternatives. If fewer than five distinct plans exist, the result gives an insufficiency reason.
2. **Review routes:** select a plan, adjust corridors by role, and inspect local length plus server-evaluated metrics. Lock in routes saves the evaluated network, settings, and corridor provenance in one revision. Leaving Review keeps the previously saved network.
3. **Refine:** add lines, edit stations and track, inspect key metrics, or download a schematic SVG.
4. **Analyse:** inspect simulated OD demand, issues, Dijkstra/A* journeys, service levels, disruptions, and train simulation.

Each launch opens a project start screen: create a named project, open one saved in this browser, or import JSON. Stages unlock when a network exists; you can always return to earlier work. The single task panel floats on desktop and becomes a collapsible bottom sheet on tablet. Projects autosave to IndexedDB; the Project menu handles new, save, import, and export. JSON schema v2 records generation settings and corridor provenance. Existing v1 projects migrate when opened.

## Architecture and run

- Kotlin 2.2.20, JVM toolchain 23, Gradle 8.14 wrapper, Ktor 3.6 Netty on the first free `127.0.0.1:5000..5010` port.
- Spatial-hash clustering and JTS 1.20 Delaunay edges keep graph storage proportional to places. A principal-axis chain handles collinear sites. A bounded corridor search uses residual demand, angular penalties, and five objective profiles. The final weighted-profile search runs within an eight-second refinement budget and reports certification/gap status.
- A compact 2024 NUMBAT regional OD resource supplies optional observed evidence; unmatched locations use deterministic gravity demand. Analysis demand remains separately simulated and is labelled as such.
- TypeScript and pinned esbuild/Vitest produce a generated browser bundle during the Gradle build. The UI uses Leaflet 1.9.4 and Leaflet Draw 1.0.4 from CDNs.

```bash
./gradlew clean build
./gradlew run
```

Open the printed URL at a width of at least 768 px. No database or account is required.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/generation/jobs` | Create a search job (`202`) |
| `GET` | `/api/v1/generation/jobs/{id}/events` | Typed SSE with monotonic IDs and replay via `Last-Event-ID` |
| `GET` | `/api/v1/generation/jobs/{id}` | Recover status and final plans |
| `DELETE` | `/api/v1/generation/jobs/{id}` | Cancel a job |
| `POST` | `/api/v1/generation/jobs/{id}/evaluate` | Evaluate a selected candidate bundle |
| `POST` | `/api/v1/networks/generate` | Synchronous balanced-plan compatibility response |
| `POST` | `/api/v1/analysis/sessions` | Compile demand, score, and issues |
| `GET` | `/api/v1/analysis/sessions/{id}/demand` | Page through simulated OD rows |
| `POST` | `/api/v1/analysis/sessions/{id}/journeys` | Route with Dijkstra or A* |
| `POST` | `/api/v1/analysis/sessions/{id}/simulations` | Compile headway-derived runs |

Jobs retain a bounded replay buffer; at most four jobs live for 30 minutes. Analysis sessions are also bounded in memory. `/suggestions` and `/metro_suggestions` remain legacy compatibility routes.

## Checks and limits

```bash
npm test
npm run build
./gradlew test
./gradlew clean build
```

The graph benchmark test records edge counts for 1k, 5k, and 10k synthetic places. See `PLANNER_MODEL.md` for objectives and certification, `calibration/numbat-2024.json` for the offline cap sweep, `DATA_SOURCES.md` for provenance, and `DEBUG_GUIDE.md` for generation checkpoints.

Population values are demand proxies. NUMBAT is historical observed rail demand and does not measure demand for a proposed route. Build costs, route geometry, capacity, and service impacts are illustrative. Alignments do not resolve geology, property, utilities, consent, or detailed engineering. Project code is MIT licensed; source datasets retain their own terms.
