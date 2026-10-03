# London Network Planner

London Network Planner is a local desktop planning workspace for designing and testing hypothetical metro networks around Greater London. It combines an editable Leaflet map with Kotlin generation, synthetic origin-destination demand, network scoring, frequency-aware journey planning, and deterministic disruption simulation.

The application is a planning model, not a live TfL service. OD journeys, capacity loads, scores, costs, and train movements are simulated and are labelled accordingly in the UI.

## What it does

- Draw a supported London study area and generate an editable metro proposal.
- Draw manual radial, cross-city trunk, orbital bypass, and core distributor lines.
- Mark each segment independently as underground or overground.
- Save up to 20 local project revisions in IndexedDB and import/export versioned JSON projects.
- Generate and inspect a downloadable synthetic point-to-point journeys/day dataset.
- Score demand served, journey time, capacity, connectivity, construction efficiency, and resilience.
- Compare Dijkstra and A* routes with boarding waits, transfer waits, dwell, and infrastructure speed.
- Change line frequency and inspect headway effects.
- Simulate trains and cancellations, track incidents, weather, signal failures, and station closures.
- Detect coverage gaps, long waits, long segments, overcrowding, and other deterministic issues.

## Architecture

- Kotlin 2.2.20, JVM toolchain 23, Gradle 8.14 wrapper.
- Ktor 2.3.12 on Netty, bound to `127.0.0.1` on the first free port in `5000..5010`.
- kotlinx serialization for versioned API and project contracts.
- Plain HTML/CSS/JavaScript with pinned Leaflet 1.9.4 and Leaflet Draw 1.0.4 CDN assets.
- No database, accounts, cloud synchronization, frontend build system, or live transport feed.
- Editable networks compile into primitive CSR-style arrays and a primitive binary min-heap for routing.

The older `MetroBuilder` remains the automatic generation engine. Planner analysis uses the new project model and `PrimitiveRoutingEngine`; the legacy `/suggestions`, `/metro_suggestions`, and `/density` contracts remain available for compatibility.

## Run

```bash
./gradlew build
./gradlew run
```

Open the URL printed by the server. The full editor requires a viewport at least 1100 pixels wide.

## Typical workflow

1. In **Design**, draw a study area within the supported green Greater London boundary.
2. Generate a network or draw lines manually. Select lines, stations, and segments to edit them.
3. Open **Demand** to inspect the explicitly simulated OD matrix and score.
4. Switch to **Operate**, set trains per hour, and compare Dijkstra/A* journeys.
5. Add disruption scenarios and play the train simulation. Weather targets list only overground segments.
6. Use **Issues** to navigate directly to inefficient or overloaded network elements.
7. Save locally or export the complete project as JSON.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/coverage` | Supported Greater London geometry and evidence metadata |
| `POST` | `/api/v1/networks/generate` | Generate an editable network for a study polygon |
| `POST` | `/api/v1/analysis/sessions` | Compile a project, synthetic demand, score, and issues |
| `GET` | `/api/v1/analysis/sessions/{id}/demand` | Page through simulated OD rows |
| `GET` | `/api/v1/analysis/sessions/{id}/demand.csv` | Download the simulated OD dataset |
| `POST` | `/api/v1/analysis/sessions/{id}/journeys` | Route with Dijkstra or A* |
| `POST` | `/api/v1/analysis/sessions/{id}/simulations` | Compile headway-derived train runs |

Analysis sessions are bounded in-memory caches. Saved browser projects contain inputs and model versions, so sessions and derived demand can be reproduced after restart.

## Verification

```bash
./gradlew test
./gradlew build
```

Tests cover primitive routing, Dijkstra/A* parity, expected waits, track incidents, infrastructure-specific weather behaviour, exact synthetic journey totals, score weights, and coverage geometry.

See `PLANNER_MODEL.md` for formulas and limitations, `DATA_SOURCES.md` for provenance and attribution, and `DEBUG_GUIDE.md` before changing automatic-generation thresholds.

## Important limitations

- Population-grid values are demand-density proxies, not asserted exact population counts.
- PTAL is 2015 evidence and applies only inside the bundled London coverage.
- The fixed score is a comparison aid, not a business case or engineering feasibility assessment.
- Speeds, capacities, cost multipliers, and surface-reference time are visible model assumptions.
- Track alignments do not account for geology, rights of way, utilities, planning consent, or construction access.
- No project licence is currently declared. Check source-data and code licensing before redistribution.
