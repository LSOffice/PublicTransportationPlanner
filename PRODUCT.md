# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The primary user is a city planner working on a desktop or tablet around Greater London. They need to sketch, generate, compare, operate, and explain hypothetical metro networks without leaving the map workspace.

## Product Purpose

PublicTransportationPlanner turns population and accessibility evidence into editable metro proposals. Success means a planner can define a supported study area, generate or draw a network, inspect explicitly simulated origin-destination demand, evaluate the network, and test journeys and disruptions through reproducible algorithms.

## Positioning

The product exposes the planning model rather than hiding it: synthetic demand, infrastructure assumptions, route-search algorithm, wait time, capacity, score components, and disruption effects remain inspectable from the same geographic workspace.

## Operating Context

Projects are single-user and local. Every launch starts with a choice to create a project, open one saved in this browser, or import JSON. Planners then move through Create, Review routes, Refine, and Analyse stages in one map workspace. Create streams candidate and plan search; Review holds a visible draft until an authoritative evaluation enables Lock in routes; Refine edits the saved network; Analyse handles demand, journeys, service, issues, and disruptions. Project files must be portable between browsers through versioned JSON export and import.

## Capabilities and Constraints

- Planning is restricted to the population-supported part of Greater London.
- Population coverage and PTAL coverage are distinct; missing PTAL uses an explicit neutral fallback.
- Historical NUMBAT OD evidence can steer candidate generation; simulated gravity fills unmatched areas. Analysis OD demand is deterministic, synthetic, and labelled as such. Live TfL conditions are out of scope.
- Lines have a planning role; each segment independently records underground or overground infrastructure.
- Service uses frequency/headway assumptions rather than exact timetables.
- Dijkstra and A* must produce equivalent optimal journey costs and include expected wait and transfer time.
- Weather disruptions apply only to overground track.
- The application remains a Kotlin/JVM service with a TypeScript browser bundle and no database.
- The planning workspace supports desktop and tablet widths from 768px; narrower screens show a minimum-width message.

## Evidence on Hand

- A bundled 2020 one-kilometre population grid covering Great Britain.
- Bundled 2015 London PTAL centroid data with LSOA identifiers.
- Existing automatic natural-network generation and checkpoint documentation.
- Bundled NUMBAT 2024 provides historical observed OD usage. Construction costs remain coarse estimates; no live operations feed, customer evidence, or production SLA is available.

## Product Principles

1. Show assumptions beside results.
2. Keep geographic evidence and network edits visually connected.
3. Make every comparison reproducible from saved project inputs.
4. Prefer explainable algorithms and raw component metrics over opaque recommendations.
5. Prevent unsupported edits instead of silently scoring invented coverage.

## Accessibility & Inclusion

Do not rely on colour alone for coverage, infrastructure, severity, or line state. All controls require keyboard focus states, textual labels, and reduced-motion behaviour; numerical tables use clear units and tabular figures.
