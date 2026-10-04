---
version: 1
slug: "src-main-resources-map-html"
primary_target: "src/main/resources/map.html"
related_targets: ["src/main/resources/planner.css","frontend/planner.ts"]
---

# Planning workspace

- Scope: `src/main/resources/map.html` and its CSS/TypeScript modules
- Mode: Operate
- Audience: city planners evaluating hypothetical Greater London metro networks
- Job: create a sparse Pareto network, refine a chosen plan, then analyse its demand and operation
- Primary task: edit the map without losing sight of coverage, assumptions, selection state, or analysis freshness
- Constraints: desktop and tablet from 768px; Leaflet remains the geographic canvas; simulated values are labelled; colour is never the sole state channel

## Direction contract

**THESIS:** The map is the working document, not a dashboard illustration. A compact planning frame surrounds one continuous geographic canvas and refuses the category-default collection of detached metric cards.

**OWN-WORLD:** Restrained slate and paper-white work surfaces use one safety-blue action colour, amber for incomplete evidence, and red only for unsupported or failed states. Crisp one-pixel rules, square-ended tool rails, compact fields, tabular numerals, line-pattern legends, and shallow offset shadows evoke a maintained GIS workstation rather than a consumer map.

**STORY:** The planner first sees where evidence is valid, then draws or generates a network, selects real map objects to edit their assumptions, opens synthetic demand evidence, and switches the same project into operation. Score changes and analysis freshness remain visible throughout.

**FIRST VIEWPORT:** A project start screen offers New project, existing browser projects, and JSON import before opening the map. The map workspace has a 52-pixel project bar and a 326-pixel paper task panel at left; it becomes a bottom sheet on tablets. The primary Create action is Draw study area, followed by a streamed generation search. A high-DPI canvas records candidate paths over the geographic map, mirrored by a textual event timeline. The signature interaction is choosing a Pareto plan, adjusting its visible draft corridors, and locking in the authoritative evaluation.

Map utilities form one top-right toolbar for Fit, zoom, and Layers. The legend sits in Layers, and task feedback stays in the panel footer.

**FORM:** Brief-pinned dense GIS workspace, first choice from the approved planning round; seed key `brief-pinned-dense-gis-operate`. Motion is limited to 180ms panel and selection-state transitions, live train interpolation, and one analysis-freshness pulse.

**FINISH:** unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
