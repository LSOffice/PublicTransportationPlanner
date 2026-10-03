const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  coverage: null,
  project: null,
  projects: [],
  session: null,
  demandRecords: [],
  selected: { type: "network", id: "network" },
  mode: "design",
  activeDrawMode: null,
  history: [],
  future: [],
  analysisTimer: null,
  saveTimer: null,
  mapLayers: new Map(),
  demandVisible: false,
  simulation: null,
  simulationMinute: 0,
  simulationPlaying: false,
  simulationFrame: null,
  simulationLastTimestamp: null,
};

const dbPromise = openDatabase();
const map = L.map("map", { zoomControl: true, preferCanvas: false, minZoom: 9, maxZoom: 18 }).setView([51.5074, -0.1278], 10);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);
const coverageLayer = L.layerGroup().addTo(map);
const studyLayer = L.layerGroup().addTo(map);
const networkLayer = L.layerGroup().addTo(map);
const demandLayer = L.layerGroup().addTo(map);
const journeyLayer = L.layerGroup().addTo(map);
const trainLayer = L.layerGroup().addTo(map);

boot().catch(showError);

async function boot() {
  bindEvents();
  await loadCoverage();
  await loadProjectRegister();
  if (!state.projects.length) {
    state.project = createProject("Central London study");
    await persistProject(false);
  } else {
    const lastId = localStorage.getItem("planner:lastProject");
    state.project = state.projects.find((project) => project.id === lastId) || state.projects[0];
  }
  renderAll();
  setNotice("Draw a study area or create a manual line. All demand shown by this tool is simulated.");
}

function bindEvents() {
  $("#designModeBtn").addEventListener("click", () => setMode("design"));
  $("#operateModeBtn").addEventListener("click", () => setMode("operate"));
  $("#drawAreaBtn").addEventListener("click", startStudyAreaDraw);
  $("#drawLineBtn").addEventListener("click", startLineDraw);
  $("#generateBtn").addEventListener("click", generateNetwork);
  $("#newProjectBtn").addEventListener("click", newProject);
  $("#saveProjectBtn").addEventListener("click", () => persistProject(true));
  $("#exportProjectBtn").addEventListener("click", exportProject);
  $("#importProjectBtn").addEventListener("click", () => $("#projectFileInput").click());
  $("#projectFileInput").addEventListener("change", importProject);
  $("#projectSelect").addEventListener("change", loadSelectedProject);
  $("#undoBtn").addEventListener("click", undo);
  $("#redoBtn").addEventListener("click", redo);
  $("#dailyJourneysInput").addEventListener("change", updateDemandConfig);
  $("#decayInput").addEventListener("change", updateDemandConfig);
  $("#fitMapBtn").addEventListener("click", fitSupportedLondon);
  $("#coverageToggle").addEventListener("click", toggleCoverage);
  $("#demandToggle").addEventListener("click", toggleDemand);
  $("#planJourneyBtn").addEventListener("click", planJourney);
  $("#disruptionType").addEventListener("change", renderDisruptionTargets);
  $("#severityInput").addEventListener("input", () => $("#severityValue").textContent = `${$("#severityInput").value}%`);
  $("#addDisruptionBtn").addEventListener("click", addDisruption);
  $("#simulationPlayBtn").addEventListener("click", toggleSimulation);
  $("#simulationResetBtn").addEventListener("click", resetSimulation);
  $("#drawerToggle").addEventListener("click", toggleDrawer);
  $("#downloadDemandBtn").addEventListener("click", downloadDemand);
  $("#coverageInfoBtn").addEventListener("click", showMethod);
  $$(".drawer-tab").forEach((button) => button.addEventListener("click", () => selectDrawer(button.dataset.drawer)));
  map.on(L.Draw.Event.CREATED, handleDrawCreated);
  window.addEventListener("beforeunload", () => state.project && persistProject(false));
}

async function loadCoverage() {
  state.coverage = await api("/api/v1/coverage");
  renderCoverage();
  $("#populationCoverageValue").textContent = `${state.coverage.populationGridCellsInLondon.toLocaleString()} cells`;
  $("#ptalCoverageValue").textContent = `${state.coverage.ptalRecords.toLocaleString()} LSOAs`;
  const [west, south, east, north] = state.coverage.bounds;
  map.setMaxBounds([[south - 0.12, west - 0.18], [north + 0.12, east + 0.18]]);
  fitSupportedLondon();
}

function renderCoverage() {
  coverageLayer.clearLayers();
  if (!state.coverage || !$("#coverageToggle").classList.contains("is-active")) return;
  const supportedRings = state.coverage.hardSupport.map((ring) => ring.map((point) => [point.lat, point.lon]));
  const [west, south, east, north] = state.coverage.bounds;
  const margin = 0.75;
  const outerRing = [
    [south - margin, west - margin],
    [north + margin, west - margin],
    [north + margin, east + margin],
    [south - margin, east + margin],
  ];
  L.polygon([outerRing, ...supportedRings], {
    color: "#b42332", weight: 1, fillColor: "#b42332", fillOpacity: 0.30, fillRule: "evenodd", interactive: false,
  }).addTo(coverageLayer);
  supportedRings.forEach((ring) => L.polygon(ring, { color: "#157066", weight: 2.5, dashArray: "8 5", fill: false, interactive: false }).addTo(coverageLayer));
}

function fitSupportedLondon() {
  if (!state.coverage) return;
  const [west, south, east, north] = state.coverage.bounds;
  map.fitBounds([[south, west], [north, east]], { padding: [22, 22] });
}

function toggleCoverage() {
  $("#coverageToggle").classList.toggle("is-active");
  const active = $("#coverageToggle").classList.contains("is-active");
  $("#coverageToggle").setAttribute("aria-pressed", String(active));
  renderCoverage();
}

function startStudyAreaDraw() {
  state.activeDrawMode = "study";
  setNotice("Click to draw the study-area boundary; click the first point to finish.");
  new L.Draw.Polygon(map, { allowIntersection: false, showArea: true, shapeOptions: { color: "#0b66d4", weight: 2, fillOpacity: 0.08 } }).enable();
}

function startLineDraw() {
  if (!state.coverage) return;
  state.activeDrawMode = "line";
  setNotice("Click each station in sequence; double-click the last station to finish.");
  new L.Draw.Polyline(map, { shapeOptions: { color: "#0b66d4", weight: 5 } }).enable();
}

function handleDrawCreated(event) {
  const mode = state.activeDrawMode;
  state.activeDrawMode = null;
  const latLngs = event.layer.getLatLngs();
  if (mode === "study") {
    const ring = Array.isArray(latLngs[0]) ? latLngs[0] : latLngs;
    const coordinates = ring.map(({ lat, lng }) => ({ lon: lng, lat }));
    if (!coordinates.every((point) => pointInCoverage(point.lon, point.lat))) {
      showError(new Error("The complete study area must remain inside the green Greater London support boundary."));
      return;
    }
    commit("Set study area", (project) => { project.studyArea = { coordinates }; });
    setNotice("Study area ready. Generate a network or begin drawing manually.");
  } else if (mode === "line") {
    const points = latLngs.map(({ lat, lng }) => ({ lon: lng, lat }));
    if (points.length < 2 || !points.every((point) => pointInCoverage(point.lon, point.lat))) {
      showError(new Error("A line needs at least two stations and must stay inside supported coverage."));
      return;
    }
    commit("Draw line", (project) => addManualLine(project, points));
    setNotice("Manual line added. Select its segments to change underground or overground construction.");
  }
}

function addManualLine(project, points) {
  const lineNumber = nextNumber(project.network.lines.map((line) => line.id), "L");
  const lineId = `L${lineNumber}`;
  const infrastructure = $("#newLineInfrastructure").value;
  const stationIds = [];
  points.forEach((point) => {
    let station = nearestStation(project.network.stations, point, 250);
    if (!station) {
      const stationNumber = nextNumber(project.network.stations.map((item) => item.id), "ST");
      station = { id: `ST${stationNumber}`, name: `Station ${stationNumber}`, lon: point.lon, lat: point.lat, demandValue: 0 };
      project.network.stations.push(station);
    }
    stationIds.push(station.id);
  });
  const segments = stationIds.slice(0, -1).map((fromStationId, index) => {
    const toStationId = stationIds[index + 1];
    const from = project.network.stations.find((station) => station.id === fromStationId);
    const to = project.network.stations.find((station) => station.id === toStationId);
    return { id: `${lineId}-S${index + 1}`, fromStationId, toStationId, infrastructure, lengthMeters: haversine(from, to) };
  });
  const colors = ["#0b66d4", "#c2415d", "#14836f", "#7b52ab", "#d97706", "#3746a5", "#8b5e34", "#1677a3"];
  project.network.lines.push({
    id: lineId, name: `Line ${lineNumber}`, color: colors[(lineNumber - 1) % colors.length], role: $("#newLineRole").value,
    isLoop: false, trainsPerHour: 12, vehicleCapacity: 850, stationIds, segments,
  });
  state.selected = { type: "line", id: lineId };
}

async function generateNetwork() {
  if (!state.project.studyArea) return;
  const button = $("#generateBtn");
  button.disabled = true;
  button.classList.add("is-loading");
  setNotice("Generating corridors from population and PTAL-weighted demand…");
  try {
    const result = await api("/api/v1/networks/generate", { method: "POST", body: { studyArea: state.project.studyArea, maxTrunkLines: 4 } });
    commit("Generate network", (project) => { project.network = result.network; });
    setNotice(`Generated ${result.network.lines.length} lines from ${result.selectedGridPoints.toLocaleString()} supported grid cells.`);
    await analyseProject();
  } catch (error) {
    showError(error);
  } finally {
    button.classList.remove("is-loading");
    button.disabled = !state.project.studyArea;
  }
}

function updateDemandConfig() {
  const totalDailyJourneys = Number($("#dailyJourneysInput").value);
  const distanceDecayKm = Number($("#decayInput").value);
  commit("Update demand model", (project) => { project.demandConfig.totalDailyJourneys = totalDailyJourneys; project.demandConfig.distanceDecayKm = distanceDecayKm; });
}

function commit(label, mutation, { analyse = true } = {}) {
  if (!state.project) return;
  state.history.push(structuredClone(state.project));
  if (state.history.length > 50) state.history.shift();
  state.future = [];
  mutation(state.project);
  state.project.revision += 1;
  state.project.updatedAt = new Date().toISOString();
  state.session = null;
  $("#analysisState").textContent = "Changes pending";
  $("#analysisState").className = "status-chip";
  renderAll();
  scheduleSave();
  if (analyse && state.project.network.lines.length) scheduleAnalysis();
  setSaveState(`${label} · saving locally…`);
}

function undo() {
  if (!state.history.length) return;
  state.future.push(structuredClone(state.project));
  state.project = state.history.pop();
  state.session = null;
  renderAll();
  scheduleSave();
}

function redo() {
  if (!state.future.length) return;
  state.history.push(structuredClone(state.project));
  state.project = state.future.pop();
  state.session = null;
  renderAll();
  scheduleSave();
}

function renderAll() {
  if (!state.project) return;
  renderProjectPicker();
  renderStudyArea();
  renderNetwork();
  renderLineList();
  renderStationSelectors();
  renderFrequencyControls();
  renderDisruptionTargets();
  renderDisruptions();
  renderSelection();
  renderHistoryState();
  $("#dailyJourneysInput").value = state.project.demandConfig.totalDailyJourneys;
  $("#decayInput").value = state.project.demandConfig.distanceDecayKm;
  $("#studyAreaStatus").textContent = state.project.studyArea ? "Area ready" : "No area";
  $("#studyAreaStatus").className = `status-chip ${state.project.studyArea ? "success" : ""}`;
  $("#generateBtn").disabled = !state.project.studyArea;
}

function renderStudyArea() {
  studyLayer.clearLayers();
  if (!state.project.studyArea) return;
  L.polygon(state.project.studyArea.coordinates.map((point) => [point.lat, point.lon]), {
    color: "#0b66d4", weight: 2, dashArray: "5 5", fillColor: "#0b66d4", fillOpacity: 0.05,
  }).addTo(studyLayer);
}

function renderNetwork() {
  networkLayer.clearLayers();
  state.mapLayers.clear();
  const stations = new Map(state.project.network.stations.map((station) => [station.id, station]));
  const lineUse = new Map();
  state.project.network.lines.forEach((line) => line.stationIds.forEach((id) => lineUse.set(id, (lineUse.get(id) || 0) + 1)));
  state.project.network.lines.forEach((line) => {
    const lineCancelled = state.project.disruptions.some((item) => item.type === "TRAIN_CANCELLATION" && item.targetId === line.id);
    line.segments.forEach((segment) => {
      const from = stations.get(segment.fromStationId);
      const to = stations.get(segment.toStationId);
      if (!from || !to) return;
      const selected = state.selected.type === "segment" && state.selected.id === segment.id;
      const segmentDisruption = state.project.disruptions.find((item) => item.targetId === segment.id);
      const polyline = L.polyline([[from.lat, from.lon], [to.lat, to.lon]], {
        color: segmentDisruption ? (segmentDisruption.type === "ADVERSE_WEATHER" ? "#b86608" : "#b42332") : line.color,
        weight: selected ? 9 : 6, opacity: lineCancelled ? 0.38 : 0.92,
        dashArray: segment.infrastructure === "OVERGROUND" ? "10 8" : null,
        lineCap: "butt", lineJoin: "round",
      }).addTo(networkLayer);
      polyline.on("click", (event) => { L.DomEvent.stopPropagation(event); selectObject("segment", segment.id); });
      polyline.bindTooltip(`${escapeHtml(line.name)} · ${segment.infrastructure.toLowerCase()}`);
      state.mapLayers.set(segment.id, polyline);
    });
  });
  state.project.network.stations.forEach((station) => {
    const selected = state.selected.type === "station" && state.selected.id === station.id;
    const closed = state.project.disruptions.some((item) => item.type === "STATION_CLOSURE" && item.targetId === station.id);
    const icon = L.divIcon({
      className: "", iconSize: lineUse.get(station.id) > 1 ? [18, 18] : [14, 14],
      html: `<div class="planner-station ${lineUse.get(station.id) > 1 ? "interchange" : ""} ${selected ? "is-selected" : ""} ${closed ? "is-closed" : ""}"></div>`,
    });
    const marker = L.marker([station.lat, station.lon], { icon, draggable: state.mode === "design", keyboard: true }).addTo(networkLayer);
    marker.bindTooltip(escapeHtml(station.name), { className: "station-tooltip", direction: "top" });
    marker.on("click", () => selectObject("station", station.id));
    marker.on("dragend", () => moveStation(station.id, marker));
    state.mapLayers.set(station.id, marker);
  });
}

function moveStation(stationId, marker) {
  const { lat, lng } = marker.getLatLng();
  if (!pointInCoverage(lng, lat)) {
    renderNetwork();
    showError(new Error("Stations cannot be moved outside supported Greater London coverage."));
    return;
  }
  commit("Move station", (project) => {
    const station = project.network.stations.find((item) => item.id === stationId);
    station.lon = lng; station.lat = lat;
    project.network.lines.forEach((line) => line.segments.forEach((segment) => {
      if (segment.fromStationId === stationId || segment.toStationId === stationId) {
        const from = project.network.stations.find((item) => item.id === segment.fromStationId);
        const to = project.network.stations.find((item) => item.id === segment.toStationId);
        segment.lengthMeters = haversine(from, to);
      }
    }));
  });
}

function renderLineList() {
  const list = $("#lineList");
  $("#lineCount").textContent = `${state.project.network.lines.length} line${state.project.network.lines.length === 1 ? "" : "s"}`;
  if (!state.project.network.lines.length) {
    list.className = "list empty-list"; list.innerHTML = "<p>Generate or draw a line to begin.</p>"; return;
  }
  list.className = "list";
  list.innerHTML = state.project.network.lines.map((line) => `
    <button class="line-row ${state.selected.type === "line" && state.selected.id === line.id ? "is-selected" : ""}" data-line-id="${escapeHtml(line.id)}" type="button">
      <span class="line-swatch" style="background:${safeColor(line.color)}"></span>
      <span><span class="row-title">${escapeHtml(line.name)}</span><span class="row-meta">${roleLabel(line.role)} · ${line.stationIds.length} stations</span></span>
      <span class="numeric">${line.trainsPerHour} tph</span>
    </button>`).join("");
  list.querySelectorAll("[data-line-id]").forEach((button) => button.addEventListener("click", () => selectObject("line", button.dataset.lineId)));
}

function renderSelection() {
  const root = $("#selectionInspector");
  $("#selectionType").textContent = titleCase(state.selected.type);
  if (state.selected.type === "line") {
    const line = state.project.network.lines.find((item) => item.id === state.selected.id);
    if (!line) return selectObject("network", "network");
    root.innerHTML = `<label>Name<input id="selectedLineName" value="${escapeAttribute(line.name)}" /></label>
      <label>Planning role<select id="selectedLineRole">${roleOptions(line.role)}</select></label>
      <label>Trains per hour<input id="selectedLineTph" type="number" min="1" max="60" value="${line.trainsPerHour}" /></label>
      <dl><div><dt>Stations</dt><dd>${line.stationIds.length}</dd></div><div><dt>Track</dt><dd>${formatKm(line.segments.reduce((sum, item) => sum + item.lengthMeters, 0))}</dd></div><div><dt>Average wait</dt><dd>${(30 / line.trainsPerHour).toFixed(1)} min</dd></div></dl>
      <button id="deleteLineBtn" class="button secondary full danger-action" type="button">Delete line</button>`;
    $("#selectedLineName").addEventListener("change", (event) => commit("Rename line", (project) => project.network.lines.find((item) => item.id === line.id).name = event.target.value.trim() || line.name));
    $("#selectedLineRole").addEventListener("change", (event) => commit("Change line role", (project) => project.network.lines.find((item) => item.id === line.id).role = event.target.value));
    $("#selectedLineTph").addEventListener("change", (event) => updateLineFrequency(line.id, Number(event.target.value)));
    $("#deleteLineBtn").addEventListener("click", () => deleteLine(line.id));
  } else if (state.selected.type === "segment") {
    const found = findSegment(state.selected.id);
    if (!found) return selectObject("network", "network");
    root.innerHTML = `<p><strong>${escapeHtml(found.line.name)}</strong><br>${escapeHtml(found.segment.id)}</p>
      <label>Infrastructure<select id="selectedInfrastructure"><option value="UNDERGROUND" ${found.segment.infrastructure === "UNDERGROUND" ? "selected" : ""}>Underground</option><option value="OVERGROUND" ${found.segment.infrastructure === "OVERGROUND" ? "selected" : ""}>Overground</option></select></label>
      <dl><div><dt>Length</dt><dd>${formatKm(found.segment.lengthMeters)}</dd></div><div><dt>Weather exposure</dt><dd>${found.segment.infrastructure === "OVERGROUND" ? "Eligible" : "Protected"}</dd></div></dl>`;
    $("#selectedInfrastructure").addEventListener("change", (event) => commit("Change infrastructure", (project) => findSegmentInProject(project, found.segment.id).segment.infrastructure = event.target.value));
  } else if (state.selected.type === "station") {
    const station = state.project.network.stations.find((item) => item.id === state.selected.id);
    if (!station) return selectObject("network", "network");
    const lines = state.project.network.lines.filter((line) => line.stationIds.includes(station.id));
    root.innerHTML = `<label>Name<input id="selectedStationName" value="${escapeAttribute(station.name)}" /></label>
      <dl><div><dt>Station ID</dt><dd>${escapeHtml(station.id)}</dd></div><div><dt>Lines</dt><dd>${lines.map((line) => escapeHtml(line.name)).join(", ") || "None"}</dd></div><div><dt>Demand proxy</dt><dd>${Math.round(station.demandValue).toLocaleString()}</dd></div><div><dt>Coordinate</dt><dd>${station.lat.toFixed(4)}, ${station.lon.toFixed(4)}</dd></div></dl>`;
    $("#selectedStationName").addEventListener("change", (event) => commit("Rename station", (project) => project.network.stations.find((item) => item.id === station.id).name = event.target.value.trim() || station.name));
  } else {
    root.innerHTML = `<p>This project contains <strong>${state.project.network.lines.length}</strong> lines and <strong>${state.project.network.stations.length}</strong> stations.</p><dl><div><dt>Project revision</dt><dd>${state.project.revision}</dd></div><div><dt>Demand model</dt><dd>${escapeHtml(state.project.demandConfig.modelVersion)}</dd></div><div><dt>Persistence</dt><dd>Local browser</dd></div></dl>`;
  }
}

function selectObject(type, id) {
  state.selected = { type, id };
  renderNetwork(); renderLineList(); renderSelection();
  const layer = state.mapLayers.get(id);
  if (layer?.getBounds) map.fitBounds(layer.getBounds(), { padding: [80, 80], maxZoom: 14 });
  if (layer?.getLatLng) map.panTo(layer.getLatLng());
}

function deleteLine(lineId) {
  commit("Delete line", (project) => {
    project.network.lines = project.network.lines.filter((line) => line.id !== lineId);
    const used = new Set(project.network.lines.flatMap((line) => line.stationIds));
    project.network.stations = project.network.stations.filter((station) => used.has(station.id));
    project.disruptions = project.disruptions.filter((item) => item.targetId !== lineId && !item.targetId.startsWith(`${lineId}-`));
    state.selected = { type: "network", id: "network" };
  });
}

function renderFrequencyControls() {
  const root = $("#frequencyControls");
  if (!state.project.network.lines.length) { root.innerHTML = "<p>Add a line to set service frequency.</p>"; return; }
  root.innerHTML = state.project.network.lines.map((line) => `<div class="frequency-control"><header><strong>${escapeHtml(line.name)}</strong><span><b class="numeric">${line.trainsPerHour}</b> tph · ${(60 / line.trainsPerHour).toFixed(1)} min</span></header><input type="range" min="2" max="40" step="2" value="${line.trainsPerHour}" data-frequency-line="${escapeHtml(line.id)}" aria-label="${escapeAttribute(line.name)} trains per hour" /></div>`).join("");
  root.querySelectorAll("[data-frequency-line]").forEach((input) => input.addEventListener("change", () => updateLineFrequency(input.dataset.frequencyLine, Number(input.value))));
}

function updateLineFrequency(lineId, value) {
  commit("Change service frequency", (project) => { project.network.lines.find((line) => line.id === lineId).trainsPerHour = Math.max(1, Math.min(60, Math.round(value))); });
}

function renderStationSelectors() {
  const options = state.project.network.stations.map((station) => `<option value="${escapeAttribute(station.id)}">${escapeHtml(station.name)}</option>`).join("");
  $("#journeyFrom").innerHTML = options || '<option value="">No stations</option>';
  $("#journeyTo").innerHTML = options || '<option value="">No stations</option>';
  if (state.project.network.stations.length > 1) $("#journeyTo").selectedIndex = state.project.network.stations.length - 1;
  $("#planJourneyBtn").disabled = state.project.network.stations.length < 2;
}

async function analyseProject() {
  if (!state.project.network.lines.length) return;
  clearTimeout(state.analysisTimer);
  $("#analysisState").textContent = "Analysing";
  $("#analysisState").className = "status-chip";
  try {
    state.session = await api("/api/v1/analysis/sessions", { method: "POST", body: { project: state.project } });
    if (state.session.revision !== state.project.revision) return;
    $("#analysisState").textContent = "Current";
    $("#analysisState").className = "status-chip success";
    renderScore(); renderIssues();
    await loadDemandPage();
  } catch (error) {
    $("#analysisState").textContent = "Analysis failed";
    $("#analysisState").className = "status-chip danger";
    showError(error);
  }
}

function scheduleAnalysis() {
  clearTimeout(state.analysisTimer);
  state.analysisTimer = setTimeout(analyseProject, 700);
}

function renderScore() {
  const score = state.session?.score;
  if (!score) return;
  $("#scoreTotal").textContent = score.total.toFixed(1);
  $("#scoreComponents").innerHTML = score.components.map((component) => `<div class="score-component" title="${escapeAttribute(component.explanation)}"><label>${escapeHtml(component.label)} <span class="count">${Math.round(component.weight * 100)}%</span></label><output>${component.score.toFixed(0)}</output><div class="score-track"><span style="width:${component.score}%"></span></div></div>`).join("");
}

async function loadDemandPage() {
  if (!state.session) return;
  const page = await api(`/api/v1/analysis/sessions/${state.session.sessionId}/demand?offset=0&limit=500`);
  state.demandRecords = page.records;
  $("#demandTotal").textContent = page.totalDailyJourneys.toLocaleString();
  $("#demandPairs").textContent = page.total.toLocaleString();
  $("#downloadDemandBtn").disabled = false;
  $("#demandTable").innerHTML = page.records.slice(0, 100).map((record) => `<tr data-demand-origin="${record.originZoneId}" data-demand-destination="${record.destinationZoneId}"><td>${record.originZoneId}</td><td>${record.destinationZoneId}</td><td>${record.dailyJourneys.toLocaleString()}</td><td>${formatKm(record.distanceMeters)}</td></tr>`).join("") || '<tr><td colspan="4">No non-zero OD pairs were generated.</td></tr>';
  $("#demandTable").querySelectorAll("tr[data-demand-origin]").forEach((row) => row.addEventListener("click", () => selectDemandPair(row.dataset.demandOrigin, row.dataset.demandDestination)));
  drawDemandChart(page.records);
  renderDemandFlows();
}

function drawDemandChart(records) {
  const canvas = $("#demandChart");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 440; const height = canvas.clientHeight || 128;
  canvas.width = width * dpr; canvas.height = height * dpr; ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  const bins = Array(12).fill(0); const maxDistance = Math.max(1, ...records.map((item) => item.distanceMeters));
  records.forEach((record) => { const index = Math.min(bins.length - 1, Math.floor(record.distanceMeters / maxDistance * bins.length)); bins[index] += record.dailyJourneys; });
  const maxBin = Math.max(1, ...bins); const gap = 5; const barWidth = (width - 34 - gap * bins.length) / bins.length;
  ctx.fillStyle = "#52606d"; ctx.font = "10px Aptos, Segoe UI, sans-serif"; ctx.fillText("Journey-distance distribution", 10, 14);
  bins.forEach((value, index) => { const barHeight = value / maxBin * (height - 42); ctx.fillStyle = index < bins.length / 2 ? "#0b66d4" : "#14836f"; ctx.fillRect(12 + index * (barWidth + gap), height - 20 - barHeight, barWidth, barHeight); });
  ctx.fillStyle = "#73808d"; ctx.fillText("0 km", 10, height - 5); ctx.fillText(`${(maxDistance / 1000).toFixed(0)} km`, width - 42, height - 5);
}

function toggleDemand() {
  state.demandVisible = !state.demandVisible;
  $("#demandToggle").classList.toggle("is-active", state.demandVisible);
  $("#demandToggle").setAttribute("aria-pressed", String(state.demandVisible));
  renderDemandFlows();
}

function renderDemandFlows() {
  demandLayer.clearLayers();
  if (!state.demandVisible || !state.session) return;
  const zones = new Map(state.session.zones.map((zone) => [zone.id, zone]));
  [...state.demandRecords].sort((a, b) => b.dailyJourneys - a.dailyJourneys).slice(0, 50).forEach((record) => {
    const origin = zones.get(record.originZoneId); const destination = zones.get(record.destinationZoneId);
    if (!origin || !destination) return;
    L.polyline([[origin.lat, origin.lon], [destination.lat, destination.lon]], { color: "#7b52ab", weight: Math.max(1, Math.min(7, Math.log10(record.dailyJourneys + 1))), opacity: 0.28, interactive: false }).addTo(demandLayer);
  });
}

function selectDemandPair(originId, destinationId) {
  const zones = new Map(state.session.zones.map((zone) => [zone.id, zone]));
  const origin = zones.get(originId); const destination = zones.get(destinationId);
  if (!origin || !destination) return;
  demandLayer.clearLayers();
  L.polyline([[origin.lat, origin.lon], [destination.lat, destination.lon]], { color: "#7b52ab", weight: 7, opacity: 0.8 }).addTo(demandLayer);
  L.circleMarker([origin.lat, origin.lon], { radius: 7, color: "#7b52ab", fillOpacity: 1 }).addTo(demandLayer);
  L.circleMarker([destination.lat, destination.lon], { radius: 7, color: "#7b52ab", fillOpacity: 1 }).addTo(demandLayer);
  map.fitBounds([[origin.lat, origin.lon], [destination.lat, destination.lon]], { padding: [80, 80] });
  state.selected = { type: "demand pair", id: `${originId}-${destinationId}` };
  $("#selectionType").textContent = "Demand pair";
  const record = state.demandRecords.find((item) => item.originZoneId === originId && item.destinationZoneId === destinationId);
  $("#selectionInspector").innerHTML = `<p><strong>${originId} → ${destinationId}</strong></p><dl><div><dt>Simulated journeys/day</dt><dd>${record?.dailyJourneys.toLocaleString() || "—"}</dd></div><div><dt>Direct distance</dt><dd>${record ? formatKm(record.distanceMeters) : "—"}</dd></div><div><dt>Data status</dt><dd>Simulated</dd></div></dl>`;
}

function renderIssues() {
  const issues = state.session?.issues || [];
  $("#issueBadge").textContent = issues.filter((issue) => issue.severity !== "INFO").length;
  const root = $("#issueList");
  root.className = issues.length ? "issue-list" : "issue-list empty-list";
  root.innerHTML = issues.map((issue) => `<button class="issue-row ${issue.severity.toLowerCase()}" data-issue-id="${issue.id}" type="button"><span class="severity">${issue.severity}</span><span class="row-title">${escapeHtml(issue.title)}</span><span class="row-meta">${escapeHtml(issue.detail)}</span></button>`).join("") || "<p>No deterministic findings.</p>";
  root.querySelectorAll("[data-issue-id]").forEach((button) => button.addEventListener("click", () => selectIssue(button.dataset.issueId)));
}

function selectIssue(issueId) {
  const issue = state.session.issues.find((item) => item.id === issueId);
  if (!issue) return;
  if (state.mapLayers.has(issue.targetId)) selectObject(findSegment(issue.targetId) ? "segment" : state.project.network.lines.some((line) => line.id === issue.targetId) ? "line" : "station", issue.targetId);
  if (issue.lon != null && issue.lat != null) map.setView([issue.lat, issue.lon], 14);
  setNotice(`${issue.title}: ${issue.detail}`);
}

async function planJourney() {
  if (!state.session) await analyseProject();
  if (!state.session) return;
  const fromStationId = $("#journeyFrom").value; const toStationId = $("#journeyTo").value;
  const algorithm = document.querySelector('input[name="algorithm"]:checked').value;
  const result = await api(`/api/v1/analysis/sessions/${state.session.sessionId}/journeys`, { method: "POST", body: { fromStationId, toStationId, algorithm, disruptions: state.project.disruptions } });
  renderJourney(result); selectDrawer("journey", true);
}

function renderJourney(result) {
  journeyLayer.clearLayers();
  const root = $("#journeyResult");
  if (!result.reachable) { root.innerHTML = `<p>${escapeHtml(result.message || "No route is available.")}</p>`; return; }
  const stations = new Map(state.project.network.stations.map((station) => [station.id, station]));
  const path = result.stationIds.map((id) => stations.get(id)).filter(Boolean);
  if (path.length > 1) L.polyline(path.map((station) => [station.lat, station.lon]), { color: "#111920", weight: 9, opacity: .72 }).addTo(journeyLayer);
  root.className = "journey-result";
  root.innerHTML = `<div class="journey-summary"><span class="metric-label">${result.algorithm} optimal route</span><strong class="journey-total">${result.totalMinutes.toFixed(1)} min</strong><div class="journey-breakdown"><span>${result.waitMinutes.toFixed(1)} wait</span><span>${result.inVehicleMinutes.toFixed(1)} ride</span><span>${result.transferMinutes.toFixed(1)} transfer</span></div><p class="microcopy">${result.visitedStates} states · ${result.runtimeMicros.toLocaleString()} μs</p></div><div class="journey-legs">${result.legs.map((leg) => `<div class="journey-leg"><span class="metric-label">${leg.kind}</span><strong>${escapeHtml(leg.lineId)}</strong><span class="row-meta">${escapeHtml(leg.fromStationId)} → ${escapeHtml(leg.toStationId)}</span><span class="numeric">${leg.minutes.toFixed(1)} min</span></div>`).join("")}</div>`;
  if (path.length > 1) map.fitBounds(path.map((station) => [station.lat, station.lon]), { padding: [90, 90] });
}

function renderDisruptionTargets() {
  if (!state.project) return;
  const type = $("#disruptionType").value;
  let targets = [];
  if (type === "TRAIN_CANCELLATION") targets = state.project.network.lines.map((line) => ({ id: line.id, label: line.name }));
  else if (type === "STATION_CLOSURE") targets = state.project.network.stations.map((station) => ({ id: station.id, label: station.name }));
  else targets = state.project.network.lines.flatMap((line) => line.segments.filter((segment) => type !== "ADVERSE_WEATHER" || segment.infrastructure === "OVERGROUND").map((segment) => ({ id: segment.id, label: `${line.name} · ${segment.id} · ${titleCase(segment.infrastructure)}` })));
  $("#disruptionTarget").innerHTML = targets.map((target) => `<option value="${escapeAttribute(target.id)}">${escapeHtml(target.label)}</option>`).join("") || '<option value="">No eligible location</option>';
  $("#addDisruptionBtn").disabled = !targets.length;
}

function addDisruption() {
  const targetId = $("#disruptionTarget").value;
  if (!targetId) return;
  commit("Add disruption", (project) => project.disruptions.push({
    id: crypto.randomUUID(), type: $("#disruptionType").value, targetId,
    startMinute: Number($("#disruptionStart").value), durationMinutes: Number($("#disruptionDuration").value),
    severity: Number($("#severityInput").value) / 100, label: $("#disruptionType").selectedOptions[0].textContent,
  }));
}

function renderDisruptions() {
  const list = $("#disruptionList");
  $("#scenarioCount").textContent = `${state.project.disruptions.length} active`;
  list.innerHTML = state.project.disruptions.map((item) => `<div class="disruption-row"><span><strong>${escapeHtml(item.label)}</strong><span class="row-meta">${escapeHtml(item.targetId)} · ${Math.round(item.severity * 100)}%</span></span><button data-remove-disruption="${item.id}" type="button">Remove</button></div>`).join("");
  list.querySelectorAll("[data-remove-disruption]").forEach((button) => button.addEventListener("click", () => commit("Remove disruption", (project) => project.disruptions = project.disruptions.filter((item) => item.id !== button.dataset.removeDisruption))));
}

async function toggleSimulation() {
  if (!state.simulationPlaying) {
    if (!state.session) await analyseProject();
    if (!state.session) return;
    state.simulation = await api(`/api/v1/analysis/sessions/${state.session.sessionId}/simulations`, { method: "POST", body: { horizonMinutes: 120, disruptions: state.project.disruptions } });
    state.simulationPlaying = true; state.simulationLastTimestamp = null;
    $("#simulationPlayBtn").textContent = "Pause";
    renderSimulationSummary();
    state.simulationFrame = requestAnimationFrame(simulationTick);
    selectDrawer("simulation", true);
  } else {
    state.simulationPlaying = false; $("#simulationPlayBtn").textContent = "Play";
    cancelAnimationFrame(state.simulationFrame);
  }
}

function simulationTick(timestamp) {
  if (!state.simulationPlaying) return;
  if (state.simulationLastTimestamp != null) state.simulationMinute += (timestamp - state.simulationLastTimestamp) / 1000 * Number($("#simulationSpeed").value);
  state.simulationLastTimestamp = timestamp;
  if (state.simulationMinute > state.simulation.horizonMinutes) { resetSimulation(); return; }
  renderTrainPositions();
  $("#simulationClock").textContent = formatClock(state.simulationMinute);
  state.simulationFrame = requestAnimationFrame(simulationTick);
}

function renderTrainPositions() {
  trainLayer.clearLayers();
  if (!state.simulation) return;
  const stations = new Map(state.project.network.stations.map((station) => [station.id, station]));
  const lines = new Map(state.project.network.lines.map((line) => [line.id, line]));
  state.simulation.trains.filter((run) => !run.cancelled).forEach((run) => {
    const elapsed = state.simulationMinute - run.departureMinute;
    if (elapsed < 0 || elapsed > run.cumulativeMinutes.at(-1)) return;
    let index = run.cumulativeMinutes.findIndex((minute) => minute >= elapsed);
    if (index <= 0) index = 1;
    const start = stations.get(run.stationIds[index - 1]); const end = stations.get(run.stationIds[index]);
    if (!start || !end) return;
    const segmentStart = run.cumulativeMinutes[index - 1]; const segmentEnd = run.cumulativeMinutes[index];
    const ratio = segmentEnd === segmentStart ? 0 : (elapsed - segmentStart) / (segmentEnd - segmentStart);
    const lat = start.lat + (end.lat - start.lat) * ratio; const lon = start.lon + (end.lon - start.lon) * ratio;
    const color = lines.get(run.lineId)?.color || "#17202b";
    L.marker([lat, lon], { icon: L.divIcon({ className: "", iconSize: [12, 12], html: `<div class="train-marker" style="color:${safeColor(color)}"></div>` }), interactive: false }).addTo(trainLayer);
  });
}

function resetSimulation() {
  state.simulationPlaying = false; state.simulationMinute = 0; state.simulationLastTimestamp = null;
  cancelAnimationFrame(state.simulationFrame); trainLayer.clearLayers();
  $("#simulationPlayBtn").textContent = "Play"; $("#simulationClock").textContent = "00:00";
}

function renderSimulationSummary() {
  if (!state.simulation) return;
  const cancelled = state.simulation.trains.filter((run) => run.cancelled).length;
  $("#simulationSummary").className = "simulation-summary";
  $("#simulationSummary").innerHTML = `<div class="simulation-stat"><span class="metric-label">Scheduled runs</span><strong>${state.simulation.trains.length.toLocaleString()}</strong></div><div class="simulation-stat"><span class="metric-label">Cancelled</span><strong>${cancelled.toLocaleString()}</strong></div><div class="simulation-stat"><span class="metric-label">Horizon</span><strong>${state.simulation.horizonMinutes} minutes</strong></div><div class="simulation-stat"><span class="metric-label">Scenario events</span><strong>${state.simulation.activeDisruptions.length}</strong></div>`;
}

function setMode(mode) {
  state.mode = mode;
  $("#designModeBtn").classList.toggle("is-active", mode === "design");
  $("#operateModeBtn").classList.toggle("is-active", mode === "operate");
  $("#designModeBtn").setAttribute("aria-selected", String(mode === "design"));
  $("#operateModeBtn").setAttribute("aria-selected", String(mode === "operate"));
  $("#designPanel").classList.toggle("is-hidden", mode !== "design");
  $("#operatePanel").classList.toggle("is-hidden", mode !== "operate");
  renderNetwork();
  if (mode === "operate" && state.project.network.lines.length && !state.session) analyseProject();
}

function toggleDrawer() {
  const drawer = $("#analysisDrawer");
  drawer.classList.toggle("is-collapsed");
  const open = !drawer.classList.contains("is-collapsed");
  $("#drawerToggle").textContent = open ? "Close analysis" : "Open analysis";
  $("#drawerToggle").setAttribute("aria-expanded", String(open));
  setTimeout(() => map.invalidateSize(), 190);
}

function selectDrawer(name, forceOpen = false) {
  $$(".drawer-tab").forEach((button) => { const active = button.dataset.drawer === name; button.classList.toggle("is-active", active); button.setAttribute("aria-selected", String(active)); });
  $$(".drawer-view").forEach((view) => view.classList.add("is-hidden"));
  $(`#${name}View`).classList.remove("is-hidden");
  if (forceOpen && $("#analysisDrawer").classList.contains("is-collapsed")) toggleDrawer();
}

async function newProject() {
  const number = state.projects.length + 1;
  state.project = createProject(`Untitled London plan ${number}`);
  state.session = null; state.history = []; state.future = [];
  await persistProject(true); renderAll(); fitSupportedLondon();
}

function createProject(name) {
  const timestamp = new Date().toISOString();
  return { schemaVersion: 1, id: crypto.randomUUID(), name, revision: 0, createdAt: timestamp, updatedAt: timestamp, studyArea: null,
    demandConfig: { totalDailyJourneys: 500000, distanceDecayKm: 8, maxZones: 400, ptalInfluence: .25, modelVersion: "gravity-ipf-v1" },
    modelSettings: { stationCatchmentMeters: 800, undergroundSpeedKph: 40, overgroundSpeedKph: 45, dwellMinutes: .5, transferWalkMinutes: 3, surfaceReferenceSpeedKph: 20, peakHourShare: .1 },
    network: { stations: [], lines: [] }, disruptions: [] };
}

async function loadProjectRegister() {
  const db = await dbPromise;
  state.projects = await transactionRequest(db, "projects", "readonly", (store) => store.getAll());
  state.projects.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

async function persistProject(showConfirmation) {
  if (!state.project) return;
  const db = await dbPromise;
  await transactionPromise(db, ["projects", "revisions"], "readwrite", (transaction) => {
    transaction.objectStore("projects").put(structuredClone(state.project));
    transaction.objectStore("revisions").put({ key: `${state.project.id}:${state.project.revision}`, projectId: state.project.id, revision: state.project.revision, savedAt: new Date().toISOString(), project: structuredClone(state.project) });
  });
  await trimRevisions(db, state.project.id);
  localStorage.setItem("planner:lastProject", state.project.id);
  await loadProjectRegister(); renderProjectPicker();
  setSaveState(showConfirmation ? "Saved locally" : `Autosaved revision ${state.project.revision}`);
}

function scheduleSave() { clearTimeout(state.saveTimer); state.saveTimer = setTimeout(() => persistProject(false), 500); }

function renderProjectPicker() {
  const select = $("#projectSelect");
  const projects = [...state.projects];
  if (state.project && !projects.some((item) => item.id === state.project.id)) projects.unshift(state.project);
  select.innerHTML = projects.map((project) => `<option value="${escapeAttribute(project.id)}" ${project.id === state.project.id ? "selected" : ""}>${escapeHtml(project.name)}</option>`).join("");
}

async function loadSelectedProject(event) {
  const project = state.projects.find((item) => item.id === event.target.value);
  if (!project) return;
  state.project = structuredClone(project); state.session = null; state.history = []; state.future = []; state.selected = { type: "network", id: "network" };
  localStorage.setItem("planner:lastProject", project.id); renderAll();
  if (project.studyArea) map.fitBounds(project.studyArea.coordinates.map((point) => [point.lat, point.lon]), { padding: [30, 30] });
}

function exportProject() { downloadBlob(`${slugify(state.project.name)}.json`, JSON.stringify(state.project, null, 2), "application/json"); }
async function importProject(event) {
  const file = event.target.files[0]; event.target.value = ""; if (!file) return;
  try {
    const project = JSON.parse(await file.text());
    if (project.schemaVersion !== 1 || !project.id || !project.network) throw new Error("This is not a supported PlannerProjectV1 file.");
    project.id = crypto.randomUUID(); project.name = `${project.name || "Imported plan"} (imported)`; project.revision = 0;
    project.createdAt = new Date().toISOString(); project.updatedAt = project.createdAt;
    state.project = project; state.session = null; state.history = []; state.future = [];
    await persistProject(true); renderAll();
  } catch (error) { showError(error); }
}

function downloadDemand() { if (state.session) window.location.href = `/api/v1/analysis/sessions/${state.session.sessionId}/demand.csv`; }

function showMethod() {
  const sources = state.coverage?.sources || [];
  $("#methodContent").innerHTML = `<p>The green boundary is the hard planning extent. Red map space is unsupported and cannot receive stations or track. PTAL is used only where a nearby bundled record exists; otherwise its multiplier is neutral.</p><h3>Demand</h3><p>Origin-destination journeys are generated with a deterministic doubly constrained gravity model. They are simulated values, not observed passenger counts.</p><h3>Routing</h3><p>Dijkstra and A* use the same station/line state graph. Generalised time includes expected initial and transfer waits, dwell time, and infrastructure-dependent travel speed.</p><h3>Sources</h3>${sources.map((source) => `<p><strong>${escapeHtml(source.title)}</strong><br>${escapeHtml(source.attribution)} · ${escapeHtml(source.licence)}</p>`).join("")}`;
  $("#methodDialog").showModal();
}

function renderHistoryState() { $("#undoBtn").disabled = !state.history.length; $("#redoBtn").disabled = !state.future.length; }
function setSaveState(message) { $("#saveState").textContent = message; }
function setNotice(message, error = false) { const notice = $("#mapNotice"); notice.textContent = message; notice.classList.toggle("is-error", error); }
function showError(error) { console.error(error); setNotice(error.message || "The requested operation failed.", true); }

async function api(path, options = {}) {
  const response = await fetch(path, { method: options.method || "GET", headers: options.body ? { "Content-Type": "application/json" } : undefined, body: options.body ? JSON.stringify(options.body) : undefined });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("json") ? await response.json() : await response.text();
  if (!response.ok) throw new Error(payload.message || payload || `Request failed (${response.status})`);
  return payload;
}

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("london-network-planner", 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains("projects")) db.createObjectStore("projects", { keyPath: "id" });
      if (!db.objectStoreNames.contains("revisions")) {
        const store = db.createObjectStore("revisions", { keyPath: "key" }); store.createIndex("projectId", "projectId", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
  });
}

function transactionRequest(db, storeName, mode, operation) {
  return new Promise((resolve, reject) => { const tx = db.transaction(storeName, mode); const request = operation(tx.objectStore(storeName)); request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
}

function transactionPromise(db, stores, mode, operation) {
  return new Promise((resolve, reject) => { const transaction = db.transaction(stores, mode); operation(transaction); transaction.oncomplete = resolve; transaction.onerror = () => reject(transaction.error); });
}

async function trimRevisions(db, projectId) {
  const revisions = await new Promise((resolve, reject) => {
    const transaction = db.transaction("revisions", "readonly");
    const request = transaction.objectStore("revisions").index("projectId").getAll(projectId);
    request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
  });
  const obsolete = revisions.sort((a, b) => b.revision - a.revision).slice(20);
  if (!obsolete.length) return;
  await transactionPromise(db, ["revisions"], "readwrite", (transaction) => {
    const store = transaction.objectStore("revisions"); obsolete.forEach((revision) => store.delete(revision.key));
  });
}

function pointInCoverage(lon, lat) { return state.coverage?.hardSupport.some((polygon) => pointInPolygon(lon, lat, polygon)); }
function pointInPolygon(lon, lat, polygon) {
  let inside = false; let previous = polygon.at(-1);
  for (const current of polygon) { const intersects = ((current.lat > lat) !== (previous.lat > lat)) && lon < (previous.lon - current.lon) * (lat - current.lat) / (previous.lat - current.lat) + current.lon; if (intersects) inside = !inside; previous = current; }
  return inside;
}

function nearestStation(stations, point, maxMeters) { let best = null; let distance = Infinity; stations.forEach((station) => { const candidate = haversine(station, point); if (candidate < distance) { best = station; distance = candidate; } }); return distance <= maxMeters ? best : null; }
function haversine(a, b) { const toRad = Math.PI / 180; const dLat = (b.lat - a.lat) * toRad; const dLon = (b.lon - a.lon) * toRad; const p = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * toRad) * Math.cos(b.lat * toRad) * Math.sin(dLon / 2) ** 2; return 6371000 * 2 * Math.atan2(Math.sqrt(p), Math.sqrt(1 - p)); }
function findSegment(id) { for (const line of state.project.network.lines) { const segment = line.segments.find((item) => item.id === id); if (segment) return { line, segment }; } return null; }
function findSegmentInProject(project, id) { for (const line of project.network.lines) { const segment = line.segments.find((item) => item.id === id); if (segment) return { line, segment }; } return null; }
function nextNumber(ids, prefix) { const values = ids.map((id) => Number(id.replace(prefix, ""))).filter(Number.isFinite); return values.length ? Math.max(...values) + 1 : 1; }
function roleLabel(role) { return ({ RADIAL: "Radial", CROSS_CITY_TRUNK: "Cross-city trunk", ORBITAL_BYPASS: "Orbital bypass", CORE_DISTRIBUTOR: "Core distributor" })[role] || titleCase(role); }
function roleOptions(selected) { return ["RADIAL", "CROSS_CITY_TRUNK", "ORBITAL_BYPASS", "CORE_DISTRIBUTOR"].map((role) => `<option value="${role}" ${role === selected ? "selected" : ""}>${roleLabel(role)}</option>`).join(""); }
function titleCase(value) { return String(value).toLowerCase().replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function formatKm(meters) { return `${(meters / 1000).toFixed(meters < 10000 ? 1 : 0)} km`; }
function formatClock(minutes) { const whole = Math.floor(minutes); return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`; }
function safeColor(value) { return /^#[0-9a-f]{6}$/i.test(value) ? value : "#0b66d4"; }
function slugify(value) { return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "planner-project"; }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]); }
function escapeAttribute(value) { return escapeHtml(value); }
function downloadBlob(name, content, type) { const url = URL.createObjectURL(new Blob([content], { type })); const anchor = document.createElement("a"); anchor.href = url; anchor.download = name; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 0); }
