// @ts-nocheck
import { migrateProject, localBundleMetrics, toggleCandidate, canLockRoutes, recoveryReference, createPlannerUiState, selectedDraft } from "./planner-state";
import { decodeGenerationEvent } from "./stream";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  ...createPlannerUiState(),
  coverage: null,
  project: null,
  projects: [],
  session: null,
  demandRecords: [],
  generation: null,
  generationJob: null,
  previewPlans: [],
  sheetOpen: true,
  searchEvents: [],
  playbackIndex: 0,
  playbackTimer: null,
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
const map = L.map("map", { zoomControl: false, preferCanvas: false, minZoom: 9, maxZoom: 18 }).setView([51.5074, -0.1278], 10);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);
const coverageLayer = L.layerGroup().addTo(map);
const studyLayer = L.layerGroup().addTo(map);
const networkLayer = L.layerGroup().addTo(map);
const draftLayer = L.layerGroup().addTo(map);
const exploreLayer = L.layerGroup().addTo(map);
const demandLayer = L.layerGroup().addTo(map);
const journeyLayer = L.layerGroup().addTo(map);
const trainLayer = L.layerGroup().addTo(map);

boot().catch((error) => { showStartError(error); setSaveState("Project setup failed"); });

let coveragePromise;
let openingProject = false;

async function boot() {
  bindEvents();
  mountTaskContent();
  $(".app-shell").inert = true;
  await loadProjectRegister();
  renderStartupProjects();
  setSaveState("Choose a project");
  $("#startProgress").textContent = "Choose a project to begin.";
  $("#startNewBtn").focus();
}

function renderStartupProjects() {
  const list = $("#startProjectList");
  const lastId = localStorage.getItem("planner:lastProject");
  $("#startProjectName").value = `Untitled London plan ${state.projects.length + 1}`;
  list.innerHTML = state.projects.length ? state.projects.map((project) => {
    const date = new Date(project.updatedAt);
    const updated = Number.isNaN(date.getTime()) ? "Saved locally" : `Edited ${date.toLocaleDateString()}`;
    const lines = project.network.lines.length;
    return `<button class="start-project-row" type="button" data-start-project="${escapeAttribute(project.id)}"><span><strong>${escapeHtml(project.name || "Untitled project")}</strong><small>${project.id === lastId ? "Last opened · " : ""}${updated} · ${lines} line${lines === 1 ? "" : "s"}</small></span><span class="open-label">Open</span></button>`;
  }).join("") : "<p>No saved projects yet. Create one or import a project file.</p>";
}

function showStartError(error) {
  console.error(error);
  const message = $("#startError");
  message.textContent = error?.message || "The local workspace could not be opened. Try again.";
  message.classList.remove("is-hidden");
  $("#startProgress").textContent = "Choose an action to retry.";
  const list = $("#startProjectList");
  if (list.textContent.includes("Loading local projects")) list.innerHTML = "<p>Could not load local projects. Refresh the page to retry.</p>";
}

async function ensureCoverage() {
  if (state.coverage) return;
  if (!coveragePromise) coveragePromise = loadCoverage().catch((error) => { coveragePromise = null; throw error; });
  await coveragePromise;
}

async function openFromStart(openProject) {
  if (openingProject) return;
  openingProject = true;
  $("#startError").classList.add("is-hidden");
  $("#startProjectList").querySelectorAll("button").forEach((button) => { button.disabled = true; });
  $("#startNewBtn").disabled = true;
  $("#startImportBtn").disabled = true;
  setSaveState("Opening project…");
  $("#startProgress").textContent = "Loading the map and opening your project…";
  try {
    await ensureCoverage();
    await openProject();
    $("#projectStart").classList.add("is-hidden");
    $(".app-shell").inert = false;
    if (!state.generation && !$("#mapNotice").classList.contains("is-error")) setNotice("Draw a study area or create a manual line. All demand shown by this tool is simulated.");
    $(state.project.network.lines.length ? "#refineModeBtn" : "#drawAreaBtn").focus();
  } catch (error) {
    showStartError(error);
    setSaveState("Project could not be opened");
  } finally {
    openingProject = false;
    $("#startProjectList").querySelectorAll("button").forEach((button) => { button.disabled = false; });
    $("#startNewBtn").disabled = false;
    $("#startImportBtn").disabled = false;
  }
}

function bindEvents() {
  $("#designModeBtn").addEventListener("click", () => setMode("create"));
  $("#refineModeBtn").addEventListener("click", () => setMode("refine"));
  $("#operateModeBtn").addEventListener("click", () => setMode("analyse"));
  $("#hidePanelBtn").addEventListener("click", () => toggleSurface("workflow-panel", "showPanelBtn"));
  $("#showPanelBtn").addEventListener("click", () => toggleSurface("workflow-panel", "showPanelBtn"));

  $("#cancelGenerationBtn").addEventListener("click", cancelGeneration);
  $("#downloadSchematicBtn").addEventListener("click", downloadSchematic);
  $("#guidanceInput").addEventListener("input", (event) => $("#guidanceValue").textContent = Number(event.target.value).toFixed(2));
  $("#playSearchBtn").addEventListener("click", toggleSearchPlayback);
  $("#stepSearchBtn").addEventListener("click", () => stepSearch(1));
  $("#searchScrub").addEventListener("input", (event) => { state.playbackIndex = Number(event.target.value); drawSearchFrame(); });
  map.on("move zoom resize", drawSearchFrame);
  window.addEventListener("resize", () => { drawSearchFrame(); updateMapPadding(); });
  $("#drawAreaBtn").addEventListener("click", startStudyAreaDraw);
  $("#drawLineBtn").addEventListener("click", startLineDraw);
  $("#addLineBtn").addEventListener("click", startLineDraw);
  $("#generateBtn").addEventListener("click", generateNetwork);
  $("#newProjectBtn").addEventListener("click", () => newProject());
  $("#startNewBtn").addEventListener("click", () => openFromStart(() => newProject($("#startProjectName").value)));
  $("#startImportBtn").addEventListener("click", () => $("#projectFileInput").click());
  $("#startProjectList").addEventListener("click", (event) => {
    const button = event.target.closest("[data-start-project]");
    if (!button) return;
    const project = state.projects.find((item) => item.id === button.dataset.startProject);
    if (project) openFromStart(() => openExistingProject(project));
  });
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
  $("#zoomInBtn").addEventListener("click", () => map.zoomIn());
  $("#zoomOutBtn").addEventListener("click", () => map.zoomOut());
  $("#layerCoverageBtn").addEventListener("click", toggleCoverage);
  $("#layerDemandBtn").addEventListener("click", toggleDemand);
  $("#coverageToggle").addEventListener("click", toggleCoverage);
  $("#demandToggle").addEventListener("click", toggleDemand);
  $("#planJourneyBtn").addEventListener("click", planJourney);
  $("#disruptionType").addEventListener("change", renderDisruptionTargets);
  $("#severityInput").addEventListener("input", () => $("#severityValue").textContent = `${$("#severityInput").value}%`);
  $("#addDisruptionBtn").addEventListener("click", addDisruption);
  $("#simulationPlayBtn").addEventListener("click", toggleSimulation);
  $("#simulationResetBtn").addEventListener("click", resetSimulation);
  $("#drawerToggle").addEventListener("click", toggleDrawer);
  $("#lockRoutesBtn").addEventListener("click", lockRoutes);
  $("#reviewPlansBtn").addEventListener("click", () => setReviewTask("plans"));
  $("#adjustRoutesBtn").addEventListener("click", () => setReviewTask("adjust"));
  $("#backSelectionBtn").addEventListener("click", () => selectObject("network", "network"));
  $("#analyseNetworkBtn").addEventListener("click", () => setMode("analyse"));
  $("#backToRefineBtn").addEventListener("click", () => setMode("refine"));
  $("#sheetToggle").addEventListener("click", toggleSheet);
  $("#sheetPrimaryBtn").addEventListener("click", () => { if (state.mode === "review") lockRoutes(); else if (state.mode === "refine") startLineDraw(); else if (state.mode === "analyse") setMode("refine"); else if (state.project.studyArea) generateNetwork(); else startStudyAreaDraw(); });
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
  map.fitBounds([[south, west], [north, east]], { paddingTopLeft: mapTopLeftPadding(), paddingBottomRight: mapBottomRightPadding() });
}

function toggleCoverage() {
  $("#coverageToggle").classList.toggle("is-active");
  const active = $("#coverageToggle").classList.contains("is-active");
  $("#coverageToggle").setAttribute("aria-pressed", String(active));
  $("#layerCoverageBtn").setAttribute("aria-pressed", String(active));
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
    setMode("refine");
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
  button.disabled = true; button.classList.add("is-loading");
  state.generation = null; state.previewPlans = []; state.searchEvents = []; state.playbackIndex = 0;
  $("#searchTimeline").innerHTML = "";
  $("#cancelGenerationBtn").classList.remove("is-hidden");
  $("#searchState").textContent = "Searching";
  setNotice("Searching sparse corridors and plan alternatives…");
  try {
    const job = await api("/api/v1/generation/jobs", { method: "POST", body: { studyArea: state.project.studyArea, settings: generationSettings() } });
    state.generationJob = job;
    saveReviewReference();
    await followGeneration(job);
  } catch (error) { showError(error); }
  finally {
    button.classList.remove("is-loading");
    button.disabled = !state.project.studyArea;
    $("#cancelGenerationBtn").classList.add("is-hidden");
  }
}

function generationSettings() {
  return { guidanceStrength: Number($("#guidanceInput").value), radialSoftCap: Number($("#radialCap").value),
    orbitalSoftCap: Number($("#orbitalCap").value), distributorSoftCap: Number($("#coreCap").value),
    demandMode: $("#demandMode").value };
}

async function followGeneration(job) {
  await new Promise((resolve, reject) => {
    const source = new EventSource(job.eventsUrl);
    const types = ["SEED_DISCOVERED", "BEAM_STEP", "BRANCH_PRUNED", "CORRIDOR_COMMITTED", "RESIDUAL_HEATMAP_UPDATED", "PLAN_FORMED", "SEARCH_REFINED", "COMPLETE", "FAILED"];
    const onEvent = async (event) => {
      const update = decodeGenerationEvent(event.data);
      state.searchEvents.push(update);
      if (update.type === "PLAN_FORMED" && update.plan) {
        state.previewPlans.push(update.plan); renderPlans();
      }
      const entry = document.createElement("li"); entry.textContent = `${update.type.replaceAll("_", " ").toLowerCase()}: ${update.message}`;
      $("#searchTimeline").append(entry);
      $("#searchTimeline").scrollTop = $("#searchTimeline").scrollHeight;
      $("#searchScrub").max = state.searchEvents.length;
      if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        state.playbackIndex = state.searchEvents.length; $("#searchScrub").value = String(state.playbackIndex); drawSearchFrame();
      }
      if (update.type === "COMPLETE" || update.type === "FAILED") {
        source.close();
        const status = await api(job.statusUrl);
        state.generationJob = status;
        if (status.state === "COMPLETE") {
          state.generation = status.result;
          setMode("review"); renderPlans(); renderCandidates();
          $("#searchState").textContent = "Complete";
          setNotice(`${status.result.plans.length} plan alternatives from ${status.result.candidates.length} sparse corridors.`);
        } else if (status.state === "CANCELLED") { $("#searchState").textContent = "Cancelled"; }
        else { $("#searchState").textContent = "Failed"; reject(new Error(status.error || "Generation failed")); return; }
        resolve();
      }
    };
    types.forEach((type) => source.addEventListener(type, onEvent));
    source.onerror = async () => {
      source.close();
      try {
        const status = await api(job.statusUrl);
        if (status.state === "COMPLETE") { state.generation = status.result; setMode("review"); renderPlans(); renderCandidates(); resolve(); }
        else if (status.state === "CANCELLED") resolve();
        else reject(new Error(status.error || "Generation stream disconnected; retry generation."));
      } catch (error) { reject(error); }
    };
  });
}

async function cancelGeneration() {
  if (!state.generationJob) return;
  await api(state.generationJob.statusUrl, { method: "DELETE" });
  $("#searchState").textContent = "Cancelling";
}

function renderPlans() {
  const result = state.generation || { plans: state.previewPlans };
  $("#planCount").textContent = String(result?.plans.length || 0);
  const root = $("#planAlternatives");
  root.innerHTML = result?.plans.length ? result.plans.slice().sort((a, b) => Number(b.profile === "BALANCED") - Number(a.profile === "BALANCED")).map((plan) => {
    const index = result.plans.indexOf(plan);
    const m = plan.metrics;
    const axes = [m.coverage, Math.max(0, 1 - m.costEstimate / 10e9), 1 - m.transferOverhead, 1 - m.duplication, m.connectivity];
    const polygon = axes.map((value, i) => { const angle = -Math.PI / 2 + i * Math.PI * 2 / 5; return `${(50 + 38 * value * Math.cos(angle)).toFixed(1)},${(50 + 38 * value * Math.sin(angle)).toFixed(1)}`; }).join(" ");
    return `<article class="plan-article"><button class="plan-option ${state.selectedPlanId === plan.id ? "is-selected" : ""}" aria-pressed="${state.selectedPlanId === plan.id}" data-plan="${index}" type="button" ${state.generation ? "" : "disabled"}><strong>${escapeHtml(titleCase(plan.profile))}</strong><span>${Math.round(m.coverage * 100)}% coverage</span><svg viewBox="0 0 100 100" role="img" aria-label="Five objective radar for ${escapeAttribute(plan.profile)}"><circle cx="50" cy="50" r="38" fill="none" stroke="#d6dde1"/><polygon points="${polygon}" fill="rgba(11,102,212,.18)" stroke="#0b66d4" stroke-width="2"/></svg><small>${plan.network.lines.length} lines · ${plan.id.startsWith("preview-") ? "Preview · refining" : plan.certified ? "Certified" : `Gap ≤ ${(100 * (plan.optimalityGap || 0)).toFixed(0)}%`}</small></button><table class="plan-metrics"><tbody><tr><th>Length</th><td>${formatKm(m.lengthMeters)}</td><th>Cost</th><td>£${(m.costEstimate / 1e9).toFixed(2)}bn</td></tr><tr><th>Transfer</th><td>${(m.transferOverhead * 100).toFixed(0)}%</td><th>Duplicate</th><td>${(m.duplication * 100).toFixed(0)}%</td></tr><tr><th>Connected</th><td>${(m.connectivity * 100).toFixed(0)}%</td><th>Guidance</th><td>${m.guidancePenalty.toFixed(2)}</td></tr></tbody></table></article>`;
  }).join("") : '<p class="help">No viable plans. Expand the study area.</p>';
  root.querySelectorAll("[data-plan]").forEach((button) => button.addEventListener("click", () => choosePlan(Number(button.dataset.plan))));
  drawSearchFrame();
}

function choosePlan(index) {
  const plan = state.generation?.plans[index]; if (!plan) return;
  Object.assign(state, selectedDraft(plan));
  setMode("review");
  if (plan.network.stations.length) map.fitBounds(plan.network.stations.map((station) => [station.lat, station.lon]), { paddingTopLeft: mapTopLeftPadding(), paddingBottomRight: mapBottomRightPadding(), maxZoom: 13 });
  setReviewTask("adjust");
  saveReviewReference();
  renderPlans(); renderCandidates(); renderDraft(); renderLockState();
}

function renderCandidates() {
  const candidates = state.generation?.candidates || [];
  $("#candidateCount").textContent = String(candidates.length);
  const groups = new Map();
  candidates.forEach((candidate) => { const role = roleLabel(candidate.role); if (!groups.has(role)) groups.set(role, []); groups.get(role).push(candidate); });
  const root = $("#candidateList");
  root.innerHTML = candidates.length ? [...groups.entries()].map(([role, items]) => `<details class="candidate-group" ${items.some((item) => state.candidateIds.includes(item.id)) ? "open" : ""}><summary>${escapeHtml(role)} <span>${items.filter((item) => state.candidateIds.includes(item.id)).length}/${items.length} selected</span></summary>${items.map((candidate) => `<label class="candidate-row"><input type="checkbox" value="${escapeAttribute(candidate.id)}" ${state.candidateIds.includes(candidate.id) ? "checked" : ""} /><span>${escapeHtml(candidate.id)} · ${formatKm(candidate.lengthMeters)}</span></label>`).join("")}</details>`).join("") : "<p>Generate candidates first.</p>";
  root.querySelectorAll("input").forEach((input) => input.addEventListener("change", evaluateBundle));
  renderLockState();
}

let bundleTimer;
let bundleSerial = 0;
function evaluateBundle(event) {
  const serial = ++bundleSerial;
  state.candidateIds = toggleCandidate(state.candidateIds, event.target.value, event.target.checked);
  state.draftEvaluation = null;
  state.evaluationStatus = state.candidateIds.length ? "pending" : "idle";
  state.evaluationError = null;
  const ids = [...state.candidateIds];
  const local = localBundleMetrics(state.generation.candidates, ids);
  $("#bundleMetrics").textContent = `${local.lineCount} lines · ${formatKm(local.lengthMeters)} selected length · ${ids.length ? "evaluating…" : "choose at least one corridor"}`;
  renderCandidates(); renderDraft(); renderLockState(); saveReviewReference();
  clearTimeout(bundleTimer);
  if (!ids.length) return;
  bundleTimer = setTimeout(async () => {
    try {
      const plan = await api(`${state.generationJob.statusUrl}/evaluate`, { method: "POST", body: { candidateIds: ids, settings: generationSettings() } });
      if (serial !== bundleSerial || state.mode !== "review") return;
      state.draftEvaluation = plan;
      state.evaluationStatus = "success";
      $("#bundleMetrics").textContent = `${Math.round(plan.metrics.coverage * 100)}% coverage · ${formatKm(plan.metrics.lengthMeters)} · £${(plan.metrics.costEstimate / 1e9).toFixed(1)}bn · ${(plan.metrics.duplication * 100).toFixed(0)}% duplicated`;
      renderDraft(); renderLockState();
    } catch (error) {
      if (serial !== bundleSerial) return;
      state.evaluationStatus = "failed"; state.evaluationError = error.message;
      $("#bundleMetrics").textContent = `Evaluation failed: ${error.message}. Change the selection to retry.`;
      renderLockState();
    }
  }, 350);
}

function renderDraft() {
  draftLayer.clearLayers(); exploreLayer.clearLayers();
  if (state.mode !== "review" || !state.generation) return;
  const selected = new Set(state.candidateIds);
  if (state.reviewTask === "plans") state.generation.candidates.filter((item) => !selected.has(item.id)).forEach((candidate) => {
    L.polyline(candidate.coordinates.map((point) => [point.lat, point.lon]), { color: "#52606d", weight: 2, opacity: .38, dashArray: "5 7", interactive: false }).addTo(exploreLayer);
  });
  const network = state.draftEvaluation?.network;
  if (network && state.evaluationStatus === "success") {
    const stations = new Map(network.stations.map((station) => [station.id, station]));
    network.lines.forEach((line) => line.segments.forEach((segment) => {
      const from = stations.get(segment.fromStationId), to = stations.get(segment.toStationId);
      if (from && to) L.polyline([[from.lat, from.lon], [to.lat, to.lon]], { color: safeColor(line.color), weight: 7, opacity: .95, dashArray: segment.infrastructure === "OVERGROUND" ? "10 8" : null, interactive: false }).addTo(draftLayer);
    }));
    network.stations.forEach((station) => L.circleMarker([station.lat, station.lon], { radius: 5, color: "#17202b", weight: 2, fillColor: "#fff", fillOpacity: 1, interactive: false }).addTo(draftLayer));
  } else {
    state.generation.candidates.filter((item) => selected.has(item.id)).forEach((candidate) => {
      L.polyline(candidate.coordinates.map((point) => [point.lat, point.lon]), { color: "#0b66d4", weight: 6, opacity: .9, interactive: false }).addTo(draftLayer);
      candidate.coordinates.forEach((point) => L.circleMarker([point.lat, point.lon], { radius: 4, color: "#17202b", weight: 2, fillColor: "#fff", fillOpacity: 1, interactive: false }).addTo(draftLayer));
    });
  }
}

function renderLockState() {
  const button = $("#lockRoutesBtn");
  button.disabled = !canLockRoutes({ status: state.evaluationStatus, ids: state.candidateIds, evaluation: state.draftEvaluation });
  updateSheetPrimary();
  $("#reviewStatus").textContent = state.evaluationStatus === "success" ? "Evaluated routes ready to save" : state.evaluationStatus === "pending" ? "Evaluating selected routes…" : state.evaluationStatus === "failed" ? "Evaluation failed" : "Choose routes to continue";
}

function lockRoutes() {
  if (!canLockRoutes({ status: state.evaluationStatus, ids: state.candidateIds, evaluation: state.draftEvaluation })) return;
  const plan = state.draftEvaluation;
  ++bundleSerial; clearTimeout(bundleTimer);
  commit("Lock in routes", (project) => {
    project.network = structuredClone(plan.network);
    project.generationSettings = generationSettings();
    project.corridorProvenance = Object.fromEntries(state.generation.candidates.filter((item) => state.candidateIds.includes(item.id)).map((item) => [item.id, item.provenance]));
  });
  clearReview(); setMode("refine");
  setNotice("Routes locked in and saved locally. Refine the network or analyse it.");
}

function clearReview() {
  ++bundleSerial; clearTimeout(bundleTimer);
  state.generation = null; state.generationJob = null; state.selectedPlanId = null; state.candidateIds = []; state.draftEvaluation = null; state.evaluationStatus = "idle";
  sessionStorage.removeItem("planner:review");
  setReviewTask("plans");
  draftLayer.clearLayers(); exploreLayer.clearLayers(); drawSearchFrame();
}

function saveReviewReference() {
  if (!state.generationJob || !state.project) return;
  sessionStorage.setItem("planner:review", JSON.stringify({ ...recoveryReference(state.project.id, state.generationJob.statusUrl, state.selectedPlanId, state.candidateIds), settings: generationSettings() }));
}

async function recoverReview() {
  let reference;
  try { reference = JSON.parse(sessionStorage.getItem("planner:review") || "null"); } catch { reference = null; }
  if (!reference || reference.projectId !== state.project.id) return;
  if (typeof reference.statusUrl !== "string" || !/^\/api\/v1\/generation\/jobs\/[a-f0-9-]{36}$/.test(reference.statusUrl)) { clearReview(); return; }
  try {
    const status = await api(reference.statusUrl);
    if (status.state !== "COMPLETE" || !status.result) throw new Error("The search is no longer available.");
    state.generationJob = status; state.generation = status.result;
    if (reference.settings) {
      const settings = reference.settings;
      if (Number.isFinite(settings.guidanceStrength)) $("#guidanceInput").value = settings.guidanceStrength;
      if (["OBSERVED_BLEND", "GRAVITY_ONLY"].includes(settings.demandMode)) $("#demandMode").value = settings.demandMode;
      [["radialCap", "radialSoftCap"], ["orbitalCap", "orbitalSoftCap"], ["coreCap", "distributorSoftCap"]].forEach(([id, key]) => { if (Number.isInteger(settings[key])) $(`#${id}`).value = settings[key]; });
    }
    setMode("review"); renderPlans(); renderCandidates();
    const index = status.result.plans.findIndex((plan) => plan.id === reference.planId);
    if (index >= 0) choosePlan(index);
    if (Array.isArray(reference.candidateIds) && reference.candidateIds.join() !== state.candidateIds.join()) {
      state.candidateIds = reference.candidateIds.filter((id) => status.result.candidates.some((candidate) => candidate.id === id));
      state.draftEvaluation = null; state.evaluationStatus = "pending"; renderCandidates(); renderDraft(); renderLockState();
      // Re-evaluate the recovered bundle without changing the saved project.
      if (state.candidateIds.length) {
        const plan = await api(`${status.statusUrl}/evaluate`, { method: "POST", body: { candidateIds: state.candidateIds, settings: generationSettings() } });
        state.draftEvaluation = plan; state.evaluationStatus = "success";
      } else state.evaluationStatus = "idle";
      saveReviewReference(); renderDraft(); renderLockState();
    }
    setNotice("Recovered your route draft from this browser session.");
  } catch (error) { clearReview(); setMode(state.project.network.lines.length ? "refine" : "create"); setNotice(`${error.message} Showing your last saved network.`, true); }
}

function setReviewTask(task) {
  state.reviewTask = task;
  $("#reviewPlansView").classList.toggle("is-hidden", task !== "plans");
  $("#reviewAdjustView").classList.toggle("is-hidden", task !== "adjust");
  [["reviewPlansBtn", "plans"], ["adjustRoutesBtn", "adjust"]].forEach(([id, value]) => { $(`#${id}`).classList.toggle("is-active", task === value); $(`#${id}`).setAttribute("aria-pressed", String(task === value)); });
  renderDraft();
}

function downloadSchematic() {
  const lines = state.project.network.lines;
  if (!lines.length) return;
  const byId = new Map(state.project.network.stations.map((station) => [station.id, station]));
  const rowHeight = 42;
  const width = 1120; const height = 95 + lines.reduce((sum, line) => sum + Math.max(1, line.stationIds.length) * rowHeight + 70, 0);
  let y = 54;
  const body = lines.map((line) => {
    const names = line.stationIds.map((id) => byId.get(id)?.name || id);
    const routeName = names[0] === names[names.length - 1] ? `${names[0]} Line` : `${names[0]}–${names[names.length - 1]} Line`;
    const start = y; y += names.length * rowHeight + 70;
    const color = safeColor(line.color);
    const stops = names.map((name, index) => {
      const sy = start + 42 + index * rowHeight;
      const interchange = lines.filter((other) => other.stationIds.includes(line.stationIds[index])).length > 1;
      return `<circle cx="110" cy="${sy}" r="${interchange ? 9 : 6}" fill="white" stroke="${color}" stroke-width="3"/><text x="138" y="${sy + 5}" font-size="16" fill="#17202b">${escapeHtml(name)}${interchange ? " · interchange" : ""}</text>`;
    }).join("");
    return `<g><text x="60" y="${start}" font-size="23" font-weight="700" fill="${color}">${escapeHtml(routeName)}</text><text x="700" y="${start}" font-size="14" fill="#52606d">${escapeHtml(roleLabel(line.role))} · ${line.trainsPerHour} tph</text><path d="M110 ${start + 42} V${start + 42 + (names.length - 1) * rowHeight}" stroke="${color}" stroke-width="9" fill="none"/>${stops}</g>`;
  }).join("");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}"><rect width="100%" height="100%" fill="white"/><text x="60" y="36" font-family="sans-serif" font-size="14" fill="#52606d">${escapeHtml(state.project.name)} · schematic planning view · simulated service</text><g font-family="sans-serif">${body}</g></svg>`;
  downloadBlob(`${slugify(state.project.name)}-schematic.svg`, svg, "image/svg+xml");
}

function toggleSurface(className, returnButtonId) {
  const surface = document.querySelector(`.${className}`);
  const collapsed = surface.classList.toggle("is-collapsed");
  document.querySelector(".workspace").classList.toggle(className === "workflow-panel" ? "panel-hidden" : "inspector-hidden", collapsed);
  $(`#${returnButtonId}`).classList.toggle("is-hidden", !collapsed);
  setTimeout(() => map.invalidateSize(), 0);
}

function toggleSearchPlayback() {
  if (state.playbackTimer) { clearInterval(state.playbackTimer); state.playbackTimer = null; $("#playSearchBtn").textContent = "Play"; return; }
  state.playbackIndex = 0; $("#playSearchBtn").textContent = "Pause";
  state.playbackTimer = setInterval(() => {
    stepSearch(1);
    if (state.playbackIndex >= state.searchEvents.length) toggleSearchPlayback();
  }, 400 / Number($("#searchSpeed").value));
}
function stepSearch(amount) { state.playbackIndex = Math.min(state.searchEvents.length, state.playbackIndex + amount); $("#searchScrub").value = String(state.playbackIndex); drawSearchFrame(); }
function drawSearchFrame() {
  const canvas = $("#searchCanvas"); const rect = canvas.getBoundingClientRect(); const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * scale)); canvas.height = Math.max(1, Math.round(rect.height * scale));
  const context = canvas.getContext("2d"); context.scale(scale, scale);
  const candidates = state.mode === "create" ? state.generation?.candidates || [] : [];
  const visible = state.searchEvents.slice(0, state.playbackIndex).filter((event) => event.type === "SEED_DISCOVERED").length;
  candidates.slice(0, visible).forEach((candidate, index) => {
    context.beginPath(); candidate.coordinates.forEach((point, i) => { const pos = map.latLngToContainerPoint([point.lat, point.lon]); if (i) context.lineTo(pos.x, pos.y); else context.moveTo(pos.x, pos.y); });
    context.strokeStyle = index < state.project.network.lines.length ? "rgba(11,102,212,.7)" : "rgba(184,102,8,.35)";
    context.lineWidth = index < state.project.network.lines.length ? 4 : 2; context.stroke();
  });
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
  renderDraft();
  renderLineList();
  renderStationSelectors();
  renderFrequencyControls();
  renderDisruptionTargets();
  renderDisruptions();
  renderSelection();
  renderHistoryState();
  $("#dailyJourneysInput").value = state.project.demandConfig.totalDailyJourneys;
  $("#decayInput").value = state.project.demandConfig.distanceDecayKm;
  const settings = state.project.generationSettings || {};
  if (settings.guidanceStrength !== undefined) { $("#guidanceInput").value = settings.guidanceStrength; $("#guidanceValue").textContent = Number(settings.guidanceStrength).toFixed(2); }
  if (settings.demandMode) $("#demandMode").value = settings.demandMode;
  if (settings.radialSoftCap !== undefined) $("#radialCap").value = settings.radialSoftCap;
  if (settings.orbitalSoftCap !== undefined) $("#orbitalCap").value = settings.orbitalSoftCap;
  if (settings.distributorSoftCap !== undefined) $("#coreCap").value = settings.distributorSoftCap;
  $("#studyAreaStatus").textContent = state.project.studyArea ? "Area ready" : "No area";
  $("#areaSummary").textContent = state.project.studyArea ? `${state.project.studyArea.coordinates.length} boundary points · supported Greater London area` : "Choose a supported area on the map.";
  $("#studyAreaStatus").className = `status-chip ${state.project.studyArea ? "success" : ""}`;
  $("#generateBtn").disabled = !state.project.studyArea;
  $("#refineModeBtn").disabled = !state.project.network.lines.length;
  $("#operateModeBtn").disabled = !state.project.network.lines.length;
  if (!state.project.network.lines.length && state.mode !== "create" && state.mode !== "review") setMode("create");
  updateSheetPrimary();
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
  if (state.mode === "review") return;
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
    const marker = L.marker([station.lat, station.lon], { icon, draggable: state.mode === "refine", keyboard: true }).addTo(networkLayer);
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
  if (type !== "network" && state.mode === "analyse") setMode("refine");
  renderTaskSelection();
  renderNetwork(); renderLineList(); renderSelection();
  const layer = state.mapLayers.get(id);
  if (layer?.getBounds) map.fitBounds(layer.getBounds(), { paddingTopLeft: mapTopLeftPadding(), paddingBottomRight: mapBottomRightPadding(), maxZoom: 14 });
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
  $("#layerDemandBtn").setAttribute("aria-pressed", String(state.demandVisible));
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
  if ((mode === "refine" || mode === "analyse") && !state.project.network.lines.length) return;
  if (state.mode === "review" && mode !== "review") clearReview();
  state.mode = mode;
  [["designModeBtn", "create"], ["refineModeBtn", "refine"], ["operateModeBtn", "analyse"]].forEach(([id, value]) => {
    $(`#${id}`).classList.toggle("is-active", mode === value || (mode === "review" && value === "create"));
    $(`#${id}`).setAttribute("aria-selected", String(mode === value || (mode === "review" && value === "create")));
  });
  $("#designPanel").classList.toggle("is-hidden", mode === "analyse");
  $("#operatePanel").classList.toggle("is-hidden", mode !== "analyse");
  $$(".stage-create").forEach((item) => item.classList.toggle("is-hidden", mode !== "create"));
  $$(".stage-review").forEach((item) => item.classList.toggle("is-hidden", mode !== "review"));
  $$(".stage-refine").forEach((item) => item.classList.toggle("is-hidden", mode !== "refine"));
  $("#reviewAction").classList.toggle("is-hidden", mode !== "review");
  renderTaskSelection(); renderNetwork(); renderDraft(); drawSearchFrame();
  if (mode === "analyse" && !state.session) analyseProject();
  if (mode === "analyse") selectDrawer("issues", true);
  updateSheetPrimary(); updateMapPadding();
}

function toggleDrawer() { selectDrawer(state.analysisTask === "overview" ? "issues" : "issues"); }
function selectDrawer(name) {
  state.analysisTask = name;
  $$(".drawer-tab").forEach((button) => { const active = button.dataset.drawer === name; button.classList.toggle("is-active", active); button.setAttribute("aria-selected", String(active)); });
  $$(".drawer-view").forEach((view) => view.classList.add("is-hidden"));
  $(`#${name}View`).classList.remove("is-hidden");
  $("#overviewWorkspace").classList.toggle("is-hidden", name !== "issues");
  $("#journeyWorkspace").classList.toggle("is-hidden", name !== "journey");
  $("#operationsWorkspace").classList.toggle("is-hidden", name !== "simulation");
}

function mountTaskContent() {
  $("#selectionWorkspace").append($("#selectionType"), $("#selectionInspector"));
  $("#analysisOverview").append($(".inspector-head"), $("#scoreComponents"));
  $("#coverageWorkspace").append($(".coverage-readout"));
  $("#analysisTasks").append($(".drawer-tabs"), $(".drawer-content"));
  $("#analysisDrawer").remove(); $(".inspector").remove();
}
function renderTaskSelection() {
  const editing = state.selected.type !== "network" && state.mode === "refine";
  $("#selectionWorkspace").classList.toggle("is-hidden", !editing);
  $("#refineWorkspace").classList.toggle("is-hidden", editing || state.mode !== "refine");
}
function toggleSheet() {
  state.sheetOpen = !state.sheetOpen;
  $(".workspace").classList.toggle("sheet-collapsed", !state.sheetOpen);
  $("#sheetToggle").textContent = state.sheetOpen ? "Collapse panel" : "Open panel";
  $("#sheetToggle").setAttribute("aria-expanded", String(state.sheetOpen));
  updateSheetPrimary(); updateMapPadding();
}
function updateSheetPrimary() {
  const button = $("#sheetPrimaryBtn");
  if (!button || !state.project) return;
  button.classList.toggle("is-hidden", state.sheetOpen);
  button.textContent = state.mode === "review" ? "Lock in routes" : state.mode === "refine" ? "Add line" : state.mode === "analyse" ? "Back to Refine" : state.project.studyArea ? "Generate network" : "Draw study area";
  button.disabled = state.mode === "review" && !canLockRoutes({ status: state.evaluationStatus, ids: state.candidateIds, evaluation: state.draftEvaluation });
}
function mapTopLeftPadding() { return window.innerWidth >= 1100 && !$(".workspace").classList.contains("panel-hidden") ? [355, 20] : [20, 20]; }
function mapBottomRightPadding() { return window.innerWidth < 1100 && state.sheetOpen ? [20, Math.min(440, window.innerHeight * .48)] : [20, 20]; }
function updateMapPadding() { map.invalidateSize(); }

async function newProject(name) {
  const number = state.projects.length + 1;
  state.project = createProject(typeof name === "string" && name.trim() ? name.trim() : `Untitled London plan ${number}`);
  state.session = null; state.history = []; state.future = [];
  clearReview(); state.previewPlans = []; setMode("create");
  await persistProject(true); renderAll(); fitSupportedLondon();
  $(".project-menu").open = false;
}

function createProject(name) {
  const timestamp = new Date().toISOString();
  return { schemaVersion: 2, id: crypto.randomUUID(), name, revision: 0, createdAt: timestamp, updatedAt: timestamp, studyArea: null, generationSettings: { guidanceStrength: .08, radialSoftCap: 3, orbitalSoftCap: 3, distributorSoftCap: 3, demandMode: "OBSERVED_BLEND" }, corridorProvenance: {},
    demandConfig: { totalDailyJourneys: 500000, distanceDecayKm: 8, maxZones: 400, ptalInfluence: .25, modelVersion: "gravity-ipf-v1" },
    modelSettings: { stationCatchmentMeters: 800, undergroundSpeedKph: 40, overgroundSpeedKph: 45, dwellMinutes: .5, transferWalkMinutes: 3, surfaceReferenceSpeedKph: 20, peakHourShare: .1 },
    network: { stations: [], lines: [] }, disruptions: [] };
}

async function loadProjectRegister() {
  const db = await dbPromise;
  state.projects = (await transactionRequest(db, "projects", "readonly", (store) => store.getAll())).map(migrateProject);
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
  await openExistingProject(project);
}

async function openExistingProject(project) {
  state.project = migrateProject(structuredClone(project)); state.session = null; state.generation = null; state.previewPlans = []; state.candidateIds = []; clearReview(); state.history = []; state.future = []; state.selected = { type: "network", id: "network" };
  localStorage.setItem("planner:lastProject", project.id); renderAll(); setMode(state.project.network.lines.length ? "refine" : "create");
  if (project.studyArea) map.fitBounds(project.studyArea.coordinates.map((point) => [point.lat, point.lon]), { padding: [30, 30] });
  await recoverReview();
  setSaveState(`Autosaved revision ${state.project.revision}`);
}

function exportProject() { downloadBlob(`${slugify(state.project.name)}.json`, JSON.stringify(state.project, null, 2), "application/json"); }
async function importProject(event) {
  const file = event.target.files[0]; event.target.value = ""; if (!file) return;
  try {
    await ensureCoverage();
    const project = JSON.parse(await file.text());
    if (![1, 2].includes(project.schemaVersion) || !project.id || !project.network) throw new Error("This is not a supported planner project file.");
    Object.assign(project, migrateProject(project));
    project.id = crypto.randomUUID(); project.name = `${project.name || "Imported plan"} (imported)`; project.revision = 0;
    project.createdAt = new Date().toISOString(); project.updatedAt = project.createdAt;
    state.project = project; state.session = null; state.generation = null; state.previewPlans = []; state.candidateIds = []; clearReview(); state.history = []; state.future = [];
    await persistProject(true); renderAll(); setMode(project.network.lines.length ? "refine" : "create");
    $("#projectStart").classList.add("is-hidden"); $(".app-shell").inert = false;
    $(".project-menu").open = false;
    setNotice("Project imported and saved locally.");
  } catch (error) { if ($("#projectStart").classList.contains("is-hidden")) showError(error); else showStartError(error); }
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
