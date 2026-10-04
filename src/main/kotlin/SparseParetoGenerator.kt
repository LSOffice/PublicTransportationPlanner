package org.lsoffice

import kotlinx.serialization.Serializable
import org.locationtech.jts.geom.Coordinate as JtsCoordinate
import org.locationtech.jts.geom.GeometryFactory
import org.locationtech.jts.triangulate.DelaunayTriangulationBuilder
import kotlin.math.*

internal fun turnPenaltyDegrees(angle: Double): Double = when {
    angle > 90.0 -> 0.0
    angle > 45.0 -> exp(-(angle - 45.0) / 30.0)
    else -> 1.0
}

internal fun residualAfterCommit(value: Double): Double = value * 0.70

@Serializable
data class GenerationSettings(
    val guidanceStrength: Double = 0.08,
    val radialSoftCap: Int = 3,
    val orbitalSoftCap: Int = 3,
    val distributorSoftCap: Int = 3,
    val demandMode: String = "OBSERVED_BLEND",
)

@Serializable
data class GenerationRequest(val studyArea: StudyArea, val settings: GenerationSettings = GenerationSettings())

@Serializable
data class CandidateCorridor(
    val id: String,
    val role: PlanningRole,
    val stationIds: List<String>,
    val lengthMeters: Double,
    val demandScore: Double,
    val provenance: String,
    val coordinates: List<Coordinate> = emptyList(),
)

@Serializable
data class PlanMetrics(
    val coverage: Double,
    val lengthMeters: Double,
    val costEstimate: Double,
    val transferOverhead: Double,
    val duplication: Double,
    val connectivity: Double,
    val guidancePenalty: Double,
)

@Serializable
data class GenerationPlan(
    val id: String,
    val profile: String,
    val candidateIds: List<String>,
    val network: PlannerNetwork,
    val metrics: PlanMetrics,
    val certified: Boolean,
    val elapsedMillis: Long,
    val optimalityGap: Double?,
)

@Serializable
data class GenerationResult(
    val candidates: List<CandidateCorridor>,
    val plans: List<GenerationPlan>,
    val selectedGridPoints: Int,
    val demandEvidence: String,
    val insufficientFrontierReason: String? = null,
)

class SparseParetoGenerator {
    internal fun profileScore(metrics: PlanMetrics, profile: String): Double = score(metrics, profile)
    internal fun scoreBundle(grid: List<GridPoint>, ids: List<String>, settings: GenerationSettings,
                             profile: String): Double {
        val places = cluster(grid)
        val paths = discover(places, triangulate(places, null), { _, _ -> }, { false })
        val selected = ids.map { it.removePrefix("C").toInt() - 1 }
        return score(metrics(selected, paths, places, settings), profile)
    }
    data class Diagnostics(val placeCount: Int, val edgeCount: Int, val usedCollinearFallback: Boolean)
    private data class Place(val id: Int, val lon: Double, val lat: Double, val value: Double, val x: Double, val y: Double)
    private data class Edge(val a: Int, val b: Int, val meters: Double, val demand: Double)
    private data class Path(val nodes: List<Int>, val role: PlanningRole, val meters: Double, val demand: Double)
    private data class State(val selected: List<Int>, val next: Int)
    private val palette = listOf("#0b66d4", "#c2415d", "#14836f", "#7b52ab", "#d97706", "#3746a5", "#8b5e34", "#1677a3")
    var diagnostics = Diagnostics(0, 0, false)
        private set

    fun inspectSparseGraph(grid: List<GridPoint>): Diagnostics {
        val places = cluster(grid)
        if (places.size < 2) return Diagnostics(places.size, 0, false)
        triangulate(places, null)
        return diagnostics
    }

    fun evaluate(grid: List<GridPoint>, ids: List<String>, settings: GenerationSettings,
                 observed: RegionDemandModel? = null): GenerationPlan {
        val places = cluster(grid)
        val paths = discover(places, triangulate(places, observed), { _, _ -> }, { false })
        val selected = ids.distinct().map { id ->
            val index = id.removePrefix("C").toIntOrNull()?.minus(1)
                ?: throw PlannerValidationException("Unknown corridor $id")
            if (index !in paths.indices) throw PlannerValidationException("Unknown corridor $id")
            index
        }
        if (selected.isEmpty()) throw PlannerValidationException("Choose at least one corridor")
        val network = network(selected, paths, places)
        return GenerationPlan("evaluation", "custom", selected.map { "C${it + 1}" },
            network, metrics(selected, paths, places, settings).copy(
                lengthMeters = network.lines.sumOf { line -> line.segments.sumOf { it.lengthMeters } },
                costEstimate = network.lines.sumOf { it.buildEstimate?.totalCost ?: 0.0 }),
            true, 0, 0.0)
    }

    fun generate(
        grid: List<GridPoint>,
        settings: GenerationSettings = GenerationSettings(),
        observed: RegionDemandModel? = null,
        onEvent: (String, String) -> Unit = { _, _ -> },
        cancelled: () -> Boolean = { false },
        refinementBudgetMillis: Long = 1_500,
        onPreview: (GenerationPlan) -> Unit = {},
    ): GenerationResult {
        if (settings.guidanceStrength !in 0.0..0.20 || settings.radialSoftCap !in 0..16 ||
            settings.orbitalSoftCap !in 0..16 || settings.distributorSoftCap !in 0..16 ||
            settings.demandMode !in setOf("OBSERVED_BLEND", "GRAVITY_ONLY")) {
            throw PlannerValidationException("Invalid generation settings")
        }
        val start = System.nanoTime()
        val places = cluster(grid)
        if (places.size < 3) throw PlannerValidationException("Fewer than three distinct places in the study area", "INSUFFICIENT_PLACES")
        val edges = triangulate(places, observed)
        val paths = discover(places, edges, onEvent, cancelled)
        val candidates = paths.mapIndexed { index, p ->
            CandidateCorridor("C${index + 1}", p.role, p.nodes.map { "P$it" }, p.meters, p.demand,
                if (observed == null) "gravity-v1" else "NUMBAT-regional-blend-v1",
                p.nodes.map { Coordinate(places[it].lon, places[it].lat) })
        }
        if (paths.isEmpty()) return GenerationResult(emptyList(), emptyList(), grid.size, evidence(observed), "No connected corridor with three distinct places")
        onEvent("BEAM_STEP", "${paths.size} candidate corridors")
        val profiles = listOf("balanced", "coverage", "budget", "low-transfer", "low-duplication")
        val plans = mutableListOf<GenerationPlan>()
        val distinct = mutableSetOf<String>()
        for (profile in profiles) {
            if (cancelled()) break
            val preview = beamSearch(profile, paths, places, settings, onEvent)
            val previewNetwork = network(preview.selected, paths, places)
            onPreview(GenerationPlan("preview-$profile", profile, preview.selected.map { "C${it + 1}" },
                previewNetwork, metrics(preview.selected, paths, places, settings).copy(
                    lengthMeters = previewNetwork.lines.sumOf { line -> line.segments.sumOf { it.lengthMeters } },
                    costEstimate = previewNetwork.lines.sumOf { it.buildEstimate?.totalCost ?: 0.0 }),
                false, (System.nanoTime() - start) / 1_000_000, null))
            val (state, certified, gap) = refine(profile, preview, paths, places, settings,
                System.nanoTime() + refinementBudgetMillis.coerceIn(0, 1_600) * 1_000_000L, onEvent, cancelled)
            val key = state.selected.sorted().joinToString(",")
            if (!distinct.add(key)) continue
            val network = network(state.selected, paths, places)
            val metrics = metrics(state.selected, paths, places, settings).copy(
                lengthMeters = network.lines.sumOf { line -> line.segments.sumOf { it.lengthMeters } },
                costEstimate = network.lines.sumOf { it.buildEstimate?.totalCost ?: 0.0 })
            val elapsed = (System.nanoTime() - start) / 1_000_000
            val plan = GenerationPlan("plan-${plans.size + 1}", profile, state.selected.map { "C${it + 1}" }, network,
                metrics, certified, elapsed, gap)
            plans += plan
            state.selected.forEach { onEvent("CORRIDOR_COMMITTED", "C${it + 1}") }
        }
        val frontier = plans.filter { a -> plans.none { b -> b !== a && dominates(b.metrics, a.metrics) } }
        val resultPlans = if (frontier.isEmpty()) plans else frontier
        val reason = if (resultPlans.size < 5) "Only ${resultPlans.size} distinct nondominated bundles emerged from ${paths.size} candidates" else null
        onEvent("SEARCH_REFINED", "${resultPlans.size} nondominated plans")
        return GenerationResult(candidates, resultPlans, grid.size, evidence(observed), reason)
    }

    private fun evidence(observed: RegionDemandModel?) = if (observed == null || observed.isEmpty()) "SIMULATED_GRAVITY" else "NUMBAT_OBSERVED_BLEND"

    private fun cluster(grid: List<GridPoint>): List<Place> {
        val lat0 = grid.map { it.lat }.average()
        val lon0 = grid.map { it.lon }.average()
        val cosLat = cos(Math.toRadians(lat0))
        val buckets = HashMap<Pair<Int, Int>, MutableList<Int>>()
        val places = mutableListOf<Place>()
        for (point in grid.sortedWith(compareByDescending<GridPoint> { it.value }.thenBy { it.lon }.thenBy { it.lat })) {
            val x = (point.lon - lon0) * cosLat * 111_320.0
            val y = (point.lat - lat0) * 110_540.0
            val bx = floor(x / 1_000).toInt(); val by = floor(y / 1_000).toInt()
            val nearby = (-1..1).flatMap { dx -> (-1..1).flatMap { dy -> buckets[bx + dx to by + dy].orEmpty() } }
                .firstOrNull { hypot(places[it].x - x, places[it].y - y) <= 1_000.0 }
            if (nearby != null) {
                val old = places[nearby]
                places[nearby] = old.copy(value = old.value + point.value.coerceAtLeast(0.0))
            } else {
                val id = places.size
                places += Place(id, point.lon, point.lat, point.value.coerceAtLeast(0.0), x, y)
                buckets.getOrPut(bx to by) { mutableListOf() }.add(id)
            }
        }
        return places
    }

    private fun triangulate(places: List<Place>, observed: RegionDemandModel?): List<Edge> {
        val coordinates = places.map { JtsCoordinate(it.x, it.y) }
        val indices = coordinates.mapIndexed { i, c -> (c.x to c.y) to i }.toMap()
        val collinear = places.drop(2).all { p ->
            val a = places[0]; val b = places[1]
            abs((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) < 1.0
        }
        val pairs = mutableSetOf<Pair<Int, Int>>()
        if (collinear) {
            val ax = places.maxOf { it.x } - places.minOf { it.x }
            val ay = places.maxOf { it.y } - places.minOf { it.y }
            val sorted = places.sortedBy { if (ax >= ay) it.x else it.y }
            sorted.zipWithNext().forEach { (a, b) -> pairs += min(a.id, b.id) to max(a.id, b.id) }
        } else {
            val builder = DelaunayTriangulationBuilder()
            builder.setSites(coordinates)
            val geometry = builder.getEdges(GeometryFactory())
            for (i in 0 until geometry.numGeometries) {
                val line = geometry.getGeometryN(i).coordinates
                val a = indices[line[0].x to line[0].y] ?: continue
                val b = indices[line[1].x to line[1].y] ?: continue
                if (a != b) pairs += min(a, b) to max(a, b)
            }
        }
        diagnostics = Diagnostics(places.size, pairs.size, collinear)
        val maxGravity = pairs.maxOfOrNull { (a, b) -> places[a].value * places[b].value /
            (hypot(places[a].x - places[b].x, places[a].y - places[b].y) / 1_000.0).coerceAtLeast(0.1).pow(1.5) } ?: 1.0
        return pairs.sortedWith(compareBy<Pair<Int, Int>> { it.first }.thenBy { it.second }).map { (a, b) ->
            val p = places[a]; val q = places[b]
            val meters = hypot(p.x - q.x, p.y - q.y).coerceAtLeast(1.0)
            val gravity = p.value * q.value / (meters / 1_000.0).coerceAtLeast(0.1).pow(1.5)
            val observedDemand = observed?.demandBetween(setOf(LondonRegionGrid.cellFor(p.lon, p.lat)),
                setOf(LondonRegionGrid.cellFor(q.lon, q.lat))) ?: 0.0
            Edge(a, b, meters, gravity + if (observedDemand > 0.0 && observed != null && observed.maxDemand > 0.0)
                observedDemand / observed.maxDemand * maxGravity else 0.0)
        }
    }

    private fun discover(places: List<Place>, edges: List<Edge>, onEvent: (String, String) -> Unit,
                         cancelled: () -> Boolean): List<Path> {
        val adjacency = Array(places.size) { mutableListOf<Edge>() }
        edges.forEach { adjacency[it.a] += it; adjacency[it.b] += it }
        val residual = edges.associate { (it.a to it.b) to it.demand }.toMutableMap()
        val paths = mutableListOf<Path>()
        val seen = mutableSetOf<String>()
        val centerX = places.sumOf { it.x * it.value } / places.sumOf { it.value }.coerceAtLeast(1.0)
        val centerY = places.sumOf { it.y * it.value } / places.sumOf { it.value }.coerceAtLeast(1.0)
        repeat(3) { round ->
            val seeds = edges.sortedByDescending { residual.getValue(it.a to it.b) / (it.meters / 1_000.0) }.take(24)
            for ((seedIndex, seed) in seeds.withIndex()) {
                if (cancelled()) return paths
                fun grow(initial: List<Int>, atStart: Boolean, steps: Int): List<Int> {
                    data class Growth(val nodes: List<Int>, val score: Double)
                    var beam = listOf(Growth(initial, 0.0))
                    repeat(steps) {
                        val expanded = beam.flatMap { growth ->
                            val nodes = growth.nodes
                            val end = if (atStart) nodes.first() else nodes.last()
                            val previous = if (atStart) nodes[1] else nodes[nodes.lastIndex - 1]
                            val options = adjacency[end].mapNotNull { edge ->
                                val next = if (edge.a == end) edge.b else edge.a
                                if (next in nodes) return@mapNotNull null
                                val u = places[previous]; val v = places[end]; val w = places[next]
                                val dot = (v.x - u.x) * (w.x - v.x) + (v.y - u.y) * (w.y - v.y)
                                val angle = Math.toDegrees(acos((dot / (hypot(v.x - u.x, v.y - u.y) *
                                    hypot(w.x - v.x, w.y - v.y)).coerceAtLeast(1.0)).coerceIn(-1.0, 1.0)))
                                val penalty = turnPenaltyDegrees(angle)
                                if (penalty == 0.0) return@mapNotNull null
                                val key = min(end, next) to max(end, next)
                                val increment = residual.getValue(key) / (edge.meters / 1_000.0) * penalty
                                if (increment <= 0.0) null else Growth(
                                    if (atStart) listOf(next) + nodes else nodes + next,
                                    growth.score + increment)
                            }
                            options + growth
                        }
                        beam = expanded.distinctBy { it.nodes }.sortedWith(
                            compareByDescending<Growth> { it.score }.thenBy { it.nodes.joinToString(",") }).take(12)
                    }
                    return beam.first().nodes
                }
                val steps = 2 + seedIndex % 4
                val chain = grow(grow(listOf(seed.a, seed.b), false, steps), true, steps)
                if (chain.size < 3) continue
                val unique = chain.sorted().joinToString(",")
                if (!seen.add(unique)) continue
                val chainEdges = chain.zipWithNext().mapNotNull { (a, b) ->
                    adjacency[a].firstOrNull { (it.a == a && it.b == b) || (it.a == b && it.b == a) }
                }
                val meters = chainEdges.sumOf { it.meters }
                val demand = chainEdges.sumOf { it.demand }
                if (meters < 2_000.0 || demand / (meters / 1_000.0) < 1.0) continue
                val coreDistances = chain.map { hypot(places[it].x - centerX, places[it].y - centerY) }
                val role = when {
                    coreDistances.min() < 3_000 && meters >= 6_000 -> PlanningRole.RADIAL
                    coreDistances.average() < 4_000 -> PlanningRole.CORE_DISTRIBUTOR
                    else -> PlanningRole.ORBITAL_BYPASS
                }
                if (paths.count { it.role == role } >= 16) continue
                paths += Path(chain.toList(), role, meters, demand)
                onEvent("SEED_DISCOVERED", "${paths.size}: ${chain.size} places")
                chainEdges.forEach { edge -> residual[edge.a to edge.b] = residualAfterCommit(residual.getValue(edge.a to edge.b)) }
                onEvent("RESIDUAL_HEATMAP_UPDATED", "round $round")
                if (paths.size == 48) return paths
            }
        }
        return paths
    }

    private fun beamSearch(profile: String, paths: List<Path>, places: List<Place>, settings: GenerationSettings,
                           onEvent: (String, String) -> Unit): State {
        var beam = listOf(State(emptyList(), 0))
        val totalValue = places.sumOf { it.value }.coerceAtLeast(1.0)
        for (i in paths.indices) {
            val expanded = beam.flatMap { listOf(it.copy(next = i + 1), State(it.selected + i, i + 1)) }
            beam = expanded.map { state -> state to score(metrics(state.selected, paths, places, settings, totalValue), profile) }
                .sortedWith(compareByDescending<Pair<State, Double>> { it.second }
                    .thenBy { it.first.selected.joinToString(",") })
                .take(256).map { it.first }
            onEvent("BEAM_STEP", "${i + 1}/${paths.size}")
        }
        return beam.firstOrNull { it.selected.isNotEmpty() } ?: State(listOf(0), paths.size)
    }

    private data class Refined(val state: State, val certified: Boolean, val gap: Double)

    private fun refine(profile: String, preview: State, paths: List<Path>, places: List<Place>,
                       settings: GenerationSettings, deadline: Long, onEvent: (String, String) -> Unit,
                       cancelled: () -> Boolean): Refined {
        val totalValue = places.sumOf { it.value }.coerceAtLeast(1.0)
        var best = preview
        var bestScore = score(metrics(preview.selected, paths, places, settings, totalValue), profile)
        val coverageSuffix = Array(paths.size + 1) { emptySet<Int>() }
        for (i in paths.indices.reversed()) coverageSuffix[i] = coverageSuffix[i + 1] + paths[i].nodes
        val weights = weights(profile)
        val rootBound = weights[0] * coverageSuffix[0].sumOf { places[it].value } / totalValue + 0.4
        var visited = 0
        var interrupted = false
        fun visit(index: Int, ids: List<Int>, covered: Set<Int>, lengthMeters: Double) {
            if (interrupted) return
            if (++visited % 256 == 0 || visited == 1) {
                if (System.nanoTime() >= deadline || cancelled()) { interrupted = true; return }
            }
            val upperCoverage = (covered + coverageSuffix[index]).sumOf { places[it].value } / totalValue
            val upper = weights[0] * upperCoverage + 0.4 - weights[1] * lengthMeters / 100_000.0
            if (upper <= bestScore + 1e-12) {
                if (visited % 256 == 0) onEvent("BRANCH_PRUNED", "upper bound below incumbent")
                return
            }
            if (index == paths.size) {
                if (ids.isNotEmpty()) {
                    val value = score(metrics(ids, paths, places, settings, totalValue), profile)
                    if (value > bestScore + 1e-12) { bestScore = value; best = State(ids, index) }
                }
                return
            }
            visit(index + 1, ids + index, covered + paths[index].nodes, lengthMeters + paths[index].meters)
            if (!interrupted) visit(index + 1, ids, covered, lengthMeters)
        }
        visit(0, emptyList(), emptySet(), 0.0)
        val certified = !interrupted
        val gap = if (certified) 0.0 else ((rootBound - bestScore) / abs(bestScore).coerceAtLeast(1.0)).coerceAtLeast(0.0)
        return Refined(best, certified, gap)
    }

    private fun metrics(ids: List<Int>, paths: List<Path>, places: List<Place>, settings: GenerationSettings,
                        totalValue: Double = places.sumOf { it.value }.coerceAtLeast(1.0)): PlanMetrics {
        val selected = ids.map { paths[it] }
        val covered = selected.flatMap { it.nodes }.toSet()
        val coverage = covered.sumOf { places[it].value } / totalValue
        val length = selected.sumOf { it.meters }
        val edgeCounts = selected.flatMap { it.nodes.zipWithNext().map { (a, b) -> min(a, b) to max(a, b) } }.groupingBy { it }.eachCount()
        val duplicate = if (edgeCounts.isEmpty()) 0.0 else edgeCounts.values.sumOf { (it - 1).coerceAtLeast(0) }.toDouble() / edgeCounts.values.sum()
        val transfer = if (selected.size <= 1) 0.0 else 1.0 - covered.count { node -> selected.count { node in it.nodes } > 1 }.toDouble() / covered.size.coerceAtLeast(1)
        val components = if (selected.isEmpty()) 0 else {
            val remaining = selected.indices.toMutableSet(); var count = 0
            while (remaining.isNotEmpty()) {
                count++; val queue = ArrayDeque<Int>(); queue.add(remaining.first()); remaining.remove(queue.first())
                while (queue.isNotEmpty()) { val current = queue.removeFirst(); val adjacent = remaining.filter { other ->
                    selected[current].nodes.any { it in selected[other].nodes } }
                    adjacent.forEach { remaining.remove(it); queue.add(it) }
                }
            }; count
        }
        val connectivity = if (selected.isEmpty()) 0.0 else 1.0 / components.coerceAtLeast(1)
        val caps = listOf(PlanningRole.RADIAL to settings.radialSoftCap, PlanningRole.ORBITAL_BYPASS to settings.orbitalSoftCap,
            PlanningRole.CORE_DISTRIBUTOR to settings.distributorSoftCap)
        val penalty = settings.guidanceStrength * caps.sumOf { (role, cap) ->
            (selected.count { it.role == role } - cap).coerceAtLeast(0).toDouble().pow(2)
        }
        return PlanMetrics(coverage, length, length / 1_000.0 * 140_000_000.0,
            transfer, duplicate, connectivity, penalty)
    }

    private fun score(m: PlanMetrics, profile: String): Double {
        val weights = weights(profile)
        return weights[0] * m.coverage + 0.4 * m.connectivity - weights[1] * m.lengthMeters / 100_000.0 -
            weights[2] * m.transferOverhead - weights[3] * m.duplication - m.guidancePenalty
    }

    private fun weights(profile: String): List<Double> = when (profile) {
            "coverage" -> listOf(2.8, 0.20, 0.25, 0.25)
            "budget" -> listOf(1.2, 1.25, 0.3, 0.3)
            "low-transfer" -> listOf(1.3, 0.35, 1.4, 0.25)
            "low-duplication" -> listOf(1.3, 0.35, 0.25, 1.4)
            else -> listOf(1.8, 0.5, 0.6, 0.6)
        }

    private fun dominates(a: PlanMetrics, b: PlanMetrics): Boolean {
        val good = a.coverage >= b.coverage && a.connectivity >= b.connectivity && a.lengthMeters <= b.lengthMeters &&
            a.costEstimate <= b.costEstimate &&
            a.transferOverhead <= b.transferOverhead && a.duplication <= b.duplication
        val strict = a.coverage > b.coverage || a.connectivity > b.connectivity || a.lengthMeters < b.lengthMeters ||
            a.costEstimate < b.costEstimate ||
            a.transferOverhead < b.transferOverhead || a.duplication < b.duplication
        return good && strict
    }

    private fun network(ids: List<Int>, paths: List<Path>, places: List<Place>): PlannerNetwork {
        val raw = ids.map { id ->
            val path = paths[id]
            Line("C${id + 1}", path.nodes.map { node ->
                val p = places[node]; Station("P$node", p.lon, p.lat, p.value)
            }, path.meters, 0.0, when (path.role) {
                PlanningRole.RADIAL, PlanningRole.CROSS_CITY_TRUNK -> LineType.RADIAL_TRUNK
                PlanningRole.ORBITAL_BYPASS -> LineType.ORBITAL
                PlanningRole.CORE_DISTRIBUTOR -> LineType.CORE_DISTRIBUTOR
            }, trainsPerHour = 12)
        }
        val builder = MetroBuilder(BuilderParams(1_000_000_000.0, 50_000_000.0), debug = false)
        val consolidated = builder.consolidateStationClusters(raw)
        val total = places.sumOf { it.value }.coerceAtLeast(1.0)
        val centerLon = places.sumOf { it.lon * it.value } / total
        val centerLat = places.sumOf { it.lat * it.value } / total
        val estimated = builder.estimateBuildTechnology(consolidated, centerLon, centerLat, 3_000.0, 6_000.0)
        val stations = estimated.flatMap { it.stations }.groupBy { it.id }.map { (id, group) ->
            val weight = group.sumOf { it.catchmentPopulation }.coerceAtLeast(1.0)
            PlannerStation(id, contextualStationName(id, group.first().lon, group.first().lat),
                group.sumOf { it.lon * it.catchmentPopulation } / weight,
                group.sumOf { it.lat * it.catchmentPopulation } / weight,
                group.maxOf { it.catchmentPopulation })
        }
        val byId = stations.associateBy { it.id }
        val lines = estimated.mapIndexed { index, line ->
            val stationIds = line.stations.map { it.id }
            val segments = stationIds.zipWithNext().mapIndexed { segmentIndex, (a, b) ->
                val from = byId.getValue(a); val to = byId.getValue(b)
                val technology = line.buildEstimate?.segments?.getOrNull(segmentIndex)?.technology
                PlannerSegment("${line.id}-S${segmentIndex + 1}", a, b,
                    if (technology == AlignmentTechnology.SURFACE_OR_ELEVATED) InfrastructureType.OVERGROUND else InfrastructureType.UNDERGROUND,
                    haversineMeters(from.lon, from.lat, to.lon, to.lat))
            }
            val first = byId.getValue(stationIds.first()).name
            val last = byId.getValue(stationIds.last()).name
            val name = if (first == last) "$first Line" else "$first–$last Line"
            PlannerLine(line.id, name, palette[index % palette.size], when (line.type) {
                LineType.RADIAL_TRUNK -> PlanningRole.RADIAL
                LineType.ORBITAL -> PlanningRole.ORBITAL_BYPASS
                LineType.CORE_DISTRIBUTOR -> PlanningRole.CORE_DISTRIBUTOR
                LineType.NOT_METRO -> PlanningRole.CROSS_CITY_TRUNK
            }, trainsPerHour = line.trainsPerHour.coerceAtLeast(12), stationIds = stationIds,
                segments = segments,
                buildEstimate = line.buildEstimate?.let { estimate -> PlannerBuildEstimate(estimate.totalCost,
                    estimate.deepBoreMeters, estimate.subsurfaceMeters, estimate.surfaceOrElevatedMeters,
                    estimate.recommendation) })
        }
        return PlannerNetwork(stations, lines)
    }

    private fun contextualStationName(id: String, lon: Double, lat: Double): String {
        val nearest = numbatStations.minByOrNull { haversineMeters(lon, lat, it.lon, it.lat) }
        return if (nearest != null && haversineMeters(lon, lat, nearest.lon, nearest.lat) <= 550.0)
            nearest.name else id.replace("P", "Place ").replace("INT_", "Interchange ")
    }

    private val numbatStations: List<NumbatStation> by lazy {
        SparseParetoGenerator::class.java.getResourceAsStream("/from-to-data/derived/numbat-stations-2024.csv")
            ?.use { NumbatDemandLoader.loadStations(it).values.toList() } ?: emptyList()
    }
}
