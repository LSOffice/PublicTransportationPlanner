package org.lsoffice

import kotlin.math.*

/*
 Metro system builder - algorithmic implementation.

 Core principles (enforced in heuristics):
  - Demand first, infrastructure second
  - Accessibility over proximity
  - Network effects and intersections
  - Hierarchy of modes (metro = backbone)
  - Land-use feedback loops

 This implementation is a heuristic, self-contained engine intended to
 convert a set of spatial zones and simple inputs into a candidate
 metro network. It favors clarity and testability over production
 optimization solver complexity.
*/

data class Zone(
    val id: String,
    val lon: Double,
    val lat: Double,
    val population: Double,
    val jobs: Double,
    val socioeconomicWeight: Double = 1.0,
    val activity: Double = 1.0,
    val growthForecast: Double = 1.0,
    val zoningAllowsGrowth: Boolean = true,
)

data class GridPoint(
    val lon: Double,
    val lat: Double,
    val value: Double, // population density
)

data class Station(
    val id: String,
    val lon: Double,
    val lat: Double,
    val catchmentPopulation: Double = 0.0,
)

data class Hub(
    val zone: Zone,
    val score: Double,
)

data class Corridor(
    val from: Hub,
    val to: Hub,
    val lengthMeters: Double,
    val demand: Double,
    val estimatedCost: Double,
    val score: Double,
)

data class Line(
    val id: String,
    val stations: List<Station>,
    val lengthMeters: Double,
    val cost: Double,
)

data class BuilderParams(
    val capitalBudget: Double,
    val operatingBudgetPerYear: Double,
    val targetCoverageFraction: Double = 0.6,
    val maxAcceptableTravelMins: Double = 60.0,
    val timeHorizonYears: Int = 30,
    val constructionCostPerKm: Double = 100_000_000.0, // 100M per km baseline
    val costPerStation: Double = 50_000_000.0,
)

class MetroBuilder(
    val params: BuilderParams,
) {
    // Step 1: Build demand graph (OD matrix) using a gravity model
    fun buildODMatrix(
        zones: List<Zone>,
        distanceExponent: Double = 1.5,
    ): Array<DoubleArray> {
        val n = zones.size
        val od = Array(n) { DoubleArray(n) { 0.0 } }
        // Normalizing constants can be learned; use simple gravity form
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j) continue
                val zij = haversineMeters(zones[i].lon, zones[i].lat, zones[j].lon, zones[j].lat)
                val distanceKm = max(0.001, zij / 1000.0)
                val trips =
                    (zones[i].population * zones[j].jobs * zones[i].socioeconomicWeight * zones[j].socioeconomicWeight) /
                        (distanceKm.pow(distanceExponent) * (1.0 + zones[j].activity * 0.1))
                od[i][j] = trips * zones[i].growthForecast // simple growth scaling
            }
        }
        return od
    }

    // Step 2: Identify high-value hubs using inbound trips and jobs/pop
    fun computeHubScores(
        zones: List<Zone>,
        od: Array<DoubleArray>,
        alpha: Double = 0.5,
        beta: Double = 0.3,
        gamma: Double = 0.15,
        delta: Double = 0.05,
    ): List<Hub> {
        val n = zones.size
        val inbound = DoubleArray(n) { 0.0 }
        for (i in 0 until n) for (j in 0 until n) inbound[j] += od[i][j]

        val hubs =
            zones.mapIndexed { idx, z ->
                val score = alpha * z.population + beta * z.jobs + gamma * inbound[idx] + delta * z.activity
                Hub(z, score)
            }
        return hubs.sortedByDescending { it.score }
    }

    // Step 3: Candidate station locations - catchment analysis
    fun generateCandidateStations(
        zones: List<Zone>,
        hubs: List<Hub>,
        catchmentMeters: Double = 800.0,
        minCatchment: Double = 500.0,
    ): List<Station> {
        val stations = mutableListOf<Station>()

        for (h in hubs) {
            // collect zones within catchment
            val nearby = mutableListOf<Zone>()
            var catchPop = 0.0
            for (z in zones) {
                val d = haversineMeters(h.zone.lon, h.zone.lat, z.lon, z.lat)
                if (d <= catchmentMeters) {
                    nearby.add(z)
                    catchPop += z.population + z.jobs
                }
            }

            if (catchPop >= minCatchment && nearby.isNotEmpty()) {
                // compute weighted centroid of nearby zones so station can sit between grid cells
                var sumW = 0.0
                var sumLon = 0.0
                var sumLat = 0.0
                for (z in nearby) {
                    val w = (z.population + z.jobs).coerceAtLeast(1.0)
                    sumW += w
                    sumLon += z.lon * w
                    sumLat += z.lat * w
                }
                var centroidLon = sumLon / sumW
                var centroidLat = sumLat / sumW

                // Randomise location within a small radius so a station can be placed between grid points.
                // TODO: Replace this random jitter with a deterministic medoid/optimization placement.
                val jitterRadiusMeters = 500.0 // up to ~500m jitter (tweakable)
                val r = kotlin.random.Random.nextDouble(0.0, jitterRadiusMeters)
                val bearing = kotlin.random.Random.nextDouble(0.0, 2.0 * PI)
                val dx = r * kotlin.math.cos(bearing)
                val dy = r * kotlin.math.sin(bearing)
                val latOffset = metersToLatDegrees(dy)
                val lonOffset = metersToLonDegrees(dx, centroidLat)
                val finalLat = centroidLat + latOffset
                val finalLon = centroidLon + lonOffset

                stations.add(Station("st_${h.zone.id}", finalLon, finalLat, catchPop))
            }
        }

        // remove close duplicates (keep highest catchment)
        val filtered = mutableListOf<Station>()
        for (s in stations) {
            val tooClose = filtered.any { existing -> haversineMeters(existing.lon, existing.lat, s.lon, s.lat) < 400 }
            if (!tooClose) filtered.add(s)
        }
        return filtered
    }

    // Step 4: Generate candidate corridors connecting hubs where trips exceed threshold
    fun generateCandidateCorridors(
        hubs: List<Hub>,
        od: Array<DoubleArray>,
        zones: List<Zone>,
        tripThresholdFactor: Double = 0.0001,
    ): List<Corridor> {
        val corridors = mutableListOf<Corridor>()
        val idxMap = zones.mapIndexed { index, z -> z.id to index }.toMap()
        for (i in hubs.indices) {
            for (j in i + 1 until hubs.size) {
                val hi = hubs[i]
                val hj = hubs[j]
                val iIdx = idxMap[hi.zone.id] ?: continue
                val jIdx = idxMap[hj.zone.id] ?: continue
                val demand = od[iIdx][jIdx] + od[jIdx][iIdx]
                val threshold = tripThresholdFactor * (hi.score + hj.score + 1.0)
                if (demand < threshold) continue
                val length = haversineMeters(hi.zone.lon, hi.zone.lat, hj.zone.lon, hj.zone.lat)
                val cost = (length / 1000.0) * params.constructionCostPerKm + params.costPerStation * 2
                val score = if (cost <= 0) 0.0 else (demand / cost)
                corridors.add(Corridor(hi, hj, length, demand, cost, score))
            }
        }
        return corridors.sortedByDescending { it.score }
    }

    // Step 5: Network optimization (greedy heuristic under budget)
    fun optimizeNetwork(
        candidateCorridors: List<Corridor>,
        maxBudget: Double = params.capitalBudget,
    ): List<Line> {
        val selected = mutableListOf<Corridor>()
        var spent = 0.0
        for (c in candidateCorridors) {
            if (spent + c.estimatedCost > maxBudget) continue
            // enhance network connectivity factor: prefer corridors that link to selected network
            val linksToNetwork =
                selected.any { s ->
                    s.from.zone.id == c.from.zone.id || s.to.zone.id == c.from.zone.id ||
                        s.from.zone.id == c.to.zone.id ||
                        s.to.zone.id == c.to.zone.id
                }
            val effectiveScore = c.score * (if (linksToNetwork) 1.5 else 1.0)
            // simple rule: pick if score above small threshold
            if (effectiveScore > 0.0) {
                selected.add(c)
                spent += c.estimatedCost
            }
        }

        // convert corridors to lines (simple two-station lines or merged chains)
        val lines = mutableListOf<Line>()
        var idx = 1
        for (c in selected) {
            val stA = Station("st_${c.from.zone.id}", c.from.zone.lon, c.from.zone.lat)
            val stB = Station("st_${c.to.zone.id}", c.to.zone.lon, c.to.zone.lat)
            val line = Line("L$idx", listOf(stA, stB), c.lengthMeters, c.estimatedCost)
            lines.add(line)
            idx += 1
        }
        return lines
    }

    // Helpers: convert small meter offsets to degree offsets (approximate, good for <~10km)
    private fun metersToLatDegrees(meters: Double): Double = meters / 111320.0

    private fun metersToLonDegrees(
        meters: Double,
        atLat: Double,
    ): Double {
        val latRad = atLat * PI / 180.0
        val metersPerDeg = 111320.0 * kotlin.math.cos(latRad)
        if (metersPerDeg == 0.0) return 0.0
        return meters / metersPerDeg
    }

    // Step 6: Phasing strategy - rank lines by ridership per dollar
    fun phaseLines(
        lines: List<Line>,
        od: Array<DoubleArray>,
        zones: List<Zone>,
    ): List<Pair<Line, Int>> {
        // naive: compute estimated ridership = sum of trips between end stations' zones
        val idxMap = zones.mapIndexed { index, z -> z.id to index }.toMap()
        val result = mutableListOf<Pair<Line, Int>>()
        val scored =
            lines
                .map { line ->
                    val aId =
                        line.stations
                            .first()
                            .id
                            .removePrefix("st_")
                    val bId =
                        line.stations
                            .last()
                            .id
                            .removePrefix("st_")
                    val ai = idxMap[aId] ?: -1
                    val bi = idxMap[bId] ?: -1
                    val ridership = if (ai >= 0 && bi >= 0) od[ai][bi] + od[bi][ai] else 0.0
                    val metric = if (line.cost <= 0) 0.0 else ridership / line.cost
                    Triple(line, metric, ridership)
                }.sortedByDescending { it.second }

        var phase1budget = params.capitalBudget * 0.5
        var spent = 0.0
        for ((i, t) in scored.withIndex()) {
            val phase =
                if (spent + t.first.cost <= phase1budget) {
                    spent += t.first.cost
                    1
                } else {
                    2
                }
            result.add(Pair(t.first, phase))
        }
        return result
    }

    // Step 7: Land-use feedback loop (adjust demand where densification allowed)
    fun applyLandUseFeedback(
        zones: List<Zone>,
        stations: List<Station>,
        upliftFactor: Double = 1.3,
    ): List<Zone> {
        val updated =
            zones.map { z ->
                var f = z.growthForecast
                // if zone gets a station and zoning allows growth, increase forecast
                val nearStation = stations.any { s -> haversineMeters(s.lon, s.lat, z.lon, z.lat) <= 800 }
                if (nearStation && z.zoningAllowsGrowth) f *= upliftFactor
                z.copy(growthForecast = f)
            }
        return updated
    }

    // Step 8: Stress testing - produce a few scenarios and evaluate stability
    fun stressTest(
        lines: List<Line>,
        zones: List<Zone>,
        od: Array<DoubleArray>,
    ): Map<String, Boolean> {
        val results = mutableMapOf<String, Boolean>()
        // scenario A: remote work reduces trips by 30%
        val odA = od.map { row -> row.map { it * 0.7 }.toDoubleArray() }.toTypedArray()
        results["remote_work_30pct"] = evaluateNetwork(lines, zones, odA)

        // scenario B: growth concentrated in outer suburbs (double outer zone population)
        val odBZones =
            zones.map { z ->
                if (haversineMeters(z.lon, z.lat, -0.1278, 51.5074) >
                    30_000
                ) {
                    z.copy(population = z.population * 2.0)
                } else {
                    z
                }
            }
        val odB = buildODMatrix(odBZones)
        results["outer_growth"] = evaluateNetwork(lines, odBZones, odB)

        // scenario C: one line fails - check disconnected major trips (naive)
        results["single_line_failure"] = true // placeholder; real sim required

        return results
    }

    private fun evaluateNetwork(
        lines: List<Line>,
        zones: List<Zone>,
        od: Array<DoubleArray>,
    ): Boolean {
        // Naive evaluation: ensure total served trips by lines is > some fraction
        val idxMap = zones.mapIndexed { index, z -> z.id to index }.toMap()
        var served = 0.0
        var total = 0.0
        for (i in od.indices) {
            for (j in od.indices) {
                total += od[i][j]
            }
        }
        for (line in lines) {
            if (line.stations.size < 2) continue
            val aId =
                line.stations
                    .first()
                    .id
                    .removePrefix("st_")
            val bId =
                line.stations
                    .last()
                    .id
                    .removePrefix("st_")
            val ai = idxMap[aId] ?: -1
            val bi = idxMap[bId] ?: -1
            if (ai >= 0 && bi >= 0) served += od[ai][bi] + od[bi][ai]
        }
        // require that selected lines serve at least 20% of total demand in a reasonable design
        return if (total <= 0.0) true else (served / total) >= 0.20
    }

// Example usage (comment):
// val builder = MetroBuilder(BuilderParams(1_000_000_000.0, 50_000_000.0))
// val od = builder.buildODMatrix(zones)
// val hubs = builder.computeHubScores(zones, od)
// val stations = builder.generateCandidateStations(zones, hubs.take(50))
    // val corridors = builder.generateCandidateCorridors(hubs.take(50), od, zones)
    // val lines = builder.optimizeNetwork(corridors)

    // New: Build metro lines from grid points using k-medoids + PCA ordering
    fun buildMetroLinesFromGrid(gridPoints: List<GridPoint>): List<Line> {
        // Step 1: Suppress nearby points — enforce minimum station spacing
        val minStationSpacing = 800.0 // meters
        val filtered = mutableListOf<GridPoint>()
        val sorted = gridPoints.sortedByDescending { it.value }
        for (point in sorted) {
            val tooClose = filtered.any { haversineMeters(it.lon, it.lat, point.lon, point.lat) < minStationSpacing }
            if (!tooClose) filtered.add(point)
        }

        if (filtered.isEmpty()) return emptyList()

        // Step 2: Compute city diameter from bounding box
        val minLat = filtered.minOf { it.lat }
        val maxLat = filtered.maxOf { it.lat }
        val minLon = filtered.minOf { it.lon }
        val maxLon = filtered.maxOf { it.lon }
        val cityDiameterMeters = haversineMeters(minLon, minLat, maxLon, maxLat)
        val cityDiameterKm = cityDiameterMeters / 1000.0
        val maxLinesGeographic = max(1, floor(cityDiameterKm / 6.0).toInt())

        // Step 3: Build distance matrix with demand weighting
        val n = filtered.size
        val dist = Array(n) { DoubleArray(n) { 0.0 } }
        val eps = 1e-6
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                val d = haversineMeters(filtered[i].lon, filtered[i].lat, filtered[j].lon, filtered[j].lat)
                val denom = sqrt((filtered[i].value.coerceAtLeast(1.0)) * (filtered[j].value.coerceAtLeast(1.0)) + eps)
                val wd = d / denom
                dist[i][j] = wd
                dist[j][i] = wd
            }
        }

        // Step 4: Cap number of lines
        val targetStationsPerLine = if (n > 40) 6 else 5
        val numLinesDemand = max(1, round(n.toDouble() / targetStationsPerLine).toInt())
        val numLines = min(maxLinesGeographic, numLinesDemand)

        // Step 5: K-medoids clustering
        val (medoids, clusters) = kMedoids(filtered, dist, numLines)

        // Step 6: For each cluster, order along PCA axis and create Line
        val lines = mutableListOf<Line>()
        var lineId = 1
        for (cluster in clusters) {
            if (cluster.isEmpty()) continue
            val ordered = orderAlongPrincipalAxis(filtered, cluster)
            // TODO: fix randomisation - current jitter is a simple uniform radial offset within ~500m
            val stations =
                ordered.mapIndexed { idx, i ->
                    // jitter within ~500m so stations don't all sit exactly on grid centroids
                    val jitterRadiusMeters = 500.0
                    val r = kotlin.random.Random.nextDouble(0.0, jitterRadiusMeters)
                    val bearing = kotlin.random.Random.nextDouble(0.0, 2.0 * PI)
                    val dx = r * kotlin.math.cos(bearing)
                    val dy = r * kotlin.math.sin(bearing)
                    val latOffset = metersToLatDegrees(dy)
                    val lonOffset = metersToLonDegrees(dx, filtered[i].lat)
                    val finalLat = filtered[i].lat + latOffset
                    val finalLon = filtered[i].lon + lonOffset
                    Station("st_${lineId}_$idx", finalLon, finalLat, filtered[i].value)
                }
            val length =
                if (stations.size >= 2) {
                    stations.zipWithNext().sumOf { (a, b) -> haversineMeters(a.lon, a.lat, b.lon, b.lat) }
                } else {
                    0.0
                }
            val cost = length / 1000.0 * params.constructionCostPerKm + stations.size * params.costPerStation
            lines.add(Line("L$lineId", stations, length, cost))
            lineId++
        }
        return lines
    }

    private fun kMedoids(
        points: List<GridPoint>,
        dist: Array<DoubleArray>,
        k: Int,
    ): Pair<List<Int>, List<List<Int>>> {
        val n = points.size
        if (k <= 0) return Pair(emptyList(), emptyList())

        // Initialize medoids: highest value, separated
        val sortedIdx = (0 until n).sortedByDescending { points[it].value }
        val medoids = mutableListOf<Int>()
        val minSeparation = 800.0
        for (idx in sortedIdx) {
            if (medoids.size >= k) break
            val ok = medoids.all { haversineMeters(points[it].lon, points[it].lat, points[idx].lon, points[idx].lat) >= minSeparation }
            if (ok) medoids.add(idx)
        }
        var p = 0
        while (medoids.size < k && p < sortedIdx.size) {
            if (!medoids.contains(sortedIdx[p])) medoids.add(sortedIdx[p])
            p++
        }

        var changed = true
        val assignments = IntArray(n) { 0 }
        var iter = 0
        val maxIter = 100
        while (changed && iter < maxIter) {
            iter++
            // Assign each point to nearest medoid index (medoid index, not medoid id)
            for (i in 0 until n) {
                var bestMi = 0
                var bestD = Double.MAX_VALUE
                for (mi in medoids.indices) {
                    val m = medoids[mi]
                    if (dist[i][m] < bestD) {
                        bestD = dist[i][m]
                        bestMi = mi
                    }
                }
                assignments[i] = bestMi
            }

            changed = false
            // Update medoids for each cluster
            for (mi in medoids.indices) {
                val members: List<Int> = assignments.mapIndexed { idx, a -> if (a == mi) idx else null }.filterNotNull()
                if (members.isEmpty()) continue
                var bestMed = medoids[mi]
                var bestCost = Double.MAX_VALUE
                for (cand in members) {
                    val cost = members.sumOf { other -> dist[cand][other] }
                    if (cost < bestCost) {
                        bestCost = cost
                        bestMed = cand
                    }
                }
                if (bestMed != medoids[mi]) {
                    medoids[mi] = bestMed
                    changed = true
                }
            }
        }

        // Build clusters as lists of point indices
        val clusters = List(medoids.size) { mutableListOf<Int>() }
        for (i in 0 until n) {
            val mi = assignments[i]
            if (mi in clusters.indices) clusters[mi].add(i)
        }

        return Pair(medoids.toList(), clusters.map { it.toList() })
    }

    private fun orderAlongPrincipalAxis(
        points: List<GridPoint>,
        indices: List<Int>,
    ): List<Int> {
        // Fallback: sort by latitude; could be replaced with PCA in future
        return indices.sortedBy { points[it].lat }
    }

    // Natural Metro Network Formation — Corridor-First Implementation
    fun buildNaturalNetworkFromGrid(
        gridPoints: List<GridPoint>,
        walkRadiusMeters: Double = 800.0,
        minStationValue: Double = 1.0,
        maxTrunkLines: Int = 4,
        minCorridorLengthMeters: Double = 5_000.0,
        minStationsPerLine: Int = 5,
    ): List<Line> {
        if (gridPoints.isEmpty()) return emptyList()

        // STEP 1 — Identify places, not points (collapse grid artifacts)
        val sortedPoints = gridPoints.sortedByDescending { it.value }
        val places = mutableListOf<GridPoint>()
        val clusterRadius = 1000.0
        val used = BooleanArray(gridPoints.size) { false }
        val pointIndices = gridPoints.indices.sortedByDescending { gridPoints[it].value }

        for (idx in pointIndices) {
            if (used[idx]) continue
            val p = gridPoints[idx]
            if (p.value < minStationValue) break

            // Sum up values in radius to create a "Place"
            var sumVal = 0.0
            for (j in gridPoints.indices) {
                if (haversineMeters(p.lon, p.lat, gridPoints[j].lon, gridPoints[j].lat) <= clusterRadius) {
                    sumVal += gridPoints[j].value
                    used[j] = true
                }
            }
            places.add(GridPoint(p.lon, p.lat, sumVal))
        }

        if (places.size < 2) return emptyList()

        // Demand Centroid (City Center)
        var totalVal = 0.0
        var weightedLon = 0.0
        var weightedLat = 0.0
        for (p in places) {
            weightedLon += p.lon * p.value
            weightedLat += p.lat * p.value
            totalVal += p.value
        }
        val centerLon = if (totalVal > 0) weightedLon / totalVal else places[0].lon
        val centerLat = if (totalVal > 0) weightedLat / totalVal else places[0].lat

        fun getMinSpacing(distFromCenterMeters: Double): Double =
            when {
                distFromCenterMeters < 3000 -> 700.0

                // CBD core
                distFromCenterMeters < 10000 -> 1200.0

                // Inner suburbs
                else -> 2500.0 // Outer corridors
            }

        // STEP 2 — Build a place-to-place demand graph
        val n = places.size
        val adj = Array(n) { DoubleArray(n) { 0.0 } }
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                val distKm = haversineMeters(places[i].lon, places[i].lat, places[j].lon, places[j].lat) / 1000.0
                // Gravity model: demand = (m1 * m2) / d^1.5
                val demand = (places[i].value * places[j].value) / (distKm.pow(1.5).coerceAtLeast(0.1))
                adj[i][j] = demand
                adj[j][i] = demand
            }
        }

        // STEP 3 — Find long demand chains (discover corridors)
        val visitedEdges = mutableSetOf<Pair<Int, Int>>()
        val candidateChains = mutableListOf<List<Int>>()

        // Start from high-demand node pairs and grow chains
        val allEdges = mutableListOf<Triple<Int, Int, Double>>()
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                allEdges.add(Triple(i, j, adj[i][j]))
            }
        }
        allEdges.sortByDescending { it.third }

        for (edge in allEdges.take(100)) { // look at top 100 demand edges
            if (visitedEdges.contains(edge.first to edge.second)) continue

            val chain = mutableListOf(edge.first, edge.second)
            visitedEdges.add(edge.first to edge.second)
            visitedEdges.add(edge.second to edge.first)

            // Grow forwards
            var current = edge.second
            var prev = edge.first
            while (true) {
                var bestNext = -1
                var bestDemand = 0.0
                for (next in 0 until n) {
                    if (chain.contains(next)) continue
                    val demand = adj[current][next]
                    // Geometry check: avoid sharp turns (> 60 degrees)
                    val d1x = places[current].lon - places[prev].lon
                    val d1y = places[current].lat - places[prev].lat
                    val d2x = places[next].lon - places[current].lon
                    val d2y = places[next].lat - places[current].lat
                    val dot = d1x * d2x + d1y * d2y
                    val mag1 = sqrt(d1x * d1x + d1y * d1y)
                    val mag2 = sqrt(d2x * d2x + d2y * d2y)
                    val cosTheta = if (mag1 * mag2 > 0) dot / (mag1 * mag2) else 1.0

                    if (cosTheta > 0.5 && demand > bestDemand) {
                        bestDemand = demand
                        bestNext = next
                    }
                }
                if (bestNext == -1 || bestDemand < edge.third * 0.1) break
                chain.add(bestNext)
                visitedEdges.add(current to bestNext)
                visitedEdges.add(bestNext to current)
                prev = current
                current = bestNext
            }

            // Grow backwards
            current = edge.first
            prev = edge.second
            while (true) {
                var bestNext = -1
                var bestDemand = 0.0
                for (next in 0 until n) {
                    if (chain.contains(next)) continue
                    val demand = adj[current][next]
                    val d1x = places[current].lon - places[prev].lon
                    val d1y = places[current].lat - places[prev].lat
                    val d2x = places[next].lon - places[current].lon
                    val d2y = places[next].lat - places[current].lat
                    val dot = d1x * d2x + d1y * d2y
                    val mag1 = sqrt(d1x * d1x + d1y * d1y)
                    val mag2 = sqrt(d2x * d2x + d2y * d2y)
                    val cosTheta = if (mag1 * mag2 > 0) dot / (mag1 * mag2) else 1.0

                    if (cosTheta > 0.5 && demand > bestDemand) {
                        bestDemand = demand
                        bestNext = next
                    }
                }
                if (bestNext == -1 || bestDemand < edge.third * 0.1) break
                chain.add(0, bestNext)
                visitedEdges.add(current to bestNext)
                visitedEdges.add(bestNext to current)
                prev = current
                current = bestNext
            }
            if (chain.size >= 2) candidateChains.add(chain)
        }

        // STEP 4b — Filter chains: reject short or redundant chains (reintroduced)
        val validChains =
            candidateChains
                .filter { chain ->
                    if (chain.size < minStationsPerLine) return@filter false
                    val dist =
                        chain.zipWithNext().sumOf { (a, b) ->
                            haversineMeters(places[a].lon, places[a].lat, places[b].lon, places[b].lat)
                        }
                    dist >= minCorridorLengthMeters
                }.sortedByDescending { it.size }

        // STEP 8 — Cap the number of lines and enforce radial hierarchy
        val finalCorridors = mutableListOf<List<Int>>()
        var radialCount = 0
        val maxRadials = (maxTrunkLines * 0.6).toInt().coerceAtLeast(1)
        val coreRadius = 2500.0

        for (chain in validChains) {
            if (finalCorridors.size >= maxTrunkLines) break

            // Redundancy check
            val alreadyCovered = chain.count { idx -> finalCorridors.any { it.contains(idx) } }
            if (alreadyCovered >= chain.size * 0.5) continue

            // Radial check: does it pass near the center?
            val isRadial = chain.any { haversineMeters(places[it].lon, places[it].lat, centerLon, centerLat) < coreRadius }

            if (isRadial && radialCount >= maxRadials) {
                // Skirt the core: allow it only if it's very high value relative to others?
                // For now, let's just skip to favor variety.
                continue
            }
            if (isRadial) radialCount++
            finalCorridors.add(chain)
        }

        // STEP 5, 6, 7 — Place stations with Variable Spacing and Terminal Thinning
        val stationRegistry = mutableListOf<Station>()
        val transferDist = 400.0

        fun findOrCreateStation(
            lon: Double,
            lat: Double,
            value: Double,
            lineId: String,
            idx: Int,
        ): Station {
            val existing = stationRegistry.find { haversineMeters(it.lon, it.lat, lon, lat) <= transferDist }
            if (existing != null) return existing
            val st = Station("st_${lineId}_$idx", lon, lat, value)
            stationRegistry.add(st)
            return st
        }

        val builtLines = mutableListOf<Line>()
        for ((li, chain) in finalCorridors.withIndex()) {
            val lineId = "L${li + 1}"

            // Find the "seed" (highest value node in the chain)
            val seedIndexInChain = chain.indices.maxByOrNull { places[chain[it]].value } ?: (chain.size / 2)
            val seedNode = chain[seedIndexInChain]

            val finalizedNodes = mutableListOf<Int>()
            finalizedNodes.add(seedNode)

            var cumulativePop = places[seedNode].value

            // Grow Forward from seed
            var lastIdx = seedIndexInChain
            for (i in seedIndexInChain + 1 until chain.size) {
                val pIdx = chain[i]
                val p = places[pIdx]
                val distLast = haversineMeters(p.lon, p.lat, places[chain[lastIdx]].lon, places[chain[lastIdx]].lat)
                val distCenter = haversineMeters(p.lon, p.lat, centerLon, centerLat)
                val distSeed = haversineMeters(p.lon, p.lat, places[seedNode].lon, places[seedNode].lat)

                val minSpace = getMinSpacing(distCenter)
                // Terminal thinning: demand requirements grow as we move away from seed
                val penaltyFactor = (1.0 + (distSeed / 8000.0).pow(1.5))
                val requiredValue = minStationValue * penaltyFactor

                if (distLast >= minSpace && p.value >= requiredValue) {
                    finalizedNodes.add(pIdx)
                    lastIdx = i
                    cumulativePop += p.value
                } else if (distSeed > 5000 && p.value < cumulativePop * 0.08) {
                    // Terminal stop: line doesn't earn this station
                    break
                }
            }

            // Grow Backward from seed
            lastIdx = seedIndexInChain
            for (i in seedIndexInChain - 1 downTo 0) {
                val pIdx = chain[i]
                val p = places[pIdx]
                val distLast = haversineMeters(p.lon, p.lat, places[chain[lastIdx]].lon, places[chain[lastIdx]].lat)
                val distCenter = haversineMeters(p.lon, p.lat, centerLon, centerLat)
                val distSeed = haversineMeters(p.lon, p.lat, places[seedNode].lon, places[seedNode].lat)

                val minSpace = getMinSpacing(distCenter)
                val penaltyFactor = (1.0 + (distSeed / 8000.0).pow(1.5))
                val requiredValue = minStationValue * penaltyFactor

                if (distLast >= minSpace && p.value >= requiredValue) {
                    finalizedNodes.add(0, pIdx)
                    lastIdx = i
                    cumulativePop += p.value
                } else if (distSeed > 5000 && p.value < cumulativePop * 0.08) {
                    break
                }
            }

            if (finalizedNodes.size >= 2) { // Allow shorter lines if they are high value
                val lineStations =
                    finalizedNodes.mapIndexed { idx, pIdx ->
                        val p = places[pIdx]
                        findOrCreateStation(p.lon, p.lat, p.value, lineId, idx)
                    }
                val length = lineStations.zipWithNext().sumOf { (a, b) -> haversineMeters(a.lon, a.lat, b.lon, b.lat) }
                val cost = length / 1000.0 * params.constructionCostPerKm + lineStations.size * params.costPerStation
                builtLines.add(Line(lineId, lineStations, length, cost))
            }
        }

        return builtLines
    }
}
