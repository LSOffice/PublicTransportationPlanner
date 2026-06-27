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

enum class LineType {
    RADIAL_TRUNK, // passes through core, long distance
    ORBITAL, // skirts core, intersects radials
    CORE_DISTRIBUTOR, // short, dense, stays in core
    NOT_METRO, // doesn't meet criteria (filtered out)
}

enum class AlignmentTechnology {
    DEEP_BORE_TUNNEL,
    SUBSURFACE_TUNNEL,
    SURFACE_OR_ELEVATED,
}

data class BuildSegmentEstimate(
    val fromStationId: String,
    val toStationId: String,
    val lengthMeters: Double,
    val technology: AlignmentTechnology,
    val civilCost: Double,
    val rationale: String,
)

data class LineBuildEstimate(
    val segments: List<BuildSegmentEstimate>,
    val stationCost: Double,
    val landInterfaceAllowance: Double,
    val contingency: Double,
    val totalCost: Double,
    val deepBoreMeters: Double,
    val subsurfaceMeters: Double,
    val surfaceOrElevatedMeters: Double,
    val recommendation: String,
)

data class Line(
    val id: String,
    val stations: List<Station>,
    val lengthMeters: Double,
    val cost: Double,
    val type: LineType = LineType.RADIAL_TRUNK,
    val isLoop: Boolean = false,
    val trainsPerHour: Int = 0,
    val buildEstimate: LineBuildEstimate? = null,
)

internal fun combineGravityAndObservedDemand(
    gravityDemand: Double,
    observedDemand: Double,
    maxGravityDemand: Double,
    maxObservedDemand: Double,
): Double {
    if (observedDemand <= 0.0 || maxGravityDemand <= 0.0 || maxObservedDemand <= 0.0) return gravityDemand
    return gravityDemand + observedDemand * (maxGravityDemand / maxObservedDemand)
}

// Hub-to-hub path analysis
data class HubPathAnalysis(
    val hubFromValue: Double,
    val hubToValue: Double,
    val directDistanceMeters: Double,
    val networkPathLengthMeters: Double,
    val detourRatio: Double, // actual / direct
    val inVehicleMinutes: Double,
    val numberOfInterchanges: Int,
    val transferPenaltyMinutes: Double, // 6 min per interchange
    val generalizedTimeMinutes: Double, // in-vehicle + transfers + detour
    val isDirect: Boolean, // on same line
    val pathDescription: String,
)

data class NetworkFitnessReport(
    val topHubPairs: List<HubPathAnalysis>,
    val violationsFound: List<String>,
    val orbitalUtility: Map<String, String>, // line ID -> utility assessment
    val overallMetricsOK: Boolean,
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
    val debug: Boolean = true,
) {
    private fun dbg(msg: String) {
        if (debug) println("[MetroBuilder] $msg")
    }

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
        if (indices.size <= 1) return indices
        // True PCA: find the dominant axis of the cluster and sort by projection onto it.
        // This ensures E-W clusters order left-to-right, NW-SE clusters order along their
        // diagonal, etc. — instead of the naive latitude sort that was always north-to-south.
        val meanLon = indices.sumOf { points[it].lon } / indices.size
        val meanLat = indices.sumOf { points[it].lat } / indices.size
        var cxx = 0.0
        var cxy = 0.0
        var cyy = 0.0
        for (i in indices) {
            val dx = points[i].lon - meanLon
            val dy = points[i].lat - meanLat
            cxx += dx * dx
            cxy += dx * dy
            cyy += dy * dy
        }
        // Analytical larger eigenvalue of the 2x2 covariance matrix [[cxx,cxy],[cxy,cyy]]
        val trace = cxx + cyy
        val disc = sqrt(max(0.0, (cxx - cyy).pow(2.0) + 4.0 * cxy * cxy)) / 2.0
        val lambda = trace / 2.0 + disc
        // Corresponding eigenvector (the principal axis direction)
        val ex = cxy
        val ey = lambda - cxx
        val mag = sqrt(ex * ex + ey * ey)
        val px = if (mag > 1e-10) ex / mag else 1.0
        val py = if (mag > 1e-10) ey / mag else 0.0
        // Sort each point by its scalar projection onto the principal axis
        return indices.sortedBy { i ->
            val dx = points[i].lon - meanLon
            val dy = points[i].lat - meanLat
            dx * px + dy * py
        }
    }

    // Helper: compute loopiness metric (Improvement 1)
    private fun computeLoopiness(
        finalizedNodes: List<Int>,
        places: List<GridPoint>,
    ): Double {
        if (finalizedNodes.size < 3) return 1.0 // not circular

        val first = finalizedNodes.first()
        val last = finalizedNodes.last()
        val endToEndDistance =
            haversineMeters(places[first].lon, places[first].lat, places[last].lon, places[last].lat)

        val totalLength =
            finalizedNodes.zipWithNext().sumOf { (a, b) ->
                haversineMeters(places[a].lon, places[a].lat, places[b].lon, places[b].lat)
            }

        if (totalLength <= 0) return 1.0
        return endToEndDistance / totalLength
    }

    // Helper: find weakest demand edge in chain for breaking (Improvement 2 Option A)
    private fun findWeakestEdgeInChain(
        chain: List<Int>,
        adj: Array<DoubleArray>,
    ): Int? {
        if (chain.size < 2) return null
        var weakestIdx = 0
        var weakestDemand = Double.MAX_VALUE
        for (i in 0 until chain.size) {
            val j = (i + 1) % chain.size
            val demand = adj[chain[i]][chain[j]]
            if (demand < weakestDemand) {
                weakestDemand = demand
                weakestIdx = i
            }
        }
        return weakestIdx
    }

    // Helper: count radial-like lines that pass through a region (for intersection test)
    private fun countRadialIntersections(
        line: List<Int>,
        allFinalizedLines: List<List<Int>>,
        places: List<GridPoint>,
        centerLon: Double,
        centerLat: Double,
        coreRadius: Double,
    ): Int {
        var intersections = 0
        for (otherLine in allFinalizedLines) {
            // Check if other line passes through core (radial-like)
            val otherInCore =
                otherLine.any { idx ->
                    haversineMeters(places[idx].lon, places[idx].lat, centerLon, centerLat) <= coreRadius
                }
            if (!otherInCore) continue

            // Check if our ring line shares nodes with this radial
            val shared = line.count { idx -> otherLine.contains(idx) }
            if (shared >= 1) intersections++
        }
        return intersections
    }

    // Helper: count high-value places on a line
    private fun countHighValuePlaces(
        line: List<Int>,
        places: List<GridPoint>,
        valueThreshold: Double,
    ): Int = line.count { idx -> places[idx].value >= valueThreshold }

    // Step 0: Classify corridor intent based on geometry of interaction with center
    private fun classifyCorridor(
        chain: List<Int>,
        places: List<GridPoint>,
        centerLon: Double,
        centerLat: Double,
        coreRadius: Double,
        ringRadius: Double,
    ): LineType {
        if (chain.isEmpty()) return LineType.NOT_METRO

        // Compute min and max distance of chain nodes from center
        val distances =
            chain.map { idx ->
                haversineMeters(places[idx].lon, places[idx].lat, centerLon, centerLat)
            }
        val minDistToCenter = distances.minOrNull() ?: Double.MAX_VALUE
        val maxDistToCenter = distances.maxOrNull() ?: 0.0

        // Count stations in core and ring
        val stationsInCore =
            chain.count { idx ->
                haversineMeters(places[idx].lon, places[idx].lat, centerLon, centerLat) <= coreRadius
            }
        val stationsInRing =
            chain.count { idx ->
                val d = haversineMeters(places[idx].lon, places[idx].lat, centerLon, centerLat)
                d > coreRadius && d <= ringRadius * 1.2
            }
        val fracInCore = stationsInCore.toDouble() / chain.size.coerceAtLeast(1)
        val fracInRing = stationsInRing.toDouble() / chain.size.coerceAtLeast(1)

        // Soft geometric classification (no length/demand constraints)
        return when {
            // Radial: enters core and extends far beyond ring
            minDistToCenter <= coreRadius && maxDistToCenter >= ringRadius -> {
                dbg("    Chain of ${chain.size}: RADIAL_TRUNK (enters core, exits far)")
                LineType.RADIAL_TRUNK
            }

            // Orbital: avoids core, stays in ring band
            minDistToCenter > coreRadius && maxDistToCenter <= ringRadius * 1.2 -> {
                dbg("    Chain of ${chain.size}: ORBITAL (skirts core, ring-bound)")
                LineType.ORBITAL
            }

            // Core distributor: mostly inside inner core
            fracInCore >= 0.6 -> {
                dbg("    Chain of ${chain.size}: CORE_DISTRIBUTOR (${(fracInCore * 100).toInt()}% in core)")
                LineType.CORE_DISTRIBUTOR
            }

            else -> {
                dbg("    Chain of ${chain.size}: NOT_METRO (no clear pattern)")
                LineType.NOT_METRO
            }
        }
    }

    // ========== POST-BUILD NETWORK FITNESS EVALUATION ==========
    // (Improvement 1–5: Hub-to-hub travel time analysis)

    // Build a lightweight station-to-station graph from lines (for shortest-path queries)
    private fun buildStationGraph(lines: List<Line>): Map<String, List<Pair<String, Double>>> {
        val graph = mutableMapOf<String, MutableList<Pair<String, Double>>>()

        // Initialize all stations
        for (line in lines) {
            for (st in line.stations) {
                graph.putIfAbsent(st.id, mutableListOf())
            }
        }

        // Add edges (each line segment + transfer edges)
        for (line in lines) {
            // Segment edges: consecutive stations on same line
            for (i in 0 until line.stations.size - 1) {
                val st1 = line.stations[i]
                val st2 = line.stations[i + 1]
                val distMeters = haversineMeters(st1.lon, st1.lat, st2.lon, st2.lat)
                // In-vehicle time: assume ~35 km/h average metro speed = 35km/3600s = 0.58 min/km
                val timeMinutes = (distMeters / 1000.0) / 35.0 * 60.0
                graph[st1.id]?.add(st2.id to timeMinutes)
                graph[st2.id]?.add(st1.id to timeMinutes)
            }
        }

        // Transfer edges: stations within 300m get a 3-min transfer
        val stations = lines.flatMap { it.stations }
        for (i in 0 until stations.size) {
            for (j in i + 1 until stations.size) {
                val dist =
                    haversineMeters(
                        stations[i].lon,
                        stations[i].lat,
                        stations[j].lon,
                        stations[j].lat,
                    )
                if (dist in 1.0..300.0 && stations[i].id != stations[j].id) {
                    val transferTime = 3.0 // minutes
                    graph[stations[i].id]?.add(stations[j].id to transferTime)
                    graph[stations[j].id]?.add(stations[i].id to transferTime)
                }
            }
        }

        return graph
    }

    // Dijkstra shortest path from source to all stations (returns time in minutes)
    private fun dijkstraShortestPath(
        graph: Map<String, List<Pair<String, Double>>>,
        source: String,
    ): Map<String, Double> {
        val distances = mutableMapOf<String, Double>()
        val unvisited = mutableSetOf<String>()

        for (node in graph.keys) {
            distances[node] = Double.MAX_VALUE
            unvisited.add(node)
        }
        distances[source] = 0.0

        while (unvisited.isNotEmpty()) {
            val current = unvisited.minByOrNull { distances[it] ?: Double.MAX_VALUE } ?: break
            unvisited.remove(current)

            for ((neighbor, edgeTime) in graph[current] ?: emptyList()) {
                if (neighbor !in unvisited) continue
                val alt = (distances[current] ?: Double.MAX_VALUE) + edgeTime
                if (alt < (distances[neighbor] ?: Double.MAX_VALUE)) {
                    distances[neighbor] = alt
                }
            }
        }

        return distances
    }

    // Compute hub-to-hub paths for the top N hubs by value
    fun evaluateNetworkFitness(
        lines: List<Line>,
        places: List<GridPoint>,
        topHubsCount: Int = 12,
        transferPenaltyMinutes: Double = 6.0,
    ): NetworkFitnessReport {
        if (lines.isEmpty() || places.isEmpty()) {
            return NetworkFitnessReport(emptyList(), emptyList(), emptyMap(), true)
        }

        // Select top hubs by value
        val topHubs = places.sortedByDescending { it.value }.take(topHubsCount)
        val stationGraph = buildStationGraph(lines)
        val metroSpeed = 35.0 / 60.0 // km/h to km/min

        // Build map of which hub is served by which stations
        val hubToStations = mutableMapOf<Int, List<Station>>()
        for ((hubIdx, hub) in topHubs.withIndex()) {
            val nearbyStations =
                lines.flatMap { it.stations }.filter { st ->
                    haversineMeters(st.lon, st.lat, hub.lon, hub.lat) <= 600.0 // within 600m
                }
            if (nearbyStations.isNotEmpty()) {
                hubToStations[hubIdx] = nearbyStations
            }
        }

        // Analyze top OD pairs
        val hubPairAnalyses = mutableListOf<HubPathAnalysis>()
        val violations = mutableListOf<String>()

        for (i in 0 until topHubs.size) {
            for (j in i + 1 until topHubs.size) {
                val hubA = topHubs[i]
                val hubB = topHubs[j]
                val stationsA = hubToStations[i]
                val stationsB = hubToStations[j]

                if (stationsA == null || stationsB == null) continue

                // Direct straight-line distance
                val directDistMeters = haversineMeters(hubA.lon, hubA.lat, hubB.lon, hubB.lat)

                // Find shortest path via network (multiple attempts from different starting stations)
                var bestNetworkTime = Double.MAX_VALUE
                var bestPathLength = 0.0
                var bestInterchanges = 0
                var isDirect = false

                for (stA in stationsA) {
                    val distances = dijkstraShortestPath(stationGraph, stA.id)

                    for (stB in stationsB) {
                        val pathTime = distances[stB.id] ?: continue
                        if (pathTime < bestNetworkTime) {
                            bestNetworkTime = pathTime

                            // Estimate path length from time (rough heuristic)
                            // Assume metro = 35km/h, transfer = 3 min, so time = dist_m/35/1000*60 + 3*interchanges
                            // For now, use a conservative estimate
                            bestPathLength =
                                if (stA.id == stB.id) 0.0 else directDistMeters * (1.0 + 0.2) // assume ~20% detour baseline

                            // Count interchanges (rough: transitions between different lines)
                            var transfers = 0
                            for (line in lines) {
                                val hasStA = line.stations.any { it.id == stA.id }
                                val hasStB = line.stations.any { it.id == stB.id }
                                if (hasStA && hasStB) {
                                    // Same line, no interchange needed
                                    isDirect = true
                                    transfers = 0
                                    break
                                }
                            }
                            if (!isDirect) {
                                transfers =
                                    if (bestNetworkTime > 10.0) 1 else 0 // rough heuristic
                            }
                            bestInterchanges = transfers
                        }
                    }
                }

                if (bestNetworkTime >= Double.MAX_VALUE) continue

                val detourRatio =
                    if (directDistMeters > 0) (bestPathLength) / directDistMeters else 1.0
                val transferPenalty = bestInterchanges * transferPenaltyMinutes
                val generalizedTime = bestNetworkTime + transferPenalty + (detourRatio - 1.0) * 5.0

                val pathDesc =
                    if (isDirect) {
                        "direct"
                    } else if (bestInterchanges == 0) {
                        "no-transfer"
                    } else {
                        "$bestInterchanges xfers"
                    }

                val analysis =
                    HubPathAnalysis(
                        hubFromValue = hubA.value,
                        hubToValue = hubB.value,
                        directDistanceMeters = directDistMeters,
                        networkPathLengthMeters = bestPathLength,
                        detourRatio = detourRatio,
                        inVehicleMinutes = bestNetworkTime,
                        numberOfInterchanges = bestInterchanges,
                        transferPenaltyMinutes = transferPenalty,
                        generalizedTimeMinutes = generalizedTime,
                        isDirect = isDirect,
                        pathDescription = pathDesc,
                    )

                hubPairAnalyses.add(analysis)

                // Check efficiency thresholds
                val maxNetworkTime = generalizedTime * 1.4
                val maxInterchanges = 2

                if (bestNetworkTime > maxNetworkTime) {
                    violations.add(
                        "⚠ Hub pair (${hubA.value.toInt()}, ${hubB.value.toInt()}): slow path " +
                            "(${bestNetworkTime.toInt()} min > ${maxNetworkTime.toInt()} min) — " +
                            "detour=${detourRatio.toInt()}x, $bestInterchanges xfers",
                    )
                }
                if (bestInterchanges > maxInterchanges) {
                    violations.add(
                        "⚠ Hub pair (${hubA.value.toInt()}, ${hubB.value.toInt()}): excessive transfers " +
                            "($bestInterchanges > $maxInterchanges)",
                    )
                }
            }
        }

        // Evaluate orbital utility (Improvement 5)
        val orbitalUtility = mutableMapOf<String, String>()
        for (line in lines) {
            if (line.type == LineType.ORBITAL || line.isLoop) {
                // Check if this orbital reduces interchange count or provides faster alternative
                val utility =
                    if (line.isLoop) "loop: assess fast-path utility" else "orbital: assess hub-interchange relief"
                orbitalUtility[line.id] = utility
            }
        }

        val metricsOK = violations.isEmpty()
        val diagnostics =
            violations.map { it } +
                orbitalUtility.map { (id, util) ->
                    "ℹ $id: $util"
                }

        dbg("NETWORK FITNESS — Hub-pair analysis:")
        for (pair in hubPairAnalyses.take(5)) {
            dbg(
                "  Hub-pair (${pair.hubFromValue.toInt()}, ${pair.hubToValue.toInt()}): " +
                    "${pair.generalizedTimeMinutes.toInt()} min gen-time, ${pair.numberOfInterchanges} xfers, " +
                    "detour=${pair.detourRatio.toInt()}x",
            )
        }
        dbg("Violations: ${violations.size}")
        violations.forEach { dbg("  $it") }

        return NetworkFitnessReport(
            topHubPairs = hubPairAnalyses,
            violationsFound = violations,
            orbitalUtility = orbitalUtility,
            overallMetricsOK = metricsOK,
        )
    }

    /**
     * Compute average user journey metrics across all station pairs in the network.
     * This uses Dijkstra's algorithm to calculate the shortest travel time between every possible
     * permutation of station A -> station B, including estimated transfer times.
     */
    fun computeAverageJourneyMetrics(lines: List<Line>): Map<String, Double> {
        if (lines.isEmpty()) return emptyMap()

        val graph = buildStationGraph(lines)
        val stations = lines.flatMap { it.stations }.distinctBy { it.id }
        if (stations.size < 2) return emptyMap()

        var totalTime = 0.0
        var reachablePairs = 0
        var maxTime = 0.0

        for (source in stations) {
            val distances = dijkstraShortestPath(graph, source.id)
            for (targetId in stations.map { it.id }) {
                if (source.id == targetId) continue
                val time = distances[targetId] ?: Double.MAX_VALUE
                if (time != Double.MAX_VALUE) {
                    totalTime += time
                    reachablePairs++
                    if (time > maxTime) maxTime = time
                }
            }
        }

        val avgTime = if (reachablePairs > 0) totalTime / reachablePairs else 0.0
        return mapOf(
            "average_time_mins" to avgTime,
            "max_time_mins" to maxTime,
            "reachable_pairs" to reachablePairs.toDouble(),
            "total_stations" to stations.size.toDouble(),
        )
    }

    /**
     * Returns a list of all station-to-station journeys and their travel times.
     */
    fun getAllJourneyTimes(lines: List<Line>): List<Triple<String, String, Double>> {
        if (lines.isEmpty()) return emptyList()

        val graph = buildStationGraph(lines)
        val stations = lines.flatMap { it.stations }.distinctBy { it.id }
        val results = mutableListOf<Triple<String, String, Double>>()

        for (source in stations) {
            val distances = dijkstraShortestPath(graph, source.id)
            for (target in stations) {
                if (source.id == target.id) continue
                val time = distances[target.id] ?: Double.MAX_VALUE
                if (time != Double.MAX_VALUE) {
                    results.add(Triple(source.id, target.id, time))
                }
            }
        }
        return results
    }

    // Natural Metro Network Formation — Corridor-First Implementation

    /**
     * Estimate the suggested trains per hour for each line by normalising the
     * average per-station catchment (log-scaled to reduce skew) across all lines
     * to a [4, 30] tph range, then snapping to the nearest even number (standard
     * headway increment — 2 min, 4 min, …).
     */
    fun computeTrainsPerHour(lines: List<Line>): List<Line> {
        if (lines.isEmpty()) return lines
        // Average log-scaled catchment across stations on each line
        val scores =
            lines.map { line ->
                line.stations.sumOf { ln(1.0 + it.catchmentPopulation) } /
                    line.stations.size.coerceAtLeast(1)
            }
        val minScore = scores.minOrNull() ?: return lines
        val maxScore = scores.maxOrNull() ?: return lines
        val scoreRange = (maxScore - minScore).coerceAtLeast(1e-9)
        val minTph = 4
        val maxTph = 30
        return lines.zip(scores).map { (line, score) ->
            val t = (score - minScore) / scoreRange
            val rawTph = minTph + round(t * (maxTph - minTph)).toInt()
            // Snap to nearest even number for realistic headways
            val tph = ((rawTph + 1) / 2 * 2).coerceIn(minTph, maxTph)
            line.copy(trainsPerHour = tph)
        }
    }

    fun buildNaturalNetworkFromGrid(
        gridPoints: List<GridPoint>,
        walkRadiusMeters: Double = 800.0,
        minStationValue: Double = 1.0,
        maxTrunkLines: Int = 4,
        minCorridorLengthMeters: Double = 5_000.0,
        minStationsPerLine: Int = 5,
        observedRegionDemand: RegionDemandModel? = null,
    ): List<Line> {
        if (gridPoints.isEmpty()) return emptyList()

        // STEP 1 — Identify places, not points (collapse grid artifacts)
        val sortedPoints = gridPoints.sortedByDescending { it.value }
        val places = mutableListOf<GridPoint>()
        val placeRegionCells = mutableListOf<Set<RegionCell>>()
        val clusterRadius = 1000.0
        val used = BooleanArray(gridPoints.size) { false }
        val pointIndices = gridPoints.indices.sortedByDescending { gridPoints[it].value }

        for (idx in pointIndices) {
            if (used[idx]) continue
            val p = gridPoints[idx]
            if (p.value < minStationValue) break

            // Sum up values in radius to create a "Place"
            var sumVal = 0.0
            val regionCells = mutableSetOf<RegionCell>()
            for (j in gridPoints.indices) {
                if (haversineMeters(p.lon, p.lat, gridPoints[j].lon, gridPoints[j].lat) <= clusterRadius) {
                    sumVal += gridPoints[j].value
                    regionCells.add(LondonRegionGrid.cellFor(gridPoints[j].lon, gridPoints[j].lat))
                    used[j] = true
                }
            }
            places.add(GridPoint(p.lon, p.lat, sumVal))
            placeRegionCells.add(regionCells)
        }

        if (places.size < 2) return emptyList()

        // STEP 1 — Instrumentation
        dbg("STEP 1 — Places identified: ${places.size}")
        dbg("Top 5 place values: ${places.take(5).map { "%.0f".format(it.value) }}")

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

        // Compute city scale dynamically (Layer 1 → Layer 2 boundary)
        val distancesFromCenter =
            places
                .map { p ->
                    haversineMeters(p.lon, p.lat, centerLon, centerLat)
                }.sorted()
        val cityRadius =
            if (distancesFromCenter.isNotEmpty()) {
                distancesFromCenter[(distancesFromCenter.size * 0.9).toInt()]
            } else {
                5000.0
            }
        val coreRadius = cityRadius * 0.35 // ~7–9 km for London
        val ringRadius = cityRadius * 0.6 // ~12–15 km for London
        dbg(
            "City scale: cityRadius=${"%.0f".format(
                cityRadius,
            )}m, coreRadius=${"%.0f".format(coreRadius)}m, ringRadius=${"%.0f".format(ringRadius)}m",
        )

        // STEP 2 — Build a place-to-place demand graph
        val n = places.size
        val gravityAdj = Array(n) { DoubleArray(n) { 0.0 } }
        val observedAdj = Array(n) { DoubleArray(n) { 0.0 } }
        val adj = Array(n) { DoubleArray(n) { 0.0 } }
        var maxGravityDemand = 0.0
        var maxObservedDemand = 0.0
        var observedMatches = 0
        var observedTotal = 0.0
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                val distKm = haversineMeters(places[i].lon, places[i].lat, places[j].lon, places[j].lat) / 1000.0
                // Gravity model: demand = (m1 * m2) / d^1.5
                val gravityDemand = (places[i].value * places[j].value) / (distKm.pow(1.5).coerceAtLeast(0.1))
                val observedDemand =
                    observedRegionDemand?.demandBetween(placeRegionCells[i], placeRegionCells[j]) ?: 0.0
                gravityAdj[i][j] = gravityDemand
                gravityAdj[j][i] = gravityDemand
                observedAdj[i][j] = observedDemand
                observedAdj[j][i] = observedDemand
                maxGravityDemand = max(maxGravityDemand, gravityDemand)
                if (observedDemand > 0.0) {
                    maxObservedDemand = max(maxObservedDemand, observedDemand)
                    observedMatches += 1
                    observedTotal += observedDemand
                }
            }
        }
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                val demand =
                    combineGravityAndObservedDemand(
                        gravityDemand = gravityAdj[i][j],
                        observedDemand = observedAdj[i][j],
                        maxGravityDemand = maxGravityDemand,
                        maxObservedDemand = maxObservedDemand,
                    )
                adj[i][j] = demand
                adj[j][i] = demand
            }
        }

        // STEP 2 — Instrumentation
        val maxDemand = adj.maxOf { row -> row.maxOrNull() ?: 0.0 }
        val avgDemand = adj.sumOf { it.sum() } / (n * n).coerceAtLeast(1)
        dbg("STEP 2 — Demand graph built: maxDemand=${"%.1f".format(maxDemand)} avgDemand=${"%.1f".format(avgDemand)}")
        if (observedRegionDemand != null) {
            dbg(
                "STEP 2 — NUMBAT observed demand matches: pairs=$observedMatches, " +
                    "max=${"%.1f".format(maxObservedDemand)}, total=${"%.1f".format(observedTotal)}",
            )
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

                    // 0.65 ≈ cos(49°): tight enough to prevent arcs curving back around,
                    // while still allowing gentle bends along real corridor geometry.
                    if (cosTheta > 0.65 && demand > bestDemand) {
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

                    if (cosTheta > 0.65 && demand > bestDemand) {
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

        // STEP 3 — Instrumentation
        dbg("STEP 3 — Candidate chains found: ${candidateChains.size}")
        dbg("Chain lengths (top 10): ${candidateChains.map { it.size }.sortedDescending().take(10)}")

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

        // STEP 4 — Instrumentation
        dbg("STEP 4 — Valid chains after filtering: ${validChains.size}")
        dbg("Valid chain lengths: ${validChains.map { it.size }.sortedByDescending { it }.take(10)}")

        // STEP 0 — Classify corridors by intent (NEW)
        dbg("STEP 0 — Classifying corridors (soft geometric rules):")
        val classifiedChains =
            validChains.map { chain ->
                chain to classifyCorridor(chain, places, centerLon, centerLat, coreRadius, ringRadius)
            }

        // Filter out non-metro corridors; separate by type
        val radialTrunks = classifiedChains.filter { it.second == LineType.RADIAL_TRUNK }.map { it.first }
        val coreDistributors = classifiedChains.filter { it.second == LineType.CORE_DISTRIBUTOR }.map { it.first }
        val orbitals = classifiedChains.filter { it.second == LineType.ORBITAL }.map { it.first }

        // STEP 0 — Instrumentation (KEY)
        dbg("STEP 0 — Corridor classification:")
        dbg("  RADIAL_TRUNK → ${radialTrunks.size} chains")
        dbg("  CORE_DISTRIBUTOR → ${coreDistributors.size} chains")
        dbg("  ORBITAL → ${orbitals.size} chains")
        dbg("  NOT_METRO → ${classifiedChains.count { it.second == LineType.NOT_METRO }} chains")

        // STEP 8 — Type-aware selection with reserved quotas (not global ranking)
        // Principle: Different line types serve different purposes and need different selection criteria.

        // Quotas (tunable but principled)
        val maxRadials = min(maxTrunkLines, 4)
        val maxDistributors = 2
        val maxOrbitals = 1

        val finalCorridors = mutableListOf<Pair<List<Int>, LineType>>()

        fun corridorBearingVector(chain: List<Int>): Pair<Double, Double> {
            if (chain.size < 2) return 0.0 to 0.0
            val first = places[chain.first()]
            val last = places[chain.last()]
            val avgLatRad = Math.toRadians((first.lat + last.lat) / 2.0)
            val dx = (last.lon - first.lon) * cos(avgLatRad) * 111_320.0
            val dy = (last.lat - first.lat) * 110_540.0
            return dx to dy
        }

        fun bearingSimilarity(a: List<Int>, b: List<Int>): Double {
            val (ax, ay) = corridorBearingVector(a)
            val (bx, by) = corridorBearingVector(b)
            val aLen = hypot(ax, ay)
            val bLen = hypot(bx, by)
            if (aLen <= 0.0 || bLen <= 0.0) return 0.0
            return abs((ax * bx + ay * by) / (aLen * bLen))
        }

        fun closeStationFraction(candidate: List<Int>, existing: List<Int>, thresholdMeters: Double): Double {
            if (candidate.isEmpty() || existing.isEmpty()) return 0.0
            val closeCount =
                candidate.count { cIdx ->
                    val c = places[cIdx]
                    existing.any { eIdx ->
                        val e = places[eIdx]
                        haversineMeters(c.lon, c.lat, e.lon, e.lat) <= thresholdMeters
                    }
                }
            return closeCount.toDouble() / candidate.size.toDouble()
        }

        fun isParallelToSelected(chain: List<Int>): Boolean =
            finalCorridors.any { (selectedChain, selectedType) ->
                val closeFraction = closeStationFraction(chain, selectedChain, thresholdMeters = 700.0)
                val similarity = bearingSimilarity(chain, selectedChain)
                val redundant = closeFraction >= 0.5 && similarity >= 0.85
                if (redundant) {
                    dbg(
                        "    ✗ Parallel corridor rejected: ${chain.size} nodes near $selectedType " +
                            "(close=${"%.0f".format(closeFraction * 100)}%, bearing=${"%.2f".format(similarity)})",
                    )
                }
                redundant
            }

        // PHASE 1: Select RADIAL_TRUNK by demand + length (global priority)
        dbg("STEP 8 — Type-aware selection (quotas: $maxRadials radials, $maxDistributors distributors, $maxOrbitals orbital)")
        dbg("  Phase 1: Selecting radial trunks (by demand)...")

        val radialScores =
            radialTrunks
                .mapIndexed { idx, chain ->
                    val totalDemand =
                        chain.zipWithNext().sumOf { (a, b) ->
                            adj[a][b]
                        }
                    val length =
                        chain.zipWithNext().sumOf { (a, b) ->
                            haversineMeters(places[a].lon, places[a].lat, places[b].lon, places[b].lat)
                        }
                    Triple(idx, chain, totalDemand * length) // demand × length as score
                }.sortedByDescending { it.third }

        for ((origIdx, chain, score) in radialScores) {
            if (finalCorridors.count { it.second == LineType.RADIAL_TRUNK } >= maxRadials) break
            val alreadyCovered = chain.count { idx -> finalCorridors.any { it.first.contains(idx) } }
            if (alreadyCovered < chain.size * 0.5 && !isParallelToSelected(chain)) {
                finalCorridors.add(chain to LineType.RADIAL_TRUNK)
                dbg("    ✓ Radial ${finalCorridors.size}: ${chain.size} nodes, demand-score=${"%.0f".format(score)}")
            }
        }

        // PHASE 2: Select CORE_DISTRIBUTOR by coverage + density (distinct from radials)
        dbg("  Phase 2: Selecting core distributors (by inner-city coverage)...")

        val distributorScores =
            coreDistributors
                .map { chain ->
                    // Score: how many high-value nodes in core region
                    val densityScore = chain.sumOf { idx -> places[idx].value }
                    chain to densityScore
                }.sortedByDescending { it.second }

        for ((chain, score) in distributorScores) {
            if (finalCorridors.count { it.second == LineType.CORE_DISTRIBUTOR } >= maxDistributors) break
            val alreadyCovered = chain.count { idx -> finalCorridors.any { it.first.contains(idx) } }
            if (alreadyCovered < chain.size * 0.4 && !isParallelToSelected(chain)) { // allow more overlap for distributors
                finalCorridors.add(chain to LineType.CORE_DISTRIBUTOR)
                dbg("    ✓ Distributor ${finalCorridors.size}: ${chain.size} nodes, density-score=${"%.0f".format(score)}")
            }
        }

        // PHASE 3: Evaluate if ORBITAL is needed (problem-driven, not demand-driven)
        dbg("  Phase 3: Evaluating orbital necessity (problem-driven)...")

        // Quick provisional fitness check to see if orbitals are needed
        val needsOrbital =
            if (finalCorridors.isNotEmpty()) {
                val provisionalLines =
                    finalCorridors.map { (chain, lineType) ->
                        // Rough line for fitness check (don't need full station placement yet)
                        val first = places[chain.first()]
                        val last = places[chain.last()]
                        val length =
                            chain.zipWithNext().sumOf { (a, b) ->
                                haversineMeters(places[a].lon, places[a].lat, places[b].lon, places[b].lat)
                            }
                        val cost = length / 1000.0 * params.constructionCostPerKm + chain.size * params.costPerStation
                        Line(
                            id = "temp_${lineType.name}",
                            stations =
                                listOf(
                                    Station("temp_first", first.lon, first.lat, first.value),
                                    Station("temp_last", last.lon, last.lat, last.value),
                                ),
                            lengthMeters = length,
                            cost = cost,
                            type = lineType,
                        )
                    }

                // Check fitness: any hub pairs with >2 transfers or >45 min generalized time?
                val fitnessReport = evaluateNetworkFitness(provisionalLines, places, topHubsCount = 12)
                val tooManyViolations = fitnessReport.violationsFound.size >= 2
                tooManyViolations.also { result ->
                    if (result) {
                        dbg("    ⚠ Network inefficiency detected: ${fitnessReport.violationsFound.size} violations → orbital justified")
                    } else {
                        dbg("    ✓ Current radials + distributors serve top hubs efficiently")
                    }
                }
            } else {
                false
            }

        // Add orbital if justified
        // If radials don't intersect enough in core, promote a CORE_DISTRIBUTOR before considering orbitals
        run {
            val radialChains = finalCorridors.filter { it.second == LineType.RADIAL_TRUNK }.map { it.first }
            var pairCount = 0
            var intersectPairs = 0
            val interThreshold = 600.0
            for (i in 0 until radialChains.size) {
                for (j in i + 1 until radialChains.size) {
                    pairCount++
                    var found = false
                    for (a in radialChains[i]) {
                        for (b in radialChains[j]) {
                            val d = haversineMeters(places[a].lon, places[a].lat, places[b].lon, places[b].lat)
                            if (d <= interThreshold) {
                                // ensure intersection occurs in core
                                val midLon = (places[a].lon + places[b].lon) / 2.0
                                val midLat = (places[a].lat + places[b].lat) / 2.0
                                val midDist = haversineMeters(midLon, midLat, centerLon, centerLat)
                                if (midDist <= coreRadius * 1.2) {
                                    found = true
                                    break
                                }
                            }
                        }
                        if (found) break
                    }
                    if (found) intersectPairs++
                }
            }
            val avgIntersections = if (pairCount > 0) intersectPairs.toDouble() / pairCount else 0.0
            if (avgIntersections < 1.5 && distributorScores.isNotEmpty()) {
                // promote best available core distributor not already selected
                val selectedDistributor =
                    distributorScores.map { it.first }.firstOrNull { chain ->
                        finalCorridors.none { it.first == chain } && !isParallelToSelected(chain)
                    }
                if (selectedDistributor != null) {
                    finalCorridors.add(selectedDistributor to LineType.CORE_DISTRIBUTOR)
                    dbg(
                        "  ⚠ Low radial interconnection (avg=${"%.2f".format(
                            avgIntersections,
                        )}) — promoted a CORE_DISTRIBUTOR to improve connectivity",
                    )
                }
            }
        }

        if (needsOrbital && orbitals.isNotEmpty() && finalCorridors.count { it.second == LineType.ORBITAL } < maxOrbitals) {
            // Select the orbital with best potential: longest ring among those not in core
            val orbitalScores =
                orbitals
                    .map { chain ->
                        val length =
                            chain.zipWithNext().sumOf { (a, b) ->
                                haversineMeters(places[a].lon, places[a].lat, places[b].lon, places[b].lat)
                            }
                        chain to length
                    }.sortedByDescending { it.second }

            for ((chain, length) in orbitalScores) {
                val alreadyCovered = chain.count { idx -> finalCorridors.any { it.first.contains(idx) } }
                if (alreadyCovered < chain.size * 0.3 && !isParallelToSelected(chain)) { // allow small overlap for orbital connectivity
                    finalCorridors.add(chain to LineType.ORBITAL)
                    dbg("    ✓ Orbital added (problem-driven): ${chain.size} nodes, length=${"%.1f".format(length / 1000)} km")
                    break
                }
            }
        }

        // STEP 8 — Instrumentation
        dbg("STEP 8 — Final corridors selected: ${finalCorridors.size}")
        dbg("  Breakdown:")
        dbg("    RADIAL_TRUNK: ${finalCorridors.count { it.second == LineType.RADIAL_TRUNK }}")
        dbg("    CORE_DISTRIBUTOR: ${finalCorridors.count { it.second == LineType.CORE_DISTRIBUTOR }}")
        dbg("    ORBITAL: ${finalCorridors.count { it.second == LineType.ORBITAL }}")

        // Enforce mandatory interchanges between crossing corridors BEFORE station placement
        fun enforceMandatoryInterchanges(
            corridors: MutableList<Pair<List<Int>, LineType>>,
            places: List<GridPoint>,
            centerLon: Double,
            centerLat: Double,
            coreRadius: Double,
        ) {
            val interchangeThreshold = 600.0 // meters
            for (a in 0 until corridors.size) {
                for (b in a + 1 until corridors.size) {
                    val chainA = corridors[a].first
                    val chainB = corridors[b].first
                    // find closest approach
                    var bestDist = Double.MAX_VALUE
                    var bestAi = -1
                    var bestBj = -1
                    for (i in chainA.indices) {
                        for (j in chainB.indices) {
                            val pa = places[chainA[i]]
                            val pb = places[chainB[j]]
                            val d = haversineMeters(pa.lon, pa.lat, pb.lon, pb.lat)
                            if (d < bestDist) {
                                bestDist = d
                                bestAi = chainA[i]
                                bestBj = chainB[j]
                            }
                        }
                    }
                    // Enforce interchange globally wherever two corridors come within threshold.
                    // Previously this was gated on the crossing being inside the core radius,
                    // which silently skipped all suburban crossings. Now any close approach forces
                    // a shared node so passengers can change trains network-wide.
                    if (bestDist <= interchangeThreshold) {
                        // Insert node from chainA into chainB at the position nearest bestBj
                        val insertIdx = bestAi
                        if (!chainB.contains(insertIdx)) {
                            val pos = corridors[b].first.indexOf(bestBj)
                            if (pos >= 0) {
                                val newChain = corridors[b].first.toMutableList()
                                newChain.add(pos + 1, insertIdx)
                                corridors[b] = newChain.toList() to corridors[b].second
                                dbg("  ✓ Interchange (${"%.0f".format(bestDist)}m): place $insertIdx -> corridor $b")
                            }
                        }
                        // Symmetric: ensure chainA also contains bestBj
                        val insertIdxB = bestBj
                        if (!chainA.contains(insertIdxB)) {
                            val posA = corridors[a].first.indexOf(bestAi)
                            if (posA >= 0) {
                                val newChainA = corridors[a].first.toMutableList()
                                newChainA.add(posA + 1, insertIdxB)
                                corridors[a] = newChainA.toList() to corridors[a].second
                                dbg("  ✓ Interchange (${"%.0f".format(bestDist)}m): place $insertIdxB -> corridor $a")
                            }
                        }
                    }
                }
            }
        }

        // Apply mandatory interchange enforcement
        enforceMandatoryInterchanges(finalCorridors, places, centerLon, centerLat, coreRadius)

        // STEP 5, 6, 7 — Place stations with type-specific rules
        val stationRegistry = mutableListOf<Station>()
        val forcedInterchangeRadius = 600.0
        val optionalTransferRadius = 300.0

        fun findOrCreateStation(
            lon: Double,
            lat: Double,
            value: Double,
            lineId: String,
            idx: Int,
            callingLineType: LineType,
        ): Station {
            // First, force snap to any existing station within forced radius (ensures core interchanges)
            val forced = stationRegistry.find { haversineMeters(it.lon, it.lat, lon, lat) <= forcedInterchangeRadius }
            if (forced != null) return forced

            // Next, allow optional transfer snapping within a smaller radius
            val optional = stationRegistry.find { haversineMeters(it.lon, it.lat, lon, lat) <= optionalTransferRadius }
            if (optional != null) return optional

            // Otherwise create a new station
            val st = Station("st_${lineId}_$idx", lon, lat, value)
            stationRegistry.add(st)
            return st
        }

        val builtLines = mutableListOf<Line>()
        for ((li, chainWithType) in finalCorridors.withIndex()) {
            var (chain, lineType) = chainWithType
            val typePrefix =
                when (lineType) {
                    LineType.RADIAL_TRUNK -> "RT"
                    LineType.CORE_DISTRIBUTOR -> "CD"
                    LineType.ORBITAL -> "O"
                    LineType.NOT_METRO -> "X"
                }
            val lineId = "${typePrefix}${li + 1}"

            // Find the "seed" (highest value node in the chain)
            val seedIndexInChain = chain.indices.maxByOrNull { places[chain[it]].value } ?: (chain.size / 2)
            val seedNode = chain[seedIndexInChain]

            dbg(
                "Line $lineId ($lineType): initial chain size=${chain.size}, seed node=$seedNode, seedValue=${"%.0f".format(
                    places[seedNode].value,
                )}",
            )

            val finalizedNodes = mutableListOf<Int>()
            finalizedNodes.add(seedNode)

            var cumulativePop = places[seedNode].value

            // Type-specific rules for terminal thinning and spacing
            val terminateThresholdFrac =
                when (lineType) {
                    LineType.RADIAL_TRUNK -> 0.08

                    LineType.CORE_DISTRIBUTOR -> 0.15

                    // stricter for core lines
                    LineType.ORBITAL -> 0.10

                    LineType.NOT_METRO -> 0.20
                }

            val terminalThresholdDistMeters =
                when (lineType) {
                    LineType.RADIAL_TRUNK -> 5000.0

                    LineType.CORE_DISTRIBUTOR -> 2000.0

                    // shorter threshold
                    LineType.ORBITAL -> 4000.0

                    LineType.NOT_METRO -> 3000.0
                }

            // Grow Forward from seed (with soft loop penalty - Improvement 4)
            var lastIdx = seedIndexInChain
            for (i in seedIndexInChain + 1 until chain.size) {
                val pIdx = chain[i]
                val p = places[pIdx]
                val distLast = haversineMeters(p.lon, p.lat, places[chain[lastIdx]].lon, places[chain[lastIdx]].lat)
                val distCenter = haversineMeters(p.lon, p.lat, centerLon, centerLat)
                val distSeed = haversineMeters(p.lon, p.lat, places[seedNode].lon, places[seedNode].lat)

                val minSpace =
                    when (lineType) {
                        LineType.CORE_DISTRIBUTOR -> 600.0

                        // dense
                        else -> getMinSpacing(distCenter)
                    }

                val penaltyFactor =
                    when (lineType) {
                        LineType.RADIAL_TRUNK -> (1.0 + (distSeed / 8000.0).pow(1.5))

                        LineType.CORE_DISTRIBUTOR -> 1.0

                        // no penalty; stay dense
                        LineType.ORBITAL -> (1.0 + (distSeed / 6000.0).pow(1.2))

                        LineType.NOT_METRO -> 1.5
                    }

                // Soft loop penalty (Improvement 4): if moving toward opposite terminus, increase threshold
                val loopPenalty = if (lineType != LineType.ORBITAL && i > chain.size * 0.7) 1.3 else 1.0

                val requiredValue = minStationValue * penaltyFactor * loopPenalty

                if (distLast >= minSpace && p.value >= requiredValue) {
                    finalizedNodes.add(pIdx)
                    lastIdx = i
                    cumulativePop += p.value
                } else if (distSeed > terminalThresholdDistMeters && p.value < cumulativePop * terminateThresholdFrac) {
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

                val minSpace =
                    when (lineType) {
                        LineType.CORE_DISTRIBUTOR -> 600.0
                        else -> getMinSpacing(distCenter)
                    }

                val penaltyFactor =
                    when (lineType) {
                        LineType.RADIAL_TRUNK -> (1.0 + (distSeed / 8000.0).pow(1.5))
                        LineType.CORE_DISTRIBUTOR -> 1.0
                        LineType.ORBITAL -> (1.0 + (distSeed / 6000.0).pow(1.2))
                        LineType.NOT_METRO -> 1.5
                    }

                // Soft loop penalty
                val loopPenalty = if (lineType != LineType.ORBITAL && i < chain.size * 0.3) 1.3 else 1.0

                val requiredValue = minStationValue * penaltyFactor * loopPenalty

                if (distLast >= minSpace && p.value >= requiredValue) {
                    finalizedNodes.add(0, pIdx)
                    lastIdx = i
                    cumulativePop += p.value
                } else if (distSeed > terminalThresholdDistMeters && p.value < cumulativePop * terminateThresholdFrac) {
                    break
                }
            }

            if (finalizedNodes.size >= 2) {
                // Anti-U-turn trim: if the finalized chain folds back on its own overall bearing
                // (a "mini-circle" that isn't fully closed), prune the reversing tail and then head.
                // Threshold 0.55: a perfectly straight line scores 1.0; a true circle scores ~0.0;
                // 0.55 catches chains that arc more than ~120° without fully closing.
                if (lineType != LineType.ORBITAL && finalizedNodes.size > minStationsPerLine) {
                    val preTrimStraightness = computeLoopiness(finalizedNodes, places)
                    if (preTrimStraightness < 0.55) {
                        // Overall bearing vector: first station -> last station
                        val odx = places[finalizedNodes.last()].lon - places[finalizedNodes.first()].lon
                        val ody = places[finalizedNodes.last()].lat - places[finalizedNodes.first()].lat

                        // Prune TAIL nodes whose step opposes the overall direction
                        while (finalizedNodes.size > minStationsPerLine) {
                            val last = finalizedNodes.last()
                            val prev2 = finalizedNodes[finalizedNodes.size - 2]
                            val sdx = places[last].lon - places[prev2].lon
                            val sdy = places[last].lat - places[prev2].lat
                            if (sdx * odx + sdy * ody < 0.0) {
                                finalizedNodes.removeAt(finalizedNodes.size - 1)
                                dbg("  ✂ Anti-U-turn tail: removed node $last")
                            } else {
                                break
                            }
                        }

                        // Recompute overall direction, then prune HEAD nodes the same way
                        if (finalizedNodes.size > minStationsPerLine) {
                            val odx2 = places[finalizedNodes.last()].lon - places[finalizedNodes.first()].lon
                            val ody2 = places[finalizedNodes.last()].lat - places[finalizedNodes.first()].lat
                            while (finalizedNodes.size > minStationsPerLine) {
                                val first = finalizedNodes.first()
                                val next2 = finalizedNodes[1]
                                val sdx = places[next2].lon - places[first].lon
                                val sdy = places[next2].lat - places[first].lat
                                if (sdx * odx2 + sdy * ody2 < 0.0) {
                                    finalizedNodes.removeAt(0)
                                    dbg("  ✂ Anti-U-turn head: removed node $first")
                                } else {
                                    break
                                }
                            }
                        }
                        dbg(
                            "  Anti-U-turn trim: straightness ${"%.2f".format(
                                preTrimStraightness,
                            )} -> ${"%.2f".format(computeLoopiness(finalizedNodes, places))}",
                        )
                    }
                }

                // Improvements 1–3: Detect and handle circularity
                val loopiness = computeLoopiness(finalizedNodes, places)
                var isCircular = loopiness < 0.15
                var finalLineType = lineType
                var finalNodeList = finalizedNodes.toList()

                dbg(
                    "Line $lineId: loopiness=${"%.3f".format(
                        loopiness,
                    )}, isCircular=$isCircular, type=$lineType",
                )

                // Improvement 2: Only allow circular if ORBITAL, otherwise intervene
                if (isCircular && lineType != LineType.ORBITAL) {
                    dbg("  ⚠ Circular non-orbital detected: attempting Option A (break weakest edge)")

                    val weakestEdgeIdx = findWeakestEdgeInChain(finalizedNodes, adj)
                    if (weakestEdgeIdx != null) {
                        // Break the chain at weakest edge
                        val broken = finalizedNodes.dropLast(finalizedNodes.size - weakestEdgeIdx - 1)
                        if (broken.size >= minStationsPerLine) {
                            finalNodeList = broken
                            isCircular = false
                            dbg("  ✓ Loop broken at edge $weakestEdgeIdx; new size=${broken.size}")
                        }
                    }

                    // Option B fallback: check if can promote to ORBITAL
                    if (isCircular) {
                        val distToCenter =
                            finalizedNodes.map { idx ->
                                haversineMeters(places[idx].lon, places[idx].lat, centerLon, centerLat)
                            }
                        val minDist = distToCenter.minOrNull() ?: Double.MAX_VALUE
                        val fracInRing =
                            finalizedNodes
                                .count { idx ->
                                    val d = haversineMeters(places[idx].lon, places[idx].lat, centerLon, centerLat)
                                    d > coreRadius && d <= ringRadius * 1.2
                                }.toDouble() / finalizedNodes.size

                        // Count intersections with radial-like lines (Improvement 5)
                        val radialCount =
                            countRadialIntersections(
                                finalizedNodes,
                                finalCorridors.take(li).map { it.first },
                                places,
                                centerLon,
                                centerLat,
                                coreRadius,
                            )
                        val highValueCount =
                            countHighValuePlaces(
                                finalizedNodes,
                                places,
                                cumulativePop / finalizedNodes.size,
                            )

                        if (minDist > coreRadius && fracInRing > 0.6 && (radialCount >= 2 || highValueCount >= 3)) {
                            finalLineType = LineType.ORBITAL
                            dbg("  ✓ Promoted to ORBITAL (intercepts $radialCount radials, $highValueCount high-value places)")
                        } else {
                            dbg(
                                "  ✗ Loop rejected: minDist=${"%.0f".format(
                                    minDist,
                                )} coreRadius=${"%.0f".format(
                                    coreRadius,
                                )}, fracInRing=${"%.2f".format(fracInRing)}, radials=$radialCount, highVal=$highValueCount",
                            )
                            isCircular = false
                        }
                    }
                }

                // Improvement 3: Snap endpoints if true loop detected
                var finalStations: List<Station>? = null
                if (isCircular && loopiness < 0.15) {
                    val first = finalNodeList.first()
                    val last = finalNodeList.last()
                    val endpointDist =
                        haversineMeters(places[first].lon, places[first].lat, places[last].lon, places[last].lat)

                    if (endpointDist <= 700.0) {
                        dbg("  ✓ Endpoints within 700m: snapping to single terminal (true loop)")
                        // Use last node's location as the shared terminal
                        val terminalStations =
                            finalNodeList.take(finalNodeList.size - 1).mapIndexed { idx, pIdx ->
                                val p = places[pIdx]
                                findOrCreateStation(p.lon, p.lat, p.value, lineId, idx, lineType)
                            }
                        // Add one more station at the junction (snapped terminal)
                        val p = places[last]
                        val junctionStation = findOrCreateStation(p.lon, p.lat, p.value, lineId, finalNodeList.size - 1, lineType)
                        finalStations = terminalStations + junctionStation
                    }
                }

                // If no special snapping, create stations normally
                if (finalStations == null) {
                    finalStations =
                        finalNodeList.mapIndexed { idx, pIdx ->
                            val p = places[pIdx]
                            findOrCreateStation(p.lon, p.lat, p.value, lineId, idx, lineType)
                        }
                }

                val length = finalStations.zipWithNext().sumOf { (a, b) -> haversineMeters(a.lon, a.lat, b.lon, b.lat) }
                val cost = length / 1000.0 * params.constructionCostPerKm + finalStations.size * params.costPerStation
                dbg("Line $lineId finalized: stations=${finalStations.size}, isLoop=$isCircular, type=$finalLineType")
                builtLines.add(Line(lineId, finalStations, length, cost, finalLineType, isCircular))
            }
        }

        // FINAL — Instrumentation
        dbg("FINAL — Built lines: ${builtLines.size}")
        builtLines.forEach {
            val loopMarker = if (it.isLoop) " [LOOP]" else ""
            dbg("  ${it.id}: type=${it.type}, stations=${it.stations.size}, length=${"%.1f".format(it.lengthMeters / 1000)} km$loopMarker")
        }

        // POST-BUILD: Evaluate network fitness (Improvement 1–5)
        dbg("")
        val fitnessReport = evaluateNetworkFitness(builtLines, places)
        dbg("NETWORK FITNESS — Summary:")
        dbg("  Violations: ${fitnessReport.violationsFound.size}")
        dbg("  Metrics OK: ${fitnessReport.overallMetricsOK}")

        // POST-BUILD: Snap interchange stations.
        // When two lines have stations within 500 m of each other, replace both with a
        // single shared station at the population-weighted midpoint. This produces a genuine
        // shared node (one map dot carrying both line colours) not two nearly-coincident dots.
        val snappedLines = snapInterchangeStations(builtLines, snapRadiusMeters = 500.0)
        val consolidatedLines =
            consolidateStationClusters(
                snappedLines,
                mergeRadiusMeters = 900.0,
                maxRouteLengthChangeRatio = 0.12,
                minStationsPerLine = minStationsPerLine,
            )

        // POST-BUILD: Estimate economically sensible construction methods and compute suggested service levels
        val linesWithBuildEstimates =
            estimateBuildTechnology(
                consolidatedLines,
                centerLon = centerLon,
                centerLat = centerLat,
                coreRadiusMeters = coreRadius,
                ringRadiusMeters = ringRadius,
            )
        val linesWithTph = computeTrainsPerHour(linesWithBuildEstimates)
        linesWithTph.forEach { line ->
            dbg("  ${line.id}: ${line.trainsPerHour} trains/hr, cost=£${"%.1f".format(line.cost / 1_000_000_000.0)}bn")
        }

        return linesWithTph
    }

    /**
     * Post-build interchange snapping.
     *
     * For every pair of lines whose closest stations are within [snapRadiusMeters] of each
     * other, replace both stations with a single shared interchange station placed at the
     * population-weighted midpoint.  Shared station IDs carry the prefix "INT_" so the
     * front-end can render them with a distinct interchange symbol.
     *
     * Multiple rounds are run so that 3- and 4-way interchanges converge correctly
     * (round 1 merges A↔B and A↔C; round 2 then merges B↔C at the same point).
     */
    private fun snapInterchangeStations(
        lines: List<Line>,
        snapRadiusMeters: Double = 500.0,
        rounds: Int = 3,
    ): List<Line> {
        val current = lines.toMutableList()

        repeat(rounds) { round ->
            var anySnapped = false
            for (i in current.indices) {
                for (j in i + 1 until current.size) {
                    val stI = current[i].stations
                    val stJ = current[j].stations

                    // Find the closest cross-line station pair
                    var bestDist = Double.MAX_VALUE
                    var bestSi: Station? = null
                    var bestSj: Station? = null
                    for (si in stI) {
                        for (sj in stJ) {
                            if (si.id == sj.id) continue // already the same shared station
                            val d = haversineMeters(si.lon, si.lat, sj.lon, sj.lat)
                            if (d < bestDist) {
                                bestDist = d
                                bestSi = si
                                bestSj = sj
                            }
                        }
                    }

                    if (bestDist > snapRadiusMeters || bestSi == null || bestSj == null) continue

                    // Population-weighted midpoint
                    val wI = bestSi.catchmentPopulation.coerceAtLeast(1.0)
                    val wJ = bestSj.catchmentPopulation.coerceAtLeast(1.0)
                    val wSum = wI + wJ
                    val sharedLon = (bestSi.lon * wI + bestSj.lon * wJ) / wSum
                    val sharedLat = (bestSi.lat * wI + bestSj.lat * wJ) / wSum
                    // Preserve an existing INT_ id across rounds for stable convergence
                    val sharedId =
                        when {
                            bestSi.id.startsWith("INT_") -> bestSi.id
                            bestSj.id.startsWith("INT_") -> bestSj.id
                            else -> "INT_${current[i].id}_${current[j].id}"
                        }
                    val shared = Station(sharedId, sharedLon, sharedLat, wSum)

                    fun rebuildLine(
                        line: Line,
                        old: Station,
                    ): Line {
                        val newSt = line.stations.map { if (it.id == old.id) shared else it }
                        val len =
                            newSt
                                .zipWithNext()
                                .sumOf { (a, b) -> haversineMeters(a.lon, a.lat, b.lon, b.lat) }
                        return line.copy(stations = newSt, lengthMeters = len)
                    }

                    current[i] = rebuildLine(current[i], bestSi)
                    current[j] = rebuildLine(current[j], bestSj)
                    anySnapped = true
                    dbg(
                        "  ✦ [r${round + 1}] Interchange snap: ${current[i].id} <-> ${current[j].id}" +
                            " -> $sharedId (${"%.0f".format(bestDist)}m)",
                    )
                }
            }
            if (!anySnapped) return@repeat
        }

        return current
    }

    internal fun consolidateStationClusters(
        lines: List<Line>,
        mergeRadiusMeters: Double = 900.0,
        maxRouteLengthChangeRatio: Double = 0.12,
        minStationsPerLine: Int = 3,
    ): List<Line> {
        data class StationRef(
            val lineIndex: Int,
            val stationIndex: Int,
            val station: Station,
        )

        var current = lines
        var clusterCounter = 1

        repeat(4) {
            val refs =
                current.flatMapIndexed { lineIndex, line ->
                    line.stations.mapIndexed { stationIndex, station ->
                        StationRef(lineIndex, stationIndex, station)
                    }
                }
            val parent = IntArray(refs.size) { it }

            fun find(i: Int): Int {
                var x = i
                while (parent[x] != x) {
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                }
                return x
            }

            fun union(
                a: Int,
                b: Int,
            ) {
                val ra = find(a)
                val rb = find(b)
                if (ra != rb) parent[rb] = ra
            }

            for (i in refs.indices) {
                for (j in i + 1 until refs.size) {
                    if (refs[i].lineIndex == refs[j].lineIndex) continue
                    val d =
                        haversineMeters(
                            refs[i].station.lon,
                            refs[i].station.lat,
                            refs[j].station.lon,
                            refs[j].station.lat,
                        )
                    if (d <= mergeRadiusMeters) union(i, j)
                }
            }

            val clusters = refs.indices.groupBy { find(it) }.values
            var changed = false
            var next = current

            for (clusterIndexes in clusters) {
                if (clusterIndexes.size < 2) continue
                val clusterRefs = clusterIndexes.map { refs[it] }
                val lineIndexes = clusterRefs.map { it.lineIndex }.toSet()
                if (lineIndexes.size < 2) continue

                // Do not merge two distinct stops on the same route; that creates accidental short-cuts.
                if (clusterRefs.groupingBy { it.lineIndex }.eachCount().any { it.value > 1 }) continue

                val wSum = clusterRefs.sumOf { it.station.catchmentPopulation.coerceAtLeast(1.0) }
                val sharedLon = clusterRefs.sumOf { it.station.lon * it.station.catchmentPopulation.coerceAtLeast(1.0) } / wSum
                val sharedLat = clusterRefs.sumOf { it.station.lat * it.station.catchmentPopulation.coerceAtLeast(1.0) } / wSum
                val sharedId =
                    clusterRefs
                        .map { it.station.id }
                        .firstOrNull { it.startsWith("INT_") }
                        ?: "INT_CLUSTER_${clusterCounter++}"
                val shared = Station(sharedId, sharedLon, sharedLat, wSum)
                val idsToReplace = clusterRefs.map { it.station.id }.toSet()

                val candidate =
                    next.map { line ->
                        if (line.stations.none { it.id in idsToReplace }) return@map line
                        val newStations = line.stations.map { if (it.id in idsToReplace) shared else it }
                        val newLength = lineLengthMeters(newStations)
                        line.copy(stations = newStations, lengthMeters = newLength)
                    }

                val valid =
                    candidate.indices.all { idx ->
                        val oldLine = next[idx]
                        val newLine = candidate[idx]
                        val hasConsecutiveDuplicate =
                            newLine.stations.zipWithNext().any { (a, b) -> a.id == b.id }
                        val enoughStations = newLine.stations.distinctBy { it.id }.size >= minStationsPerLine
                        val oldLength = oldLine.lengthMeters.coerceAtLeast(1.0)
                        val changeRatio = abs(newLine.lengthMeters - oldLine.lengthMeters) / oldLength
                        !hasConsecutiveDuplicate && enoughStations && changeRatio <= maxRouteLengthChangeRatio
                    }

                if (valid) {
                    next = candidate
                    changed = true
                    dbg("  ✦ Consolidated ${clusterRefs.size} nearby cross-line stations into $sharedId")
                }
            }

            current = next
            if (!changed) return@repeat
        }

        return current
    }

    internal fun estimateBuildTechnology(
        lines: List<Line>,
        centerLon: Double,
        centerLat: Double,
        coreRadiusMeters: Double,
        ringRadiusMeters: Double,
    ): List<Line> {
        if (lines.isEmpty()) return lines
        val stationValues =
            lines
                .flatMap { it.stations }
                .map { it.catchmentPopulation }
                .filter { it.isFinite() && it > 0.0 }
                .sorted()
        val highValueThreshold =
            if (stationValues.isEmpty()) {
                Double.POSITIVE_INFINITY
            } else {
                stationValues[(stationValues.size * 0.75).coerceAtMost(stationValues.lastIndex.toDouble()).toInt()]
            }

        return lines.map { line ->
            val segments =
                line.stations.zipWithNext().map { (from, to) ->
                    val length = haversineMeters(from.lon, from.lat, to.lon, to.lat)
                    val midLon = (from.lon + to.lon) / 2.0
                    val midLat = (from.lat + to.lat) / 2.0
                    val distToCenter = haversineMeters(midLon, midLat, centerLon, centerLat)
                    val avgValue = (from.catchmentPopulation + to.catchmentPopulation) / 2.0
                    val technology =
                        when {
                            distToCenter <= coreRadiusMeters * 0.9 -> AlignmentTechnology.DEEP_BORE_TUNNEL
                            line.type == LineType.CORE_DISTRIBUTOR && distToCenter <= coreRadiusMeters * 1.15 ->
                                AlignmentTechnology.DEEP_BORE_TUNNEL
                            avgValue >= highValueThreshold && distToCenter <= ringRadiusMeters ->
                                AlignmentTechnology.DEEP_BORE_TUNNEL
                            distToCenter <= ringRadiusMeters * 1.15 -> AlignmentTechnology.SUBSURFACE_TUNNEL
                            else -> AlignmentTechnology.SURFACE_OR_ELEVATED
                        }
                    val perKm =
                        when (technology) {
                            AlignmentTechnology.DEEP_BORE_TUNNEL -> 450_000_000.0
                            AlignmentTechnology.SUBSURFACE_TUNNEL -> 250_000_000.0
                            AlignmentTechnology.SURFACE_OR_ELEVATED -> 90_000_000.0
                        }
                    BuildSegmentEstimate(
                        fromStationId = from.id,
                        toStationId = to.id,
                        lengthMeters = length,
                        technology = technology,
                        civilCost = length / 1000.0 * perKm,
                        rationale = rationaleForTechnology(technology),
                    )
                }

            val civilCost = segments.sumOf { it.civilCost }
            val stationCost =
                line.stations.sumOf { station ->
                    val adjacentTech =
                        segments
                            .filter { it.fromStationId == station.id || it.toStationId == station.id }
                            .map { it.technology }
                    val stationTech =
                        when {
                            AlignmentTechnology.DEEP_BORE_TUNNEL in adjacentTech -> AlignmentTechnology.DEEP_BORE_TUNNEL
                            AlignmentTechnology.SUBSURFACE_TUNNEL in adjacentTech -> AlignmentTechnology.SUBSURFACE_TUNNEL
                            else -> AlignmentTechnology.SURFACE_OR_ELEVATED
                        }
                    when (stationTech) {
                        AlignmentTechnology.DEEP_BORE_TUNNEL -> 300_000_000.0
                        AlignmentTechnology.SUBSURFACE_TUNNEL -> 150_000_000.0
                        AlignmentTechnology.SURFACE_OR_ELEVATED -> 60_000_000.0
                    }
                }
            val landAllowance =
                segments.sumOf { segment ->
                    val allowance =
                        when (segment.technology) {
                            AlignmentTechnology.DEEP_BORE_TUNNEL -> 0.08
                            AlignmentTechnology.SUBSURFACE_TUNNEL -> 0.15
                            AlignmentTechnology.SURFACE_OR_ELEVATED -> 0.10
                        }
                    segment.civilCost * allowance
                }
            val preContingency = civilCost + stationCost + landAllowance
            val contingency = preContingency * 0.30
            val estimate =
                LineBuildEstimate(
                    segments = segments,
                    stationCost = stationCost,
                    landInterfaceAllowance = landAllowance,
                    contingency = contingency,
                    totalCost = preContingency + contingency,
                    deepBoreMeters =
                        segments
                            .filter { it.technology == AlignmentTechnology.DEEP_BORE_TUNNEL }
                            .sumOf { it.lengthMeters },
                    subsurfaceMeters =
                        segments
                            .filter { it.technology == AlignmentTechnology.SUBSURFACE_TUNNEL }
                            .sumOf { it.lengthMeters },
                    surfaceOrElevatedMeters =
                        segments
                            .filter { it.technology == AlignmentTechnology.SURFACE_OR_ELEVATED }
                            .sumOf { it.lengthMeters },
                    recommendation = recommendationForSegments(segments),
                )

            line.copy(cost = estimate.totalCost, buildEstimate = estimate)
        }
    }

    private fun lineLengthMeters(stations: List<Station>): Double =
        stations.zipWithNext().sumOf { (a, b) -> haversineMeters(a.lon, a.lat, b.lon, b.lat) }

    private fun rationaleForTechnology(technology: AlignmentTechnology): String =
        when (technology) {
            AlignmentTechnology.DEEP_BORE_TUNNEL -> "Dense central/interchange segment: avoid demolition and severe surface disruption."
            AlignmentTechnology.SUBSURFACE_TUNNEL -> "Inner urban segment: shallow tunnelling balances cost and surface constraints."
            AlignmentTechnology.SURFACE_OR_ELEVATED -> "Outer lower-density segment: surface or viaduct is the economical default if corridor land is available."
        }

    private fun recommendationForSegments(segments: List<BuildSegmentEstimate>): String {
        val deepKm = segments.filter { it.technology == AlignmentTechnology.DEEP_BORE_TUNNEL }.sumOf { it.lengthMeters } / 1000.0
        val shallowKm = segments.filter { it.technology == AlignmentTechnology.SUBSURFACE_TUNNEL }.sumOf { it.lengthMeters } / 1000.0
        val surfaceKm = segments.filter { it.technology == AlignmentTechnology.SURFACE_OR_ELEVATED }.sumOf { it.lengthMeters } / 1000.0
        return when {
            deepKm >= shallowKm && deepKm >= surfaceKm -> "Build mainly as deep-bore tunnel through dense central areas; it costs more per km but avoids large-scale demolition."
            shallowKm >= surfaceKm -> "Use mostly subsurface tunnel, with deep-bore sections where central constraints are highest."
            else -> "Use surface/elevated track on outer sections and tunnel only where urban constraints require it."
        }
    }
}
