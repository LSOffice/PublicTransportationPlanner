package org.lsoffice

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.BufferedReader
import java.io.InputStreamReader
import java.security.MessageDigest
import java.util.LinkedHashMap
import java.util.UUID
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class PlannerValidationException(
    message: String,
    val code: String = "VALIDATION_ERROR",
) : IllegalArgumentException(message)

class MissingSessionException(
    val sessionId: String,
) : IllegalStateException("Analysis session '$sessionId' is missing or was evicted")

class StaleSessionException(
    val expectedRevision: Long,
    val actualRevision: Long,
) : IllegalStateException("Analysis session revision $actualRevision does not match project revision $expectedRevision")

class PlannerService(
    private val json: Json,
) {
    data class AnalysisSession(
        val response: AnalysisSessionResponse,
        val project: PlannerProject,
        val demand: DemandMatrix,
        val router: PrimitiveRoutingEngine,
        val segmentPeakLoads: Map<String, Double>,
    )

    data class DemandMatrix(
        val zones: List<DemandZone>,
        val origins: IntArray,
        val destinations: IntArray,
        val dailyJourneys: IntArray,
        val distanceMeters: DoubleArray,
        val totalDailyJourneys: Int,
    ) {
        val size: Int get() = dailyJourneys.size

        fun record(index: Int): DemandRecord =
            DemandRecord(
                originZoneId = zones[origins[index]].id,
                destinationZoneId = zones[destinations[index]].id,
                dailyJourneys = dailyJourneys[index],
                distanceMeters = distanceMeters[index],
            )
    }

    @Serializable
    private data class CoverageAsset(
        val polygons: List<List<Coordinate>>,
        val sourceSha256: String,
    )

    private val sessions =
        object : LinkedHashMap<String, AnalysisSession>(8, 0.75f, true) {
            override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, AnalysisSession>?): Boolean = size > 4
        }
    private val coverageAsset: CoverageAsset by lazy { loadCoverageAsset() }
    private val londonPopulationPoints: List<GridPoint> by lazy { loadLondonPopulationPoints() }

    fun coverage(): CoverageResponse {
        val polygons = coverageAsset.polygons
        val all = polygons.flatten()
        val bounds =
            listOf(
                all.minOf { it.lon },
                all.minOf { it.lat },
                all.maxOf { it.lon },
                all.maxOf { it.lat },
            )
        return CoverageResponse(
            hardSupport = polygons,
            ptalSupport = polygons,
            bounds = bounds,
            populationGridCellsInLondon = londonPopulationPoints.size,
            ptalRecords = PtalLookup.records.size,
            sources =
                listOf(
                    CoverageSource(
                        id = "gla-boundary-2025",
                        title = "Greater London boundary",
                        url = "https://data.london.gov.uk/dataset/statistical-gis-boundary-files-for-london-20od9",
                        licence = "Open Government Licence v2",
                        attribution =
                            "Contains National Statistics data © Crown copyright and database right 2015; " +
                                "Contains Ordnance Survey data © Crown copyright and database right 2015",
                    ),
                    CoverageSource(
                        id = "gbr-population-2020",
                        title = "Great Britain 1 km population grid",
                        url = "bundled:gbr_pd_2020_1km_ASCII_XYZ.csv",
                        licence = "Repository-provided dataset; verify provenance before redistribution",
                        attribution = "Used as a demand-density proxy, not an exact journey count",
                    ),
                    CoverageSource(
                        id = "tfl-ptal-2015",
                        title = "TfL PTAL 2015, LSOA 2011",
                        url = "bundled:ptal_lsoa2011.json",
                        licence = "Open Government Licence v2",
                        attribution = "Transport for London accessibility evidence",
                    ),
                ),
        )
    }

    fun generateNetwork(request: GenerateNetworkRequest): GenerateNetworkResponse {
        validateStudyArea(request.studyArea)
        val points = londonPopulationPoints.filter { pointInPolygon(it.lon, it.lat, request.studyArea.coordinates) }
        if (points.size < 3) throw PlannerValidationException("The study area contains too few population-grid cells")
        val enriched =
            points.map { point ->
                val multiplier = PtalLookup.demandWeight(point.lon, point.lat)
                point.copy(value = point.value * multiplier)
            }
        val builder =
            MetroBuilder(
                params = BuilderParams(1_000_000_000.0, 50_000_000.0),
                debug = false,
            )
        val generated =
            builder.buildNaturalNetworkFromGrid(
                gridPoints = enriched,
                minStationValue = 0.0,
                maxTrunkLines = request.maxTrunkLines.coerceIn(1, 8),
                minCorridorLengthMeters = 2_000.0,
                minStationsPerLine = 3,
            )
        if (generated.isEmpty()) {
            throw PlannerValidationException(
                "Automatic generation produced no viable lines. Draw a larger area or include more populated cells.",
                "NO_VIABLE_NETWORK",
            )
        }
        return GenerateNetworkResponse(
            network = convertGeneratedNetwork(generated),
            selectedGridPoints = points.size,
            warnings =
                listOf(
                    "Infrastructure labels are a density-based planning assumption and remain editable.",
                    "Demand values are synthetic planning proxies.",
                ),
        )
    }

    fun createSession(request: AnalysisSessionRequest): AnalysisSessionResponse {
        validateProject(request.project)
        val fingerprint = fingerprint(request.project)
        synchronized(sessions) {
            sessions.values.firstOrNull { it.response.fingerprint == fingerprint }?.let { return it.response }
        }
        val demand = generateDemand(request.project)
        val router = PrimitiveRoutingEngine(request.project.network, request.project.modelSettings)
        val analysis = analyse(request.project, demand, router)
        val response =
            AnalysisSessionResponse(
                sessionId = UUID.randomUUID().toString(),
                projectId = request.project.id,
                revision = request.project.revision,
                fingerprint = fingerprint,
                score = analysis.score,
                zones = demand.zones,
                demandRecordCount = demand.size,
                totalDailyJourneys = demand.totalDailyJourneys,
                issues = analysis.issues,
            )
        synchronized(sessions) {
            sessions[response.sessionId] =
                AnalysisSession(response, request.project, demand, router, analysis.segmentPeakLoads)
        }
        return response
    }

    fun demandPage(
        sessionId: String,
        offset: Int,
        limit: Int,
    ): DemandPage {
        val session = session(sessionId)
        val safeOffset = offset.coerceIn(0, session.demand.size)
        val safeLimit = limit.coerceIn(1, 1_000)
        val end = min(session.demand.size, safeOffset + safeLimit)
        return DemandPage(
            records = (safeOffset until end).map(session.demand::record),
            offset = safeOffset,
            limit = safeLimit,
            total = session.demand.size,
            totalDailyJourneys = session.demand.totalDailyJourneys,
        )
    }

    fun demandCsv(sessionId: String): String {
        val demand = session(sessionId).demand
        return buildString {
            appendLine("origin_zone_id,destination_zone_id,daily_journeys,distance_m")
            for (index in 0 until demand.size) {
                val row = demand.record(index)
                append(row.originZoneId).append(',')
                append(row.destinationZoneId).append(',')
                append(row.dailyJourneys).append(',')
                append("%.1f".format(java.util.Locale.ROOT, row.distanceMeters)).appendLine()
            }
        }
    }

    fun issues(sessionId: String): List<NetworkIssue> = session(sessionId).response.issues

    fun journey(
        sessionId: String,
        request: JourneyRequest,
    ): JourneyResult {
        validateDisruptions(session(sessionId).project, request.disruptions)
        return session(sessionId).router.route(request)
    }

    fun simulation(
        sessionId: String,
        request: SimulationRequest,
    ): SimulationResponse {
        val session = session(sessionId)
        validateDisruptions(session.project, request.disruptions)
        val horizon = request.horizonMinutes.coerceIn(10.0, 720.0)
        val runs = mutableListOf<TrainRun>()
        session.project.network.lines.forEach { line ->
            if (line.stationIds.size < 2 || line.trainsPerHour <= 0) return@forEach
            val headway = 60.0 / line.trainsPerHour
            var departure = 0.0
            var departureIndex = 0
            while (departure <= horizon) {
                for (direction in listOf(1, -1)) {
                    val activeAtDeparture =
                        request.disruptions.filter { disruption ->
                            departure >= disruption.startMinute && departure < disruption.startMinute + disruption.durationMinutes
                        }
                    val orderedStations = if (direction == 1) line.stationIds else line.stationIds.reversed()
                    val orderedSegments = if (direction == 1) line.segments else line.segments.reversed()
                    val cumulative = mutableListOf(0.0)
                    var elapsed = 0.0
                    orderedSegments.forEach { segment ->
                        val travelMinutes = session.router.segmentTravelMinutes(segment, activeAtDeparture)
                        elapsed = if (travelMinutes.isFinite()) elapsed + travelMinutes else horizon + 60.0
                        cumulative += elapsed
                    }
                    val cancellationSeverity =
                        activeAtDeparture
                            .filter { it.type == DisruptionType.TRAIN_CANCELLATION && it.targetId == line.id }
                            .maxOfOrNull { it.severity }
                            ?.coerceIn(0.0, 1.0) ?: 0.0
                    val cancelEvery = if (cancellationSeverity <= 0.0) Int.MAX_VALUE else max(1, (1.0 / cancellationSeverity).toInt())
                    val cancelled = departureIndex % cancelEvery == 0 && cancellationSeverity > 0.0
                    runs +=
                        TrainRun(
                            id = "${line.id}-${direction}-${departureIndex}",
                            lineId = line.id,
                            direction = direction,
                            departureMinute = departure,
                            stationIds = orderedStations,
                            cumulativeMinutes = cumulative,
                            cancelled = cancelled,
                        )
                }
                departureIndex++
                departure += headway
            }
        }
        return SimulationResponse(horizon, runs, request.disruptions)
    }

    fun nearestDensity(
        lon: Double,
        lat: Double,
        maxMeters: Double,
    ): GridPoint? =
        londonPopulationPoints
            .asSequence()
            .map { it to haversineMeters(lon, lat, it.lon, it.lat) }
            .filter { it.second <= maxMeters }
            .minByOrNull { it.second }
            ?.first

    private data class AnalysisResult(
        val score: NetworkScore,
        val issues: List<NetworkIssue>,
        val segmentPeakLoads: Map<String, Double>,
    )

    private fun analyse(
        project: PlannerProject,
        demand: DemandMatrix,
        router: PrimitiveRoutingEngine,
    ): AnalysisResult {
        val settings = project.modelSettings
        val nearestStation =
            demand.zones.map { zone ->
                project.network.stations
                    .asSequence()
                    .map { station -> station to haversineMeters(zone.lon, zone.lat, station.lon, station.lat) }
                    .filter { it.second <= settings.stationCatchmentMeters }
                    .minByOrNull { it.second }
                    ?.first
            }
        val candidateIndices =
            (0 until demand.size)
                .filter { demand.dailyJourneys[it] > 0 }
                .sortedByDescending { demand.dailyJourneys[it] }
                .take(5_000)
        val routeCache = mutableMapOf<Pair<String, String>, JourneyResult>()
        val segmentLoads = mutableMapOf<String, Double>()
        var considered = 0.0
        var served = 0.0
        var timeBenefitWeighted = 0.0
        var directnessWeighted = 0.0
        var transferPenaltyWeighted = 0.0
        candidateIndices.forEach { index ->
            val journeys = demand.dailyJourneys[index].toDouble()
            considered += journeys
            val originStation = nearestStation[demand.origins[index]] ?: return@forEach
            val destinationStation = nearestStation[demand.destinations[index]] ?: return@forEach
            if (originStation.id == destinationStation.id) {
                served += journeys
                timeBenefitWeighted += journeys
                directnessWeighted += journeys
                return@forEach
            }
            val key = originStation.id to destinationStation.id
            val result =
                routeCache.getOrPut(key) {
                    router.route(JourneyRequest(originStation.id, destinationStation.id, RouteAlgorithm.DIJKSTRA))
                }
            if (!result.reachable || result.totalMinutes == null) return@forEach
            served += journeys
            val referenceMinutes = demand.distanceMeters[index] / 1000.0 / settings.surfaceReferenceSpeedKph * 60.0 + 5.0
            val benefit = ((referenceMinutes - result.totalMinutes) / referenceMinutes).coerceIn(0.0, 1.0)
            timeBenefitWeighted += benefit * journeys
            val routedDistance =
                result.legs
                    .mapNotNull { leg -> project.network.lines.flatMap { it.segments }.firstOrNull { it.id == leg.segmentId } }
                    .sumOf { it.lengthMeters }
            val directness =
                if (routedDistance <= 0.0) 1.0 else (demand.distanceMeters[index] / routedDistance).coerceIn(0.0, 1.0)
            directnessWeighted += directness * journeys
            transferPenaltyWeighted += (result.transferMinutes / max(result.totalMinutes, 1.0)).coerceIn(0.0, 1.0) * journeys
            val peakJourneys = journeys * settings.peakHourShare
            result.legs.filter { it.kind == "RIDE" && it.segmentId != null }.forEach { leg ->
                segmentLoads[leg.segmentId!!] = (segmentLoads[leg.segmentId] ?: 0.0) + peakJourneys
            }
        }

        val sampleCoverage = if (considered > 0.0) served / considered else 0.0
        val demandServedScore = sampleCoverage * 100.0
        val timeScore = if (served > 0.0) timeBenefitWeighted / served * 100.0 else 0.0
        val directnessScore = if (served > 0.0) directnessWeighted / served * 100.0 else 0.0
        val transferQuality = if (served > 0.0) (1.0 - transferPenaltyWeighted / served) * 100.0 else 0.0
        val connectivityScore = (directnessScore * 0.6 + transferQuality * 0.4).coerceIn(0.0, 100.0)

        var capacityPassengerKm = 0.0
        var overflowPassengerKm = 0.0
        val segmentById = project.network.lines.flatMap { it.segments }.associateBy { it.id }
        val lineBySegment = project.network.lines.flatMap { line -> line.segments.map { it.id to line } }.toMap()
        segmentLoads.forEach { (segmentId, load) ->
            val segment = segmentById[segmentId] ?: return@forEach
            val line = lineBySegment[segmentId] ?: return@forEach
            val capacity = line.trainsPerHour * line.vehicleCapacity.toDouble()
            capacityPassengerKm += load * segment.lengthMeters
            overflowPassengerKm += max(0.0, load - capacity) * segment.lengthMeters
        }
        val capacityScore =
            if (capacityPassengerKm <= 0.0) 100.0 else (1.0 - overflowPassengerKm / capacityPassengerKm).coerceIn(0.0, 1.0) * 100.0

        val equivalentTrackKm =
            project.network.lines
                .flatMap { it.segments }
                .distinctBy { it.id }
                .sumOf { segment ->
                    segment.lengthMeters / 1000.0 * if (segment.infrastructure == InfrastructureType.UNDERGROUND) 3.0 else 1.0
                }
        val servedPerEquivalentKm = if (equivalentTrackKm > 0.0) served / equivalentTrackKm else 0.0
        val constructionScore = (servedPerEquivalentKm / 10_000.0 * 100.0).coerceIn(0.0, 100.0)
        val interchangeCount = project.network.stations.count { station -> project.network.lines.count { station.id in it.stationIds } > 1 }
        val resilienceScore =
            if (project.network.lines.isEmpty()) 0.0 else min(100.0, 35.0 + interchangeCount * 12.0 + (project.network.lines.size - 1) * 5.0)

        val components =
            listOf(
                contribution("demand", "Demand served", sampleCoverage, demandServedScore, 0.35, "%", "OD journeys with both ends in catchment and a valid route"),
                contribution("time", "Journey-time benefit", timeBenefitWeighted / max(served, 1.0), timeScore, 0.20, "%", "Passenger-weighted saving against the disclosed surface reference"),
                contribution("capacity", "Capacity adequacy", 1.0 - overflowPassengerKm / max(capacityPassengerKm, 1.0), capacityScore, 0.15, "%", "Peak passenger-distance carried within scheduled capacity"),
                contribution("connectivity", "Connectivity & directness", connectivityScore / 100.0, connectivityScore, 0.10, "%", "Route directness with transfer-time penalty"),
                contribution("construction", "Construction efficiency", servedPerEquivalentKm, constructionScore, 0.10, "journeys/equivalent km", "Underground kilometres count as three construction-equivalent kilometres"),
                contribution("resilience", "Resilience", resilienceScore / 100.0, resilienceScore, 0.10, "%", "Structural alternatives from lines and shared interchanges"),
            )
        val total = components.sumOf { it.score * it.weight }.coerceIn(0.0, 100.0)
        val issues = buildIssues(project, segmentLoads, segmentById, lineBySegment, sampleCoverage)
        return AnalysisResult(NetworkScore(total, components), issues, segmentLoads)
    }

    private fun buildIssues(
        project: PlannerProject,
        segmentLoads: Map<String, Double>,
        segmentById: Map<String, PlannerSegment>,
        lineBySegment: Map<String, PlannerLine>,
        coverage: Double,
    ): List<NetworkIssue> {
        val issues = mutableListOf<NetworkIssue>()
        if (coverage < 0.6) {
            issues +=
                NetworkIssue(
                    "demand-gap",
                    IssueSeverity.CRITICAL,
                    "UNSERVED_DEMAND",
                    "Large demand coverage gap",
                    "Only ${(coverage * 100).toInt()}% of sampled simulated journeys can use the network.",
                    "network",
                )
        }
        project.network.lines.forEach { line ->
            if (line.trainsPerHour < 6) {
                issues +=
                    NetworkIssue(
                        "frequency-${line.id}",
                        IssueSeverity.WARNING,
                        "LONG_WAIT",
                        "Long expected wait on ${line.name}",
                        "${line.trainsPerHour} tph gives an average initial wait of ${"%.1f".format(30.0 / max(line.trainsPerHour, 1))} minutes.",
                        line.id,
                    )
            }
        }
        segmentLoads.forEach { (segmentId, load) ->
            val segment = segmentById[segmentId] ?: return@forEach
            val line = lineBySegment[segmentId] ?: return@forEach
            val capacity = line.trainsPerHour * line.vehicleCapacity
            val from = project.network.stations.firstOrNull { it.id == segment.fromStationId }
            val to = project.network.stations.firstOrNull { it.id == segment.toStationId }
            if (load > capacity) {
                issues +=
                    NetworkIssue(
                        "capacity-$segmentId",
                        IssueSeverity.CRITICAL,
                        "OVERCROWDING",
                        "Peak load exceeds capacity",
                        "${load.toInt()} assigned journeys/hour exceed ${capacity.toInt()} places/hour on ${line.name}.",
                        segmentId,
                        lon = listOfNotNull(from?.lon, to?.lon).averageOrNull(),
                        lat = listOfNotNull(from?.lat, to?.lat).averageOrNull(),
                    )
            }
            if (segment.lengthMeters > 5_000.0) {
                issues +=
                    NetworkIssue(
                        "spacing-$segmentId",
                        IssueSeverity.WARNING,
                        "LONG_SEGMENT",
                        "Unusually long station spacing",
                        "The segment is ${"%.1f".format(segment.lengthMeters / 1000.0)} km long.",
                        segmentId,
                        lon = listOfNotNull(from?.lon, to?.lon).averageOrNull(),
                        lat = listOfNotNull(from?.lat, to?.lat).averageOrNull(),
                    )
            }
        }
        if (issues.isEmpty()) {
            issues +=
                NetworkIssue(
                    "no-critical-findings",
                    IssueSeverity.INFO,
                    "SUMMARY",
                    "No high-severity structural issues",
                    "The current deterministic checks found no crowding, spacing, wait, or coverage exceptions.",
                    "network",
                )
        }
        return issues.take(100)
    }

    private fun contribution(
        id: String,
        label: String,
        raw: Double,
        score: Double,
        weight: Double,
        unit: String,
        explanation: String,
    ) = MetricContribution(id, label, raw, score.coerceIn(0.0, 100.0), weight, unit, explanation)

    private fun generateDemand(project: PlannerProject): DemandMatrix {
        val area = project.studyArea
        val sourcePoints =
            if (area == null) {
                project.network.stations.map { GridPoint(it.lon, it.lat, max(it.demandValue, 1.0)) }
            } else {
                londonPopulationPoints.filter { pointInPolygon(it.lon, it.lat, area.coordinates) }
            }
        if (sourcePoints.size < 2) {
            return DemandMatrix(emptyList(), IntArray(0), IntArray(0), IntArray(0), DoubleArray(0), 0)
        }
        val zones = aggregateZones(sourcePoints, project.demandConfig.maxZones.coerceIn(2, 400))
        val n = zones.size
        if (n < 2) return DemandMatrix(zones, IntArray(0), IntArray(0), IntArray(0), DoubleArray(0), 0)
        val total = project.demandConfig.totalDailyJourneys.coerceIn(1_000, 10_000_000)
        val productions = normalizeTargets(zones.map { it.demandWeight }, total)
        val attractions =
            normalizeTargets(
                zones.map { zone ->
                    zone.demandWeight * (1.0 + project.demandConfig.ptalInfluence * (zone.ptalMultiplier - 1.0))
                },
                total,
            )
        val matrix = DoubleArray(n * n)
        val decayMeters = project.demandConfig.distanceDecayKm.coerceIn(0.5, 100.0) * 1000.0
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i == j) continue
                val distance = haversineMeters(zones[i].lon, zones[i].lat, zones[j].lon, zones[j].lat)
                matrix[i * n + j] = exp(-distance / decayMeters).coerceAtLeast(1e-12)
            }
        }
        repeat(60) {
            for (i in 0 until n) {
                var rowSum = 0.0
                for (j in 0 until n) rowSum += matrix[i * n + j]
                val scale = if (rowSum > 0.0) productions[i] / rowSum else 0.0
                for (j in 0 until n) matrix[i * n + j] *= scale
            }
            for (j in 0 until n) {
                var columnSum = 0.0
                for (i in 0 until n) columnSum += matrix[i * n + j]
                val scale = if (columnSum > 0.0) attractions[j] / columnSum else 0.0
                for (i in 0 until n) matrix[i * n + j] *= scale
            }
        }
        val rounded = IntArray(matrix.size) { floor(matrix[it]).toInt() }
        var remaining = total - rounded.sum()
        val fractions =
            matrix.indices
                .filter { matrix[it] > 0.0 }
                .sortedByDescending { matrix[it] - floor(matrix[it]) }
        var cursor = 0
        while (remaining > 0 && fractions.isNotEmpty()) {
            rounded[fractions[cursor % fractions.size]]++
            remaining--
            cursor++
        }
        val recordCount = rounded.count { it > 0 }
        val origins = IntArray(recordCount)
        val destinations = IntArray(recordCount)
        val journeys = IntArray(recordCount)
        val distances = DoubleArray(recordCount)
        var output = 0
        for (i in 0 until n) {
            for (j in 0 until n) {
                val count = rounded[i * n + j]
                if (count <= 0) continue
                origins[output] = i
                destinations[output] = j
                journeys[output] = count
                distances[output] = haversineMeters(zones[i].lon, zones[i].lat, zones[j].lon, zones[j].lat)
                output++
            }
        }
        return DemandMatrix(zones, origins, destinations, journeys, distances, journeys.sum())
    }

    private fun aggregateZones(
        points: List<GridPoint>,
        maxZones: Int,
    ): List<DemandZone> {
        val minLon = points.minOf { it.lon }
        val maxLon = points.maxOf { it.lon }
        val minLat = points.minOf { it.lat }
        val maxLat = points.maxOf { it.lat }
        val side = ceil(sqrt(maxZones.toDouble())).toInt().coerceAtLeast(1)
        val lonStep = ((maxLon - minLon) / side).coerceAtLeast(1e-6)
        val latStep = ((maxLat - minLat) / side).coerceAtLeast(1e-6)
        data class Accumulator(var weight: Double = 0.0, var lon: Double = 0.0, var lat: Double = 0.0)
        val groups = linkedMapOf<Pair<Int, Int>, Accumulator>()
        points.forEach { point ->
            val x = min(side - 1, floor((point.lon - minLon) / lonStep).toInt())
            val y = min(side - 1, floor((point.lat - minLat) / latStep).toInt())
            val acc = groups.getOrPut(x to y) { Accumulator() }
            val weight = max(point.value, 1.0)
            acc.weight += weight
            acc.lon += point.lon * weight
            acc.lat += point.lat * weight
        }
        return groups.values.mapIndexed { index, acc ->
            val lon = acc.lon / acc.weight
            val lat = acc.lat / acc.weight
            DemandZone(
                id = "DZ-${(index + 1).toString().padStart(3, '0')}",
                lon = lon,
                lat = lat,
                demandWeight = acc.weight,
                ptalMultiplier = PtalLookup.demandWeight(lon, lat),
            )
        }
    }

    private fun normalizeTargets(
        weights: List<Double>,
        total: Int,
    ): DoubleArray {
        val sum = weights.sum().coerceAtLeast(1e-12)
        return DoubleArray(weights.size) { weights[it] / sum * total }
    }

    private fun convertGeneratedNetwork(lines: List<Line>): PlannerNetwork {
        val stations =
            lines.flatMap { it.stations }
                .groupBy { it.id }
                .map { (id, group) ->
                    val totalWeight = group.sumOf { max(it.catchmentPopulation, 1.0) }
                    PlannerStation(
                        id = id,
                        name = id.replace('_', ' '),
                        lon = group.sumOf { it.lon * max(it.catchmentPopulation, 1.0) } / totalWeight,
                        lat = group.sumOf { it.lat * max(it.catchmentPopulation, 1.0) } / totalWeight,
                        demandValue = group.sumOf { it.catchmentPopulation },
                    )
                }
        val stationById = stations.associateBy { it.id }
        val stationDemand = stations.map { it.demandValue }.sorted()
        val threshold = stationDemand.getOrElse((stationDemand.size * 0.75).toInt().coerceAtMost(stationDemand.lastIndex)) { 0.0 }
        val palette = listOf("#0b66d4", "#c2415d", "#14836f", "#7b52ab", "#d97706", "#3746a5", "#8b5e34", "#1677a3")
        val converted =
            lines.mapIndexed { lineIndex, line ->
                val stationIds = line.stations.map { it.id }
                val segmentPairs =
                    buildList {
                        addAll(stationIds.zipWithNext())
                        if (line.isLoop && stationIds.size > 2) add(stationIds.last() to stationIds.first())
                    }
                val segments =
                    segmentPairs.mapIndexed { index, (fromId, toId) ->
                        val from = stationById.getValue(fromId)
                        val to = stationById.getValue(toId)
                        val infrastructure =
                            if ((from.demandValue + to.demandValue) / 2.0 >= threshold) {
                                InfrastructureType.UNDERGROUND
                            } else {
                                InfrastructureType.OVERGROUND
                            }
                        PlannerSegment(
                            id = "${line.id}-S${index + 1}",
                            fromStationId = fromId,
                            toStationId = toId,
                            infrastructure = infrastructure,
                            lengthMeters = haversineMeters(from.lon, from.lat, to.lon, to.lat),
                        )
                    }
                PlannerLine(
                    id = line.id,
                    name = line.id.replace('_', ' '),
                    color = palette[lineIndex % palette.size],
                    role =
                        when (line.type) {
                            LineType.RADIAL_TRUNK -> PlanningRole.RADIAL
                            LineType.ORBITAL -> PlanningRole.ORBITAL_BYPASS
                            LineType.CORE_DISTRIBUTOR -> PlanningRole.CORE_DISTRIBUTOR
                            LineType.NOT_METRO -> PlanningRole.CROSS_CITY_TRUNK
                        },
                    isLoop = line.isLoop,
                    trainsPerHour = line.trainsPerHour.coerceIn(2, 40),
                    stationIds = stationIds,
                    segments = segments,
                )
            }
        return PlannerNetwork(stations, converted)
    }

    private fun validateProject(project: PlannerProject) {
        if (project.schemaVersion != 1) throw PlannerValidationException("Unsupported project schema ${project.schemaVersion}")
        project.studyArea?.let(::validateStudyArea)
        val stationIds = project.network.stations.map { it.id }
        if (stationIds.size != stationIds.distinct().size) throw PlannerValidationException("Station IDs must be unique")
        project.network.stations.forEach { station ->
            if (!pointInCoverage(station.lon, station.lat)) {
                throw PlannerValidationException("Station '${station.name}' lies outside supported Greater London coverage")
            }
        }
        project.network.lines.forEach { line ->
            if (line.trainsPerHour !in 1..60) throw PlannerValidationException("${line.name} must run between 1 and 60 tph")
            if (line.stationIds.size < 2) throw PlannerValidationException("${line.name} requires at least two stations")
            if (line.stationIds.any { it !in stationIds }) throw PlannerValidationException("${line.name} references an unknown station")
        }
        validateDisruptions(project, project.disruptions)
    }

    private fun validateDisruptions(
        project: PlannerProject,
        disruptions: List<DisruptionScenario>,
    ) {
        val segments = project.network.lines.flatMap { it.segments }.associateBy { it.id }
        disruptions.forEach { scenario ->
            if (scenario.durationMinutes <= 0.0) throw PlannerValidationException("Disruption duration must be positive")
            if (scenario.type == DisruptionType.ADVERSE_WEATHER) {
                val segment = segments[scenario.targetId]
                    ?: throw PlannerValidationException("Weather must target an existing overground segment")
                if (segment.infrastructure != InfrastructureType.OVERGROUND) {
                    throw PlannerValidationException("Weather cannot affect underground segment '${segment.id}'", "INVALID_WEATHER_TARGET")
                }
            }
        }
    }

    private fun validateStudyArea(area: StudyArea) {
        if (area.coordinates.size < 3) throw PlannerValidationException("A study area requires at least three coordinates")
        area.coordinates.forEach { coordinate ->
            if (!pointInCoverage(coordinate.lon, coordinate.lat)) {
                throw PlannerValidationException("Study-area coordinates must remain inside supported Greater London coverage")
            }
        }
    }

    private fun session(sessionId: String): AnalysisSession =
        synchronized(sessions) { sessions[sessionId] } ?: throw MissingSessionException(sessionId)

    private fun fingerprint(project: PlannerProject): String {
        val bytes = json.encodeToString(project).toByteArray()
        return MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    }

    private fun pointInCoverage(
        lon: Double,
        lat: Double,
    ): Boolean = coverageAsset.polygons.any { pointInPolygon(lon, lat, it) }

    private fun loadCoverageAsset(): CoverageAsset {
        val stream = PlannerService::class.java.getResourceAsStream("/london_coverage.json")
            ?: throw IllegalStateException("london_coverage.json is missing")
        return stream.bufferedReader().use { json.decodeFromString<CoverageAsset>(it.readText()) }
    }

    private fun loadLondonPopulationPoints(): List<GridPoint> {
        val stream = PlannerService::class.java.getResourceAsStream("/gbr_pd_2020_1km_ASCII_XYZ.csv")
            ?: throw IllegalStateException("Population grid resource is missing")
        val points = mutableListOf<GridPoint>()
        BufferedReader(InputStreamReader(stream)).use { reader ->
            reader.readLine()
            while (true) {
                val line = reader.readLine() ?: break
                val parts = line.split(',')
                if (parts.size < 3) continue
                val lon = parts[0].toDoubleOrNull() ?: continue
                val lat = parts[1].toDoubleOrNull() ?: continue
                val value = parts[2].toDoubleOrNull() ?: continue
                if (lon !in -0.55..0.35 || lat !in 51.25..51.75) continue
                if (pointInCoverage(lon, lat)) points += GridPoint(lon, lat, value)
            }
        }
        return points
    }

    companion object {
        fun pointInPolygon(
            lon: Double,
            lat: Double,
            polygon: List<Coordinate>,
        ): Boolean {
            if (polygon.size < 3) return false
            var inside = false
            var previous = polygon.last()
            for (current in polygon) {
                val intersects =
                    ((current.lat > lat) != (previous.lat > lat)) &&
                        (lon < (previous.lon - current.lon) * (lat - current.lat) / (previous.lat - current.lat) + current.lon)
                if (intersects) inside = !inside
                previous = current
            }
            return inside
        }

        private fun List<Double>.averageOrNull(): Double? = if (isEmpty()) null else average()
    }
}
