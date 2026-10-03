package org.lsoffice

import kotlin.math.max

/**
 * Immutable, primitive-array transit graph compiled from the editable project model.
 * A graph state is a station/line pair. Boarding and transfer waits are applied at
 * state entry, which keeps Dijkstra and A* on the same generalized-time model.
 */
class PrimitiveRoutingEngine(
    private val network: PlannerNetwork,
    private val settings: ModelSettings,
) {
    private data class State(
        val stationId: String,
        val lineIndex: Int,
    )

    private data class EdgeDraft(
        val from: Int,
        val to: Int,
        val baseMinutes: Double,
        val kind: Int,
        val lineIndex: Int,
        val segmentIndex: Int,
    )

    private val stationById = network.stations.associateBy { it.id }
    private val lineIndexById = network.lines.mapIndexed { index, line -> line.id to index }.toMap()
    private val segmentIndexById: Map<String, Int>
    private val segments: List<PlannerSegment>
    private val states: List<State>
    private val statesByStation: Map<String, IntArray>
    private val offsets: IntArray
    private val destinations: IntArray
    private val baseMinutes: DoubleArray
    private val edgeKinds: IntArray
    private val edgeLineIndices: IntArray
    private val edgeSegmentIndices: IntArray
    private val maxSpeedKph = max(settings.undergroundSpeedKph, settings.overgroundSpeedKph)

    init {
        val allSegments = mutableListOf<PlannerSegment>()
        network.lines.forEach { allSegments += it.segments }
        segments = allSegments
        segmentIndexById = segments.mapIndexed { index, segment -> segment.id to index }.toMap()

        val stateList = mutableListOf<State>()
        val stateIndex = mutableMapOf<Pair<String, Int>, Int>()
        network.lines.forEachIndexed { lineIndex, line ->
            line.stationIds.forEach { stationId ->
                val key = stationId to lineIndex
                if (key !in stateIndex && stationId in stationById) {
                    stateIndex[key] = stateList.size
                    stateList += State(stationId, lineIndex)
                }
            }
        }
        states = stateList
        statesByStation =
            states.indices
                .groupBy { states[it].stationId }
                .mapValues { (_, indices) -> indices.toIntArray() }

        val drafts = mutableListOf<EdgeDraft>()
        network.lines.forEachIndexed { lineIndex, line ->
            line.segments.forEach { segment ->
                val from = stateIndex[segment.fromStationId to lineIndex] ?: return@forEach
                val to = stateIndex[segment.toStationId to lineIndex] ?: return@forEach
                val segmentIndex = segmentIndexById.getValue(segment.id)
                val speed =
                    when (segment.infrastructure) {
                        InfrastructureType.UNDERGROUND -> settings.undergroundSpeedKph
                        InfrastructureType.OVERGROUND -> settings.overgroundSpeedKph
                    }.coerceAtLeast(1.0)
                val travelMinutes = segment.lengthMeters / 1000.0 / speed * 60.0 + settings.dwellMinutes
                drafts += EdgeDraft(from, to, travelMinutes, EDGE_RIDE, lineIndex, segmentIndex)
                drafts += EdgeDraft(to, from, travelMinutes, EDGE_RIDE, lineIndex, segmentIndex)
            }
        }

        statesByStation.values.forEach { stationStates ->
            for (from in stationStates) {
                for (to in stationStates) {
                    if (from == to) continue
                    drafts += EdgeDraft(from, to, settings.transferWalkMinutes, EDGE_TRANSFER, states[to].lineIndex, -1)
                }
            }
        }

        drafts.sortWith(compareBy<EdgeDraft> { it.from }.thenBy { it.to })
        offsets = IntArray(states.size + 1)
        drafts.forEach { offsets[it.from + 1]++ }
        for (index in 1 until offsets.size) offsets[index] += offsets[index - 1]
        destinations = IntArray(drafts.size)
        baseMinutes = DoubleArray(drafts.size)
        edgeKinds = IntArray(drafts.size)
        edgeLineIndices = IntArray(drafts.size)
        edgeSegmentIndices = IntArray(drafts.size)
        drafts.forEachIndexed { index, edge ->
            destinations[index] = edge.to
            baseMinutes[index] = edge.baseMinutes
            edgeKinds[index] = edge.kind
            edgeLineIndices[index] = edge.lineIndex
            edgeSegmentIndices[index] = edge.segmentIndex
        }
    }

    fun route(
        request: JourneyRequest,
    ): JourneyResult {
        val started = System.nanoTime()
        val originStates = statesByStation[request.fromStationId]
            ?: return unreachable(request, started, "Unknown origin station")
        val destinationStates = statesByStation[request.toStationId]
            ?: return unreachable(request, started, "Unknown destination station")
        if (request.fromStationId == request.toStationId) {
            return JourneyResult(
                reachable = true,
                algorithm = request.algorithm,
                totalMinutes = 0.0,
                stationIds = listOf(request.fromStationId),
                runtimeMicros = (System.nanoTime() - started) / 1_000,
            )
        }

        val disruptions = ActiveDisruptions(request.disruptions)
        if (request.fromStationId in disruptions.closedStations || request.toStationId in disruptions.closedStations) {
            return unreachable(request, started, "Origin or destination station is closed")
        }

        val destinationSet = BooleanArray(states.size)
        destinationStates.forEach { destinationSet[it] = true }
        val destinationStation = stationById.getValue(request.toStationId)
        val lineServesDestination = BooleanArray(network.lines.size)
        destinationStates.forEach { lineServesDestination[states[it].lineIndex] = true }

        val distance = DoubleArray(states.size) { Double.POSITIVE_INFINITY }
        val previousState = IntArray(states.size) { -1 }
        val previousEdge = IntArray(states.size) { -1 }
        val heap = PrimitiveMinHeap(max(16, destinations.size + states.size))

        originStates.forEach { state ->
            val lineIndex = states[state].lineIndex
            val wait = expectedWaitMinutes(lineIndex, disruptions)
            if (wait.isFinite()) {
                distance[state] = wait
                heap.push(state, wait + heuristic(state, destinationStation, lineServesDestination, disruptions, request.algorithm))
            }
        }

        var visited = 0
        var bestDestination = -1
        while (!heap.isEmpty()) {
            val item = heap.pop()
            val state = item.node
            val expectedPriority =
                distance[state] + heuristic(state, destinationStation, lineServesDestination, disruptions, request.algorithm)
            if (item.priority > expectedPriority + EPSILON) continue
            visited++
            if (destinationSet[state]) {
                bestDestination = state
                break
            }

            for (edge in offsets[state] until offsets[state + 1]) {
                val next = destinations[edge]
                val nextStationId = states[next].stationId
                if (nextStationId in disruptions.closedStations) continue
                val edgeMinutes = dynamicEdgeMinutes(edge, disruptions)
                if (!edgeMinutes.isFinite()) continue
                val candidate = distance[state] + edgeMinutes
                if (candidate + EPSILON < distance[next]) {
                    distance[next] = candidate
                    previousState[next] = state
                    previousEdge[next] = edge
                    heap.push(
                        next,
                        candidate + heuristic(next, destinationStation, lineServesDestination, disruptions, request.algorithm),
                    )
                }
            }
        }

        if (bestDestination < 0) return unreachable(request, started, "No route is available", visited)

        val statePath = mutableListOf<Int>()
        var cursor = bestDestination
        while (cursor >= 0) {
            statePath += cursor
            cursor = previousState[cursor]
        }
        statePath.reverse()

        val legs = mutableListOf<JourneyLeg>()
        val initialWait = expectedWaitMinutes(states[statePath.first()].lineIndex, disruptions)
        legs +=
            JourneyLeg(
                fromStationId = request.fromStationId,
                toStationId = request.fromStationId,
                lineId = network.lines[states[statePath.first()].lineIndex].id,
                minutes = initialWait,
                kind = "WAIT",
            )
        var inVehicle = 0.0
        var transfer = 0.0
        var transferWait = 0.0
        for (index in 1 until statePath.size) {
            val state = statePath[index]
            val edge = previousEdge[state]
            val from = statePath[index - 1]
            val line = network.lines[edgeLineIndices[edge]]
            val minutes = dynamicEdgeMinutes(edge, disruptions)
            if (edgeKinds[edge] == EDGE_TRANSFER) {
                val wait = expectedWaitMinutes(edgeLineIndices[edge], disruptions)
                transfer += settings.transferWalkMinutes
                transferWait += wait
                legs +=
                    JourneyLeg(
                        fromStationId = states[from].stationId,
                        toStationId = states[state].stationId,
                        lineId = line.id,
                        minutes = minutes,
                        kind = "TRANSFER",
                    )
            } else {
                inVehicle += minutes
                val segment = segments[edgeSegmentIndices[edge]]
                legs +=
                    JourneyLeg(
                        fromStationId = states[from].stationId,
                        toStationId = states[state].stationId,
                        lineId = line.id,
                        segmentId = segment.id,
                        minutes = minutes,
                        kind = "RIDE",
                    )
            }
        }

        return JourneyResult(
            reachable = true,
            algorithm = request.algorithm,
            totalMinutes = distance[bestDestination],
            waitMinutes = initialWait + transferWait,
            inVehicleMinutes = inVehicle,
            transferMinutes = transfer,
            visitedStates = visited,
            runtimeMicros = (System.nanoTime() - started) / 1_000,
            stationIds = statePath.map { states[it].stationId }.distinct(),
            legs = legs,
        )
    }

    fun segmentTravelMinutes(
        segment: PlannerSegment,
        disruptions: List<DisruptionScenario> = emptyList(),
    ): Double {
        val index = segmentIndexById[segment.id] ?: return Double.POSITIVE_INFINITY
        val edge = edgeSegmentIndices.indexOf(index)
        return if (edge >= 0) dynamicEdgeMinutes(edge, ActiveDisruptions(disruptions)) else Double.POSITIVE_INFINITY
    }

    private fun heuristic(
        stateIndex: Int,
        destination: PlannerStation,
        lineServesDestination: BooleanArray,
        disruptions: ActiveDisruptions,
        algorithm: RouteAlgorithm,
    ): Double {
        if (algorithm == RouteAlgorithm.DIJKSTRA) return 0.0
        val state = states[stateIndex]
        val station = stationById.getValue(state.stationId)
        val geographicLowerBound =
            haversineMeters(station.lon, station.lat, destination.lon, destination.lat) / 1000.0 / maxSpeedKph * 60.0
        val unavoidableWaitLowerBound =
            if (lineServesDestination[state.lineIndex]) {
                0.0
            } else {
                network.lines.indices
                    .asSequence()
                    .map { expectedWaitMinutes(it, disruptions) }
                    .filter { it.isFinite() }
                    .minOrNull() ?: 0.0
            }
        return geographicLowerBound + unavoidableWaitLowerBound
    }

    private fun dynamicEdgeMinutes(
        edge: Int,
        disruptions: ActiveDisruptions,
    ): Double {
        if (edgeKinds[edge] == EDGE_TRANSFER) {
            return baseMinutes[edge] + expectedWaitMinutes(edgeLineIndices[edge], disruptions)
        }
        val segment = segments[edgeSegmentIndices[edge]]
        if (segment.id in disruptions.blockedSegments) return Double.POSITIVE_INFINITY
        var multiplier = 1.0
        disruptions.signalSeverity[segment.id]?.let { multiplier *= 1.0 + it.coerceIn(0.0, 1.0) * 2.0 }
        if (segment.infrastructure == InfrastructureType.OVERGROUND) {
            disruptions.weatherSeverity[segment.id]?.let { multiplier *= 1.0 + it.coerceIn(0.0, 1.0) }
        }
        return baseMinutes[edge] * multiplier
    }

    private fun expectedWaitMinutes(
        lineIndex: Int,
        disruptions: ActiveDisruptions,
    ): Double {
        val line = network.lines[lineIndex]
        val cancellation = disruptions.lineCancellationSeverity[line.id] ?: 0.0
        val effectiveTph = line.trainsPerHour * (1.0 - cancellation.coerceIn(0.0, 0.95))
        return if (effectiveTph <= 0.0) Double.POSITIVE_INFINITY else 30.0 / effectiveTph
    }

    private fun unreachable(
        request: JourneyRequest,
        started: Long,
        message: String,
        visited: Int = 0,
    ) = JourneyResult(
        reachable = false,
        algorithm = request.algorithm,
        visitedStates = visited,
        runtimeMicros = (System.nanoTime() - started) / 1_000,
        message = message,
    )

    private inner class ActiveDisruptions(
        scenarios: List<DisruptionScenario>,
    ) {
        val closedStations = mutableSetOf<String>()
        val blockedSegments = mutableSetOf<String>()
        val weatherSeverity = mutableMapOf<String, Double>()
        val signalSeverity = mutableMapOf<String, Double>()
        val lineCancellationSeverity = mutableMapOf<String, Double>()

        init {
            scenarios.forEach { scenario ->
                when (scenario.type) {
                    DisruptionType.STATION_CLOSURE -> closedStations += scenario.targetId
                    DisruptionType.TRACK_INCIDENT -> blockedSegments += scenario.targetId
                    DisruptionType.ADVERSE_WEATHER -> {
                        val segment = segmentIndexById[scenario.targetId]?.let { segments[it] }
                        if (segment?.infrastructure == InfrastructureType.OVERGROUND) {
                            weatherSeverity[scenario.targetId] = scenario.severity
                        }
                    }
                    DisruptionType.SIGNAL_FAILURE -> signalSeverity[scenario.targetId] = scenario.severity
                    DisruptionType.TRAIN_CANCELLATION -> lineCancellationSeverity[scenario.targetId] = scenario.severity
                }
            }
        }
    }

    private class PrimitiveMinHeap(
        initialCapacity: Int,
    ) {
        private var nodes = IntArray(initialCapacity)
        private var priorities = DoubleArray(initialCapacity)
        private var size = 0

        data class Item(
            val node: Int,
            val priority: Double,
        )

        fun isEmpty(): Boolean = size == 0

        fun push(
            node: Int,
            priority: Double,
        ) {
            ensureCapacity()
            var index = size++
            while (index > 0) {
                val parent = (index - 1) ushr 1
                if (priorities[parent] <= priority) break
                nodes[index] = nodes[parent]
                priorities[index] = priorities[parent]
                index = parent
            }
            nodes[index] = node
            priorities[index] = priority
        }

        fun pop(): Item {
            val result = Item(nodes[0], priorities[0])
            val lastIndex = --size
            if (lastIndex == 0) return result
            val lastNode = nodes[lastIndex]
            val lastPriority = priorities[lastIndex]
            var index = 0
            while (true) {
                val left = index * 2 + 1
                if (left >= size) break
                val right = left + 1
                val child = if (right < size && priorities[right] < priorities[left]) right else left
                if (priorities[child] >= lastPriority) break
                nodes[index] = nodes[child]
                priorities[index] = priorities[child]
                index = child
            }
            nodes[index] = lastNode
            priorities[index] = lastPriority
            return result
        }

        private fun ensureCapacity() {
            if (size < nodes.size) return
            nodes = nodes.copyOf(nodes.size * 2)
            priorities = priorities.copyOf(priorities.size * 2)
        }
    }

    private companion object {
        const val EDGE_RIDE = 0
        const val EDGE_TRANSFER = 1
        const val EPSILON = 1e-9
    }
}
