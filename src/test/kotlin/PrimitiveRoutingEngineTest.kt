package org.lsoffice

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PrimitiveRoutingEngineTest {
    private val settings = ModelSettings(dwellMinutes = 0.0, transferWalkMinutes = 3.0)

    @Test
    fun `dijkstra and a star return the same generalized time`() {
        val engine = PrimitiveRoutingEngine(twoLineNetwork(), settings)
        val dijkstra = engine.route(JourneyRequest("A", "C", RouteAlgorithm.DIJKSTRA))
        val aStar = engine.route(JourneyRequest("A", "C", RouteAlgorithm.ASTAR))

        assertTrue(dijkstra.reachable)
        assertTrue(aStar.reachable)
        assertEquals(dijkstra.totalMinutes!!, aStar.totalMinutes!!, 1e-9)
        assertEquals(5.0, dijkstra.waitMinutes, 1e-9)
        assertEquals(3.0, dijkstra.transferMinutes, 1e-9)
        assertTrue(aStar.visitedStates <= dijkstra.visitedStates)
    }

    @Test
    fun `track incident makes the only path unreachable`() {
        val engine = PrimitiveRoutingEngine(twoLineNetwork(), settings)
        val result =
            engine.route(
                JourneyRequest(
                    "A",
                    "C",
                    RouteAlgorithm.DIJKSTRA,
                    listOf(DisruptionScenario("incident", DisruptionType.TRACK_INCIDENT, "L2-S1")),
                ),
            )

        assertFalse(result.reachable)
    }

    @Test
    fun `weather changes overground travel but not underground travel`() {
        val network = twoLineNetwork()
        val engine = PrimitiveRoutingEngine(network, settings)
        val baseUnderground = engine.segmentTravelMinutes(network.lines[0].segments[0])
        val weatherUnderground =
            engine.segmentTravelMinutes(
                network.lines[0].segments[0],
                listOf(DisruptionScenario("weather-u", DisruptionType.ADVERSE_WEATHER, "L1-S1", severity = 1.0)),
            )
        val baseOverground = engine.segmentTravelMinutes(network.lines[1].segments[0])
        val weatherOverground =
            engine.segmentTravelMinutes(
                network.lines[1].segments[0],
                listOf(DisruptionScenario("weather-o", DisruptionType.ADVERSE_WEATHER, "L2-S1", severity = 1.0)),
            )

        assertEquals(baseUnderground, weatherUnderground, 1e-9)
        assertTrue(abs(weatherOverground - baseOverground * 2.0) < 1e-9)
    }

    private fun twoLineNetwork(): PlannerNetwork {
        val stations =
            listOf(
                PlannerStation("A", "Alpha", -0.20, 51.50),
                PlannerStation("B", "Bravo", -0.10, 51.50),
                PlannerStation("C", "Charlie", 0.00, 51.50),
            )
        return PlannerNetwork(
            stations,
            listOf(
                PlannerLine(
                    "L1",
                    "West line",
                    "#0b66d4",
                    PlanningRole.RADIAL,
                    trainsPerHour = 12,
                    stationIds = listOf("A", "B"),
                    segments = listOf(PlannerSegment("L1-S1", "A", "B", InfrastructureType.UNDERGROUND, 4_000.0)),
                ),
                PlannerLine(
                    "L2",
                    "East line",
                    "#14836f",
                    PlanningRole.CROSS_CITY_TRUNK,
                    trainsPerHour = 12,
                    stationIds = listOf("B", "C"),
                    segments = listOf(PlannerSegment("L2-S1", "B", "C", InfrastructureType.OVERGROUND, 4_500.0)),
                ),
            ),
        )
    }
}
