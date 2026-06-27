package org.lsoffice

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class MetroBuildPlanningTest {
    private val builder =
        MetroBuilder(
            BuilderParams(
                capitalBudget = 1_000_000_000.0,
                operatingBudgetPerYear = 50_000_000.0,
            ),
            debug = false,
        )

    private fun routeLength(stations: List<Station>): Double =
        stations.zipWithNext().sumOf { (a, b) -> haversineMeters(a.lon, a.lat, b.lon, b.lat) }

    @Test
    fun consolidatesNearbyCrossLineStationsIntoSharedInterchange() {
        val stationsA =
            listOf(
                Station("a1", -0.1500, 51.5000, 100.0),
                Station("a2", -0.1000, 51.5000, 200.0),
                Station("a3", -0.0500, 51.5000, 100.0),
            )
        val stationsB =
            listOf(
                Station("b1", -0.1000, 51.4500, 100.0),
                Station("b2", -0.1004, 51.5002, 300.0),
                Station("b3", -0.1000, 51.5500, 100.0),
            )
        val lineA =
            Line(
                "RT1",
                stationsA,
                routeLength(stationsA),
                0.0,
            )
        val lineB =
            Line(
                "RT2",
                stationsB,
                routeLength(stationsB),
                0.0,
            )

        val result = builder.consolidateStationClusters(listOf(lineA, lineB), minStationsPerLine = 3)

        assertEquals(result[0].stations[1].id, result[1].stations[1].id)
        assertTrue(result[0].stations[1].id.startsWith("INT_"))
    }

    @Test
    fun rejectsClusterThatWouldMergeTwoStopsOnTheSameLine() {
        val stationsA =
            listOf(
                Station("a1", -0.1500, 51.5000, 100.0),
                Station("a2", -0.1000, 51.5000, 200.0),
                Station("a3", -0.0995, 51.5002, 200.0),
            )
        val stationsB =
            listOf(
                Station("b1", -0.1000, 51.4500, 100.0),
                Station("b2", -0.1002, 51.5001, 300.0),
                Station("b3", -0.1000, 51.5500, 100.0),
            )
        val lineA =
            Line(
                "RT1",
                stationsA,
                routeLength(stationsA),
                0.0,
            )
        val lineB =
            Line(
                "RT2",
                stationsB,
                routeLength(stationsB),
                0.0,
            )

        val result = builder.consolidateStationClusters(listOf(lineA, lineB), minStationsPerLine = 3)

        assertNotEquals(result[0].stations[1].id, result[1].stations[1].id)
        assertNotEquals(result[0].stations[2].id, result[1].stations[1].id)
    }

    @Test
    fun rejectsNearbyClusterThatWouldDragAStationTooFar() {
        val stationsA =
            listOf(
                Station("a1", -0.1500, 51.5000, 100.0),
                Station("a2", -0.1000, 51.5000, 50.0),
                Station("a3", -0.0500, 51.5000, 100.0),
            )
        val stationsB =
            listOf(
                Station("b1", -0.1000, 51.4600, 100.0),
                Station("b2", -0.1000, 51.5072, 900.0),
                Station("b3", -0.1000, 51.5500, 100.0),
            )
        val lineA =
            Line(
                "RT1",
                stationsA,
                routeLength(stationsA),
                0.0,
            )
        val lineB =
            Line(
                "RT2",
                stationsB,
                routeLength(stationsB),
                0.0,
            )

        val result = builder.consolidateStationClusters(listOf(lineA, lineB), minStationsPerLine = 3)

        assertNotEquals(result[0].stations[1].id, result[1].stations[1].id)
    }

    @Test
    fun classifiesCentralRingAndOuterSegmentsByMostEconomicalTechnology() {
        val centerLon = -0.1278
        val centerLat = 51.5074
        val lines =
            listOf(
                Line(
                    "CD1",
                    listOf(
                        Station("c1", centerLon, centerLat, 1000.0),
                        Station("c2", centerLon + 0.005, centerLat, 1000.0),
                    ),
                    1.0,
                    0.0,
                    LineType.CORE_DISTRIBUTOR,
                ),
                Line(
                    "RT1",
                    listOf(
                        Station("r1", centerLon + 0.160, centerLat, 100.0),
                        Station("r2", centerLon + 0.170, centerLat, 100.0),
                    ),
                    1.0,
                    0.0,
                    LineType.RADIAL_TRUNK,
                ),
                Line(
                    "RT2",
                    listOf(
                        Station("o1", centerLon + 0.300, centerLat, 100.0),
                        Station("o2", centerLon + 0.310, centerLat, 100.0),
                    ),
                    1.0,
                    0.0,
                    LineType.RADIAL_TRUNK,
                ),
            )

        val estimates =
            builder.estimateBuildTechnology(
                lines,
                centerLon = centerLon,
                centerLat = centerLat,
                coreRadiusMeters = 3000.0,
                ringRadiusMeters = 12_000.0,
            )

        assertEquals(AlignmentTechnology.DEEP_BORE_TUNNEL, estimates[0].buildEstimate?.segments?.single()?.technology)
        assertEquals(AlignmentTechnology.SUBSURFACE_TUNNEL, estimates[1].buildEstimate?.segments?.single()?.technology)
        assertEquals(AlignmentTechnology.SURFACE_OR_ELEVATED, estimates[2].buildEstimate?.segments?.single()?.technology)
        assertTrue((estimates[0].buildEstimate?.totalCost ?: 0.0) > (estimates[0].buildEstimate?.segments?.sumOf { it.civilCost } ?: 0.0))
    }
}
