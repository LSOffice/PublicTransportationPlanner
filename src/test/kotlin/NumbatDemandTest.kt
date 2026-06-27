package org.lsoffice

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NumbatDemandTest {
    @Test
    fun assignsCoordinatesToFixed500mLondonCells() {
        assertEquals(RegionCell(0, 0), LondonRegionGrid.cellFor(-0.1278, 51.5074))
        assertEquals(RegionCell(1, 0), LondonRegionGrid.cellFor(-0.1198, 51.5074))
        assertEquals(RegionCell(0, 1), LondonRegionGrid.cellFor(-0.1278, 51.5120))
    }

    @Test
    fun parsesDerivedStationLookupCsv() {
        val csv =
            """
            master_nlc,station_name,latitude,longitude
            500,"Acton Town",51.50270600,-0.27997061
            501,"Baker Street, LU",51.52288300,-0.15713000
            """.trimIndent()

        val stations = NumbatDemandLoader.loadStations(csv.byteInputStream())

        assertEquals(2, stations.size)
        assertEquals("Acton Town", stations[500]?.name)
        assertEquals("Baker Street, LU", stations[501]?.name)
        assertEquals(51.522883, stations[501]?.lat)
    }

    @Test
    fun appliesDayTypeWeightingAndAggregatesTimeBands() {
        val stations =
            """
            master_nlc,station_name,latitude,longitude
            100,Origin,51.50740000,-0.12780000
            101,Destination,51.50740000,-0.11980000
            """.trimIndent()
        val monday =
            """
            mnlc_o,mnlc_d,tb_o,vol
            100,101,2,10.0
            """.trimIndent()
        val typicalWeekday =
            """
            mnlc_o,mnlc_d,tb_o,vol
            100,101,2,2.0
            101,100,3,3.0
            """.trimIndent()

        val model =
            NumbatDemandLoader.load(
                stations.byteInputStream(),
                listOf(
                    NumbatDemandLoader.WeightedOdInput(monday.byteInputStream(), 1.0),
                    NumbatDemandLoader.WeightedOdInput(typicalWeekday.byteInputStream(), 3.0),
                ),
            )

        val originCell = LondonRegionGrid.cellFor(-0.1278, 51.5074)
        val destinationCell = LondonRegionGrid.cellFor(-0.1198, 51.5074)
        assertEquals(25.0, model.demandBetween(setOf(originCell), setOf(destinationCell)), 1e-9)
        assertEquals(3, model.stats.sourceRows)
        assertEquals(3, model.stats.matchedRows)
        assertEquals(1, model.stats.aggregatedPairs)
    }

    @Test
    fun skipsRowsWithUnknownStationCodesSafely() {
        val stations =
            """
            master_nlc,station_name,latitude,longitude
            100,Origin,51.50740000,-0.12780000
            101,Destination,51.50740000,-0.11980000
            """.trimIndent()
        val od =
            """
            mnlc_o,mnlc_d,tb_o,vol
            100,101,2,4.0
            100,999,2,99.0
            """.trimIndent()

        val model =
            NumbatDemandLoader.load(
                stations.byteInputStream(),
                listOf(NumbatDemandLoader.WeightedOdInput(od.byteInputStream(), 1.0)),
            )

        assertEquals(2, model.stats.sourceRows)
        assertEquals(1, model.stats.matchedRows)
        assertEquals(1, model.stats.skippedUnknownStations)
        assertTrue(model.stats.totalDemand < 99.0)
    }

    @Test
    fun gravityDemandIsUnchangedWhenObservedDemandIsAbsent() {
        assertEquals(
            42.0,
            combineGravityAndObservedDemand(
                gravityDemand = 42.0,
                observedDemand = 0.0,
                maxGravityDemand = 100.0,
                maxObservedDemand = 50.0,
            ),
        )
    }
}
