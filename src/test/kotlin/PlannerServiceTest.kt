package org.lsoffice

import kotlinx.serialization.json.Json
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PlannerServiceTest {
    private val json = Json { encodeDefaults = true }
    private val service = PlannerService(json)

    @Test
    fun `synthetic demand preserves the configured journey total`() {
        val response = service.createSession(AnalysisSessionRequest(project()))

        assertEquals(12_345, response.totalDailyJourneys)
        assertTrue(response.demandRecordCount > 0)
        assertEquals(100.0, response.score.components.sumOf { it.weight } * 100.0, 1e-9)
    }

    @Test
    fun `weather targeting underground infrastructure is rejected`() {
        val project =
            project().copy(
                disruptions =
                    listOf(
                        DisruptionScenario(
                            id = "invalid-weather",
                            type = DisruptionType.ADVERSE_WEATHER,
                            targetId = "L1-S1",
                        ),
                    ),
            )

        val error = assertFailsWith<PlannerValidationException> { service.createSession(AnalysisSessionRequest(project)) }
        assertEquals("INVALID_WEATHER_TARGET", error.code)
    }

    @Test
    fun `coverage polygon contains central London and excludes Birmingham`() {
        assertTrue(PlannerService.pointInPolygon(-0.1278, 51.5074, service.coverage().hardSupport.first()))
        assertTrue(service.coverage().hardSupport.none { PlannerService.pointInPolygon(-1.8904, 52.4862, it) })
    }

    private fun project(): PlannerProject {
        val stations =
            listOf(
                PlannerStation("A", "Alpha", -0.20, 51.50, 100.0),
                PlannerStation("B", "Bravo", -0.10, 51.50, 120.0),
                PlannerStation("C", "Charlie", 0.00, 51.50, 90.0),
            )
        val line =
            PlannerLine(
                id = "L1",
                name = "Test line",
                color = "#0b66d4",
                role = PlanningRole.CROSS_CITY_TRUNK,
                trainsPerHour = 12,
                stationIds = listOf("A", "B", "C"),
                segments =
                    listOf(
                        PlannerSegment("L1-S1", "A", "B", InfrastructureType.UNDERGROUND, 4_000.0),
                        PlannerSegment("L1-S2", "B", "C", InfrastructureType.OVERGROUND, 4_500.0),
                    ),
            )
        return PlannerProject(
            id = "project-test",
            name = "Test project",
            revision = 1,
            createdAt = "2026-10-03T00:00:00Z",
            updatedAt = "2026-10-03T00:00:00Z",
            demandConfig = DemandConfig(totalDailyJourneys = 12_345, maxZones = 10),
            network = PlannerNetwork(stations, listOf(line)),
        )
    }
}
