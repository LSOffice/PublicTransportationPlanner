package org.lsoffice

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class Coordinate(
    val lon: Double,
    val lat: Double,
)

@Serializable
data class StudyArea(
    val coordinates: List<Coordinate>,
)

@Serializable
enum class PlanningRole {
    RADIAL,
    CROSS_CITY_TRUNK,
    ORBITAL_BYPASS,
    CORE_DISTRIBUTOR,
}

@Serializable
enum class InfrastructureType {
    UNDERGROUND,
    OVERGROUND,
}

@Serializable
data class PlannerStation(
    val id: String,
    val name: String,
    val lon: Double,
    val lat: Double,
    val demandValue: Double = 0.0,
)

@Serializable
data class PlannerSegment(
    val id: String,
    val fromStationId: String,
    val toStationId: String,
    val infrastructure: InfrastructureType,
    val lengthMeters: Double,
)

@Serializable
data class PlannerLine(
    val id: String,
    val name: String,
    val color: String,
    val role: PlanningRole,
    val isLoop: Boolean = false,
    val trainsPerHour: Int = 12,
    val vehicleCapacity: Int = 850,
    val stationIds: List<String>,
    val segments: List<PlannerSegment>,
)

@Serializable
data class PlannerNetwork(
    val stations: List<PlannerStation> = emptyList(),
    val lines: List<PlannerLine> = emptyList(),
)

@Serializable
data class DemandConfig(
    val totalDailyJourneys: Int = 500_000,
    val distanceDecayKm: Double = 8.0,
    val maxZones: Int = 400,
    val ptalInfluence: Double = 0.25,
    val modelVersion: String = "gravity-ipf-v1",
)

@Serializable
data class ModelSettings(
    val stationCatchmentMeters: Double = 800.0,
    val undergroundSpeedKph: Double = 40.0,
    val overgroundSpeedKph: Double = 45.0,
    val dwellMinutes: Double = 0.5,
    val transferWalkMinutes: Double = 3.0,
    val surfaceReferenceSpeedKph: Double = 20.0,
    val peakHourShare: Double = 0.10,
)

@Serializable
data class DisruptionScenario(
    val id: String,
    val type: DisruptionType,
    val targetId: String,
    val startMinute: Double = 0.0,
    val durationMinutes: Double = 30.0,
    val severity: Double = 0.5,
    val label: String = "",
)

@Serializable
enum class DisruptionType {
    TRAIN_CANCELLATION,
    TRACK_INCIDENT,
    ADVERSE_WEATHER,
    SIGNAL_FAILURE,
    STATION_CLOSURE,
}

@Serializable
data class PlannerProject(
    val schemaVersion: Int = 1,
    val id: String,
    val name: String,
    val revision: Long,
    val createdAt: String,
    val updatedAt: String,
    val studyArea: StudyArea? = null,
    val demandConfig: DemandConfig = DemandConfig(),
    val modelSettings: ModelSettings = ModelSettings(),
    val network: PlannerNetwork = PlannerNetwork(),
    val disruptions: List<DisruptionScenario> = emptyList(),
)

@Serializable
data class CoverageSource(
    val id: String,
    val title: String,
    val url: String,
    val licence: String,
    val attribution: String,
)

@Serializable
data class CoverageResponse(
    val hardSupport: List<List<Coordinate>>,
    val ptalSupport: List<List<Coordinate>>,
    val bounds: List<Double>,
    val populationGridCellsInLondon: Int,
    val ptalRecords: Int,
    val sources: List<CoverageSource>,
)

@Serializable
data class GenerateNetworkRequest(
    val studyArea: StudyArea,
    val maxTrunkLines: Int = 4,
)

@Serializable
data class GenerateNetworkResponse(
    val network: PlannerNetwork,
    val selectedGridPoints: Int,
    val warnings: List<String> = emptyList(),
)

@Serializable
data class DensityResponse(
    val lon: Double,
    val lat: Double,
    val value: Double,
    @SerialName("distance_m") val distanceMeters: Double,
)

@Serializable
data class AnalysisSessionRequest(
    val project: PlannerProject,
)

@Serializable
data class MetricContribution(
    val id: String,
    val label: String,
    val rawValue: Double,
    val score: Double,
    val weight: Double,
    val unit: String,
    val explanation: String,
)

@Serializable
data class NetworkScore(
    val total: Double,
    val components: List<MetricContribution>,
    val methodology: String = "planner-score-v1",
)

@Serializable
data class DemandZone(
    val id: String,
    val lon: Double,
    val lat: Double,
    val demandWeight: Double,
    val ptalMultiplier: Double,
)

@Serializable
data class DemandRecord(
    val originZoneId: String,
    val destinationZoneId: String,
    val dailyJourneys: Int,
    val distanceMeters: Double,
)

@Serializable
data class AnalysisSessionResponse(
    val sessionId: String,
    val projectId: String,
    val revision: Long,
    val fingerprint: String,
    val score: NetworkScore,
    val zones: List<DemandZone>,
    val demandRecordCount: Int,
    val totalDailyJourneys: Int,
    val issues: List<NetworkIssue>,
)

@Serializable
data class DemandPage(
    val records: List<DemandRecord>,
    val offset: Int,
    val limit: Int,
    val total: Int,
    val totalDailyJourneys: Int,
)

@Serializable
enum class IssueSeverity {
    INFO,
    WARNING,
    CRITICAL,
}

@Serializable
data class NetworkIssue(
    val id: String,
    val severity: IssueSeverity,
    val type: String,
    val title: String,
    val detail: String,
    val targetId: String,
    val lon: Double? = null,
    val lat: Double? = null,
)

@Serializable
enum class RouteAlgorithm {
    DIJKSTRA,
    ASTAR,
}

@Serializable
data class JourneyRequest(
    val fromStationId: String,
    val toStationId: String,
    val algorithm: RouteAlgorithm = RouteAlgorithm.ASTAR,
    val disruptions: List<DisruptionScenario> = emptyList(),
)

@Serializable
data class JourneyLeg(
    val fromStationId: String,
    val toStationId: String,
    val lineId: String,
    val segmentId: String? = null,
    val minutes: Double,
    val kind: String,
)

@Serializable
data class JourneyResult(
    val reachable: Boolean,
    val algorithm: RouteAlgorithm,
    val totalMinutes: Double? = null,
    val waitMinutes: Double = 0.0,
    val inVehicleMinutes: Double = 0.0,
    val transferMinutes: Double = 0.0,
    val visitedStates: Int = 0,
    val runtimeMicros: Long = 0,
    val stationIds: List<String> = emptyList(),
    val legs: List<JourneyLeg> = emptyList(),
    val message: String? = null,
)

@Serializable
data class SimulationRequest(
    val horizonMinutes: Double = 120.0,
    val disruptions: List<DisruptionScenario> = emptyList(),
)

@Serializable
data class TrainRun(
    val id: String,
    val lineId: String,
    val direction: Int,
    val departureMinute: Double,
    val stationIds: List<String>,
    val cumulativeMinutes: List<Double>,
    val cancelled: Boolean,
)

@Serializable
data class SimulationResponse(
    val horizonMinutes: Double,
    val trains: List<TrainRun>,
    val activeDisruptions: List<DisruptionScenario>,
)

@Serializable
data class ApiError(
    val code: String,
    val message: String,
    val details: Map<String, String> = emptyMap(),
)

@Serializable
data class LegacyStationResponse(
    val id: String,
    val lon: Double,
    val lat: Double,
    val value: Double,
)

@Serializable
data class LegacyLineResponse(
    val id: String,
    val type: String,
    val isLoop: Boolean,
    @SerialName("length_m") val lengthMeters: Double,
    val cost: Double,
    @SerialName("trains_per_hour") val trainsPerHour: Int,
    val stations: List<LegacyStationResponse>,
)

@Serializable
data class LegacySuggestionsResponse(
    val lines: List<LegacyLineResponse>,
)
