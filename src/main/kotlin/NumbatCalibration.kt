package org.lsoffice

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.security.MessageDigest
import kotlin.math.hypot

@Serializable
private data class SweepRow(
    val radialCap: Int,
    val orbitalCap: Int,
    val distributorCap: Int,
    val coverage: Double,
    val costEstimate: Double,
    val lineCount: Int,
    val transferOverhead: Double,
    val duplication: Double,
)

@Serializable
private data class CalibrationReport(
    val algorithmVersion: String,
    val datasetSha256: String,
    val studyArea: StudyArea,
    val guidanceStrength: Double,
    val sweep: List<SweepRow>,
    val chosenParetoKnee: SweepRow,
    val note: String,
)

/** Reproducible offline preview sweep. Run with ./gradlew calibrateNumbat. */
fun main(args: Array<String>) {
    val output = File(args.firstOrNull() ?: error("Output path required"))
    val source = File("src/main/resources/from-to-data/derived/numbat-regional-2024.csv")
    val hash = MessageDigest.getInstance("SHA-256").digest(source.readBytes()).joinToString("") { "%02x".format(it) }
    val area = StudyArea(listOf(Coordinate(-0.20, 51.46), Coordinate(-0.06, 51.46),
        Coordinate(-0.06, 51.55), Coordinate(-0.20, 51.55)))
    val service = PlannerService(Json { encodeDefaults = true })
    val rows = (0..5).map { cap ->
        val settings = GenerationSettings(guidanceStrength = 0.08, radialSoftCap = cap,
            orbitalSoftCap = cap, distributorSoftCap = cap)
        val plan = service.generatePareto(GenerationRequest(area, settings), refinementBudgetMillis = 0)
            .plans.firstOrNull { it.profile == "balanced" } ?: error("No balanced plan for cap $cap")
        SweepRow(cap, cap, cap, plan.metrics.coverage, plan.metrics.costEstimate,
            plan.network.lines.size, plan.metrics.transferOverhead, plan.metrics.duplication)
    }
    val minCoverage = rows.minOf { it.coverage }; val maxCoverage = rows.maxOf { it.coverage }
    val minCost = rows.minOf { it.costEstimate }; val maxCost = rows.maxOf { it.costEstimate }
    val chosen = rows.minWith(compareBy<SweepRow> { row ->
        val coverageLoss = (maxCoverage - row.coverage) / (maxCoverage - minCoverage).coerceAtLeast(1e-9)
        val costExcess = (row.costEstimate - minCost) / (maxCost - minCost).coerceAtLeast(1e-9)
        hypot(coverageLoss, costExcess)
    }.thenBy { it.radialCap })
    val report = CalibrationReport("sparse-pareto-v1-preview-sweep", hash, area, 0.08,
        rows, chosen, "Preview sweep on one medium London polygon; caps are guidance, not a research constant.")
    output.parentFile.mkdirs()
    output.writeText(Json { prettyPrint = true }.encodeToString(report) + "\n")
    println("Chosen common cap ${chosen.radialCap}; wrote $output")
}
