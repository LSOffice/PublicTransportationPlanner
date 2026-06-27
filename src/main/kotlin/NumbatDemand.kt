package org.lsoffice

import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import kotlin.math.cos
import kotlin.math.floor

data class RegionCell(
    val x: Int,
    val y: Int,
)

data class NumbatStation(
    val masterNlc: Int,
    val name: String,
    val lat: Double,
    val lon: Double,
)

data class RegionDemandPair(
    val a: RegionCell,
    val b: RegionCell,
) {
    companion object {
        fun unordered(
            first: RegionCell,
            second: RegionCell,
        ): RegionDemandPair =
            if (first.x < second.x || (first.x == second.x && first.y <= second.y)) {
                RegionDemandPair(first, second)
            } else {
                RegionDemandPair(second, first)
            }
    }
}

data class RegionDemandStats(
    val stationCount: Int,
    val sourceRows: Long,
    val matchedRows: Long,
    val skippedUnknownStations: Long,
    val aggregatedPairs: Int,
    val totalDemand: Double,
)

class RegionDemandModel(
    private val demandByPair: Map<RegionDemandPair, Double>,
    val stats: RegionDemandStats,
) {
    val maxDemand: Double = demandByPair.values.maxOrNull() ?: 0.0
    val pairCount: Int = demandByPair.size

    fun isEmpty(): Boolean = demandByPair.isEmpty()

    fun demandBetween(
        originCells: Set<RegionCell>,
        destinationCells: Set<RegionCell>,
    ): Double {
        if (originCells.isEmpty() || destinationCells.isEmpty()) return 0.0

        val seenPairs = mutableSetOf<RegionDemandPair>()
        var total = 0.0
        for (origin in originCells) {
            for (destination in destinationCells) {
                val pair = RegionDemandPair.unordered(origin, destination)
                if (seenPairs.add(pair)) {
                    total += demandByPair[pair] ?: 0.0
                }
            }
        }
        return total
    }
}

object LondonRegionGrid {
    private const val originLon = -0.1278
    private const val originLat = 51.5074
    private const val cellSizeMeters = 500.0
    private val originLatRadians = Math.toRadians(originLat)

    fun cellFor(
        lon: Double,
        lat: Double,
    ): RegionCell {
        val xMeters = (lon - originLon) * cos(originLatRadians) * 111_320.0
        val yMeters = (lat - originLat) * 110_540.0
        return RegionCell(
            floor(xMeters / cellSizeMeters).toInt(),
            floor(yMeters / cellSizeMeters).toInt(),
        )
    }
}

object NumbatDemandLoader {
    private data class WeightedResource(
        val path: String,
        val weight: Double,
    )

    private val defaultOdResources =
        listOf(
            WeightedResource("/from-to-data/raw/NBT24MON5d_od_network_tb_lf_o.csv", 1.0),
            WeightedResource("/from-to-data/raw/NBT24TWT5d_od_network_tb_lf_o.csv", 3.0),
            WeightedResource("/from-to-data/raw/NBT24FRI5d_od_network_tb_lf_o.csv", 1.0),
            WeightedResource("/from-to-data/raw/NBT24SAT5d_od_network_tb_lf_o.csv", 1.0),
            WeightedResource("/from-to-data/raw/NBT24SUN5d_od_network_tb_lf_o.csv", 1.0),
        )

    fun loadFromResources(): RegionDemandModel? =
        try {
            val stationStream =
                NumbatDemandLoader::class.java.getResourceAsStream("/from-to-data/derived/numbat-stations-2024.csv")
                    ?: run {
                        System.err.println("[NumbatDemand] Station lookup not found; NUMBAT demand disabled")
                        return null
                    }

            val odStreams =
                defaultOdResources.map { resource ->
                    val stream =
                        NumbatDemandLoader::class.java.getResourceAsStream(resource.path)
                            ?: run {
                                System.err.println("[NumbatDemand] Missing ${resource.path}; NUMBAT demand disabled")
                                stationStream.close()
                                return null
                            }
                    WeightedOdInput(stream, resource.weight)
                }

            load(stationStream, odStreams).also { model ->
                val stats = model.stats
                println(
                    "[NumbatDemand] Loaded ${stats.sourceRows} OD rows, matched ${stats.matchedRows}, " +
                        "aggregated ${stats.aggregatedPairs} region pairs, total weekly demand=${"%.0f".format(stats.totalDemand)}",
                )
                if (stats.skippedUnknownStations > 0) {
                    println("[NumbatDemand] Skipped ${stats.skippedUnknownStations} rows with unknown station codes")
                }
            }
        } catch (e: Exception) {
            System.err.println("[NumbatDemand] Failed to load NUMBAT demand: ${e.message}")
            null
        }

    data class WeightedOdInput(
        val stream: InputStream,
        val weight: Double,
    )

    fun load(
        stationCsv: InputStream,
        odInputs: List<WeightedOdInput>,
    ): RegionDemandModel {
        stationCsv.use { stationsInput ->
            val stations = loadStations(stationsInput)
            val demandByPair = mutableMapOf<RegionDemandPair, Double>()
            var sourceRows = 0L
            var matchedRows = 0L
            var skippedUnknown = 0L

            for (input in odInputs) {
                input.stream.use { odStream ->
                    val reader = BufferedReader(InputStreamReader(odStream))
                    val headerLine = reader.readLine()
                    if (headerLine == null) {
                        return@use
                    }
                    val header = parseCsvLine(headerLine)
                    val mnlcOIdx = header.indexOf("mnlc_o")
                    val mnlcDIdx = header.indexOf("mnlc_d")
                    val volIdx = header.indexOf("vol")
                    if (mnlcOIdx < 0 || mnlcDIdx < 0 || volIdx < 0) continue

                    while (true) {
                        val line = reader.readLine() ?: break
                        if (line.isBlank()) continue
                        sourceRows += 1

                        val fields = parseCsvLine(line)
                        val originCode = fields.getOrNull(mnlcOIdx)?.toIntOrNull()
                        val destinationCode = fields.getOrNull(mnlcDIdx)?.toIntOrNull()
                        val volume = fields.getOrNull(volIdx)?.toDoubleOrNull()
                        if (originCode == null || destinationCode == null || volume == null) continue

                        val origin = stations[originCode]
                        val destination = stations[destinationCode]
                        if (origin == null || destination == null) {
                            skippedUnknown += 1
                            continue
                        }

                        val originCell = LondonRegionGrid.cellFor(origin.lon, origin.lat)
                        val destinationCell = LondonRegionGrid.cellFor(destination.lon, destination.lat)
                        val pair = RegionDemandPair.unordered(originCell, destinationCell)
                        demandByPair[pair] = (demandByPair[pair] ?: 0.0) + volume * input.weight
                        matchedRows += 1
                    }
                }
            }

            val totalDemand = demandByPair.values.sum()
            return RegionDemandModel(
                demandByPair = demandByPair,
                stats =
                    RegionDemandStats(
                        stationCount = stations.size,
                        sourceRows = sourceRows,
                        matchedRows = matchedRows,
                        skippedUnknownStations = skippedUnknown,
                        aggregatedPairs = demandByPair.size,
                        totalDemand = totalDemand,
                    ),
            )
        }
    }

    fun loadStations(stationCsv: InputStream): Map<Int, NumbatStation> {
        val reader = BufferedReader(InputStreamReader(stationCsv))
        val header = reader.readLine()?.let(::parseCsvLine) ?: return emptyMap()
        val codeIdx = header.indexOf("master_nlc")
        val nameIdx = header.indexOf("station_name")
        val latIdx = header.indexOf("latitude")
        val lonIdx = header.indexOf("longitude")
        if (codeIdx < 0 || nameIdx < 0 || latIdx < 0 || lonIdx < 0) return emptyMap()

        val stations = mutableMapOf<Int, NumbatStation>()
        while (true) {
            val line = reader.readLine() ?: break
            if (line.isBlank()) continue
            val fields = parseCsvLine(line)
            val code = fields.getOrNull(codeIdx)?.toIntOrNull() ?: continue
            val name = fields.getOrNull(nameIdx)?.trim().orEmpty()
            val lat = fields.getOrNull(latIdx)?.toDoubleOrNull() ?: continue
            val lon = fields.getOrNull(lonIdx)?.toDoubleOrNull() ?: continue
            stations[code] = NumbatStation(code, name, lat, lon)
        }
        return stations
    }

    private fun parseCsvLine(line: String): List<String> {
        val fields = mutableListOf<String>()
        val current = StringBuilder()
        var inQuotes = false
        var i = 0
        while (i < line.length) {
            val c = line[i]
            when {
                c == '"' && inQuotes && i + 1 < line.length && line[i + 1] == '"' -> {
                    current.append('"')
                    i += 1
                }
                c == '"' -> inQuotes = !inQuotes
                c == ',' && !inQuotes -> {
                    fields.add(current.toString())
                    current.clear()
                }
                else -> current.append(c)
            }
            i += 1
        }
        fields.add(current.toString())
        return fields
    }
}
