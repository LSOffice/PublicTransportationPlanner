package org.lsoffice

import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Loads PTAL (Public Transport Accessibility Level) data at first use and
 * provides a fast nearest-neighbour lookup by WGS84 coordinate.
 *
 * Data source: Transport for London (TfL) – PTAL 2015, LSOA 2011.
 * Licence: UK Open Government Licence (OGL v2).
 * Spatial extent: Greater London only (~4 835 LSOAs).
 *
 * When a zone falls outside London the lookup returns null and the caller
 * should fall back to its default weights.
 */
object PtalLookup {
    /** A single LSOA entry from ptal_spatial.csv. */
    data class Record(
        val lon: Double,
        val lat: Double,
        /** Average PTAI index value (0 – ~130).  Higher = better access. */
        val avgPtai2015: Double,
        /** Numeric band index: 0 (band "0") … 8 (band "6b"). */
        val ptalNumeric: Double,
    )

    /** All records loaded from the classpath resource. Empty if file is absent. */
    val records: List<Record> by lazy { loadRecords() }

    private fun loadRecords(): List<Record> {
        val stream =
            PtalLookup::class.java.getResourceAsStream("/ptal_spatial.csv")
                ?: run {
                    System.err.println("[PtalLookup] WARNING: ptal_spatial.csv not found on classpath – PTAL enrichment disabled")
                    return emptyList()
                }
        val reader = BufferedReader(InputStreamReader(stream))
        reader.readLine() // skip header: lon,lat,avgPtai2015,ptalNumeric
        val result = mutableListOf<Record>()
        while (true) {
            val line = reader.readLine() ?: break
            val p = line.split(',')
            if (p.size < 4) continue
            val lon = p[0].toDoubleOrNull() ?: continue
            val lat = p[1].toDoubleOrNull() ?: continue
            val avg = p[2].toDoubleOrNull() ?: continue
            val num = p[3].toDoubleOrNull() ?: continue
            result.add(Record(lon, lat, avg, num))
        }
        reader.close()
        println("[PtalLookup] Loaded ${result.size} PTAL records (London LSOAs)")
        return result
    }

    /**
     * Returns the nearest PTAL record to (lon, lat), or null if no record
     * is within [maxSearchMeters] (default 3 km – covers London gaps between
     * 1 km population grid cells and the ~500 m LSOA centroids).
     */
    fun nearest(
        lon: Double,
        lat: Double,
        maxSearchMeters: Double = 3_000.0,
    ): Record? {
        var best: Record? = null
        var bestDist = Double.MAX_VALUE
        for (r in records) {
            val d = haversineMeters(lon, lat, r.lon, r.lat)
            if (d < bestDist) {
                bestDist = d
                best = r
            }
        }
        return if (bestDist <= maxSearchMeters) best else null
    }

    /**
     * Maps avgPtai2015 (0–~130) to a socioeconomic demand weight centred on 1.0.
     *
     * Calibration: London mean PTAI ≈ 25–30.  A PTAI of 50 → weight 1.0.
     * Areas with excellent access (PTAI 100+) get weight ≈ 2.0; car-dependent
     * fringes (PTAI < 5) get weight ≈ 0.1.  Clamped to [0.1, 3.0].
     */
    fun ptaiToSocioWeight(ptai: Double): Double = (ptai / 50.0).coerceIn(0.1, 3.0)

    /**
     * Maps avgPtai2015 to an activity multiplier in [0.0, 1.0].
     * Used for Zone.activity in the gravity-model OD matrix.
     */
    fun ptaiToActivity(ptai: Double): Double = (ptai / 100.0).coerceIn(0.0, 1.0)

    /**
     * Convenience: look up and return a demand multiplier for a coordinate,
     * returning [fallback] if the point is outside London or the file is missing.
     */
    fun demandWeight(
        lon: Double,
        lat: Double,
        fallback: Double = 1.0,
    ): Double {
        val r = nearest(lon, lat) ?: return fallback
        return ptaiToSocioWeight(r.avgPtai2015)
    }
}
