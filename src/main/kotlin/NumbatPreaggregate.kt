package org.lsoffice

import java.io.File

/** Offline data preparation; never run from a request handler. */
fun main(args: Array<String>) {
    val output = File(args.firstOrNull() ?: error("Output path required"))
    val model = NumbatDemandLoader.loadRawFromResources() ?: error("Raw NUMBAT resources are unavailable")
    output.parentFile.mkdirs()
    output.bufferedWriter().use { writer ->
        writer.appendLine("origin_x,origin_y,destination_x,destination_y,weekly_demand")
        model.entries().toSortedMap(compareBy<RegionDemandPair> { it.a.x }.thenBy { it.a.y }
            .thenBy { it.b.x }.thenBy { it.b.y }).forEach { (pair, demand) ->
            writer.appendLine("${pair.a.x},${pair.a.y},${pair.b.x},${pair.b.y},$demand")
        }
    }
    println("Wrote ${model.pairCount} compact regional pairs to $output")
}
