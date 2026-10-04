package org.lsoffice

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SparseParetoGeneratorTest {
    @Test
    fun `triangulation stays sparse and generation is deterministic`() {
        val grid = (0 until 8).flatMap { x -> (0 until 8).map { y ->
            GridPoint(-0.2 + x * 0.017, 51.45 + y * 0.011, (x * y + 30).toDouble())
        } }
        val generator = SparseParetoGenerator()
        val first = generator.generate(grid)
        assertTrue(generator.diagnostics.edgeCount <= generator.diagnostics.placeCount * 3)
        assertTrue(first.candidates.isNotEmpty())
        assertTrue(first.plans.isNotEmpty())
        assertEquals(first.candidates.size, first.candidates.map { it.stationIds.sorted() }.distinct().size)
        assertEquals("SIMULATED_GRAVITY", first.demandEvidence)
        assertEquals(first.candidates, SparseParetoGenerator().generate(grid).candidates)
    }

    @Test
    fun `collinear places use a chain`() {
        val grid = (0 until 12).map { GridPoint(-0.2 + it * 0.02, 51.5, 100.0 + it) }
        val generator = SparseParetoGenerator()
        generator.generate(grid)
        assertTrue(generator.diagnostics.usedCollinearFallback)
        assertEquals(generator.diagnostics.placeCount - 1, generator.diagnostics.edgeCount)
    }

    @Test
    fun `search distinguishes certified and timed results`() {
        val grid = (0 until 5).map { GridPoint(-0.2 + it * 0.03, 51.5, 100.0) }
        val certified = SparseParetoGenerator().generate(grid)
        val timed = SparseParetoGenerator().generate(grid, refinementBudgetMillis = 0)
        assertTrue(certified.plans.all { it.certified && it.optimalityGap == 0.0 })
        assertTrue(timed.plans.all { !it.certified && it.optimalityGap != null })
    }

    @Test
    fun `residual and angle rules are continuous at the turning threshold`() {
        assertEquals(70.0, residualAfterCommit(100.0), 1e-9)
        assertEquals(1.0, turnPenaltyDegrees(45.0), 1e-9)
        assertTrue(turnPenaltyDegrees(46.0) < 1.0)
        assertTrue(turnPenaltyDegrees(89.0) > 0.0)
        assertEquals(0.0, turnPenaltyDegrees(91.0))
    }

    @Test
    fun `certified search matches brute force on a small frontier`() {
        val grid = (0 until 5).map { GridPoint(-0.2 + it * 0.03, 51.5, 100.0) }
        val generator = SparseParetoGenerator()
        val result = generator.generate(grid)
        val balanced = result.plans.first { it.profile == "balanced" }
        val ids = result.candidates.map { it.id }
        val exhaustive = (1 until (1 shl ids.size)).maxOf { mask ->
            val bundle = ids.filterIndexed { index, _ -> mask and (1 shl index) != 0 }
            generator.scoreBundle(grid, bundle, GenerationSettings(), "balanced")
        }
        assertEquals(exhaustive, generator.scoreBundle(grid, balanced.candidateIds, GenerationSettings(), "balanced"), 1e-6)
    }

    @Test
    fun `soft caps add quadratic guidance penalty`() {
        val grid = (0 until 8).flatMap { x -> (0 until 8).map { y ->
            GridPoint(-0.2 + x * 0.017, 51.45 + y * 0.011, 100.0)
        } }
        val generator = SparseParetoGenerator()
        val result = generator.generate(grid)
        val ids = result.candidates.take(3).map { it.id }
        val loose = generator.evaluate(grid, ids, GenerationSettings(guidanceStrength = 0.0, radialSoftCap = 0, orbitalSoftCap = 0, distributorSoftCap = 0))
        val guided = generator.evaluate(grid, ids, GenerationSettings(guidanceStrength = 0.08, radialSoftCap = 0, orbitalSoftCap = 0, distributorSoftCap = 0))
        assertEquals(0.0, loose.metrics.guidancePenalty)
        assertTrue(guided.metrics.guidancePenalty > 0.0)
    }

    @Test
    fun `sparse graph operation counts scale linearly for large grids`() {
        for (size in listOf(1_000, 5_000, 10_000)) {
            val width = 100
            val grid = (0 until size).map { index -> GridPoint(-0.5 + index % width * 0.016,
                51.0 + index / width * 0.011, 100.0) }
            val begin = System.nanoTime()
            val graph = SparseParetoGenerator().inspectSparseGraph(grid)
            val millis = (System.nanoTime() - begin) / 1_000_000
            println("SPARSE_BENCHMARK size=$size places=${graph.placeCount} edges=${graph.edgeCount} millis=$millis")
            assertTrue(graph.edgeCount <= graph.placeCount * 3)
        }
    }
}
