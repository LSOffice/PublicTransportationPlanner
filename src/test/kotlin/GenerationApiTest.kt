package org.lsoffice

import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.server.testing.testApplication
import kotlinx.serialization.json.Json
import kotlin.test.*

class GenerationApiTest {
    @Test
    fun `invalid settings are rejected before job creation`() = testApplication {
        application { plannerModule() }
        val area = StudyArea(listOf(Coordinate(-0.20, 51.46), Coordinate(-0.06, 51.46),
            Coordinate(-0.06, 51.55), Coordinate(-0.20, 51.55)))
        val json = Json { ignoreUnknownKeys = true }
        val response = client.post("/api/v1/generation/jobs") {
            contentType(ContentType.Application.Json)
            setBody(json.encodeToString(GenerationRequest.serializer(), GenerationRequest(area,
                GenerationSettings(guidanceStrength = 0.5))))
        }
        assertEquals(HttpStatusCode.BadRequest, response.status)
    }

    @Test
    fun `running generation can be cancelled`() = testApplication {
        application { plannerModule() }
        val area = StudyArea(listOf(Coordinate(-0.20, 51.46), Coordinate(-0.06, 51.46),
            Coordinate(-0.06, 51.55), Coordinate(-0.20, 51.55)))
        val json = Json { ignoreUnknownKeys = true }
        val created = client.post("/api/v1/generation/jobs") {
            contentType(ContentType.Application.Json)
            setBody(json.encodeToString(GenerationRequest.serializer(), GenerationRequest(area,
                GenerationSettings(demandMode = "GRAVITY_ONLY"))))
        }
        val job = json.decodeFromString(GenerationJobStatus.serializer(), created.bodyAsText())
        val cancelled = client.delete(job.statusUrl)
        assertEquals(HttpStatusCode.OK, cancelled.status)
        assertEquals("CANCELLED", json.decodeFromString(GenerationJobStatus.serializer(), cancelled.bodyAsText()).state)
    }

    @Test
    fun `generation job reaches a terminal result and evaluates a bundle`() = testApplication {
        application { plannerModule() }
        val area = StudyArea(listOf(Coordinate(-0.20, 51.46), Coordinate(-0.06, 51.46),
            Coordinate(-0.06, 51.55), Coordinate(-0.20, 51.55)))
        val json = Json { ignoreUnknownKeys = true }
        val response = client.post("/api/v1/generation/jobs") {
            contentType(ContentType.Application.Json)
            setBody(json.encodeToString(GenerationRequest.serializer(), GenerationRequest(area,
                GenerationSettings())))
        }
        assertEquals(HttpStatusCode.Accepted, response.status)
        val job = json.decodeFromString(GenerationJobStatus.serializer(), response.bodyAsText())
        var final = job
        repeat(150) {
            if (final.state in setOf("COMPLETE", "FAILED", "CANCELLED")) return@repeat
            Thread.sleep(100)
            final = json.decodeFromString(GenerationJobStatus.serializer(), client.get(job.statusUrl).bodyAsText())
        }
        assertEquals("COMPLETE", final.state, final.error)
        assertTrue(final.result!!.candidates.isNotEmpty())
        assertEquals("NUMBAT_OBSERVED_BLEND", final.result!!.demandEvidence)
        val balanced = final.result!!.plans.firstOrNull { it.profile == "balanced" }
        assertTrue((balanced?.network?.lines?.size ?: 0) > 1 ||
            final.result!!.insufficientFrontierReason?.startsWith("Only ") == true)
        assertTrue(final.result!!.plans.flatMap { it.network.lines }.all { it.buildEstimate != null })
        val stream = client.get(job.eventsUrl)
        assertEquals(HttpStatusCode.OK, stream.status)
        assertTrue(stream.headers[HttpHeaders.ContentType].orEmpty().startsWith("text/event-stream"))
        val frames = stream.bodyAsText()
        assertTrue(frames.contains("event: COMPLETE"))
        val previewEvents = frames.lineSequence().filter { it.startsWith("data: ") }
            .mapNotNull { line -> runCatching { json.decodeFromString(GenerationEvent.serializer(), line.removePrefix("data: ")) }.getOrNull() }
            .filter { it.type == "PLAN_FORMED" && it.plan != null }.toList()
        assertTrue(previewEvents.isNotEmpty())
        println("FIRST_PREVIEW_MILLIS ${previewEvents.first().plan!!.elapsedMillis}")
        println("FINAL_REFINEMENT_MILLIS ${final.result!!.plans.maxOf { it.elapsedMillis }}")
        val ids = Regex("(?m)^id: (\\d+)$").findAll(frames).map { it.groupValues[1].toLong() }.toList()
        assertEquals(ids.sorted(), ids)
        assertEquals(ids.distinct(), ids)
        val replay = client.get(job.eventsUrl) { header("Last-Event-ID", ids.last().toString()) }.bodyAsText()
        assertTrue(!replay.contains("event: PLAN_FORMED"))
        val bundle = EvaluateBundleRequest(final.result!!.candidates.take(2).map { it.id })
        val evaluation = client.post("${job.statusUrl}/evaluate") {
            contentType(ContentType.Application.Json)
            setBody(json.encodeToString(EvaluateBundleRequest.serializer(), bundle))
        }
        assertEquals(HttpStatusCode.OK, evaluation.status)
        assertTrue(json.decodeFromString(GenerationPlan.serializer(), evaluation.bodyAsText()).metrics.lengthMeters > 0.0)
    }
}
