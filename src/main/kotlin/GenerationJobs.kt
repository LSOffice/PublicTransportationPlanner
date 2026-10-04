package org.lsoffice

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import java.util.LinkedHashMap
import java.util.UUID

@Serializable
data class GenerationEvent(val id: Long, val type: String, val message: String, val plan: GenerationPlan? = null)

@Serializable
data class GenerationJobStatus(
    val id: String,
    val state: String,
    val eventsUrl: String,
    val statusUrl: String,
    val result: GenerationResult? = null,
    val error: String? = null,
    val lastEventId: Long = 0,
)

@Serializable
data class EvaluateBundleRequest(val candidateIds: List<String>, val settings: GenerationSettings = GenerationSettings())

class GenerationJobs(private val service: PlannerService) {
    private data class Job(
        val id: String,
        val request: GenerationRequest,
        val createdAtMillis: Long = System.currentTimeMillis(),
        val events: ArrayDeque<GenerationEvent> = ArrayDeque(),
        var nextId: Long = 1,
        var state: String = "QUEUED",
        var result: GenerationResult? = null,
        var error: String? = null,
        @Volatile var cancelled: Boolean = false,
        var lastVisualMillis: Long = 0,
    )
    private val jobs = object : LinkedHashMap<String, Job>(4, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Job>?): Boolean {
            if (size <= 4) return false
            eldest?.value?.cancelled = true
            return true
        }
    }
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    @Synchronized
    fun create(request: GenerationRequest): GenerationJobStatus {
        service.validateGenerationRequest(request)
        expire()
        val job = Job(UUID.randomUUID().toString(), request)
        jobs[job.id] = job
        scope.launch {
            synchronized(job) { job.state = "RUNNING" }
            try {
                val result = service.generatePareto(request,
                    onEvent = { type, message -> emit(job, type, message) },
                    cancelled = { job.cancelled },
                    onPreview = { plan -> emit(job, "PLAN_FORMED", plan.id, plan) })
                synchronized(job) {
                    if (job.cancelled) {
                        job.state = "CANCELLED"
                    } else {
                        job.result = result
                        job.state = "COMPLETE"
                        emit(job, "COMPLETE", "${result.plans.size} plans ready")
                    }
                }
            } catch (exception: Exception) {
                synchronized(job) {
                    job.state = if (job.cancelled) "CANCELLED" else "FAILED"
                    job.error = exception.message ?: "Generation failed"
                    emit(job, "FAILED", job.error!!)
                }
            }
        }
        return status(job.id)
    }

    @Synchronized
    fun status(id: String): GenerationJobStatus {
        expire()
        val job = jobs[id] ?: throw PlannerValidationException("Generation job '$id' is unavailable", "JOB_NOT_FOUND")
        synchronized(job) {
            val base = "/api/v1/generation/jobs/$id"
            return GenerationJobStatus(id, job.state, "$base/events", base, job.result, job.error, job.nextId - 1)
        }
    }

    @Synchronized
    fun eventsSince(id: String, after: Long): List<GenerationEvent> {
        expire()
        val job = jobs[id] ?: throw PlannerValidationException("Generation job '$id' is unavailable", "JOB_NOT_FOUND")
        synchronized(job) { return job.events.filter { it.id > after } }
    }

    @Synchronized
    fun cancel(id: String): GenerationJobStatus {
        val job = jobs[id] ?: throw PlannerValidationException("Generation job '$id' is unavailable", "JOB_NOT_FOUND")
        synchronized(job) {
            if (job.state == "RUNNING" || job.state == "QUEUED") {
                job.cancelled = true
                job.state = "CANCELLED"
                emit(job, "COMPLETE", "Generation cancelled")
            }
        }
        return status(id)
    }

    @Synchronized
    fun evaluate(id: String, bundle: EvaluateBundleRequest): GenerationPlan {
        val job = jobs[id] ?: throw PlannerValidationException("Generation job '$id' is unavailable", "JOB_NOT_FOUND")
        if (job.state != "COMPLETE") throw PlannerValidationException("Generation is not complete", "JOB_NOT_COMPLETE")
        val valid = job.result?.candidates?.map { it.id }?.toSet().orEmpty()
        if (!valid.containsAll(bundle.candidateIds)) throw PlannerValidationException("Unknown corridor in bundle")
        return service.evaluatePareto(job.request, bundle)
    }

    private fun emit(job: Job, type: String, message: String, plan: GenerationPlan? = null) {
        synchronized(job) {
            val now = System.currentTimeMillis()
            if (type in setOf("BEAM_STEP", "BRANCH_PRUNED", "RESIDUAL_HEATMAP_UPDATED") && now - job.lastVisualMillis < 100) return
            if (type in setOf("BEAM_STEP", "BRANCH_PRUNED", "RESIDUAL_HEATMAP_UPDATED")) job.lastVisualMillis = now
            job.events.addLast(GenerationEvent(job.nextId++, type, message, plan))
            while (job.events.size > 512) job.events.removeFirst()
        }
    }

    private fun expire() {
        val cutoff = System.currentTimeMillis() - 30 * 60 * 1000
        jobs.entries.removeIf { it.value.createdAtMillis < cutoff }
    }
}
