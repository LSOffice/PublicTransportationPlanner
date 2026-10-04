package org.lsoffice

import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationCall
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.compression.Compression
import io.ktor.server.plugins.compression.gzip
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.statuspages.StatusPages
import io.ktor.server.request.receive
import io.ktor.server.request.receiveText
import io.ktor.server.response.respond
import io.ktor.server.response.respondBytes
import io.ktor.server.response.respondText
import io.ktor.server.routing.get
import io.ktor.server.routing.delete
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import io.ktor.server.sse.SSE
import io.ktor.sse.ServerSentEvent
import io.ktor.server.sse.heartbeat
import io.ktor.server.sse.sse
import kotlinx.coroutines.delay
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.net.InetAddress
import java.net.ServerSocket
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

fun haversineMeters(
    lon1: Double,
    lat1: Double,
    lon2: Double,
    lat2: Double,
): Double {
    val earthRadiusMeters = 6_371_000.0
    val phi1 = Math.toRadians(lat1)
    val phi2 = Math.toRadians(lat2)
    val deltaPhi = Math.toRadians(lat2 - lat1)
    val deltaLambda = Math.toRadians(lon2 - lon1)
    val a = sin(deltaPhi / 2).pow(2.0) + cos(phi1) * cos(phi2) * sin(deltaLambda / 2).pow(2.0)
    return earthRadiusMeters * 2 * atan2(sqrt(a), sqrt(1 - a))
}

private val plannerJson =
    Json {
        prettyPrint = false
        ignoreUnknownKeys = false
        encodeDefaults = true
    }

fun main() {
    val port = findFreePort(5000..5010)
        ?: error("Failed to bind to any port in range 5000..5010. Please free a port and try again.")
    println("PublicTransportationPlanner running at http://127.0.0.1:$port")
    embeddedServer(Netty, host = "127.0.0.1", port = port, module = Application::plannerModule).start(wait = true)
}

fun Application.plannerModule(service: PlannerService = PlannerService(plannerJson)) {
    val jobs = GenerationJobs(service)
    install(ContentNegotiation) { json(plannerJson) }
    install(Compression) { gzip() }
    install(SSE)
    install(StatusPages) {
        exception<PlannerValidationException> { call, cause ->
            call.respond(HttpStatusCode.BadRequest, ApiError(cause.code, cause.message ?: "Invalid request"))
        }
        exception<MissingSessionException> { call, cause ->
            call.respond(
                HttpStatusCode.Conflict,
                ApiError("SESSION_MISSING", cause.message ?: "Analysis session is unavailable", mapOf("sessionId" to cause.sessionId)),
            )
        }
        exception<Throwable> { call, cause ->
            cause.printStackTrace()
            call.respond(HttpStatusCode.InternalServerError, ApiError("INTERNAL_ERROR", "The planner could not complete the request"))
        }
    }

    routing {
        route("/api/v1") {
            get("/coverage") { call.respond(service.coverage()) }
            post("/networks/generate") { call.respond(service.generateNetwork(call.receive())) }
            post("/generation/jobs") { call.respond(HttpStatusCode.Accepted, jobs.create(call.receive())) }
            get("/generation/jobs/{id}") {
                call.respond(jobs.status(call.parameters["id"] ?: throw PlannerValidationException("Missing job ID")))
            }
            delete("/generation/jobs/{id}") {
                call.respond(jobs.cancel(call.parameters["id"] ?: throw PlannerValidationException("Missing job ID")))
            }
            post("/generation/jobs/{id}/evaluate") {
                call.respond(jobs.evaluate(call.parameters["id"] ?: throw PlannerValidationException("Missing job ID"), call.receive()))
            }
            sse("/generation/jobs/{id}/events") {
                val id = call.parameters["id"] ?: throw PlannerValidationException("Missing job ID")
                var last = call.request.headers["Last-Event-ID"]?.toLongOrNull()
                    ?: call.request.queryParameters["after"]?.toLongOrNull() ?: 0L
                heartbeat { period = kotlin.time.Duration.parse("15s") }
                while (true) {
                    val events = jobs.eventsSince(id, last)
                    for (event in events) {
                        send(ServerSentEvent(data = plannerJson.encodeToString(event), event = event.type, id = event.id.toString()))
                        last = event.id
                    }
                    val state = jobs.status(id).state
                    if (state in setOf("COMPLETE", "FAILED", "CANCELLED") && events.isEmpty()) break
                    delay(100)
                }
            }
            post("/analysis/sessions") { call.respond(HttpStatusCode.Created, service.createSession(call.receive())) }
            get("/analysis/sessions/{id}/demand") {
                val id = call.parameters["id"] ?: throw PlannerValidationException("Missing session ID")
                val offset = call.request.queryParameters["offset"]?.toIntOrNull() ?: 0
                val limit = call.request.queryParameters["limit"]?.toIntOrNull() ?: 100
                call.respond(service.demandPage(id, offset, limit))
            }
            get("/analysis/sessions/{id}/demand.csv") {
                val id = call.parameters["id"] ?: throw PlannerValidationException("Missing session ID")
                call.response.headers.append(HttpHeaders.ContentDisposition, "attachment; filename=simulated-demand.csv")
                call.respondText(service.demandCsv(id), ContentType.parse("text/csv; charset=utf-8"))
            }
            get("/analysis/sessions/{id}/issues") {
                val id = call.parameters["id"] ?: throw PlannerValidationException("Missing session ID")
                call.respond(service.issues(id))
            }
            post("/analysis/sessions/{id}/journeys") {
                val id = call.parameters["id"] ?: throw PlannerValidationException("Missing session ID")
                call.respond(service.journey(id, call.receive()))
            }
            post("/analysis/sessions/{id}/simulations") {
                val id = call.parameters["id"] ?: throw PlannerValidationException("Missing session ID")
                call.respond(service.simulation(id, call.receive()))
            }
        }

        get("/density") {
            val lon = call.request.queryParameters["lon"]?.toDoubleOrNull()
                ?: throw PlannerValidationException("Missing or invalid lon")
            val lat = call.request.queryParameters["lat"]?.toDoubleOrNull()
                ?: throw PlannerValidationException("Missing or invalid lat")
            val maxMeters = call.request.queryParameters["max_m"]?.toDoubleOrNull()?.coerceIn(1.0, 20_000.0) ?: 1_000.0
            val point = service.nearestDensity(lon, lat, maxMeters)
            if (point == null) {
                call.respond(HttpStatusCode.NotFound, ApiError("DENSITY_NOT_FOUND", "No supported grid cell is within the requested radius"))
            } else {
                call.respond(DensityResponse(point.lon, point.lat, point.value, haversineMeters(lon, lat, point.lon, point.lat)))
            }
        }

        post("/suggestions") {
            val points = parseLegacyCsv(call.receiveText())
            if (points.isEmpty()) throw PlannerValidationException("No valid points received")
            call.respond(buildLegacySuggestions(points))
        }

        get("/metro_suggestions") {
            val centralLondon =
                StudyArea(
                    listOf(
                        Coordinate(-0.31, 51.41),
                        Coordinate(0.08, 51.41),
                        Coordinate(0.08, 51.62),
                        Coordinate(-0.31, 51.62),
                    ),
                )
            val generated = service.generateNetwork(GenerateNetworkRequest(centralLondon))
            call.respond(toLegacy(generated.network))
        }

        get("/proxy") { proxyNominatim(call.request.queryParameters["url"], call) }

        get("/") { call.respondResource("map.html") }
        get("/{path...}") {
            val path = call.parameters.getAll("path")?.joinToString("/") ?: ""
            if (path.contains("..") || path.startsWith("api/")) {
                call.respond(HttpStatusCode.NotFound, ApiError("NOT_FOUND", "Resource not found"))
            } else {
                call.respondResource(path)
            }
        }
    }
}

private suspend fun ApplicationCall.respondResource(path: String) {
    val stream = object {}.javaClass.getResourceAsStream("/$path")
    if (stream == null) {
        respond(HttpStatusCode.NotFound, ApiError("NOT_FOUND", "Resource not found"))
        return
    }
    val type =
        when (path.substringAfterLast('.', "")) {
            "html" -> ContentType.Text.Html
            "css" -> ContentType.Text.CSS
            "js" -> ContentType.parse("application/javascript")
            "json", "geojson" -> ContentType.Application.Json
            "csv" -> ContentType.parse("text/csv")
            else -> ContentType.Application.OctetStream
        }
    respondBytes(stream.use { it.readAllBytes() }, type)
}

private fun buildLegacySuggestions(points: List<GridPoint>): LegacySuggestionsResponse {
    val enriched = points.map { point -> point.copy(value = point.value * PtalLookup.demandWeight(point.lon, point.lat)) }
    val lines =
        MetroBuilder(BuilderParams(1_000_000_000.0, 50_000_000.0), debug = false)
            .buildNaturalNetworkFromGrid(enriched, minStationValue = 0.0, minCorridorLengthMeters = 2_000.0, minStationsPerLine = 3)
    return LegacySuggestionsResponse(
        lines.map { line ->
            LegacyLineResponse(
                id = line.id,
                type = line.type.name,
                isLoop = line.isLoop,
                lengthMeters = line.lengthMeters,
                cost = line.cost,
                trainsPerHour = line.trainsPerHour,
                stations = line.stations.map { LegacyStationResponse(it.id, it.lon, it.lat, it.catchmentPopulation) },
            )
        },
    )
}

private fun toLegacy(network: PlannerNetwork): LegacySuggestionsResponse {
    val stations = network.stations.associateBy { it.id }
    return LegacySuggestionsResponse(
        network.lines.map { line ->
            LegacyLineResponse(
                id = line.id,
                type = line.role.name,
                isLoop = line.isLoop,
                lengthMeters = line.segments.sumOf { it.lengthMeters },
                cost = line.segments.sumOf { it.lengthMeters } * 100_000.0,
                trainsPerHour = line.trainsPerHour,
                stations = line.stationIds.mapNotNull(stations::get).map { LegacyStationResponse(it.id, it.lon, it.lat, it.demandValue) },
            )
        },
    )
}

private fun parseLegacyCsv(body: String): List<GridPoint> =
    body.lineSequence().mapNotNull { line ->
        val parts = line.trim().split(',')
        if (parts.size < 3) return@mapNotNull null
        val lon = parts[0].toDoubleOrNull() ?: return@mapNotNull null
        val lat = parts[1].toDoubleOrNull() ?: return@mapNotNull null
        val value = parts[2].toDoubleOrNull() ?: return@mapNotNull null
        GridPoint(lon, lat, value)
    }.toList()

private suspend fun proxyNominatim(
    rawUrl: String?,
    call: io.ktor.server.application.ApplicationCall,
) {
    if (rawUrl.isNullOrBlank()) throw PlannerValidationException("Missing url query parameter")
    val uri = runCatching { URI(rawUrl) }.getOrElse { throw PlannerValidationException("Invalid proxy URL") }
    if (uri.scheme != "https" || uri.host.lowercase() != "nominatim.openstreetmap.org") {
        throw PlannerValidationException("Only HTTPS requests to nominatim.openstreetmap.org are allowed", "PROXY_HOST_REJECTED")
    }
    val request =
        HttpRequest.newBuilder(uri)
            .timeout(Duration.ofSeconds(10))
            .header("User-Agent", "PublicTransportationPlanner/2.0 (local planner)")
            .header("Accept", "application/json")
            .GET()
            .build()
    val response =
        HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build()
            .send(request, HttpResponse.BodyHandlers.ofByteArray())
    call.respondBytes(
        response.body(),
        ContentType.parse(response.headers().firstValue("content-type").orElse("application/json")),
        HttpStatusCode.fromValue(response.statusCode()),
    )
}

private fun findFreePort(range: IntRange): Int? =
    range.firstOrNull { port ->
        runCatching {
            ServerSocket(port, 1, InetAddress.getByName("127.0.0.1")).use { }
            true
        }.getOrDefault(false)
    }
