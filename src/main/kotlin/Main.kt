package org.lsoffice

import com.sun.net.httpserver.HttpServer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.net.BindException
import java.net.HttpURLConnection
import java.net.InetSocketAddress
import java.net.URL
import java.net.URLDecoder
import kotlin.math.*

fun haversineMeters(lon1: Double, lat1: Double, lon2: Double, lat2: Double): Double {
    val R = 6371000.0
    val phi1 = Math.toRadians(lat1)
    val phi2 = Math.toRadians(lat2)
    val dphi = Math.toRadians(lat2 - lat1)
    val dlambda = Math.toRadians(lon2 - lon1)
    val a = sin(dphi / 2).pow(2.0) + cos(phi1) * cos(phi2) * sin(dlambda / 2).pow(2.0)
    val c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
}

fun main() {
    var port = 5000
    val maxPort = 5010
    var server: HttpServer? = null

    while (server == null && port <= maxPort) {
        try {
            server = HttpServer.create(InetSocketAddress(port), 0)
        } catch (e: BindException) {
            System.err.println("Port $port is in use, trying ${port + 1}...")
            port += 1
        }
    }

    if (server == null) {
        System.err.println("Failed to bind to any port in range 5000..$maxPort. Please free a port and try again.")
        return
    }

    // Serve / -> map.html and static resources
    server.createContext("/") { exchange ->
        try {
            val uri = exchange.requestURI.path
            val path = if (uri == "/" || uri.isEmpty()) "/map.html" else uri
            val resourceStream = object {}.javaClass.getResourceAsStream(path)
            if (resourceStream == null) {
                val notFound = "404 Not Found"
                exchange.sendResponseHeaders(404, notFound.toByteArray().size.toLong())
                exchange.responseBody.use { it.write(notFound.toByteArray()) }
            } else {
                val content = resourceStream.readAllBytes()
                val contentType = when {
                    path.endsWith(".html") -> "text/html; charset=utf-8"
                    path.endsWith(".js") -> "application/javascript"
                    path.endsWith(".css") -> "text/css"
                    path.endsWith(".png") -> "image/png"
                    path.endsWith(".jpg") || path.endsWith(".jpeg") -> "image/jpeg"
                    else -> "application/octet-stream"
                }
                exchange.responseHeaders.add("Content-Type", contentType)
                exchange.sendResponseHeaders(200, content.size.toLong())
                exchange.responseBody.use { it.write(content) }
            }
        } catch (e: IOException) {
            e.printStackTrace()
            try {
                exchange.sendResponseHeaders(500, -1)
            } catch (_: Exception) {}
        } finally {
            try { exchange.close() } catch (_: Exception) {}
        }
    }

    // Proxy endpoint (existing)
    server.createContext("/proxy") { exchange ->
        try {
            val query = exchange.requestURI.query ?: ""
            val params = query.split("&").mapNotNull { part ->
                val idx = part.indexOf('=')
                if (idx <= 0) null else part.substring(0, idx) to part.substring(idx + 1)
            }.toMap()

            val rawUrl = params["url"]?.let { URLDecoder.decode(it, "UTF-8") }
            if (rawUrl == null || rawUrl.isBlank()) {
                val msg = "Missing 'url' query parameter"
                exchange.sendResponseHeaders(400, msg.toByteArray().size.toLong())
                exchange.responseBody.use { it.write(msg.toByteArray()) }
                return@createContext
            }

            var target = rawUrl
            val lat = params["lat"]?.let { URLDecoder.decode(it, "UTF-8") }
            val lon = params["lon"]?.let { URLDecoder.decode(it, "UTF-8") }
            if (lat != null && lon != null) {
                target = target.replace("{lat}", lat).replace("{lon}", lon)
            }

            if (!(target.startsWith("http://") || target.startsWith("https://"))) {
                val msg = "Invalid target URL"
                exchange.sendResponseHeaders(400, msg.toByteArray().size.toLong())
                exchange.responseBody.use { it.write(msg.toByteArray()) }
                return@createContext
            }

            val url = URL(target)
            println("[proxy] Requesting: $target")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "GET"
            conn.connectTimeout = 10000
            conn.readTimeout = 10000
            conn.instanceFollowRedirects = true

            try {
                conn.setRequestProperty("User-Agent", "PublicTransportationPlanner/1.0")
                val ghToken = System.getenv("GITHUB_TOKEN")
                if (!ghToken.isNullOrBlank()) {
                    conn.setRequestProperty("Authorization", "token $ghToken")
                }
                if (url.host.contains("api.github.com")) {
                    conn.setRequestProperty("Accept", "application/vnd.github.v3+json")
                } else if (url.host.contains("raw.githubusercontent.com") || url.host.contains("githubusercontent.com")) {
                    conn.setRequestProperty("Accept", "text/plain, application/json, */*")
                } else {
                    conn.setRequestProperty("Accept", "application/json, text/plain, */*")
                }
            } catch (_: Exception) {
            }

            val code = conn.responseCode
            println("[proxy] Response code $code for $target")
            val contentType = conn.contentType ?: "application/octet-stream"
            val input = if (code in 200..299) conn.inputStream else conn.errorStream
            val body = input.readAllBytes()

            exchange.responseHeaders.add("Content-Type", contentType)
            exchange.responseHeaders.add("Access-Control-Allow-Origin", "*")
            exchange.sendResponseHeaders(code, body.size.toLong())
            exchange.responseBody.use { it.write(body) }

        } catch (e: Exception) {
            e.printStackTrace()
            try {
                val err = "Proxy error"
                exchange.sendResponseHeaders(500, err.toByteArray().size.toLong())
                exchange.responseBody.use { it.write(err.toByteArray()) }
            } catch (_: Exception) {}
        } finally {
            try { exchange.close() } catch (_: Exception) {}
        }
    }

    // New: density lookup endpoint (no Python) - reads CSV resource and finds nearest point
    server.createContext("/density") { exchange ->
        try {
            val q = exchange.requestURI.query ?: ""
            val params = q.split("&").mapNotNull { part ->
                val idx = part.indexOf('=')
                if (idx <= 0) null else part.substring(0, idx) to URLDecoder.decode(part.substring(idx + 1), "UTF-8")
            }.toMap()

            val lonStr = params["lon"]
            val latStr = params["lat"]
            val maxM = params["max_m"]?.toDoubleOrNull() ?: 1000.0
            if (lonStr == null || latStr == null) {
                val msg = "Missing lon or lat query parameters"
                exchange.sendResponseHeaders(400, msg.toByteArray().size.toLong())
                exchange.responseBody.use { it.write(msg.toByteArray()) }
                return@createContext
            }

            val qlon = lonStr.toDoubleOrNull()
            val qlat = latStr.toDoubleOrNull()
            if (qlon == null || qlat == null) {
                val msg = "Invalid lon/lat"
                exchange.sendResponseHeaders(400, msg.toByteArray().size.toLong())
                exchange.responseBody.use { it.write(msg.toByteArray()) }
                return@createContext
            }

            // Stream the CSV resource from classpath
            val resourceStream = object {}.javaClass.getResourceAsStream("/gbr_pd_2020_1km_ASCII_XYZ.csv")
            if (resourceStream == null) {
                val msg = "CSV resource not found"
                exchange.sendResponseHeaders(500, msg.toByteArray().size.toLong())
                exchange.responseBody.use { it.write(msg.toByteArray()) }
                return@createContext
            }

            val reader = BufferedReader(InputStreamReader(resourceStream))
            var line = reader.readLine() // header
            var bestLon = 0.0
            var bestLat = 0.0
            var bestZ = Double.NaN
            var bestDist = Double.POSITIVE_INFINITY

            while (true) {
                line = reader.readLine() ?: break
                val parts = line.split(',')
                if (parts.size < 3) continue
                val x = parts[0].toDoubleOrNull() ?: continue
                val y = parts[1].toDoubleOrNull() ?: continue
                val z = parts[2].toDoubleOrNull() ?: continue
                val d = haversineMeters(qlon, qlat, x, y)
                if (d < bestDist) {
                    bestDist = d
                    bestLon = x
                    bestLat = y
                    bestZ = z
                }
            }

            reader.close()

            if (bestDist <= maxM) {
                val json = "{" +
                        "\"lon\":$bestLon,\"lat\":$bestLat,\"value\":$bestZ,\"distance_m\":$bestDist" +
                        "}"
                exchange.responseHeaders.add("Content-Type", "application/json; charset=utf-8")
                exchange.responseHeaders.add("Access-Control-Allow-Origin", "*")
                exchange.sendResponseHeaders(200, json.toByteArray().size.toLong())
                exchange.responseBody.use { it.write(json.toByteArray()) }
            } else {
                val msg = "No grid cell within $maxM meters (nearest ${"%.1f".format(bestDist)} m)"
                exchange.sendResponseHeaders(404, msg.toByteArray().size.toLong())
                exchange.responseBody.use { it.write(msg.toByteArray()) }
            }

        } catch (e: Exception) {
            e.printStackTrace()
            try {
                val err = "Density lookup error"
                exchange.sendResponseHeaders(500, err.toByteArray().size.toLong())
                exchange.responseBody.use { it.write(err.toByteArray()) }
            } catch (_: Exception) {}
        } finally {
            try { exchange.close() } catch (_: Exception) {}
        }
    }

    server.executor = null
    server.start()
    println("Server started at http://localhost:$port/")
}