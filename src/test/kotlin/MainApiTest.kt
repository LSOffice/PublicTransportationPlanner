package org.lsoffice

import io.ktor.client.request.get
import io.ktor.http.HttpStatusCode
import io.ktor.server.testing.testApplication
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MainApiTest {
    @Test
    fun `coverage endpoint returns the planning boundary`() = testApplication {
        application { plannerModule() }

        val response = client.get("/api/v1/coverage")

        assertEquals(HttpStatusCode.OK, response.status)
        assertTrue(response.headers["Content-Type"].orEmpty().startsWith("application/json"))
    }

    @Test
    fun `proxy rejects hosts outside the allowlist`() = testApplication {
        application { plannerModule() }

        val response = client.get("/proxy?url=https%3A%2F%2Fexample.com")

        assertEquals(HttpStatusCode.BadRequest, response.status)
    }
}
