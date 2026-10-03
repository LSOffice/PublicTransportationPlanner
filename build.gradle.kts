plugins {
    kotlin("jvm") version "2.2.20"
    kotlin("plugin.serialization") version "2.2.20"
    application
}

group = "org.lsoffice"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("io.ktor:ktor-server-core:2.3.12")
    implementation("io.ktor:ktor-server-netty:2.3.12")
    implementation("io.ktor:ktor-server-content-negotiation:2.3.12")
    implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.12")
    implementation("io.ktor:ktor-server-cors:2.3.12")
    implementation("io.ktor:ktor-server-status-pages:2.3.12")
    implementation("io.ktor:ktor-server-compression:2.3.12")
    runtimeOnly("org.slf4j:slf4j-nop:1.7.36")
    testImplementation(kotlin("test"))
    testImplementation("io.ktor:ktor-server-test-host:2.3.12")
}

application {
    // Main function lives in Main.kt
    mainClass.set("org.lsoffice.MainKt")
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(23)
}
