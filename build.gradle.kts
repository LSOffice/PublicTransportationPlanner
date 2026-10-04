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
    implementation("io.ktor:ktor-server-core:3.6.0")
    implementation("io.ktor:ktor-server-netty:3.6.0")
    implementation("io.ktor:ktor-server-content-negotiation:3.6.0")
    implementation("io.ktor:ktor-serialization-kotlinx-json:3.6.0")
    implementation("io.ktor:ktor-server-cors:3.6.0")
    implementation("io.ktor:ktor-server-status-pages:3.6.0")
    implementation("io.ktor:ktor-server-compression:3.6.0")
    implementation("io.ktor:ktor-server-sse:3.6.0")
    implementation("org.locationtech.jts:jts-core:1.20.0")
    runtimeOnly("org.slf4j:slf4j-nop:2.0.17")
    testImplementation(kotlin("test"))
    testImplementation("io.ktor:ktor-server-test-host:3.6.0")
}

application {
    // Main function lives in Main.kt
    mainClass.set("org.lsoffice.MainKt")
}

val npmCi by tasks.registering(Exec::class) {
    commandLine("npm", "ci", "--no-audit", "--no-fund")
    inputs.files("package.json", "package-lock.json")
    outputs.dir("node_modules")
}
val npmTest by tasks.registering(Exec::class) {
    dependsOn(npmCi)
    commandLine("npm", "test")
    inputs.dir("frontend")
    inputs.files("package.json", "package-lock.json", "vitest.config.ts")
}
val npmBuild by tasks.registering(Exec::class) {
    dependsOn(npmTest)
    commandLine("npm", "run", "build")
    inputs.dir("frontend")
    inputs.files("package.json", "package-lock.json")
    outputs.file("build/generated-resources/planner.js")
}
tasks.processResources {
    dependsOn(npmBuild)
    from("build/generated-resources")
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(23)
}

tasks.register<JavaExec>("aggregateNumbat") {
    dependsOn(tasks.classes)
    classpath = sourceSets.main.get().runtimeClasspath
    mainClass.set("org.lsoffice.NumbatPreaggregateKt")
    args("src/main/resources/from-to-data/derived/numbat-regional-2024.csv")
}

tasks.register<JavaExec>("calibrateNumbat") {
    dependsOn(tasks.classes)
    classpath = sourceSets.main.get().runtimeClasspath
    mainClass.set("org.lsoffice.NumbatCalibrationKt")
    args("calibration/numbat-2024.json")
}
