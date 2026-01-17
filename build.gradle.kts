plugins {
    kotlin("jvm") version "2.2.20"
    application
}

group = "org.lsoffice"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    testImplementation(kotlin("test"))
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