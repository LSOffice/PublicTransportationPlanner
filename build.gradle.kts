plugins {
    kotlin("jvm") version "2.2.20"
    application
    id("org.openjfx.javafxplugin") version "0.0.13"
}

group = "org.lsoffice"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.openjfx:javafx-controls:20.0.2")
    implementation("org.openjfx:javafx-web:20.0.2")
    testImplementation(kotlin("test"))
}

javafx {
    version = "20.0.2"
    modules = listOf("javafx.controls", "javafx.web")
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