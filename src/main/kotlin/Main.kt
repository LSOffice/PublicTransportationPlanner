package org.lsoffice

import javafx.application.Application
import javafx.concurrent.Worker
import javafx.geometry.Insets
import javafx.scene.Scene
import javafx.scene.control.Button
import javafx.scene.control.Label
import javafx.scene.control.TextField
import javafx.scene.layout.BorderPane
import javafx.scene.layout.HBox
import javafx.scene.layout.VBox
import javafx.scene.web.WebEngine
import javafx.scene.web.WebView
import javafx.stage.Stage

class MainApp : Application() {
    override fun start(primaryStage: Stage) {
        val latField = TextField("0")
        val lonField = TextField("0")
        val zoomField = TextField("2")
        latField.prefColumnCount = 8
        lonField.prefColumnCount = 8
        zoomField.prefColumnCount = 3

        val setViewBtn = Button("Set View")
        val getBoundsBtn = Button("Get Bounds")

        val webView = WebView()
        val engine: WebEngine = webView.engine
        val mapHtml = javaClass.getResource("/map.html")?.toExternalForm()
        if (mapHtml != null) engine.load(mapHtml) else engine.loadContent("<html><body>map not found</body></html>")

        setViewBtn.setOnAction {
            val lat = latField.text.toDoubleOrNull() ?: 0.0
            val lon = lonField.text.toDoubleOrNull() ?: 0.0
            val zoom = zoomField.text.toIntOrNull() ?: 2
            // Wait until the web engine has loaded the page
            if (engine.loadWorker.state == Worker.State.SUCCEEDED) {
                engine.executeScript("setView($lat, $lon, $zoom);")
            } else {
                engine.loadWorker.stateProperty().addListener { _, _, new ->
                    if (new == Worker.State.SUCCEEDED) {
                        engine.executeScript("setView($lat, $lon, $zoom);")
                    }
                }
            }
        }

        getBoundsBtn.setOnAction {
            // execute a small script to return the bounds
            if (engine.loadWorker.state == Worker.State.SUCCEEDED) {
                val boundsObj = engine.executeScript("(function(){ const b = getBounds(); return JSON.stringify(b); })()") as String
                println("Map bounds: $boundsObj")
            }
        }

        val topControls = HBox(8.0, Label("Lat:"), latField, Label("Lon:"), lonField, Label("Zoom:"), zoomField, setViewBtn, getBoundsBtn)
        topControls.padding = Insets(8.0)

        val root = BorderPane()
        root.top = topControls
        root.center = webView

        val scene = Scene(root, 900.0, 600.0)
        primaryStage.scene = scene
        primaryStage.title = "Public Transport Planner - Step 1"
        primaryStage.show()
    }
}

fun main() {
    Application.launch(MainApp::class.java)
}