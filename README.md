# PublicTransportationPlanner

PublicTransportationPlanner is a small Kotlin-based utility to analyze geospatial population data and generate simple public-transport planning visualizations (map output). It reads CSV population/coordinate data, builds a basic transport model, and produces an HTML map and related outputs for exploration.

**Features**

- **Input:** Reads CSV population/coordinate files (example: `src/main/resources/gbr_pd_2020_1km_ASCII_XYZ.csv`).
- **Output:** Produces `map.html` (visualization) and other derived artifacts.
- **Implementation:** Command-line Kotlin application using Gradle build.

**Requirements**

- JDK 11 or newer
- Gradle (wrapper included)

**Quickstart**

- Build: `./gradlew build`
- Run: `./gradlew run` or run the fat jar under `build/libs` (if configured).

If you prefer to run the compiled classes directly with the Gradle run task:

```
./gradlew run
```

**Project layout**

- `src/main/kotlin` : Kotlin source (entrypoint `Main.kt`, helper `MetroBuilder.kt`).
- `src/main/resources` : Example input CSV and `map.html` template.
- `build/` : Gradle build outputs.

**Usage notes**

- Replace or provide CSV input files in `src/main/resources` or adjust the runtime arguments if `Main.kt` supports custom input paths.
- After running, open the generated `map.html` in a browser to view the visualization.

**Development**

- Use the Gradle wrapper (`./gradlew`) to build and run locally.
- Tests (if present) run with `./gradlew test`.

**Contact / License**

- No license specified — add a `LICENSE` file if you intend to open-source this project.
