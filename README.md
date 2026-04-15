# PublicTransportationPlanner

PublicTransportationPlanner is a Kotlin-based web application that analyses geospatial population data and generates public-transport network suggestions and visualisations. It reads CSV population/coordinate data, builds a metro-style transport model using a gravity demand model and corridor classification algorithm, and serves an interactive HTML map via a built-in HTTP server.

## Features

- **Interactive map** – Serves `map.html` (Leaflet-based) on `http://localhost:5000` for exploring generated network proposals.
- **Automatic network generation** – `MetroBuilder` clusters population grid cells, builds a demand graph, identifies corridor chains (radial trunks, core distributors, orbitals), and selects non-overlapping lines.
- **PTAL enrichment** – `PtalLookup` provides Public Transport Accessibility Level data for Greater London (TfL PTAL 2015, ~4 835 LSOAs) to weight demand calculations.
- **Population density lookup** – `/density` endpoint finds the nearest grid point in a CSV population dataset for any WGS84 coordinate.
- **CORS proxy** – `/proxy` endpoint forwards external data requests (e.g. GitHub-hosted GeoJSON) to avoid browser CORS restrictions.
- **Transport suggestions** – `/suggestions` endpoint returns proposed metro lines as GeoJSON for a given geographic polygon.
- **Haversine distances** – All spatial calculations use accurate great-circle distances.
- **Dijkstra journey metrics** – Average travel times across all station pairs are computed and logged.

## Requirements

- JDK 17 or newer (project targets JVM toolchain 23)
- Gradle wrapper included – no separate Gradle installation needed

## Quickstart

```bash
# Build
./gradlew build

# Run (starts HTTP server on port 5000–5010)
./gradlew run
```

Then open **http://localhost:5000** in your browser to view the interactive map.

To run the packaged jar directly (after `./gradlew build`):

```bash
java -jar build/libs/PublicTransportationPlanner-1.0-SNAPSHOT.jar
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Serves the interactive `map.html` visualisation |
| `GET /density?lon=&lat=&max_m=` | GET | Nearest-neighbour population density lookup |
| `GET /proxy?url=&lat=&lon=` | GET | CORS proxy for external data sources |
| `GET /suggestions` | GET/POST | Returns proposed metro lines as GeoJSON |

## Project Layout

```
PublicTransportationPlanner/
├── src/main/kotlin/
│   ├── Main.kt           # HTTP server, endpoints, haversine helper
│   ├── MetroBuilder.kt   # Network generation algorithm
│   └── PtalLookup.kt     # TfL PTAL 2015 nearest-neighbour lookup
├── src/main/resources/
│   ├── map.html          # Leaflet-based interactive map
│   ├── ptal_spatial.csv  # PTAL data (Greater London LSOAs)
│   └── *.csv             # Population grid datasets
├── scripts/
│   └── sample.csv        # Example input CSV
├── build.gradle.kts
├── settings.gradle.kts
├── DEBUG_GUIDE.md        # Tuning guide for the network algorithm
└── README.md
```

## Configuration & Data

- Population data CSVs should be placed in `src/main/resources/` (example: `gbr_pd_2020_1km_ASCII_XYZ.csv`).
- PTAL enrichment is automatically enabled when `ptal_spatial.csv` is present on the classpath; otherwise it degrades gracefully.
- A `GITHUB_TOKEN` environment variable can be set to authenticate proxied requests to the GitHub API.

## Development

```bash
# Build and test
./gradlew build

# Run tests only
./gradlew test

# Run with debug output (MetroBuilder logs are on by default)
./gradlew run
```

See [DEBUG_GUIDE.md](DEBUG_GUIDE.md) for a detailed walkthrough of the network algorithm's diagnostic output and how to tune its parameters.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
