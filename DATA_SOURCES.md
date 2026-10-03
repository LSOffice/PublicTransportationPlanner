# Data sources and attribution

## Greater London coverage

- Source: Greater London Authority, [Statistical GIS Boundary Files for London](https://data.london.gov.uk/dataset/statistical-gis-boundary-files-for-london-20od9).
- Resource used: `gla.zip`, downloaded from the dataset's “Greater London boundary” resource.
- Source archive SHA-256: `6c2c52af0a1b6e921c0b2c5ec3cda6256dd470527cfa21ea79521bfc97be1fee`.
- Processing: the British National Grid shapefile was transformed from EPSG:27700 to EPSG:4326 and topology-preserving simplification of 0.00035 degrees was applied. The resulting validation/display polygon is `london_coverage.json`.
- Derived resource SHA-256: `bbb3d2342fcd801b80fb62455a5a14ecfbd6e9008280dd4f123df949f6292e6b`.
- Licence: Open Government Licence v2.
- Required attribution: “Contains National Statistics data © Crown copyright and database right 2015” and “Contains Ordnance Survey data © Crown copyright and database right 2015”.

## Population grid

- Resource: `gbr_pd_2020_1km_ASCII_XYZ.csv`.
- Shape: 499,366 data rows plus header; columns `X,Y,Z` (`longitude,latitude,value`).
- SHA-256: `d2dd664fd2b37dddae48acfb75dd0b1c72f574f52b5679c13e87e3e095f88deb`.
- Runtime treatment: only cells whose centres fall inside the authoritative Greater London polygon are loaded into the planner service. `value` is described as a density/demand proxy.
- Provenance/licence: not established by the repository. Verify before redistribution.

## PTAL

- Source description in the bundled data: Transport for London PTAL 2015, LSOA 2011.
- Runtime resource: `ptal_spatial.csv`, 4,835 data rows plus header.
- Runtime SHA-256: `7cbee2fb962c1edb075defab89da7abc6b4b30c45e041fc1f94f8afbb69cb3d7`.
- Metadata-rich source SHA-256: `adabc4e37ac489165c0915c61f356586af77802cca98a581783a200677039025`.
- Licence recorded by the project: Open Government Licence v2.
- Runtime treatment: nearest record within 3 km; otherwise the PTAL multiplier is exactly neutral (`1.0`).

## Map tiles and station names

- Base map: OpenStreetMap tiles and attribution, loaded at runtime.
- Optional station naming: OpenStreetMap Nominatim. The local proxy accepts only HTTPS requests to `nominatim.openstreetmap.org` and does not forward credentials.
