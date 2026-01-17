# Metro Algorithm Debug Guide

## Instrumentation Added

The `buildNaturalNetworkFromGrid` function now logs at 8 critical checkpoints. Debug output is on by default (pass `debug=false` to MetroBuilder constructor to disable).

## Reading the Output

### STEP 1 — Places Identified

```
[MetroBuilder] STEP 1 — Places identified: 47
[MetroBuilder] Top 5 place values: [52140, 31850, 18902, 15670, 12450]
```

**What it tells you:**

- Number of grid cells collapsed into place clusters
- **< 5 places**: Grid collapse too aggressive (clusterRadius too large)
- **Flat values**: Demand calculations will fail downstream

---

### STEP 2 — Demand Graph

```
[MetroBuilder] STEP 2 — Demand graph built: maxDemand=1250000.0 avgDemand=125.4
```

**What it tells you:**

- `maxDemand ≈ 0` → Gravity model starved (tiny place values or distance exponent too high)
- `Very high max, low avg` → Only 1–2 dominant corridors (normal in monocentric cities like London)

---

### STEP 3 — Candidate Chains

```
[MetroBuilder] STEP 3 — Candidate chains found: 12
[MetroBuilder] Chain lengths (top 10): [8, 7, 6, 5, 5, 4, 4, 3, 2, 2]
```

**What it tells you:**

- **0 chains**: Demand threshold or geometry constraint (angle > 60°) too strict
- **Only size 2–3**: Geometry constraint stopping growth early
- **Plenty of 5+ length chains**: Good, demand chains are developing

---

### STEP 4 — Valid Chains Filtering

```
[MetroBuilder] STEP 4 — Valid chains after filtering: 8
[MetroBuilder] Valid chain lengths: [8, 7, 6, 5, 5, 4, 4, 3]
```

**What it tells you:**

- **Drops to 0 from STEP 3**: `minStationsPerLine` or `minCorridorLengthMeters` killing everything
- **Only 1 survives**: Explains "single line only" problem

---

### STEP 0 — Corridor Classification (THE KEY ONE)

```
[MetroBuilder] STEP 0 — Corridor classification:
[MetroBuilder]   RADIAL_TRUNK → 5 chains
[MetroBuilder]   CORE_DISTRIBUTOR → 2 chains
[MetroBuilder]   ORBITAL → 1 chains
[MetroBuilder]   NOT_METRO → 0 chains
```

**What it tells you:**

- **All RADIAL_TRUNK**:
  - `fracInCore >= 0.3` too permissive
  - `coreRadius` too large relative to city size
  - OR: City is genuinely monocentric (valid!)
- **Everything NOT_METRO**:
  - Classification thresholds too strict
  - Check `totalLength >= 8000`, `avgDemandPerKm > 500`

- **ORBITAL = 0**:
  - Very common. Geometry rule `distMidFromCenter in 2000..6000` not satisfied
  - Suggests places don't form orbital patterns (need more spatial spread)

---

### STEP 8 — Final Corridor Selection

```
[MetroBuilder] STEP 8 — Final corridors selected: 4
[MetroBuilder]   Line type RADIAL_TRUNK with 8 nodes
[MetroBuilder]   Line type RADIAL_TRUNK with 7 nodes
[MetroBuilder]   Line type CORE_DISTRIBUTOR with 5 nodes
[MetroBuilder]   Line type ORBITAL with 4 nodes
```

**What it tells you:**

- **Empty**: Redundancy filter `alreadyCovered < chain.size * 0.5` rejecting everything
  - Places are sparse; overlapping is high
- **Only 1**: Overlap threshold too strict

---

### STEP 5–7 — Station Pruning Per Line

```
[MetroBuilder] Line RT1 (RADIAL_TRUNK): initial chain size=8, seed node=2, seedValue=52140
[MetroBuilder] Line RT1 finalized: stations=6
```

**What it tells you:**

- Initial chain vs. finalized stations delta:
  - **Big drop (8 → 2)**: Terminal thinning firing too aggressively
  - **Small drop (8 → 7)**: Healthy pruning
  - **RADIAL dies to 1–2**: Check `terminateThresholdDistMeters` too small

---

### FINAL — Built Lines

```
[MetroBuilder] FINAL — Built lines: 4
[MetroBuilder]   RT1: type=RADIAL_TRUNK, stations=6, length=12.5 km
[MetroBuilder]   RT2: type=RADIAL_TRUNK, stations=5, length=9.8 km
[MetroBuilder]   CD1: type=CORE_DISTRIBUTOR, stations=4, length=3.2 km
[MetroBuilder]   O1: type=ORBITAL, stations=3, length=8.1 km
```

**What it tells you:**

- Success metric: Mix of types, stations > 3 per line, reasonable lengths

---

## The Three Failure Modes Decoded

### ❌ "Only Radial Trunks" (No CD or O)

**Look for:**

- STEP 0 output showing only RADIAL_TRUNK
- Check `coreRadius` vs. actual city extent
- Check `fracInCore >= 0.3` — is it matching everything?

**Fix:**

- Tighten `fracInCore` threshold (e.g., >= 0.5)
- Reduce `coreRadius` relative to city diameter
- Increase `totalLength >= 8000` if city is small

---

### ❌ "Only One Line" (STEP 8 shows 1 line selected)

**Look for:**

- STEP 4: Valid chains > 1
- STEP 8: Final corridors = 1
- Means overlap rejection is working too well

**Fix:**

- Increase overlap threshold from `0.5` to `0.6` or `0.7`:
  ```kotlin
  if (alreadyCovered < chain.size * 0.7) {  // was 0.5
      finalCorridors.add(chain to type)
  }
  ```

---

### ❌ "No Lines at All"

**Look for in order:**

1. STEP 1: Are places > 5?
2. STEP 3: Are candidate chains > 0?
3. STEP 4: Do valid chains exist after filtering?
4. STEP 0: Do any chains pass classification?
5. STEP 8: Why was STEP 8 empty?

**Culprits (priority order):**

1. `minCorridorLengthMeters` too high
2. `minStationsPerLine` too high
3. `avgDemandPerKm > 500` in classification killing everything
4. Terminal thinning collapsing both directions

---

## How to Use

When you see unexpected behavior:

1. **Enable debug** (on by default)
2. **Run `/suggestions` with test polygon**
3. **Watch terminal output** — find the first step that collapses
4. **Read this guide** for that step
5. **Adjust ONE threshold** based on guidance
6. **Repeat**

Never blindly tweak. The logs tell you exactly where to look.
