# Planner model

## Synthetic demand

The selected population cells are aggregated into at most 400 deterministic rectangular zones. Each zone retains a weighted centroid, population-demand proxy, and PTAL multiplier.

For zones `i` and `j`, the initial interaction is:

```text
F(i,j) = exp(-distance(i,j) / distanceDecay)
```

The default distance decay is 8 km. Production targets are proportional to population-demand weight. Attraction targets additionally apply the configured PTAL influence. Sixty iterative proportional-fitting passes scale rows and columns to their targets. Largest-remainder rounding preserves the exact configured journeys/day total. Intrazonal journeys are excluded.

The generated matrix is directed and explicitly labelled simulated. Saved projects store its inputs and `gravity-ipf-v1`, not thousands of derived rows.

## Routing graph

An editable project is compiled into station/line states. Primitive arrays store CSR offsets, destinations, base minutes, edge kinds, line indexes, and segment indexes. Ride edges connect adjacent stops in both directions; transfer edges connect different line states at a shared station.

Generalised time includes:

- initial expected wait `30 / trainsPerHour` minutes;
- in-vehicle time from segment distance and infrastructure speed;
- dwell time at each ride edge;
- transfer walking time;
- another expected wait when boarding a different line.

The custom binary min-heap stores node IDs in an `IntArray` and priorities in a `DoubleArray`. It uses lazy duplicate pushes: a popped entry is discarded when its priority is stale. This avoids boxed queue entries and the index-maintenance complexity of decrease-key.

Dijkstra uses a zero heuristic. A* uses straight-line distance divided by maximum active speed, plus a minimum boarding wait only when the current line cannot reach the destination without a transfer. The heuristic is a lower bound in the deterministic expected-time graph. Tests require Dijkstra and A* to return the same optimal cost.

## Service and disruptions

Service is frequency-based rather than timetable-based. Train runs depart each terminus at regular headways; loop and reverse operation use the line's ordered stations. Scenarios are deterministic:

- cancellation reduces effective line frequency and suppresses departures during its time window;
- track incident blocks a segment;
- weather slows only overground segments;
- signal failure applies a segment multiplier;
- station closure prevents routing through the station.

Simulation interpolation is geographic straight-line movement between station coordinates. It is not signalling, rolling-stock, platform, or microscopic passenger simulation.

## Fixed score

The score is bounded to 0–100:

```text
35% demand served
20% passenger-weighted journey-time benefit
15% peak capacity adequacy
10% connectivity and directness
10% construction efficiency
10% structural resilience
```

Raw component values, component scores, weights, and descriptions are returned together. Underground distance counts as three construction-equivalent kilometres. The journey-time reference is direct surface travel at the configured reference speed plus five minutes. Capacity assignment uses the configured peak-hour share and an all-or-nothing shortest route.

The score is useful for comparing variants under identical assumptions. It is not calibrated construction cost, revenue, economic benefit, demand forecasting, or safety assurance.
