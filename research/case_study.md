## Case Study – Storm at Delhi and Cascading Delays

This case study illustrates how the **Aviation Delay Intelligence System** behaves under a major disruption at a hub airport. The scenario is intentionally stylized but reflects realistic operational dynamics.

### Scenario Description

- **Network:** Synthetic Indian domestic schedule with flights between DEL, BOM, BLR, MAA, HYD, CCU, and other airports.
- **Shock Event:** A convective storm system affects **Indira Gandhi International Airport (DEL)**.
- **Primary Impact:** All flights departing from DEL experience an initial **90-minute departure delay** due to ground stops and reduced arrival rates.

### Step 1 – Baseline Prediction

1. The system first trains a **RandomForestRegressor** on historical-style synthetic data.
2. Each flight receives a **baseline delay prediction** based on:
   - Weather severity along the route,
   - Congestion level,
   - Previous leg delay of the aircraft,
   - Route distance,
   - Airport load.
3. Model metrics (e.g. MAE and \(R^2\)) are printed to validate predictive performance.

At this point, delays are **per-flight** and **local**; network interactions are not yet applied.

### Step 2 – Graph Construction

The schedule is converted into a **directed network**:

- Nodes represent airports (DEL, BOM, BLR, etc.).
- Edges represent individual flights with attributes:
  - `duration_min`, `predicted_delay_min`,
  - `aircraft_id`, `turnaround_min`,
  - `passenger_connections`, `congestion_level`, `airport_load`.

This network forms the substrate on which delay propagation and optimization will operate.

### Step 3 – Shock Injection and Propagation

1. A **90-minute shock** is injected to all flights departing DEL.
2. The delay propagation engine runs several iterations:
   - For each flight, propagated delay is updated using:
     - Delays on previous legs of the same aircraft (tail rotation).
     - Delays on inbound connecting flights at the same airport.
   - The exponential-decay model ensures that:
     - Short connections carry higher propagation risk.
     - Distant connections (in time) see diminished influence.
3. Congestion and load multipliers amplify delays at already-stressed hubs.

**Observation:** After a few iterations, the system converges to a stable pattern of propagated delays across the network, revealing:

- Delays radiating out of DEL to cities such as BOM, BLR, and MAA.
- Secondary disruptions on outbound flights from those airports due to late arrivals and missed connections.

### Step 4 – Passenger Route Optimization

We examine a passenger planning to travel from **Delhi (DEL)** to **Bengaluru (BLR)**.

1. Dijkstra’s algorithm is run on the propagated-delay-weighted network.
2. The system may select:
   - A **direct DEL–BLR flight** if, despite delay, it remains the fastest option.
   - An **indirect route** (e.g. DEL–HYD–BLR or DEL–BOM–BLR) if those paths yield better effective travel time.

The console reports:

- The chosen path (sequence of airports).
- Total travel time including propagated delays.

This mimics how an OCC might propose **re-accommodation options** for disrupted passengers.

### Step 5 – Connectivity Backbone via MST

Using Kruskal’s algorithm on the undirected version of the network:

- The system constructs a **minimum spanning tree (MST)** that preserves connectivity with minimal effective travel time.
- This MST can be interpreted as a **resilient backbone** – a minimal set of routes that keep the network connected even under disruption.

Key outputs:

- Number of airports and flights in the original network.
- Total weight of the MST (sum of effective travel times).

### Step 6 – Linear Programming Control

The linear programming model introduces **binary decisions** to:

- Operate or cancel / retime individual flights.

Objective:

- Minimize a weighted combination of:
  - Propagated delay,
  - Fuel proxy (distance),
  - Passenger disruption (connecting passengers).

Constraint:

- At least a given fraction (e.g. 60%) of flights must remain operational.

Outcomes:

- A subset of flights is **cancelled** or significantly retimed to break problematic feedback loops.
- Total composite cost (delay + fuel proxy + disruption) is reduced by a measurable percentage.
- The system reports:
  - Number of operated vs. cancelled flights.
  - Estimated passenger connections disrupted by cancellations.

### Step 7 – Visual Analytics

The visualization engine produces figures that can be embedded into a report or presentation:

- **Network delay graph:** Airports as nodes; edges coloured by propagated delay.
- **Airport delay heatmap:** Bar chart of accumulated delay per airport, highlighting hotspots.
- **Before/after optimization comparison:** Bar chart showing:
  - Network delay before vs. after propagation.
  - LP objective before vs. after control.

### Interpretation and Discussion

This pipeline demonstrates how an integrated AI/OR system can:

- Quantify the impact of a major weather event at a hub.
- Reveal non-obvious secondary and tertiary delay effects.
- Suggest passenger-aware rerouting strategies.
- Provide principled guidance on where to cancel flights to maximize system-wide benefit.

In a real deployment, the same framework can be connected to:

- Live operational data feeds,
- More sophisticated ML models (e.g. XGBoost, deep learning),
- Richer LP/MIP formulations with fleet and crew constraints.

For a final-year project, this case study provides a **coherent narrative** from data to algorithms to operational insight, suitable for discussion with faculty, examiners, and potential research collaborators.

