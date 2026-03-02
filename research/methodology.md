## Methodology – AI-Driven Flight Delay Propagation and Network Optimization

This document describes the methodology used to design, model, and evaluate the **Aviation Delay Intelligence System**. The project integrates **machine learning**, **graph theory**, and **operations research** into a unified pipeline suitable for research and advanced engineering coursework.

### 1. Data Design and Feature Engineering

The system uses a synthetic yet realistic dataset modelled on domestic Indian aviation traffic:

- **Entities:**
  - Flights: identified by `flight_id`, with origin, destination, distance, schedule, and assigned aircraft tail.
  - Airports: major Indian hubs and metros (DEL, BOM, BLR, MAA, HYD, etc.).

- **Operational features:**
  - `weather_severity` – a [0,1] scalar capturing disruption risk due to storms, visibility, etc.
  - `congestion_level` – a [0,1] scalar representing airspace and ground congestion.
  - `prev_delay_min` – delay of the aircraft’s previous leg (late inbound).
  - `airport_load` – proxy for airport utilization (0–1).
  - `distance_km` – great-circle distance between origin and destination.
  - `sched_dep_minute_of_day` – departure time in minutes since midnight.

The target variable is **arrival delay in minutes** (`actual_delay_min`), generated using a ground-truth model that combines these factors plus stochastic noise.

This design allows:

- Controlled experiments without privacy or data availability issues.
- Reproducible training and evaluation.
- Straightforward swapping to real airline datasets in future work.

### 2. Delay Prediction Model

We train a **RandomForestRegressor** to approximate the mapping:

\[
(\text{weather severity}, \text{congestion}, \text{prev delay}, \text{distance}, \text{airport load}) \rightarrow \text{arrival delay (min)}.
\]

- **Model choice:** Random Forests are robust to non-linearities and interactions, handle mixed scales well, and provide feature importance signals.
- **Training protocol:**
  - Simple random train/test split.
  - Evaluation using MAE and \(R^2\) to capture absolute error and explained variance.
- **Outputs:**
  - Per-flight predicted delay.
  - Global metrics used later in the simulation summary.

The learned delay is treated as a **baseline prediction** prior to network-level interactions.

### 3. Network Graph Construction

We construct a directed graph \(G = (V, E)\) using NetworkX:

- **Nodes (V):** airports.
- **Edges (E):** scheduled flights.

Each edge is enriched with:

- `duration_min` – from distance and nominal cruise speed.
- `predicted_delay_min` – output from the ML model.
- `aircraft_id` – to track tail-rotation constraints.
- `turnaround_min` – function of `airport_load`.
- `passenger_connections` – proxy for connecting passengers affected by delays.
- `sched_dep_minute_of_day`, `congestion_level`, `airport_load`.

This creates a **multi-layer representation** combining schedule, operations, and predicted delays.

### 4. Delay Propagation Modelling

The delay propagation engine implements a **multi-iteration cascading model**:

- An exogenous **shock** is injected at a chosen hub (e.g. 90 minutes at DEL).
- Each flight’s **propagated delay** starts from its baseline predicted delay plus any injected shock for departures at the shock airport.
- On each iteration, delays propagate via:
  1. **Same-aircraft continuation:** late inbound flights causing late outbound departures on the same tail.
  2. **Passenger connections:** disrupted connections at the same airport within a time-window.

The mathematical model is based on:

\[
\Delta_j \mathrel{+}= \Delta_i \cdot \alpha \cdot e^{-\Delta t / \beta},
\]

where:

- \(\Delta_i\) is the current delay of flight \(i\),
- \(\Delta_j\) is the delay of a downstream flight \(j\),
- \(\Delta t\) is the connection gap (arrival of \(i\) to departure of \(j\)),
- \(\alpha\) is a propagation factor (0.5–0.8),
- \(\beta\) is a decay constant (minutes).

Additional multipliers model:

- **Airport congestion** and **load** (amplifying factor).
- **Passenger volume** (stronger effect for heavily connecting flights).

Iterations continue until the **maximum change per edge** falls below a tolerance or a maximum number of iterations is reached.

### 5. Optimization Algorithms

Once delays are propagated, we apply three optimization techniques:

1. **Dijkstra’s Algorithm (Passenger Route Optimization):**
   - Edge weights = `duration_min` + `propagated_delay_min`.
   - Computes least-delay passenger routes between origin–destination pairs.
   - Highlights how routing decisions change when the network is disrupted.

2. **Kruskal’s Algorithm (Connectivity Backbone):**
   - Applied to the undirected version of the network with the same edge weights.
   - Derives a minimum spanning tree as a **backbone connectivity network** with minimal effective travel time.

3. **Linear Programming (Network-Level Control):**
   - Binary decision variable \(x_f\) decides whether a flight is operated vs. cancelled/retimed.
   - Objective trades off:
     - Propagated delay,
     - Fuel proxy (distance),
     - Passenger disruption (connecting passengers).
   - Constraint enforces a minimum fraction of flights must remain in operation.

### 6. Simulation Protocol

The **simulation engine** runs the following pipeline:

1. Load/generate data.
2. Train and evaluate the ML delay model.
3. Predict delays for the entire schedule.
4. Build the graph and inject an airport-level shock.
5. Run iterative delay propagation until convergence.
6. Compute:
   - Least-delay passenger route.
   - MST connectivity backbone.
7. Solve the linear program to determine which flights to operate.
8. Aggregate metrics:
   - Total delay before/after propagation.
   - Composite cost before/after LP control.
   - Number of flights and passenger connections disrupted.

### 7. Evaluation and Outputs

The system reports:

- **Model metrics:** MAE, \(R^2\).
- **Network metrics:** total propagated delay and reduction percentages.
- **Route metrics:** travel time along optimized routes.
- **Control metrics:** LP objective reduction and cancelled flights.
- **Passenger metrics:** disrupted connecting passengers.

Visualizations (network graph, heatmap, comparison chart) complement the numeric results and are suitable for inclusion in a project report or research paper.

