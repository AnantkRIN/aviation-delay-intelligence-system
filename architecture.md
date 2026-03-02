## System Architecture – Aviation Delay Intelligence System

This document explains the architecture of the **AI-Driven Flight Delay Propagation and Network Optimization System**. The goal is to approximate the behaviour of an airline Operations Control Centre (OCC) with:

- A **data layer** for ingesting and synthesizing flight schedules.
- A **machine learning layer** for delay prediction.
- A **graph + propagation layer** for modelling network effects.
- An **optimization layer** for control decisions.
- A **simulation and visualization layer** for experimentation and insight.

### High-Level Flow

1. **Data Engine (`data_engine.py`):**
   - Generates or loads flight-level data (origin, destination, distance, weather, congestion, airport load, previous delay, aircraft tail, schedule times).
   - Provides train/test splits and feature matrices for ML.

2. **Delay Prediction Model (`delay_prediction_model.py`):**
   - RandomForestRegressor learns a mapping:
     \[
     f(\text{weather}, \text{congestion}, \text{prev\_delay}, \text{distance}, \text{airport\_load}) \rightarrow \text{expected delay (minutes)}
     \]
   - Outputs predicted delays for each flight in the dataset.

3. **Network Graph (`network_graph.py`):**
   - Builds a directed graph \(G = (V, E)\) where:
     - \(V\): airports (nodes).
     - \(E\): flights (directed edges with attributes).
   - Edge attributes include:
     - `duration_min` (from distance and cruise speed).
     - `predicted_delay_min` (from ML model).
     - `aircraft_id` (for tail-rotation constraints).
     - `turnaround_min` (function of load and congestion).
     - `passenger_connections` (proxy for connecting flows).
     - `sched_dep_minute_of_day`, `congestion_level`, `airport_load`.

4. **Delay Propagation Engine (`delay_propagation_engine.py`):**
   - Injects an exogenous shock (e.g. 90-minute weather delay at DEL).
   - Propagates delays iteratively across:
     - **Same-aircraft continuations** (tail rotations).
     - **Passenger connections** at the same airport.
   - Uses the exponential-decay model:
     \[
     \Delta_j \mathrel{+}= \Delta_i \cdot \alpha \cdot e^{-\Delta t / \beta}
     \]
     with \(\alpha\) a propagation factor and \(\beta\) a decay constant.

5. **Route & Connectivity Optimizers (`route_optimizer.py`):**
   - **Dijkstra:** Computes least-delay passenger routes using weights \( w = \text{duration} + \text{propagated delay} \).
   - **Kruskal MST:** Derives a minimum spanning connectivity backbone to minimize effective travel time while preserving reachability.

6. **Linear Programming Engine (`linear_programming_engine.py`):**
   - Decision variable \(x_f \in \{0,1\}\) indicates whether flight \(f\) is operated.
   - Objective:
     \[
     \min \sum_f x_f \left( w_d \cdot \Delta_f + w_f \cdot \text{distance}_f + w_p \cdot \text{passengers}_f \right)
     \]
   - Constraint: operate at least a fraction of flights to maintain network connectivity.
   - Solved using PuLP’s CBC solver.

7. **Simulation Engine (`simulation_engine.py`):**
   - Orchestrates the complete flow: data → ML → graph → propagation → optimization.
   - Returns a structured `SimulationOutputs` object with:
     - Delay metrics.
     - Best route information.
     - MST statistics.
     - LP control decisions and impact.

8. **Visualization Engine (`visualization_engine.py`):**
   - Generates:
     - **Network delay graph**: nodes as airports, edges coloured by propagated delay.
     - **Delay heatmap**: accumulated delay per airport.
     - **Optimization comparison chart**: before/after propagation and LP objective.

9. **Entry Point (`main.py`):**
   - Ties everything together and produces research-style console logs and figures.

### Design Principles

- **Modularity:** Each concern (data, ML, graph, propagation, optimization) is in its own module with clean interfaces.
- **Explainability:** The delay propagation and optimization models are expressed with explicit mathematical forms to support research and teaching.
- **Extensibility:** Real-world data from sources such as BTS / DGCA can replace the synthetic generator with minimal code changes.
- **Research Orientation:** The code is structured so that:
  - Propagation parameters (\(\alpha, \beta\)) can be tuned or learned.
  - Alternative ML models (e.g. XGBoost) can be swapped in.
  - New constraints/cost terms can be added to the LP model.

For methodological and mathematical details, see `research/methodology.md` and `research/mathematical_model.md`.

