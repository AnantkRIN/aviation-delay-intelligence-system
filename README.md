## AI-Driven Flight Delay Propagation and Network Optimization System

This repository contains a research-grade prototype of an airline Operations Control Centre (OCC) decision support system. It combines **machine learning**, **graph algorithms**, and **operations research** to understand, simulate, and optimize **flight delay propagation** across an aviation network.

- **Delay prediction:** RandomForest regression model using weather, congestion, historical delay, distance, and airport load.
- **Network modelling:** Flights are represented as edges in a directed airport graph (NetworkX).
- **Cascading propagation:** Multi-hop delay propagation model using exponential decay and congestion/connection multipliers.
- **Optimization layer:** Dijkstra, Kruskal MST, and a linear programming model to reduce overall delay, fuel proxy, and passenger disruption.
- **Simulation engine:** End-to-end scenario with a configurable “storm” at a major hub (e.g. Delhi).
- **Visual dashboards:** Network delay graph, airport delay heatmap, and optimization impact comparison.

### Quick Start

```bash
pip install -r requirements.txt
python -m src.main
```

On first run, a synthetic yet realistic Indian domestic flight dataset is generated under `data/`. The system then:

1. Trains a delay prediction model.
2. Builds the airport flight network.
3. Injects a **90-minute storm delay at Delhi (DEL)**.
4. Propagates delays across connecting flights and aircraft rotations.
5. Optimizes passenger routing and network connectivity.
6. Solves a linear program that decides which flights to operate vs. cancel/retime.
7. Produces figures under `data/figures/`.

### Console Outputs (High-Level)

Running `python -m src.main` prints:

- **Predicted delay quality** – train/test MAE and \(R^2\).
- **Network propagation summary** – total delay before vs. after, convergence iterations, reduction percentage.
- **Passenger-centric route** – least-delay path between two airports using Dijkstra on propagated delays.
- **Connectivity optimization** – MST weight as an effective connectivity baseline.
- **LP control impact** – composite objective before/after and % reduction.
- **Passenger impact analysis** – estimated disrupted connecting passengers from cancelled flights.

### Repository Layout

- `src/data_engine.py` – synthetic data generator and preprocessing utilities.
- `src/delay_prediction_model.py` – RandomForest delay regressor.
- `src/network_graph.py` – airport–flight graph construction with rich edge attributes.
- `src/delay_propagation_engine.py` – exponential-decay cascading propagation model.
- `src/route_optimizer.py` – Dijkstra and Kruskal-based network optimizers.
- `src/linear_programming_engine.py` – PuLP-based schedule control model.
- `src/simulation_engine.py` – orchestration of ML, graph, propagation, and optimization.
- `src/visualization_engine.py` – visual analytics dashboards using matplotlib / seaborn.
- `research/` – research-style documentation (methodology, models, and case study).
- `architecture.md` – detailed software and data-flow architecture.

### Real-World Relevance

This prototype is inspired by real airline OCC workflows:

- **Predictive control:** Estimating delay risk before departure.
- **Network-aware reasoning:** Understanding how disruptions at a hub ripple through rotations and connections.
- **Decision support:** Evaluating trade-offs between operating a late flight, cancelling it, or rerouting passengers.

The design is suitable as a **final-year engineering project** at the level of an IIT/DRDO lab prototype and can be extended into publishable research.

For full mathematical details, algorithms, and a narrative case study, see the documents in the `research/` folder and `architecture.md`.

