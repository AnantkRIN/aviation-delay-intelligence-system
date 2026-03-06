## AI Aviation Delay Propagation and Network Optimization System

Industry-level aviation operations decision support system combining **machine learning**, **network science**, **operations research**, **simulation modeling**, **decision support**, and **interactive visualization**.

### Features

- **Delay prediction:** RandomForest regression with confidence intervals (25th–75th percentile)
- **Weather risk integration:** Simulated `data/weather_risk.csv` with storm/wind risk; penalty = storm_probability × 20
- **Aircraft rotation modeling:** Delay propagation via tail sequences (e.g. DEL→BOM→BLR→MAA) with turnaround buffer (default 30 min)
- **Network modelling:** Airport–flight graph (NetworkX) with rich edge attributes
- **Cascading propagation:** Multi-hop delay propagation with exponential decay
- **Optimization:** Dijkstra routing, Kruskal MST, LP-based operate/cancel decisions
- **Scenario simulation:** Compare multiple disruption scenarios (single/multi-shock, weather)
- **Decision support:** Airport risk ranking, authority alerts, recommended actions
- **Interactive dashboard:** Streamlit with network map, heatmaps, scenario comparison
- **REST API:** FastAPI for programmatic simulation
- **Structured logging:** `logs/simulation.log`

### Quick Start

```bash
pip install -r requirements.txt
python -m src.main
```

**Dashboard:**
```bash
python -m streamlit run dashboard/app.py
```
(Or `streamlit run dashboard/app.py` if streamlit is on PATH)

**API:**
```bash
uvicorn api_layer.main:app --reload
```

### Project Structure

```
aviation-delay-intelligence-system/
├── src/
│   ├── data_layer/          # Flight data, weather risk
│   ├── prediction_engine/   # ML delay model + confidence
│   ├── propagation_engine/ # Delay propagation + aircraft rotation
│   ├── optimization_engine/# Dijkstra, MST, LP
│   ├── decision_engine/    # Alerts, risk ranking, recommendations
│   ├── simulation_engine/  # Orchestration + scenario engine
│   ├── visualization_engine/# Advanced plots
│   ├── api_layer/          # (FastAPI at project root)
│   └── utils/              # Logging
├── api_layer/              # FastAPI app (uvicorn api_layer.main:app)
├── dashboard/              # Streamlit app
├── data/                   # sample_flights.csv, weather_risk.csv, figures/
├── logs/                   # simulation.log
└── research/
```

### API Example

```bash
POST /simulate_delay
{
  "origin": "DEL",
  "destination": "BOM",
  "delay": 90,
  "passengers": 210
}

Response:
{
  "affected_airports": ["DEL", "BOM", ...],
  "flights_affected": 54,
  "passenger_impact": 2258,
  "recommended_actions": [...],
  "total_network_delay": 23183.65,
  "optimization_reduction_pct": 58.79
}
```

### Console Output (Operations Dashboard Style)

```
====================================================
  AI AVIATION OPERATIONS CONTROL SYSTEM
====================================================

--- Delay Prediction Performance ---
--- Network Delay Impact ---
--- Passenger Disruption ---
--- Optimized Passenger Route (Dijkstra) ---
--- Airport Risk Ranking ---
--- Optimization Effectiveness ---
--- Recommended Actions ---
====================================================
```

### Research Documentation

See `research/` and `architecture.md` for methodology, mathematical models, and case studies.
