"""FastAPI application for aviation delay simulation."""

from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

# Ensure src is on path when running: uvicorn api_layer.main:app
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation_engine.orchestrator import run_end_to_end_simulation

app = FastAPI(
    title="AI Aviation Operations Control API",
    description="Simulate delay propagation and get recommended actions",
    version="2.0.0",
)


class SimulateDelayRequest(BaseModel):
    origin: str = "DEL"
    destination: str = "BOM"
    delay: float = 90.0
    passengers: int = 210


class SimulateDelayResponse(BaseModel):
    affected_airports: List[str]
    flights_affected: int
    passenger_impact: float
    recommended_actions: List[str]
    total_network_delay: float
    optimization_reduction_pct: float


@app.post("/simulate_delay", response_model=SimulateDelayResponse)
def simulate_delay(req: SimulateDelayRequest) -> SimulateDelayResponse:
    """
    Simulate delay propagation from origin airport and return impact metrics.
    """
    outputs = run_end_to_end_simulation(
        shock_airport=req.origin,
        shock_delay_min=req.delay,
    )

    per_airport_delay = {}
    for u, v, data in outputs.network.edges(data=True):
        per_airport_delay.setdefault(u, 0.0)
        per_airport_delay[u] += float(data.get("propagated_delay_min", 0.0))

    affected = [a for a, d in per_airport_delay.items() if d > 0]
    flights_affected = outputs.network.number_of_edges()
    passenger_impact = outputs.delay_metrics.get("cancelled_passenger_connections", 0.0)
    passenger_impact += req.passengers  # Add input passengers as affected

    return SimulateDelayResponse(
        affected_airports=affected,
        flights_affected=flights_affected,
        passenger_impact=passenger_impact,
        recommended_actions=outputs.recommended_actions,
        total_network_delay=outputs.delay_metrics["total_delay_after"],
        optimization_reduction_pct=outputs.delay_metrics.get("lp_reduction_pct", 0.0),
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "aviation-ops-api"}
