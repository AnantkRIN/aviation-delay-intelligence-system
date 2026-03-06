"""Scenario simulation: single and multi-scenario comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd

from .orchestrator import run_end_to_end_simulation, SimulationOutputs


@dataclass
class ScenarioResult:
    scenario_name: str
    total_network_delay: float
    passenger_impact: float
    recommended_actions: List[str]
    outputs: SimulationOutputs


def run_scenario_simulation(
    scenario_name: str,
    shock_airport: str = "DEL",
    shock_delay_min: float = 90.0,
    additional_shocks: Optional[Dict[str, float]] = None,
    weather_disruption_airport: Optional[str] = None,
) -> ScenarioResult:
    """
    Run a single scenario simulation.
    additional_shocks: e.g. {"BOM": 45} for BOM delay 45 min
    weather_disruption_airport: inject extra weather penalty at this airport
    """
    outputs = run_end_to_end_simulation(
        shock_airport=shock_airport,
        shock_delay_min=shock_delay_min,
        additional_shocks=additional_shocks,
        weather_disruption_airport=weather_disruption_airport,
    )

    passenger_impact = outputs.delay_metrics.get("cancelled_passenger_connections", 0.0)

    return ScenarioResult(
        scenario_name=scenario_name,
        total_network_delay=outputs.delay_metrics["total_delay_after"],
        passenger_impact=passenger_impact,
        recommended_actions=outputs.recommended_actions,
        outputs=outputs,
    )


def run_multi_scenario_comparison(
    scenarios: List[Dict],
) -> List[ScenarioResult]:
    """
    Run multiple scenarios and return comparison results.

    scenarios: list of dicts with keys:
        name, shock_airport, shock_delay_min, additional_shocks (optional)
    """
    results: List[ScenarioResult] = []
    for s in scenarios:
        r = run_scenario_simulation(
            scenario_name=s.get("name", "Unnamed"),
            shock_airport=s.get("shock_airport", "DEL"),
            shock_delay_min=s.get("shock_delay_min", 90.0),
            additional_shocks=s.get("additional_shocks"),
            weather_disruption_airport=s.get("weather_disruption_airport"),
        )
        results.append(r)
    return results
