"""LP-based network operations optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import pulp


@dataclass
class LinearProgrammingResult:
    objective_before: float
    objective_after: float
    operated_flights: List[str]
    cancelled_flights: List[str]


def optimize_network_operations(
    G: nx.DiGraph,
    delay_weight: float = 1.0,
    fuel_weight: float = 0.2,
    passenger_weight: float = 0.5,
) -> LinearProgrammingResult:
    """
    Formulate and solve a simplified network-wide optimization problem.

    Decision variable:
        x_f in {0,1}  -> 1 if flight is operated as planned, 0 if cancelled.

    Objective:
        Minimize sum_f x_f * (w_delay * propagated_delay
                              + w_fuel * distance
                              + w_pass * passenger_connections)
    """
    prob = pulp.LpProblem("NetworkDelayFuelPassengerOptimization", pulp.LpMinimize)

    flights: Dict[str, Dict] = {}
    for u, v, data in G.edges(data=True):
        fid = data.get("flight_id", f"{u}->{v}")
        flights[fid] = {
            "delay": float(
                data.get("propagated_delay_min", data.get("predicted_delay_min", 0.0))
            ),
            "distance": float(data.get("distance_km", 800.0)),
            "passengers": float(data.get("passenger_connections", 100)),
        }

    x = {fid: pulp.LpVariable(f"x_{fid}", lowBound=0, upBound=1, cat="Binary") for fid in flights}

    objective_terms = []
    for fid, attrs in flights.items():
        cost = (
            delay_weight * attrs["delay"]
            + fuel_weight * attrs["distance"]
            + passenger_weight * attrs["passengers"]
        )
        objective_terms.append(cost * x[fid])

    prob += pulp.lpSum(objective_terms)
    min_operated_fraction = 0.6
    prob += pulp.lpSum(x.values()) >= min_operated_fraction * len(x)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    operated, cancelled = [], []
    for fid, var in x.items():
        if var.value() >= 0.5:
            operated.append(fid)
        else:
            cancelled.append(fid)

    objective_before = 0.0
    for fid, attrs in flights.items():
        cost = (
            delay_weight * attrs["delay"]
            + fuel_weight * attrs["distance"]
            + passenger_weight * attrs["passengers"]
        )
        objective_before += cost

    objective_after = pulp.value(prob.objective)

    return LinearProgrammingResult(
        objective_before=objective_before,
        objective_after=objective_after,
        operated_flights=operated,
        cancelled_flights=cancelled,
    )
