from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx


@dataclass
class PropagationConfig:
    alpha: float = 0.7  # propagation factor
    beta: float = 60.0  # decay constant (minutes)
    max_iterations: int = 5
    tolerance: float = 1e-2
    passenger_window_min: float = 180.0


@dataclass
class PropagationSummary:
    iterations: int
    total_delay_before: float
    total_delay_after: float
    per_airport_delay: Dict[str, float]


def _edge_key(u: str, v: str, data: Dict) -> str:
    return data.get("flight_id", f"{u}->{v}")


def initialize_propagated_delay(G: nx.DiGraph, shock_airport: str, shock_minutes: float):
    """
    Initialise propagated delay on edges based on model predictions and an exogenous shock.

    All flights departing from the shock_airport receive an additional delay.
    """
    for u, v, data in G.edges(data=True):
        base = float(data.get("predicted_delay_min", 0.0))
        if u == shock_airport:
            base += shock_minutes
        data["propagated_delay_min"] = base


def _compute_arrival_time(edge_data: Dict) -> float:
    """
    Approximate arrival time in minutes of day including propagated delay.
    """
    dep = float(edge_data["sched_dep_minute_of_day"])
    duration = float(edge_data["duration_min"])
    delay = float(edge_data.get("propagated_delay_min", 0.0))
    return dep + duration + delay


def propagate_delays(G: nx.DiGraph, config: PropagationConfig) -> PropagationSummary:
    """
    Perform iterative cascading delay propagation across the network until stable.

    The propagation model is:
        Delay_j += Delay_i * alpha * exp(-Δt / beta)

    where Δt is the connection time gap either through:
        - same-aircraft continuation, or
        - passenger connections at the same airport.
    """
    # Baseline total delay using initial propagated delays.
    total_before = sum(
        float(d.get("propagated_delay_min", d.get("predicted_delay_min", 0.0)))
        for _, _, d in G.edges(data=True)
    )

    for u, v, data in G.edges(data=True):
        if "propagated_delay_min" not in data:
            data["propagated_delay_min"] = float(
                data.get("predicted_delay_min", 0.0)
            )

    # Precompute aircraft sequences (ordered by departure time).
    aircraft_to_edges: Dict[str, List[Tuple[str, str, Dict]]] = {}
    for u, v, data in G.edges(data=True):
        ac = data["aircraft_id"]
        aircraft_to_edges.setdefault(ac, []).append((u, v, data))

    for ac, edges in aircraft_to_edges.items():
        edges.sort(key=lambda e: e[2]["sched_dep_minute_of_day"])

    for iteration in range(1, config.max_iterations + 1):
        max_change = 0.0

        # Snapshot current delays to avoid intra-iteration feedback.
        current_delays: Dict[str, float] = {}
        for u, v, data in G.edges(data=True):
            key = _edge_key(u, v, data)
            current_delays[key] = float(data["propagated_delay_min"])

        for u, v, data in G.edges(data=True):
            key_j = _edge_key(u, v, data)
            base_pred = float(data.get("predicted_delay_min", 0.0))
            current = current_delays[key_j]

            dep_j = float(data["sched_dep_minute_of_day"])
            duration_j = float(data["duration_min"])
            congestion = float(data.get("congestion_level", 0.5))
            airport_load = float(data.get("airport_load", 0.5))

            amplification = 1.0 + 0.5 * congestion + 0.5 * airport_load
            contribution = 0.0

            # 1) Same-aircraft continuation.
            ac = data["aircraft_id"]
            for u_i, v_i, data_i in aircraft_to_edges[ac]:
                key_i = _edge_key(u_i, v_i, data_i)
                if key_i == key_j:
                    continue
                dep_i = float(data_i["sched_dep_minute_of_day"])
                if dep_i >= dep_j:
                    continue
                arr_i = dep_i + float(data_i["duration_min"]) + current_delays[key_i]
                gap = dep_j - arr_i
                if gap < 0:
                    gap = 0.0
                delta = current_delays[key_i] * config.alpha * math.exp(
                    -gap / config.beta
                )
                contribution += delta

            # 2) Passenger connections at the same airport.
            for pred_u in G.predecessors(u):
                data_i = G[pred_u][u]
                key_i = _edge_key(pred_u, u, data_i)
                dep_i = float(data_i["sched_dep_minute_of_day"])
                arr_i = dep_i + float(data_i["duration_min"]) + current_delays[key_i]
                gap = dep_j - arr_i
                if 0.0 <= gap <= config.passenger_window_min:
                    passenger_factor = (
                        float(data_i.get("passenger_connections", 0)) / 200.0
                    )
                    delta = (
                        current_delays[key_i]
                        * config.alpha
                        * passenger_factor
                        * math.exp(-gap / config.beta)
                    )
                    contribution += delta

            new_delay = base_pred + amplification * contribution
            change = abs(new_delay - current)
            if change > max_change:
                max_change = change
            data["propagated_delay_min"] = new_delay

        if max_change < config.tolerance:
            break

    total_after = sum(
        float(d.get("propagated_delay_min", 0.0)) for _, _, d in G.edges(data=True)
    )

    per_airport: Dict[str, float] = {}
    for u, v, data in G.edges(data=True):
        per_airport.setdefault(u, 0.0)
        per_airport[u] += float(data["propagated_delay_min"])

    return PropagationSummary(
        iterations=iteration,
        total_delay_before=total_before,
        total_delay_after=total_after,
        per_airport_delay=per_airport,
    )


__all__ = ["PropagationConfig", "PropagationSummary", "initialize_propagated_delay", "propagate_delays"]

