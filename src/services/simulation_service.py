"""
Shared simulation service for terminal (full network) and dashboard (single flight disruption).

Terminal: simulate_full_network() -> full ML, propagation, LP, same as current CLI.
Dashboard: simulate_single_flight_disruption() -> build from historical data, inject delay, propagate 3 hops.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import networkx as nx
import pandas as pd

# Passenger impact constants (Phase 7)
AVG_PASSENGERS_PER_FLIGHT = 150
CONNECTION_RATIO = 0.35

# Propagation for single-flight disruption (Phase 6)
PROPAGATION_FACTOR = 0.5
MAX_PROPAGATION_HOPS = 3


@dataclass
class SingleDisruptionResult:
    """Result of simulate_single_flight_disruption."""

    network: nx.DiGraph
    origin: str
    destination: str
    delay_minutes: float
    weather_risk: float
    initial_delay: float  # delay + weather_delay
    affected_airports: List[str] = field(default_factory=list)
    affected_flights: int = 0
    total_passengers_affected: float = 0.0
    recommended_actions: List[str] = field(default_factory=list)
    per_airport_delay: Dict[str, float] = field(default_factory=dict)


def build_network_from_dataset() -> nx.DiGraph:
    """
    Build the base aviation network from the historical dataset.

    Steps:
    1. Load dataset using pandas
    2. Extract: origin, destination, distance, delay
    3. Construct directed graph (NetworkX)

    Nodes: airports
    Edges: flight routes
    Edge attributes: distance_km, historical_delay, duration_min (derived)
    """
    from ..data_layer import load_or_create_dataset

    df = load_or_create_dataset()
    # Aggregate by route (origin, destination)
    routes = (
        df.groupby(["origin", "destination"])
        .agg(
            distance_km=("distance_km", "mean"),
            historical_delay=("actual_delay_min", "mean"),
        )
        .reset_index()
    )

    G = nx.DiGraph()
    for _, row in routes.iterrows():
        u, v = row["origin"], row["destination"]
        G.add_node(u)
        G.add_node(v)
        distance_km = float(row["distance_km"])
        historical_delay = float(row["historical_delay"])
        duration_min = (distance_km / 800.0) * 60.0
        G.add_edge(
            u,
            v,
            distance_km=distance_km,
            historical_delay=historical_delay,
            duration_min=duration_min,
            predicted_delay_min=historical_delay,
            propagated_delay_min=0.0,
        )
    return G


def _propagate_delay_limited_hops(
    G: nx.DiGraph,
    origin: str,
    propagation_factor: float = PROPAGATION_FACTOR,
    max_hops: int = MAX_PROPAGATION_HOPS,
) -> Tuple[Set[str], int]:
    """
    Propagate delay through graph neighbors with limited hops.
    BFS: from edges with delay, propagate to outgoing edges with factor per hop.
    Returns (affected_airports, affected_flights count).
    """
    for u, v, data in G.edges(data=True):
        data["propagated_delay_min"] = float(data.get("propagated_delay_min", 0.0))

    # BFS by edges: level 0 = edges with delay; level k+1 = edges leaving head of level-k edges
    # Delay on edge (u,v) at level k = delay_at_u * (propagation_factor ** k)
    node_delay: Dict[str, float] = {}
    for u, v, data in G.edges(data=True):
        d = float(data["propagated_delay_min"])
        if d > 0:
            node_delay[v] = max(node_delay.get(v, 0), d)

    frontier = list(node_delay.keys())  # nodes that have delay
    if origin not in node_delay:
        node_delay[origin] = 0.0
    for u, v, data in G.edges(data=True):
        if float(data["propagated_delay_min"]) > 0:
            node_delay[origin] = max(
                node_delay.get(origin, 0),
                float(data["propagated_delay_min"]),
            )
            break

    for hop in range(1, max_hops + 1):
        factor = propagation_factor ** hop
        next_frontier = []
        for u in frontier:
            in_d = node_delay.get(u, 0)
            if in_d <= 0:
                continue
            for v in G.successors(u):
                edge_data = G[u][v]
                new_d = in_d * propagation_factor
                cur = float(edge_data.get("propagated_delay_min", 0))
                edge_data["propagated_delay_min"] = max(cur, new_d)
                node_delay[v] = max(node_delay.get(v, 0), edge_data["propagated_delay_min"])
                next_frontier.append(v)
        frontier = list(set(next_frontier))

    affected_nodes: Set[str] = set()
    for u, v, data in G.edges(data=True):
        if float(data.get("propagated_delay_min", 0)) > 0:
            affected_nodes.add(u)
            affected_nodes.add(v)
    affected_edges = sum(
        1 for _, _, d in G.edges(data=True) if float(d.get("propagated_delay_min", 0)) > 0
    )
    return affected_nodes, affected_edges


def _recommendations_for_disruption(
    affected_airports: List[str],
    affected_flights: int,
    total_passengers_affected: float,
) -> List[str]:
    """Generate operational recommendations for single-flight disruption (Phase 10)."""
    actions: List[str] = []
    actions.append("Deploy backup aircraft if available on the disrupted route")
    actions.append("Delay connecting departures to maintain crew and passenger connections")
    actions.append("Reroute passengers to alternate flights or carriers")
    actions.append("Prioritize high passenger flights for recovery resources")
    if affected_flights > 5:
        actions.append("Consider network-wide delay advisories for downstream airports")
    if total_passengers_affected > 500:
        actions.append("Activate passenger rebooking and compensation protocols")
    if affected_airports:
        actions.append(f"Monitor affected airports: {', '.join(affected_airports[:8])}")
    return actions


def simulate_single_flight_disruption(
    origin: str,
    destination: str,
    delay_minutes: float,
    passengers: int,
    weather_risk: float,
) -> SingleDisruptionResult:
    """
    Simulate a single flight disruption from the dashboard.

    1. Build network from historical dataset
    2. Inject delay on selected route; adjust with weather: weather_delay = delay * weather_risk, initial_delay = delay + weather_delay
    3. Propagate through graph neighbors (factor=0.5, max 3 hops)
    4. Track affected_airports, affected_flights
    5. Passenger impact: connecting_passengers = passengers * 0.35; total = passengers + connecting_passengers + affected_flights * 150
    6. Generate recommendations
    """
    G = build_network_from_dataset()

    weather_delay = delay_minutes * weather_risk
    initial_delay = delay_minutes + weather_delay

    # Inject delay on (origin, destination) edge(s)
    if G.has_edge(origin, destination):
        data = G[origin][destination]
        base = float(data.get("historical_delay", 0.0))
        data["propagated_delay_min"] = base + initial_delay
    else:
        # Add edge if missing (e.g. user picked a route not in dataset)
        G.add_edge(origin, destination, propagated_delay_min=initial_delay, historical_delay=0.0, duration_min=60.0, distance_km=800.0)

    affected_airports_set, affected_flights = _propagate_delay_limited_hops(
        G, origin, propagation_factor=PROPAGATION_FACTOR, max_hops=MAX_PROPAGATION_HOPS
    )
    affected_airports = sorted(affected_airports_set)

    # Ensure origin is in affected for display
    if origin not in affected_airports:
        affected_airports.insert(0, origin)

    # Passenger impact (Phase 7)
    connecting_passengers = passengers * CONNECTION_RATIO
    total_passengers_affected = (
        passengers
        + connecting_passengers
        + affected_flights * AVG_PASSENGERS_PER_FLIGHT
    )

    per_airport_delay = {}
    for u, v, data in G.edges(data=True):
        d = float(data.get("propagated_delay_min", 0.0))
        if d > 0:
            per_airport_delay[u] = per_airport_delay.get(u, 0) + d
            per_airport_delay[v] = per_airport_delay.get(v, 0) + d * 0.5  # downstream share

    recommended_actions = _recommendations_for_disruption(
        affected_airports, affected_flights, total_passengers_affected
    )

    return SingleDisruptionResult(
        network=G,
        origin=origin,
        destination=destination,
        delay_minutes=delay_minutes,
        weather_risk=weather_risk,
        initial_delay=initial_delay,
        affected_airports=affected_airports,
        affected_flights=affected_flights,
        total_passengers_affected=total_passengers_affected,
        recommended_actions=recommended_actions,
        per_airport_delay=per_airport_delay,
    )


def simulate_full_network(
    shock_airport: str = "DEL",
    shock_delay_min: float = 90.0,
    additional_shocks: Dict[str, float] | None = None,
    weather_disruption_airport: str | None = None,
    use_weather_penalty: bool = True,
    use_confidence: bool = True,
):
    """
    Run the full network simulation (ML, propagation, LP). Used by the terminal CLI.

    Returns the same SimulationOutputs as run_end_to_end_simulation so that
    main.py behavior remains unchanged.
    """
    from ..simulation_engine.orchestrator import run_end_to_end_simulation

    return run_end_to_end_simulation(
        shock_airport=shock_airport,
        shock_delay_min=shock_delay_min,
        additional_shocks=additional_shocks,
        weather_disruption_airport=weather_disruption_airport,
        use_weather_penalty=use_weather_penalty,
        use_confidence=use_confidence,
    )
