from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from .data_engine import (
    load_or_create_dataset,
    train_test_split,
    get_feature_target_matrices,
)
from .delay_prediction_model import DelayPredictionModel
from .network_graph import build_flight_network
from .delay_propagation_engine import (
    initialize_propagated_delay,
    propagate_delays,
    PropagationConfig,
)
from .route_optimizer import (
    compute_least_delay_route,
    compute_minimum_spanning_connectivity,
)
from .linear_programming_engine import optimize_network_operations


@dataclass
class SimulationOutputs:
    raw_data: pd.DataFrame
    network: nx.DiGraph
    origin: str
    destination: str
    delay_metrics: Dict[str, float]
    least_delay_route: Dict[str, object]
    mst_summary: Dict[str, object]
    lp_summary: Dict[str, object]


def run_end_to_end_simulation(
    shock_airport: str = "DEL",
    shock_delay_min: float = 90.0,
) -> SimulationOutputs:
    """
    Run an end-to-end scenario:
        1. Load / create dataset
        2. Train ML model and predict delays
        3. Build network graph
        4. Inject delay shock and run propagation
        5. Optimize routes (Dijkstra, Kruskal)
        6. Run LP-based network optimization
    """
    df = load_or_create_dataset()
    train_df, test_df = train_test_split(df)
    X_train, y_train = get_feature_target_matrices(train_df)
    X_test, y_test = get_feature_target_matrices(test_df)

    model = DelayPredictionModel()
    result = model.fit(X_train, y_train, X_test, y_test)

    # Predict for the full dataset (operational deployment).
    X_full, _ = get_feature_target_matrices(df)
    predicted_full = model.predict(X_full)
    df["predicted_delay_min"] = predicted_full

    # Build network.
    G = build_flight_network(df, predicted_full)
    initialize_propagated_delay(G, shock_airport=shock_airport, shock_minutes=shock_delay_min)
    propagation_config = PropagationConfig()
    propagation_summary = propagate_delays(G, propagation_config)

    delay_reduction_pct = max(
        0.0,
        100.0
        * (propagation_summary.total_delay_before - propagation_summary.total_delay_after)
        / max(propagation_summary.total_delay_before, 1e-6),
    )

    # Choose an OD pair among busy airports.
    candidate_airports = ["DEL", "BOM", "BLR", "MAA", "HYD"]
    origin, destination = "DEL", "BLR"
    if not nx.has_path(G, origin, destination):
        # Fallback to any pair connected in the graph.
        nodes = list(G.nodes())
        for o in nodes:
            for d in nodes:
                if o != d and nx.has_path(G, o, d):
                    origin, destination = o, d
                    break

    route = compute_least_delay_route(G, origin, destination)
    mst = compute_minimum_spanning_connectivity(G)

    # Linear programming optimization.
    lp_result = optimize_network_operations(G)
    lp_reduction_pct = max(
        0.0,
        100.0
        * (lp_result.objective_before - lp_result.objective_after)
        / max(lp_result.objective_before, 1e-6),
    )

    # Passenger impact: how many connecting passengers on cancelled flights.
    cancelled_passengers = 0.0
    for u, v, data in G.edges(data=True):
        if data.get("flight_id") in lp_result.cancelled_flights:
            cancelled_passengers += float(data.get("passenger_connections", 0.0))

    delay_metrics = {
        "train_mae": result.train_mae,
        "test_mae": result.test_mae,
        "test_r2": result.test_r2,
        "total_delay_before": propagation_summary.total_delay_before,
        "total_delay_after": propagation_summary.total_delay_after,
        "propagation_iterations": propagation_summary.iterations,
        "network_delay_reduction_pct": delay_reduction_pct,
        "lp_objective_before": lp_result.objective_before,
        "lp_objective_after": lp_result.objective_after,
        "lp_reduction_pct": lp_reduction_pct,
        "cancelled_passenger_connections": cancelled_passengers,
    }

    least_delay_route = {
        "origin": origin,
        "destination": destination,
        "path": route.path,
        "total_travel_time_min": route.total_travel_time_min,
    }

    mst_summary = {
        "num_nodes": len(G.nodes()),
        "num_edges": len(G.edges()),
        "mst_edges": mst.mst_edges[:15],  # short preview
        "mst_total_weight": mst.total_weight,
    }

    lp_summary = {
        "operated_flights": lp_result.operated_flights[:15],
        "num_operated": len(lp_result.operated_flights),
        "cancelled_flights": lp_result.cancelled_flights[:15],
        "num_cancelled": len(lp_result.cancelled_flights),
    }

    return SimulationOutputs(
        raw_data=df,
        network=G,
        origin=origin,
        destination=destination,
        delay_metrics=delay_metrics,
        least_delay_route=least_delay_route,
        mst_summary=mst_summary,
        lp_summary=lp_summary,
    )


__all__ = ["SimulationOutputs", "run_end_to_end_simulation"]

