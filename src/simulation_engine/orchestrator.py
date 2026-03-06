"""End-to-end simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from ..data_layer import (
    load_or_create_dataset,
    train_test_split,
    get_feature_target_matrices,
    load_weather_risk,
    get_weather_risk_by_airport,
)
from ..prediction_engine import DelayPredictionModel
from ..optimization_engine.network_graph import build_flight_network
from ..propagation_engine import (
    initialize_propagated_delay,
    propagate_delays,
    PropagationConfig,
)
from ..optimization_engine import (
    compute_least_delay_route,
    compute_minimum_spanning_connectivity,
    optimize_network_operations,
)
from ..decision_engine import (
    compute_airport_risk_ranking,
    generate_recommended_actions,
)
from ..utils.logger import get_logger

logger = get_logger()


@dataclass
class SimulationOutputs:
    raw_data: pd.DataFrame
    network: nx.DiGraph
    origin: str
    destination: str
    delay_metrics: Dict[str, object]
    least_delay_route: Dict[str, object]
    mst_summary: Dict[str, object]
    lp_summary: Dict[str, object]
    airport_risk_ranking: List[tuple]
    recommended_actions: List[str]
    prediction_confidence_sample: Optional[object] = None


def run_end_to_end_simulation(
    shock_airport: str = "DEL",
    shock_delay_min: float = 90.0,
    additional_shocks: Optional[Dict[str, float]] = None,
    weather_disruption_airport: Optional[str] = None,
    use_weather_penalty: bool = True,
    use_confidence: bool = True,
) -> SimulationOutputs:
    """
    Run an end-to-end scenario:
        1. Load / create dataset
        2. Train ML model and predict delays (with weather penalty)
        3. Build network graph
        4. Inject delay shock and run propagation (with aircraft rotation)
        5. Optimize routes (Dijkstra, Kruskal)
        6. Run LP-based network optimization
    """
    logger.info("Starting end-to-end simulation")
    df = load_or_create_dataset()
    train_df, test_df = train_test_split(df)
    X_train, y_train = get_feature_target_matrices(train_df)
    X_test, y_test = get_feature_target_matrices(test_df)

    model = DelayPredictionModel()
    result = model.fit(X_train, y_train, X_test, y_test)

    X_full, _ = get_feature_target_matrices(df)
    if use_confidence:
        predicted_full, confidence_list = model.predict_with_confidence(X_full)
        df["predicted_delay_min"] = predicted_full
        df["confidence_lower"] = [c.confidence_lower for c in confidence_list]
        df["confidence_upper"] = [c.confidence_upper for c in confidence_list]
        confidence_sample = confidence_list[0] if confidence_list else None
    else:
        predicted_full = model.predict(X_full)
        df["predicted_delay_min"] = predicted_full
        confidence_sample = None

    # Apply weather penalty: predicted_delay = model_prediction + weather_penalty
    if use_weather_penalty:
        weather_df = load_weather_risk()
        for i, row in df.iterrows():
            from ..data_layer.weather_risk import get_weather_penalty
            penalty = get_weather_penalty(str(row["origin"]), weather_df)
            df.at[i, "predicted_delay_min"] = float(df.at[i, "predicted_delay_min"]) + penalty

    # Extra weather disruption at specific airport (e.g. BLR storm)
    if weather_disruption_airport:
        from ..data_layer.weather_risk import STORM_PENALTY_MULTIPLIER
        weather_df = load_weather_risk()
        extra = STORM_PENALTY_MULTIPLIER * 0.5  # simulate 50% storm
        for i, row in df.iterrows():
            if str(row["origin"]) == weather_disruption_airport:
                df.at[i, "predicted_delay_min"] = float(df.at[i, "predicted_delay_min"]) + extra

    predicted_full = df["predicted_delay_min"].values
    G = build_flight_network(df, predicted_full)

    shocks_msg = f"{shock_airport} {shock_delay_min:.0f} min"
    if additional_shocks:
        shocks_msg += " + " + ", ".join(f"{a} {d:.0f} min" for a, d in additional_shocks.items())
    logger.info(f"Flight delay injected {shocks_msg}")
    initialize_propagated_delay(
        G,
        shock_airport=shock_airport,
        shock_minutes=shock_delay_min,
        additional_shocks=additional_shocks,
    )
    propagation_config = PropagationConfig()
    propagation_summary = propagate_delays(G, propagation_config)
    logger.info(f"Propagation iteration {propagation_summary.iterations}")

    delay_reduction_pct = max(
        0.0,
        100.0
        * (propagation_summary.total_delay_before - propagation_summary.total_delay_after)
        / max(propagation_summary.total_delay_before, 1e-6),
    )
    logger.info(f"Optimization reduced delay by {delay_reduction_pct:.1f}%")

    candidate_airports = ["DEL", "BOM", "BLR", "MAA", "HYD"]
    origin, destination = "DEL", "BLR"
    if not nx.has_path(G, origin, destination):
        nodes = list(G.nodes())
        for o in nodes:
            for d in nodes:
                if o != d and nx.has_path(G, o, d):
                    origin, destination = o, d
                    break

    route = compute_least_delay_route(G, origin, destination)
    mst = compute_minimum_spanning_connectivity(G)
    lp_result = optimize_network_operations(G)
    lp_reduction_pct = max(
        0.0,
        100.0
        * (lp_result.objective_before - lp_result.objective_after)
        / max(lp_result.objective_before, 1e-6),
    )

    cancelled_passengers = 0.0
    for u, v, data in G.edges(data=True):
        if data.get("flight_id") in lp_result.cancelled_flights:
            cancelled_passengers += float(data.get("passenger_connections", 0.0))

    weather_risk = get_weather_risk_by_airport()
    risk_ranking = compute_airport_risk_ranking(
        propagation_summary.per_airport_delay, weather_risk
    )
    high_risk = [a for a, _ in risk_ranking[:3]]
    recommended_actions = generate_recommended_actions(
        propagation_summary,
        lp_result.cancelled_flights,
        high_risk,
        cancelled_passengers,
    )

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
        "mst_edges": mst.mst_edges[:15],
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
        airport_risk_ranking=risk_ranking,
        recommended_actions=recommended_actions,
        prediction_confidence_sample=confidence_sample,
    )
