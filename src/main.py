from __future__ import annotations

from pprint import pprint

from .simulation_engine import run_end_to_end_simulation
from .delay_propagation_engine import PropagationConfig, propagate_delays
from .visualization_engine import (
    plot_network_delays,
    plot_delay_heatmap,
    plot_optimization_comparison,
)


def main() -> None:
    """
    Entry point for the aviation delay intelligence system.

    When executed, this runs an end-to-end scenario and prints:
        - ML delay prediction metrics
        - Network-wide delay propagation summary
        - Optimized passenger route
        - Connectivity optimization (MST)
        - Linear programming impact and passenger disruption
    """
    print("=== AI-Driven Flight Delay Propagation and Network Optimization ===")
    outputs = run_end_to_end_simulation()

    print("\n--- Delay Prediction Performance ---")
    print(
        f"Train MAE: {outputs.delay_metrics['train_mae']:.2f} min | "
        f"Test MAE: {outputs.delay_metrics['test_mae']:.2f} min | "
        f"Test R^2: {outputs.delay_metrics['test_r2']:.3f}"
    )

    print("\n--- Network Delay Propagation ---")
    print(
        f"Total delay before shock & propagation: "
        f"{outputs.delay_metrics['total_delay_before']:.2f} min"
    )
    print(
        f"Total delay after propagation: "
        f"{outputs.delay_metrics['total_delay_after']:.2f} min"
    )
    print(
        f"Iterations to convergence: "
        f"{outputs.delay_metrics['propagation_iterations']}"
    )
    print(
        f"Network delay reduction (model vs shock scenario): "
        f"{outputs.delay_metrics['network_delay_reduction_pct']:.2f}%"
    )

    print("\n--- Optimized Passenger Route (Dijkstra on propagated delays) ---")
    print(
        f"Origin: {outputs.least_delay_route['origin']} "
        f"-> Destination: {outputs.least_delay_route['destination']}"
    )
    print(f"Path: {outputs.least_delay_route['path']}")
    print(
        f"Total passenger travel time (including delays): "
        f"{outputs.least_delay_route['total_travel_time_min']:.2f} minutes"
    )

    print("\n--- Network Connectivity Optimization (Kruskal MST) ---")
    print(
        f"Nodes in network: {outputs.mst_summary['num_nodes']} | "
        f"Edges in network: {outputs.mst_summary['num_edges']}"
    )
    print(
        f"Total MST weight (minutes of effective travel time): "
        f"{outputs.mst_summary['mst_total_weight']:.2f}"
    )

    print("\n--- Linear Programming-Based Network Control ---")
    print(
        f"LP objective before control: {outputs.delay_metrics['lp_objective_before']:.2f}"
    )
    print(
        f"LP objective after control:  {outputs.delay_metrics['lp_objective_after']:.2f}"
    )
    print(
        f"LP reduction in composite cost: "
        f"{outputs.delay_metrics['lp_reduction_pct']:.2f}%"
    )
    print(
        f"Flights operated: {outputs.lp_summary['num_operated']} | "
        f"Flights cancelled/re-timed: {outputs.lp_summary['num_cancelled']}"
    )
    print(
        f"Estimated passenger connections disrupted: "
        f"{outputs.delay_metrics['cancelled_passenger_connections']:.0f}"
    )

    print("\n--- Generating Visual Intelligence Dashboards ---")
    # For the heatmap we need per-airport delay. We approximate from outgoing edges.
    per_airport_delay = {}
    for u, v, data in outputs.network.edges(data=True):
        per_airport_delay.setdefault(u, 0.0)
        per_airport_delay[u] += float(
            data.get("propagated_delay_min", data.get("predicted_delay_min", 0.0))
        )

    plot_network_delays(outputs.network)
    plot_delay_heatmap(per_airport_delay)
    plot_optimization_comparison(
        outputs.delay_metrics["total_delay_before"],
        outputs.delay_metrics["total_delay_after"],
        outputs.delay_metrics["lp_objective_before"],
        outputs.delay_metrics["lp_objective_after"],
    )
    print(
        "Saved figures under data/figures/: network_delays.png, "
        "delay_heatmap.png, optimization_comparison.png"
    )

    print("\n=== Simulation complete. This prototype emulates an airline OCC decision system. ===")


if __name__ == "__main__":
    main()

