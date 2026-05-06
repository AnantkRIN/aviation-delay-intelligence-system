"""
Entry point for the AI Aviation Operations Control System.

Run with: python -m src.main
"""

from __future__ import annotations

from .services.simulation_service import simulate_full_network
from .visualization_engine import (
    clear_figure_outputs,
    plot_network_delays,
    plot_delay_heatmap,
    plot_optimization_comparison,
    plot_network_delay_map,
    plot_airport_risk_ranking,
)
from .utils.logger import setup_logging, get_logger

# Initialize logging (Phase 8)
setup_logging()
logger = get_logger()


def _print_banner() -> None:
    """Professional operations dashboard banner (Phase 10)."""
    print()
    print("=" * 52)
    print("  AI AVIATION OPERATIONS CONTROL SYSTEM")
    print("=" * 52)
    print()


def _print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    _print_banner()

    outputs = simulate_full_network()

    _print_section("Delay Prediction Performance")
    print(
        f"Train MAE: {outputs.delay_metrics['train_mae']:.2f} min | "
        f"Test MAE: {outputs.delay_metrics['test_mae']:.2f} min | "
        f"Test R^2: {outputs.delay_metrics['test_r2']:.3f}"
    )
    if outputs.prediction_confidence_sample:
        c = outputs.prediction_confidence_sample
        print(f"Sample prediction: {c.predicted_delay:.0f} min | CI: {c.confidence_lower:.0f}-{c.confidence_upper:.0f} min")

    _print_section("Network Delay Impact")
    print(f"Total delay before shock & propagation: {outputs.delay_metrics['total_delay_before']:.2f} min")
    print(f"Total delay after propagation: {outputs.delay_metrics['total_delay_after']:.2f} min")
    print(f"Iterations to convergence: {outputs.delay_metrics['propagation_iterations']}")
    print(f"Network delay reduction: {outputs.delay_metrics['network_delay_reduction_pct']:.2f}%")

    _print_section("Passenger Disruption")
    print(f"Estimated passenger connections disrupted: {outputs.delay_metrics['cancelled_passenger_connections']:.0f}")

    _print_section("Optimized Passenger Route (Dijkstra)")
    print(f"Origin: {outputs.least_delay_route['origin']} -> Destination: {outputs.least_delay_route['destination']}")
    print(f"Path: {outputs.least_delay_route['path']}")
    print(f"Total travel time (including delays): {outputs.least_delay_route['total_travel_time_min']:.2f} minutes")

    _print_section("Airport Risk Ranking")
    for airport, score in outputs.airport_risk_ranking[:8]:
        print(f"  {airport}: {score:.1f}")

    _print_section("Network Connectivity (Kruskal MST)")
    print(f"Nodes: {outputs.mst_summary['num_nodes']} | Edges: {outputs.mst_summary['num_edges']}")
    print(f"MST total weight: {outputs.mst_summary['mst_total_weight']:.2f} min")

    _print_section("Optimization Effectiveness")
    print(f"LP objective before: {outputs.delay_metrics['lp_objective_before']:.2f}")
    print(f"LP objective after:  {outputs.delay_metrics['lp_objective_after']:.2f}")
    print(f"LP reduction: {outputs.delay_metrics['lp_reduction_pct']:.2f}%")
    print(f"Flights operated: {outputs.lp_summary['num_operated']} | Cancelled: {outputs.lp_summary['num_cancelled']}")

    _print_section("Recommended Actions")
    for action in outputs.recommended_actions:
        print(f"  * {action}")

    _print_section("Generating Visual Intelligence Dashboards")
    clear_figure_outputs()
    per_airport_delay = {}
    for u, v, data in outputs.network.edges(data=True):
        per_airport_delay.setdefault(u, 0.0)
        per_airport_delay[u] += float(data.get("propagated_delay_min", data.get("predicted_delay_min", 0.0)))

    plot_network_delays(outputs.network)
    plot_delay_heatmap(per_airport_delay)
    plot_optimization_comparison(
        outputs.delay_metrics["total_delay_before"],
        outputs.delay_metrics["total_delay_after"],
        outputs.delay_metrics["lp_objective_before"],
        outputs.delay_metrics["lp_objective_after"],
    )
    plot_network_delay_map(outputs.network, per_airport_delay)
    plot_airport_risk_ranking(outputs.airport_risk_ranking)

    print("Saved figures: network_delays.png, delay_heatmap.png, optimization_comparison.png, network_delay_map.png, airport_risk_ranking.png")
    print("\n" + "=" * 52)
    print("  Simulation complete. Prototype airline OCC decision system.")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()
