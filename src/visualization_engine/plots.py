"""Advanced visualization plots using matplotlib, seaborn, networkx."""

from __future__ import annotations

import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "figures"


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_network_delays(G: nx.DiGraph, filename: str = "network_delays.png") -> None:
    """Visualize the airport network with edge colours representing propagated delay."""
    _ensure_output_dir()
    pos = nx.spring_layout(G, seed=42)
    delays = [
        float(d.get("propagated_delay_min", d.get("predicted_delay_min", 0.0)))
        for _, _, d in G.edges(data=True)
    ]
    if not delays:
        return
    delays_arr = np.array(delays)
    norm_delays = (delays_arr - delays_arr.min()) / max(
        (delays_arr.max() - delays_arr.min()), 1e-6
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color="#1f78b4", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color=norm_delays, edge_cmap=plt.cm.viridis,
        arrows=False, width=2.0, ax=ax,
    )
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=delays_arr.min(), vmax=delays_arr.max()),
    )
    sm.set_array(delays_arr)
    fig.colorbar(sm, ax=ax, label="Propagated delay (min)")
    ax.set_title("Airport Network with Propagated Delays")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def plot_delay_heatmap(
    per_airport_delay: Dict[str, float], filename: str = "delay_heatmap.png"
) -> None:
    """Plot heatmap of accumulated delay per airport."""
    _ensure_output_dir()
    airports = list(per_airport_delay.keys())
    delays = [per_airport_delay[a] for a in airports]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=airports, y=delays, hue=airports, palette="Reds", legend=False)
    plt.ylabel("Total propagated delay (min)")
    plt.xlabel("Airport")
    plt.title("Accumulated Network Delay by Airport")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def plot_optimization_comparison(
    total_before: float,
    total_after: float,
    lp_before: float,
    lp_after: float,
    filename: str = "optimization_comparison.png",
) -> None:
    """Compare pre-/post-propagation and LP objective metrics."""
    _ensure_output_dir()
    labels = [
        "Network Delay\nBefore",
        "Network Delay\nAfter",
        "LP Objective\nBefore",
        "LP Objective\nAfter",
    ]
    values = [total_before, total_after, lp_before, lp_after]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=labels, y=values, hue=labels, palette="Blues", legend=False)
    plt.ylabel("Minutes / composite cost")
    plt.title("Impact of Propagation and Optimization")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def plot_network_delay_map(
    G: nx.DiGraph,
    per_airport_delay: Dict[str, float],
    filename: str = "network_delay_map.png",
) -> None:
    """Network delay map with node size proportional to airport delay."""
    _ensure_output_dir()
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [300 + per_airport_delay.get(n, 0) * 2 for n in G.nodes()]
    delays = [
        float(d.get("propagated_delay_min", 0.0)) for _, _, d in G.edges(data=True)
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#2e86ab", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
    if delays:
        delays_arr = np.array(delays)
        norm_d = (delays_arr - delays_arr.min()) / max(
            delays_arr.max() - delays_arr.min(), 1e-6
        )
        nx.draw_networkx_edges(
            G, pos, edge_color=norm_d, edge_cmap=plt.cm.YlOrRd,
            width=2.0, ax=ax,
        )
    ax.set_title("Network Delay Map")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def plot_passenger_disruption(
    per_airport_delay: Dict[str, float],
    passenger_impact: float,
    filename: str = "passenger_disruption.png",
) -> None:
    """Passenger disruption chart."""
    _ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    airports = list(per_airport_delay.keys())
    delays = [per_airport_delay[a] for a in airports]
    axes[0].bar(airports, delays, color="coral", alpha=0.8)
    axes[0].set_ylabel("Delay (min)")
    axes[0].set_title("Delay by Airport")
    axes[1].bar(["Total Passenger Impact"], [passenger_impact], color="steelblue")
    axes[1].set_ylabel("Passengers Affected")
    axes[1].set_title("Passenger Disruption")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def plot_airport_risk_ranking(
    risk_ranking: List[tuple],
    filename: str = "airport_risk_ranking.png",
) -> None:
    """Airport risk ranking chart."""
    _ensure_output_dir()
    if not risk_ranking:
        return
    airports = [r[0] for r in risk_ranking]
    scores = [r[1] for r in risk_ranking]
    plt.figure(figsize=(8, 5))
    colors = plt.cm.Reds(np.linspace(0.4, 1.0, len(airports)))
    plt.barh(airports[::-1], scores[::-1], color=colors)
    plt.xlabel("Risk Score")
    plt.title("Airport Risk Ranking")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def plot_system_effectiveness_comparison(
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float],
    filename: str = "system_effectiveness.png",
) -> None:
    """System effectiveness comparison chart."""
    _ensure_output_dir()
    labels = list(before_metrics.keys())
    before_vals = [before_metrics[k] for k in labels]
    after_vals = [after_metrics.get(k, 0) for k in labels]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, before_vals, width, label="Before", color="coral")
    ax.bar(x + width / 2, after_vals, width, label="After", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_title("System Effectiveness Comparison")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def plot_scenario_comparison(
    scenario_results: List[object],
    filename: str = "scenario_comparison.png",
) -> None:
    """Scenario comparison chart: total delay and passenger impact."""
    _ensure_output_dir()
    names = [getattr(r, "scenario_name", f"Scenario {i}") for i, r in enumerate(scenario_results)]
    delays = [getattr(r, "total_network_delay", 0) for r in scenario_results]
    passengers = [getattr(r, "passenger_impact", 0) for r in scenario_results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, delays, color="teal", alpha=0.8)
    axes[0].set_ylabel("Total Network Delay (min)")
    axes[0].set_title("Total Network Delay by Scenario")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(names, passengers, color="darkorange", alpha=0.8)
    axes[1].set_ylabel("Passenger Impact")
    axes[1].set_title("Passenger Impact by Scenario")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)
