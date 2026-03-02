from __future__ import annotations

import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "figures"


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_network_delays(G: nx.DiGraph, filename: str = "network_delays.png") -> None:
    """
    Visualize the airport network with edge colours representing propagated delay.
    """
    _ensure_output_dir()
    pos = nx.spring_layout(G, seed=42)

    delays = [
        float(d.get("propagated_delay_min", d.get("predicted_delay_min", 0.0)))
        for _, _, d in G.edges(data=True)
    ]
    norm_delays = np.array(delays)
    if norm_delays.size > 0:
        norm_delays = (norm_delays - norm_delays.min()) / max(
            (norm_delays.max() - norm_delays.min()), 1e-6
        )

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color="#1f78b4")
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=norm_delays,
        edge_cmap=plt.cm.viridis,
        arrows=False,
        width=2.0,
    )
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(delays), vmax=max(delays))
    )
    sm.set_array([])
    plt.colorbar(sm, label="Propagated delay (min)")
    plt.title("Airport Network with Propagated Delays")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def plot_delay_heatmap(per_airport_delay: Dict[str, float], filename: str = "delay_heatmap.png") -> None:
    """
    Plot a simple heatmap of accumulated delay per airport.
    """
    _ensure_output_dir()
    airports = list(per_airport_delay.keys())
    delays = [per_airport_delay[a] for a in airports]

    plt.figure(figsize=(8, 4))
    sns.barplot(x=airports, y=delays, palette="Reds")
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
    """
    Compare pre-/post-propagation and LP objective metrics.
    """
    _ensure_output_dir()

    labels = ["Network Delay\nBefore", "Network Delay\nAfter", "LP Objective\nBefore", "LP Objective\nAfter"]
    values = [total_before, total_after, lp_before, lp_after]

    plt.figure(figsize=(8, 4))
    sns.barplot(x=labels, y=values, palette="Blues")
    plt.ylabel("Minutes / composite cost")
    plt.title("Impact of Propagation and Optimization")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


__all__ = ["plot_network_delays", "plot_delay_heatmap", "plot_optimization_comparison"]

