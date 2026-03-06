"""Dijkstra routing and Kruskal MST connectivity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx


@dataclass
class RouteResult:
    path: List[str]
    total_travel_time_min: float


def _edge_weight(G: nx.DiGraph, u: str, v: str) -> float:
    data = G[u][v]
    duration = float(data["duration_min"])
    delay = float(data.get("propagated_delay_min", data.get("predicted_delay_min", 0.0)))
    return duration + delay


def compute_least_delay_route(G: nx.DiGraph, origin: str, destination: str) -> RouteResult:
    """
    Use Dijkstra's algorithm to compute the least-delay path between two airports.
    """
    def weight(u: str, v: str, data: dict) -> float:
        duration = float(data["duration_min"])
        delay = float(data.get("propagated_delay_min", data.get("predicted_delay_min", 0.0)))
        return duration + delay

    path = nx.dijkstra_path(G, origin, destination, weight=weight)
    total_time = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        total_time += _edge_weight(G, u, v)

    return RouteResult(path=path, total_travel_time_min=total_time)


@dataclass
class ConnectivityOptimizationResult:
    mst_edges: List[Tuple[str, str]]
    total_weight: float


def compute_minimum_spanning_connectivity(G: nx.DiGraph) -> ConnectivityOptimizationResult:
    """
    Compute a minimum spanning tree on the undirected version of the network
    using Kruskal's algorithm (via NetworkX).
    """
    undirected = nx.Graph()
    for u, v, data in G.edges(data=True):
        w = _edge_weight(G, u, v)
        if undirected.has_edge(u, v):
            if w < undirected[u][v]["weight"]:
                undirected[u][v]["weight"] = w
        else:
            undirected.add_edge(u, v, weight=w)

    mst = nx.minimum_spanning_tree(undirected, algorithm="kruskal")
    total_weight = sum(d["weight"] for _, _, d in mst.edges(data=True))
    mst_edges = [(u, v) for u, v in mst.edges()]
    return ConnectivityOptimizationResult(mst_edges=mst_edges, total_weight=total_weight)
