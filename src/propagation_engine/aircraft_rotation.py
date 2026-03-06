"""Aircraft rotation modeling: delay propagation via tail number sequences."""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

# Default turnaround buffer (minutes) - delay absorbed if previous flight arrives early enough
DEFAULT_TURNAROUND_BUFFER = 30.0


def _get_aircraft_rotation_sequences(G: nx.DiGraph) -> Dict[str, List[Tuple[str, str, dict]]]:
    """
    Build ordered flight sequences per aircraft by scheduled departure time.
    Returns: aircraft_id -> [(origin, dest, edge_data), ...] sorted by sched_dep.
    """
    aircraft_to_edges: Dict[str, List[Tuple[str, str, dict]]] = {}
    for u, v, data in G.edges(data=True):
        ac = data.get("aircraft_id", "UNKNOWN")
        aircraft_to_edges.setdefault(ac, []).append((u, v, data))

    for ac, edges in aircraft_to_edges.items():
        edges.sort(key=lambda e: e[2].get("sched_dep_minute_of_day", 0))

    return aircraft_to_edges


def apply_aircraft_rotation_propagation(
    G: nx.DiGraph,
    turnaround_buffer_min: float = DEFAULT_TURNAROUND_BUFFER,
) -> None:
    """
    Apply aircraft rotation delay propagation.

    Rule: delay_next_flight = max(delay_previous_flight - turnaround_time, 0)

    For each aircraft rotation (e.g. DEL -> BOM -> BLR -> MAA):
    - If flight i is delayed, flight i+1 inherits residual delay after turnaround.
    - turnaround_time = edge's turnaround_min (or default buffer).
    """
    sequences = _get_aircraft_rotation_sequences(G)

    for _ac_id, edges in sequences.items():
        for i in range(1, len(edges)):
            u_prev, v_prev, data_prev = edges[i - 1]
            u_curr, v_curr, data_curr = edges[i]

            prev_delay = float(data_prev.get("propagated_delay_min", 0.0))
            turnaround = float(data_curr.get("turnaround_min", turnaround_buffer_min))

            # Propagation rule: delay_next_flight inherits max(prev_delay - turnaround, 0)
            inherited = max(prev_delay - turnaround, 0.0)

            base = float(data_curr.get("propagated_delay_min", data_curr.get("predicted_delay_min", 0.0)))
            data_curr["propagated_delay_min"] = base + inherited
