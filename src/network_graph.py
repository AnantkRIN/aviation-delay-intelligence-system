from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import networkx as nx
import pandas as pd


@dataclass
class FlightEdgeData:
    flight_id: str
    duration_min: float
    predicted_delay_min: float
    aircraft_id: str
    turnaround_min: float
    passenger_connections: int


def build_flight_network(
    flights: pd.DataFrame, predicted_delays: Iterable[float]
) -> nx.DiGraph:
    """
    Create a directed network of airports and flights.

    Nodes: airports (IATA-like codes)
    Edges: individual flights with operational attributes.
    """
    G = nx.DiGraph()

    flights = flights.reset_index(drop=True).copy()
    flights["predicted_delay_min"] = list(predicted_delays)

    for _, row in flights.iterrows():
        origin = row["origin"]
        dest = row["destination"]
        G.add_node(origin)
        G.add_node(dest)

        # Approximate duration as distance / 800 km/h, converted to minutes.
        duration_min = (row["distance_km"] / 800.0) * 60.0

        # Simple turnaround time model: heavier loads and congestion need more time.
        turnaround_min = 30.0 + 40.0 * float(row["airport_load"])

        # Approximate passenger connections as a function of congestion and load.
        passenger_connections = int(50 + 150 * float(row["congestion_level"]))

        edge_data: Dict[str, object] = {
            "flight_id": row["flight_id"],
            "duration_min": duration_min,
            "predicted_delay_min": float(row["predicted_delay_min"]),
            "aircraft_id": row["aircraft_id"],
            "turnaround_min": turnaround_min,
            "passenger_connections": passenger_connections,
            "sched_dep_minute_of_day": int(row["sched_dep_minute_of_day"]),
            "congestion_level": float(row["congestion_level"]),
            "airport_load": float(row["airport_load"]),
        }

        G.add_edge(origin, dest, **edge_data)

    return G


def get_aircraft_flight_sequence(G: nx.DiGraph) -> Dict[str, List[str]]:
    """
    Construct sequences of flights for each aircraft based on departure time.
    """
    aircraft_to_flights: Dict[str, List[tuple]] = {}
    for u, v, data in G.edges(data=True):
        aircraft_id = data["aircraft_id"]
        aircraft_to_flights.setdefault(aircraft_id, []).append(
            (data["sched_dep_minute_of_day"], data["flight_id"], u, v)
        )

    aircraft_sequences: Dict[str, List[str]] = {}
    for ac_id, flights in aircraft_to_flights.items():
        flights_sorted = sorted(flights, key=lambda x: x[0])
        aircraft_sequences[ac_id] = [f[1] for f in flights_sorted]
    return aircraft_sequences


__all__ = ["build_flight_network", "get_aircraft_flight_sequence", "FlightEdgeData"]

