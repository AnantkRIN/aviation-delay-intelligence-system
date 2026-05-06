"""
AI Aviation Disruption Control Simulator.

Live dashboard renders figures in browser only (no disk save).
Run with: python -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.simulation_service import (
    AVG_PASSENGERS_PER_FLIGHT,
    CONNECTION_RATIO,
    SingleDisruptionResult,
    simulate_single_flight_disruption,
)

AIRPORTS = ["DEL", "BOM", "BLR", "MAA", "HYD", "CCU", "AMD", "GOI"]


def build_live_network_figure(result: SingleDisruptionResult):
    """Build a live network figure for browser display only."""
    G = result.network
    origin = result.origin
    affected_set = set(result.affected_airports)

    pos = nx.spring_layout(G, seed=42)
    node_colors = []
    for n in G.nodes():
        if n == origin:
            node_colors.append("#ff9900")  # orange
        elif n in affected_set:
            node_colors.append("#e74c3c")  # red
        else:
            node_colors.append("#3498db")  # blue

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#888", arrows=True, ax=ax)
    ax.set_title("Live Network Map — Origin (orange) | Affected (red) | Normal (blue)")
    ax.axis("off")
    fig.tight_layout()
    return fig


def build_airport_delay_bar(result: SingleDisruptionResult):
    """Bar chart of propagated delay by airport."""
    items = sorted(result.per_airport_delay.items(), key=lambda x: x[1], reverse=True)
    airports = [k for k, _ in items]
    delays = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(airports, delays, color="#e74c3c")
    ax.set_title("Airport Delay Distribution")
    ax.set_xlabel("Airport")
    ax.set_ylabel("Propagated Delay (min)")
    fig.tight_layout()
    return fig


def build_top_routes_delay_chart(result: SingleDisruptionResult):
    """Horizontal bar chart of top delayed routes."""
    route_delays = []
    for u, v, data in result.network.edges(data=True):
        d = float(data.get("propagated_delay_min", 0.0))
        if d > 0:
            route_delays.append((f"{u}->{v}", d))
    route_delays.sort(key=lambda x: x[1], reverse=True)
    top = route_delays[:10]
    labels = [r[0] for r in top][::-1]
    values = [r[1] for r in top][::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, values, color="#ff9900")
    ax.set_title("Top 10 Most Delayed Routes")
    ax.set_xlabel("Delay (min)")
    fig.tight_layout()
    return fig


def build_passenger_impact_breakdown(passengers: int, affected_flights: int):
    """Stacked-like breakdown chart of passenger impact components."""
    connecting = passengers * CONNECTION_RATIO
    network_component = affected_flights * AVG_PASSENGERS_PER_FLIGHT
    labels = ["Direct", "Connecting", "Network Ripple"]
    values = [passengers, connecting, network_component]
    colors = ["#3498db", "#9b59b6", "#e67e22"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=colors)
    ax.set_title("Passenger Impact Breakdown")
    ax.set_ylabel("Passengers")
    fig.tight_layout()
    return fig


def build_recommendation_priority_chart(recommendations: list[str]):
    """Simple priority score visualization for recommended actions."""
    if not recommendations:
        recommendations = ["No action needed"]
    scores = np.linspace(len(recommendations), 1, len(recommendations))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh([f"Action {i+1}" for i in range(len(recommendations))], scores, color="#2ecc71")
    ax.set_title("Recommendation Priority View")
    ax.set_xlabel("Priority Score")
    fig.tight_layout()
    return fig


st.set_page_config(page_title="AI Aviation Disruption Control Simulator", page_icon="✈️", layout="wide")
st.title("AI Aviation Disruption Control Simulator")
st.markdown("Real-time single flight disruption simulation using shared backend logic.")

st.subheader("Flight Input")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    origin = st.selectbox("Origin Airport", AIRPORTS, index=0)
with col2:
    destination = st.selectbox("Destination Airport", AIRPORTS, index=1)
with col3:
    delay_minutes = st.number_input("Delay Time (minutes)", min_value=0, max_value=300, value=90)
with col4:
    passengers = st.number_input("Passengers on Flight", min_value=1, max_value=500, value=210)
with col5:
    weather_risk = st.slider("Weather Risk Probability (0–1)", 0.0, 1.0, 0.6, step=0.1)

run_clicked = st.button("Simulate Disruption")

if run_clicked:
    with st.spinner("Running disruption simulation..."):
        result = simulate_single_flight_disruption(
            origin=origin,
            destination=destination,
            delay_minutes=float(delay_minutes),
            passengers=passengers,
            weather_risk=weather_risk,
        )

    st.success("Simulation complete.")

    st.subheader("Flight Input Summary")
    st.json({
        "Origin Airport": result.origin,
        "Destination Airport": result.destination,
        "Delay (minutes)": result.delay_minutes,
        "Passengers": passengers,
        "Weather Risk": result.weather_risk,
        "Initial Delay (delay + weather_delay)": round(result.initial_delay, 1),
    })

    st.subheader("Network Impact")
    st.metric("Affected Airports", len(result.affected_airports))
    st.write("**Affected airports:** " + ", ".join(result.affected_airports) if result.affected_airports else "None")
    st.metric("Affected Flights (routes)", result.affected_flights)

    st.subheader("Passenger Impact")
    connecting = passengers * CONNECTION_RATIO
    st.caption(f"Connecting passengers (passengers x {CONNECTION_RATIO}) = {connecting:.0f}")
    st.caption(f"Affected flights x {AVG_PASSENGERS_PER_FLIGHT} avg = {result.affected_flights * AVG_PASSENGERS_PER_FLIGHT:.0f}")
    st.metric("Total Passengers Affected", f"{result.total_passengers_affected:.0f}")

    st.subheader("Operational Recommendations")
    for action in result.recommended_actions:
        st.write(f"- {action}")

    st.subheader("Live Network Map")
    fig = build_live_network_figure(result)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Rendered live in browser only (not saved to folder).")

    st.subheader("Airport Delay Chart")
    fig_delay = build_airport_delay_bar(result)
    st.pyplot(fig_delay, use_container_width=True)
    plt.close(fig_delay)

    st.subheader("Top Delayed Routes")
    fig_routes = build_top_routes_delay_chart(result)
    st.pyplot(fig_routes, use_container_width=True)
    plt.close(fig_routes)

    st.subheader("Passenger Impact Breakdown")
    fig_pax = build_passenger_impact_breakdown(passengers, result.affected_flights)
    st.pyplot(fig_pax, use_container_width=True)
    plt.close(fig_pax)

    st.subheader("Recommendation Priority")
    fig_actions = build_recommendation_priority_chart(result.recommended_actions)
    st.pyplot(fig_actions, use_container_width=True)
    plt.close(fig_actions)
else:
    st.info("Enter flight details above and click Simulate Disruption.")
