"""Streamlit dashboard for AI Aviation Operations Control System."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation_engine.orchestrator import run_end_to_end_simulation
from src.simulation_engine.scenario_engine import run_multi_scenario_comparison
from src.data_layer import load_weather_risk, get_weather_risk_by_airport

st.set_page_config(
    page_title="AI Aviation Operations Control",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ AI Aviation Operations Control System")
st.markdown("Industry-level aviation delay propagation and network optimization prototype")

# Sidebar inputs
st.sidebar.header("Live Disruption Simulator")
origin = st.sidebar.selectbox("Origin Airport", ["DEL", "BOM", "BLR", "MAA", "HYD", "CCU", "AMD", "GOI"])
destination = st.sidebar.selectbox("Destination Airport", ["DEL", "BOM", "BLR", "MAA", "HYD", "CCU", "AMD", "GOI"])
delay_min = st.sidebar.slider("Delay (minutes)", 0, 180, 90)
passengers = st.sidebar.number_input("Passengers", min_value=1, max_value=500, value=210)
run_sim = st.sidebar.button("Run Simulation")

# Run simulation
if run_sim:
    with st.spinner("Running simulation..."):
        outputs = run_end_to_end_simulation(
            shock_airport=origin,
            shock_delay_min=float(delay_min),
        )

    st.success("Simulation complete!")

    # 1. Aviation network map
    st.subheader("1. Aviation Network Map")
    G = outputs.network
    pos = nx.spring_layout(G, seed=42)
    edge_trace = []
    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=list(G.nodes()),
        textposition="top center", marker=dict(size=20, color="lightblue"),
    )
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"), mode="lines")
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title="Network Graph", showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # 2. Delay heatmap
    st.subheader("2. Delay Heatmap")
    per_airport = {}
    for u, v, data in outputs.network.edges(data=True):
        per_airport.setdefault(u, 0.0)
        per_airport[u] += float(data.get("propagated_delay_min", 0.0))
    df_heat = pd.DataFrame({"Airport": list(per_airport.keys()), "Delay (min)": list(per_airport.values())})
    fig2 = px.bar(df_heat, x="Airport", y="Delay (min)", color="Delay (min)", color_continuous_scale="Reds")
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Airport risk ranking
    st.subheader("3. Airport Risk Ranking")
    risk_df = pd.DataFrame(
        outputs.airport_risk_ranking,
        columns=["Airport", "Risk Score"],
    )
    fig3 = px.bar(risk_df, x="Airport", y="Risk Score", color="Risk Score", color_continuous_scale="Oranges")
    st.plotly_chart(fig3, use_container_width=True)

    # 4. Passenger impact
    st.subheader("4. Passenger Impact")
    impact = outputs.delay_metrics.get("cancelled_passenger_connections", 0) + passengers
    st.metric("Passengers Affected", f"{impact:.0f}")

    # 5. System effectiveness
    st.subheader("5. System Effectiveness Comparison")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Delay Before", f"{outputs.delay_metrics['total_delay_before']:.0f} min")
    with col2:
        st.metric("Delay After", f"{outputs.delay_metrics['total_delay_after']:.0f} min")
    with col3:
        st.metric("LP Reduction", f"{outputs.delay_metrics['lp_reduction_pct']:.1f}%")
    with col4:
        st.metric("Propagation Iterations", outputs.delay_metrics["propagation_iterations"])

    # 6. Recommended actions
    st.subheader("6. Recommended Actions")
    for action in outputs.recommended_actions:
        st.write(f"• {action}")

else:
    st.info("Configure parameters in the sidebar and click **Run Simulation** to see results.")

# Weather risk section (always visible)
st.sidebar.header("Weather Risk")
weather_df = load_weather_risk()
st.sidebar.dataframe(weather_df, use_container_width=True)

# Scenario comparison (optional)
st.sidebar.header("Scenario Comparison")
if st.sidebar.button("Run Multi-Scenario Comparison"):
    with st.spinner("Running 3 scenarios..."):
        scenarios = [
            {"name": "DEL 90 min", "shock_airport": "DEL", "shock_delay_min": 90},
            {"name": "DEL 90 + BOM 45", "shock_airport": "DEL", "shock_delay_min": 90,
             "additional_shocks": {"BOM": 45}},
            {"name": "Weather BLR", "shock_airport": "DEL", "shock_delay_min": 0,
             "weather_disruption_airport": "BLR"},
        ]
        scenario_results = run_multi_scenario_comparison(scenarios)
    st.subheader("Scenario Comparison Chart")
    import pandas as pd
    comp_df = pd.DataFrame([
        {"Scenario": r.scenario_name, "Network Delay": r.total_network_delay,
         "Passenger Impact": r.passenger_impact}
        for r in scenario_results
    ])
    fig_sc = px.bar(comp_df, x="Scenario", y=["Network Delay", "Passenger Impact"],
                    barmode="group", title="Scenario Comparison")
    st.plotly_chart(fig_sc, use_container_width=True)
