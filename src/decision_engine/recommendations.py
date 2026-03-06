"""Recommended actions based on simulation results."""

from __future__ import annotations

from typing import Dict, List


def generate_recommended_actions(
    propagation_summary: object,
    lp_cancelled: List[str],
    high_risk_airports: List[str],
    passenger_impact: float,
) -> List[str]:
    """
    Generate recommended actions for operations control.
    """
    actions: List[str] = []
    total_after = getattr(propagation_summary, "total_delay_after", 0.0)

    if total_after > 500:
        actions.append("Consider network-wide delay mitigation: re-time or cancel low-priority flights")
    if lp_cancelled:
        actions.append(f"LP optimization suggests cancelling {len(lp_cancelled)} flights to reduce composite cost")
    if high_risk_airports:
        actions.append(f"Monitor high-risk airports: {', '.join(high_risk_airports[:5])}")
    if passenger_impact > 1000:
        actions.append("Significant passenger disruption expected - activate passenger rebooking protocols")
    actions.append("Review delay propagation report and adjust crew/aircraft rotations")
    return actions
