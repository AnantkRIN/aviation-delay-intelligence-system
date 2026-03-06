"""Authority alert generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Alert:
    level: str  # INFO, WARNING, CRITICAL
    airport: str
    message: str
    metric_value: float


def generate_authority_alerts(
    per_airport_delay: Dict[str, float],
    weather_risk: Dict[str, Dict[str, float]],
    delay_threshold_warning: float = 100.0,
    delay_threshold_critical: float = 200.0,
    storm_threshold: float = 0.5,
) -> List[Alert]:
    """
    Generate operational alerts for aviation authorities.
    """
    alerts: List[Alert] = []
    for airport, delay in per_airport_delay.items():
        if delay >= delay_threshold_critical:
            alerts.append(
                Alert(
                    level="CRITICAL",
                    airport=airport,
                    message=f"Critical delay accumulation: {delay:.0f} min",
                    metric_value=delay,
                )
            )
        elif delay >= delay_threshold_warning:
            alerts.append(
                Alert(
                    level="WARNING",
                    airport=airport,
                    message=f"Elevated delay: {delay:.0f} min",
                    metric_value=delay,
                )
            )

    for airport, risk in weather_risk.items():
        storm_prob = risk.get("storm_probability", 0.0)
        if storm_prob >= storm_threshold:
            alerts.append(
                Alert(
                    level="WARNING",
                    airport=airport,
                    message=f"Weather risk: storm probability {storm_prob:.2f}",
                    metric_value=storm_prob,
                )
            )
    return alerts
