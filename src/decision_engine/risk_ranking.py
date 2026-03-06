"""Airport risk ranking based on delay and weather."""

from __future__ import annotations

from typing import Dict, List, Tuple


def compute_airport_risk_ranking(
    per_airport_delay: Dict[str, float],
    weather_risk: Dict[str, Dict[str, float]],
    delay_weight: float = 1.0,
    weather_weight: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Rank airports by combined risk (delay + weather).
    Returns list of (airport, risk_score) sorted descending by risk.
    """
    scores: Dict[str, float] = {}
    for airport, delay in per_airport_delay.items():
        storm = weather_risk.get(airport, {}).get("storm_probability", 0.0)
        wind = weather_risk.get(airport, {}).get("wind_risk_index", 0.0)
        weather_score = storm * 100 + wind * 50  # scale to comparable range
        scores[airport] = delay_weight * delay + weather_weight * weather_score

    for airport, risk in weather_risk.items():
        if airport not in scores:
            storm = risk.get("storm_probability", 0.0)
            wind = risk.get("wind_risk_index", 0.0)
            scores[airport] = weather_weight * (storm * 100 + wind * 50)

    return sorted(scores.items(), key=lambda x: -x[1])
