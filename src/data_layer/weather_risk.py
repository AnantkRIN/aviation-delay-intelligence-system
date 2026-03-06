"""Weather risk data and penalty computation."""

from __future__ import annotations

import pathlib
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
WEATHER_RISK_PATH = DATA_DIR / "weather_risk.csv"

# Default storm penalty multiplier (minutes added per unit storm_probability)
STORM_PENALTY_MULTIPLIER = 20.0


def _generate_weather_risk_data() -> pd.DataFrame:
    """Generate simulated weather risk dataset if not present."""
    airports = ["DEL", "BOM", "BLR", "MAA", "HYD", "CCU", "AMD", "GOI"]
    rng = np.random.default_rng(42)
    storm_prob = rng.uniform(0.0, 0.6, size=len(airports))
    wind_risk = rng.uniform(0.0, 1.0, size=len(airports))
    return pd.DataFrame({
        "airport": airports,
        "storm_probability": storm_prob,
        "wind_risk_index": wind_risk,
    })


def load_weather_risk() -> pd.DataFrame:
    """
    Load weather risk data from data/weather_risk.csv.
    Creates the file with simulated data if it does not exist.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if WEATHER_RISK_PATH.exists():
        return pd.read_csv(WEATHER_RISK_PATH)
    df = _generate_weather_risk_data()
    df.to_csv(WEATHER_RISK_PATH, index=False)
    return df


def get_weather_penalty(
    airport: str,
    weather_df: Optional[pd.DataFrame] = None,
    storm_multiplier: float = STORM_PENALTY_MULTIPLIER,
) -> float:
    """
    Compute weather penalty for an airport.

    weather_penalty = storm_probability * storm_multiplier
    """
    if weather_df is None:
        weather_df = load_weather_risk()
    row = weather_df[weather_df["airport"] == airport]
    if row.empty:
        return 0.0
    storm_prob = float(row["storm_probability"].iloc[0])
    return storm_prob * storm_multiplier


def get_weather_risk_by_airport() -> Dict[str, Dict[str, float]]:
    """Return dict of airport -> {storm_probability, wind_risk_index}."""
    df = load_weather_risk()
    return {
        row["airport"]: {
            "storm_probability": float(row["storm_probability"]),
            "wind_risk_index": float(row["wind_risk_index"]),
        }
        for _, row in df.iterrows()
    }
