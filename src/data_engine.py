import pathlib
from typing import Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_flight_data(n_flights: int = 200) -> pd.DataFrame:
    """
    Generate a synthetic flight dataset resembling real-world operations.

    Columns:
        flight_id, origin, destination, distance_km,
        weather_severity (0-1), congestion_level (0-1),
        prev_delay_min, airport_load (0-1),
        actual_delay_min (target for ML),
        aircraft_id, sched_dep_minute_of_day.
    """
    rng = np.random.default_rng(42)

    airports = ["DEL", "BOM", "BLR", "MAA", "HYD", "CCU", "AMD", "GOI"]
    n_airports = len(airports)

    origins = rng.choice(airports, size=n_flights)
    destinations = []
    for o in origins:
        choices = [a for a in airports if a != o]
        destinations.append(rng.choice(choices))
    destinations = np.array(destinations)

    base_distances = rng.integers(500, 2200, size=n_flights)
    weather = rng.uniform(0.0, 1.0, size=n_flights)
    congestion = rng.uniform(0.0, 1.0, size=n_flights)
    prev_delay = rng.normal(loc=20.0, scale=15.0, size=n_flights).clip(min=0.0)
    airport_load = rng.uniform(0.2, 1.0, size=n_flights)
    sched_dep = rng.integers(0, 24 * 60, size=n_flights)  # minutes of day

    # Ground-truth delay model (used only to synthesize labels)
    delay = (
        10 * weather
        + 30 * congestion
        + 0.05 * base_distances
        + 0.6 * prev_delay
        + 20 * airport_load
        + rng.normal(0.0, 10.0, size=n_flights)
    )
    delay = delay.clip(min=0.0)

    aircraft_ids = [f"A{idx % 15:03d}" for idx in range(n_flights)]

    return pd.DataFrame(
        {
            "flight_id": [f"F{1000 + i}" for i in range(n_flights)],
            "origin": origins,
            "destination": destinations,
            "distance_km": base_distances,
            "weather_severity": weather,
            "congestion_level": congestion,
            "prev_delay_min": prev_delay,
            "airport_load": airport_load,
            "sched_dep_minute_of_day": sched_dep,
            "actual_delay_min": delay,
            "aircraft_id": aircraft_ids,
        }
    )


def load_or_create_dataset(csv_name: str = "sample_flights.csv") -> pd.DataFrame:
    """
    Load a CSV from the data directory, or create a realistic synthetic dataset
    if it does not exist yet.
    """
    _ensure_data_dir()
    csv_path = DATA_DIR / csv_name

    if csv_path.exists():
        return pd.read_csv(csv_path)

    df = generate_synthetic_flight_data()
    df.to_csv(csv_path, index=False)
    return df


def train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple train-test split by random permutation.
    """
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    split = int(len(df) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(
        drop=True
    )


def get_feature_target_matrices(
    df: pd.DataFrame,
):
    """
    Extract feature matrix X and target vector y for delay prediction.
    """
    feature_cols = [
        "weather_severity",
        "congestion_level",
        "prev_delay_min",
        "distance_km",
        "airport_load",
    ]
    X = df[feature_cols].values
    y = df["actual_delay_min"].values
    return X, y


__all__ = [
    "DATA_DIR",
    "load_or_create_dataset",
    "train_test_split",
    "get_feature_target_matrices",
]

