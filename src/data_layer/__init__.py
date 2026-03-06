"""Data layer: flight data, weather risk, dataset loading."""

from .flight_data import (
    DATA_DIR,
    load_or_create_dataset,
    train_test_split,
    get_feature_target_matrices,
    generate_synthetic_flight_data,
)
from .weather_risk import load_weather_risk, get_weather_penalty, get_weather_risk_by_airport

__all__ = [
    "DATA_DIR",
    "load_or_create_dataset",
    "train_test_split",
    "get_feature_target_matrices",
    "generate_synthetic_flight_data",
    "load_weather_risk",
    "get_weather_penalty",
    "get_weather_risk_by_airport",
]
