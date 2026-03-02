from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass
class DelayPredictionResult:
    model: RandomForestRegressor
    train_mae: float
    test_mae: float
    test_r2: float


class DelayPredictionModel:
    """
    RandomForest-based regression model for flight delay prediction.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> DelayPredictionResult:
        """
        Train the model and compute performance metrics.
        """
        self.model.fit(X_train, y_train)
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)

        return DelayPredictionResult(
            model=self.model,
            train_mae=train_mae,
            test_mae=test_mae,
            test_r2=test_r2,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict delays in minutes for the given feature matrix.
        """
        return self.model.predict(X)


__all__ = ["DelayPredictionModel", "DelayPredictionResult"]

