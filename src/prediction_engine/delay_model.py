"""RandomForest delay prediction with confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass
class DelayPredictionResult:
    model: RandomForestRegressor
    train_mae: float
    test_mae: float
    test_r2: float


@dataclass
class PredictionWithConfidence:
    """Prediction output with confidence interval."""

    predicted_delay: float
    confidence_lower: float
    confidence_upper: float
    confidence_interval: Tuple[float, float]

    def __str__(self) -> str:
        return (
            f"Predicted Delay: {self.predicted_delay:.0f} min | "
            f"Confidence Interval: {self.confidence_lower:.0f}–{self.confidence_upper:.0f} min"
        )


class DelayPredictionModel:
    """
    RandomForest-based regression model for flight delay prediction.
    Returns predicted delay and confidence interval (based on tree variance).
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

    def predict_with_confidence(
        self,
        X: np.ndarray,
        percentile_lower: float = 25.0,
        percentile_upper: float = 75.0,
    ) -> Tuple[np.ndarray, List[PredictionWithConfidence]]:
        """
        Predict delays with confidence intervals using RandomForest tree predictions.

        Uses the spread of individual tree predictions to estimate uncertainty.
        Returns (raw predictions array, list of PredictionWithConfidence).
        """
        tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean_pred = np.mean(tree_predictions, axis=0)
        lower = np.percentile(tree_predictions, percentile_lower, axis=0)
        upper = np.percentile(tree_predictions, percentile_upper, axis=0)

        results = []
        for i in range(len(mean_pred)):
            results.append(
                PredictionWithConfidence(
                    predicted_delay=float(mean_pred[i]),
                    confidence_lower=float(lower[i]),
                    confidence_upper=float(upper[i]),
                    confidence_interval=(float(lower[i]), float(upper[i])),
                )
            )
        return mean_pred, results
