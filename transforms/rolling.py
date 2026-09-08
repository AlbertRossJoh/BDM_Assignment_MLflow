import math

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class RollingMean(BaseEstimator, TransformerMixin):
    def __init__(self, window: int = 12, alpha: float = 0.3):
        self.window: int = window
        self.alpha: float = alpha

    def _weights(self):
        raw = [math.exp(-self.alpha * k) for k in range(self.window)]
        total = sum(raw)
        return [w / total for w in reversed(raw)]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pl.DataFrame):
        col = X.columns[0]
        return X.select(
            pl.col(col)
            .rolling_mean(
                window_size=self.window,
                weights=self._weights(),
            )
            .fill_null(strategy="mean")
            .alias("rolling_mean")
        )

    def get_feature_names_out(self, input_features=None):
        return ["rolling_mean"]
