from typing import Iterable
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin, _SetOutputMixin


class PolynomialColumns(BaseEstimator, TransformerMixin, _SetOutputMixin):
    def __init__(self, col: str = "Speed", degrees: Iterable[int] = (2, 3)):
        self.col: str = col
        self.degrees: Iterable[int] = degrees
        self.features: list[str] = []

    def fit(self, X, y=None):
        return self

    def transform(self, X: pl.DataFrame):
        df = X.with_columns(
            [(pl.col(self.col) ** d).alias(f"{self.col}_p{d}") for d in self.degrees]
        )
        for col in df.columns:
            self.features.append(col)
        return df

    def get_feature_names_out(self, input_features=None):
        return self.features
