from typing import Iterable
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class PolynomialColumns(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "Speed", degrees: Iterable[int] = (2, 3)):
        self.col: str = col
        self.degrees: Iterable[int] = degrees

    def fit(self, X, y=None):
        return self

    def transform(self, X: pl.DataFrame):
        return X.with_columns(
            [(pl.col(self.col) ** d).alias(f"{self.col}_p{d}") for d in self.degrees]
        )
