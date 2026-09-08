import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "time"):
        self.col: str = col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pl.DataFrame):
        return X.with_columns(
            pl.col(self.col).dt.hour().alias("hour"),
            pl.col(self.col).dt.ordinal_day().alias("doy"),
        ).drop(self.col)
