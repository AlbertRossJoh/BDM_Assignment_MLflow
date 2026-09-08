import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class TimeFeatures(BaseEstimator, TransformerMixin):
    """Adds ``hour`` and ``doy`` (ordinal day) from a datetime column and drops
    that column. Passes every other column through unchanged.
    """

    def __init__(self, col="time"):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.with_columns(
            pl.col(self.col).dt.hour().alias("hour"),
            pl.col(self.col).dt.ordinal_day().alias("doy"),
        ).drop(self.col)
