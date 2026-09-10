import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "time"):
        self.col: str = col

    def fit(self, X: pl.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        self.features_ = [c for c in X.columns if c != self.col] + ["hour", "doy"]
        return self

    def transform(self, X: pl.DataFrame):
        return X.with_columns(
            pl.col(self.col).dt.hour().alias("hour"),
            pl.col(self.col).dt.ordinal_day().alias("doy"),
        ).drop(self.col)

    def get_feature_names_out(self, input_features=None):
        return self.features_
