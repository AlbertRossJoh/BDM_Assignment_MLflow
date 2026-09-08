import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

COMPASS_16 = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]
DIR_TO_IDX = {d: i for i, d in enumerate(COMPASS_16)}
IDX_TO_DIR = dict(enumerate(COMPASS_16))


class DirectionSinCos(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        angle = (
            pl.col("Direction").replace_strict(
                DIR_TO_IDX, return_dtype=pl.Float64
            )
            * 22.5
        ).radians()
        return X.select(
            angle.sin().alias("direction_sin"),
            angle.cos().alias("direction_cos"),
        )

    def get_feature_names_out(self, input_features=None):
        return ["direction_sin", "direction_cos"]
