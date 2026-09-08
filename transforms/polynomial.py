import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

_SUFFIX = {2: "sq", 3: "cu"}


class PolynomialColumns(BaseEstimator, TransformerMixin):
    """Appends integer powers of one column, e.g. ``Speed`` -> ``Speed sq`` /
    ``Speed cu``. Passes every other column through unchanged.
    """

    def __init__(self, col="Speed", degrees=(2, 3)):
        self.col = col
        self.degrees = degrees

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.with_columns(
            [
                (pl.col(self.col) ** d).alias(
                    f"{self.col} {_SUFFIX.get(d, f'p{d}')}"
                )
                for d in self.degrees
            ]
        )
