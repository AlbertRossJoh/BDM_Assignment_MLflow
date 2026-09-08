import mlflow
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit

_SKOPS_TRUSTED_TYPES = [
    "sklearn._loss.link.IdentityLink",
    "sklearn._loss.link.Interval",
    "sklearn._loss.loss.AbsoluteError",
    "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn._loss.loss.HalfSquaredError",
    "functools.partial",
    "sklearn.compose._column_transformer._RemainderColsList",
    "sklearn.utils.validation.check_array",
    "numpy.dtype",
    "sklearn.model_selection._split.TimeSeriesSplit",
    "transforms.direction.DirectionSinCos",
    "transforms.polynomial.PolynomialColumns",
    "transforms.rolling.RollingMean",
    "transforms.select.numeric_columns",
    "transforms.time_features.TimeFeatures",
]


class PipelineRunner:
    """Runs a fixed pipeline through several masked time-series CV splits and
    logs the aggregate metrics + model to the active MLflow run.

    ``split_counts`` is the set of ``n_splits`` values to evaluate; the first
    one is the primary split that ``run`` returns a score for.
    """

    def __init__(self, pipeline, mask, split_counts=(3, 5, 8)):
        self.pipeline = pipeline
        self.mask = np.asarray(mask)
        self.split_counts = list(split_counts)

    def masked_cv_metrics(self, X, y, n_splits):
        r2, rmse, mae, mse = [], [], [], []
        idx = np.arange(len(X))
        for tr, va in TimeSeriesSplit(n_splits=n_splits).split(idx):
            keep = va[self.mask[va]]
            if keep.size == 0:
                continue
            est = clone(self.pipeline)
            est.fit(X[tr.tolist()], y[tr.tolist()].to_numpy().ravel())
            y_true = y[keep.tolist()].to_numpy().ravel()
            y_pred = est.predict(X[keep.tolist()])
            r2.append(r2_score(y_true, y_pred))
            rmse.append(root_mean_squared_error(y_true, y_pred))
            mae.append(mean_absolute_error(y_true, y_pred))
            mse.append(mean_squared_error(y_true, y_pred))
        return {
            "r2": np.asarray(r2),
            "rmse": np.asarray(rmse),
            "mae": np.asarray(mae),
            "mse": np.asarray(mse),
        }

    def run(self, X, y):
        results = {
            n: self.masked_cv_metrics(X, y, n) for n in self.split_counts
        }
        primary = results[self.split_counts[0]]
        score = primary["r2"].mean()

        self.pipeline.fit(X, y.to_numpy().ravel())
        train_r2 = r2_score(y.to_numpy().ravel(), self.pipeline.predict(X))

        def _safe(a, fn):
            return float(fn(a)) if a.size else float("nan")

        metrics = {
            "MSE": primary["mse"].mean(),
            "MAE": primary["mae"].mean(),
            "RMSE": primary["rmse"].mean(),
            "R2": score,
            "CV_R2_min": _safe(primary["r2"], min),
            "CV_R2_std": _safe(primary["r2"], lambda x: x.std()),
            "train_R2": train_r2,
            "overfit_gap": train_r2 - score,
            "n_real": int(self.mask.sum()),
            "n_total": int(self.mask.size),
        }
        for n in self.split_counts[1:]:
            metrics[f"CV_R2_{n}fold"] = results[n]["r2"].mean()

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            self.pipeline,
            name="model",
            skops_trusted_types=_SKOPS_TRUSTED_TYPES,
        )
        return score
