from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import mlflow
import numpy as np
import polars as pl
from mlflow.sklearn import log_model
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

_SKOPS_TRUSTED_TYPES: list[str] = [
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

    def __init__(
        self,
        pipeline: Pipeline,
        mask: NDArray[np.bool_] | Sequence[bool],
        split_counts: Iterable[int] = (3, 5, 8),
    ) -> None:
        self.pipeline: Pipeline = pipeline
        self.mask: NDArray[np.bool_] = np.asarray(mask)
        self.split_counts: list[int] = list(split_counts)

    def masked_cv_metrics(
        self, X: pl.DataFrame, y: pl.DataFrame, n_splits: int
    ) -> dict[str, NDArray[np.float64]]:
        r2: list[float]
        rmse: list[float]
        mae: list[float]
        mse: list[float]
        r2, rmse, mae, mse = [], [], [], []
        idx = np.arange(len(X))
        for tr, va in TimeSeriesSplit(n_splits=n_splits).split(idx):
            keep = va[self.mask[va]]
            if keep.size == 0:
                continue
            est = cast(Pipeline, clone(self.pipeline))
            est.fit(X[tr.tolist()], y[tr.tolist()].to_numpy().ravel())
            y_true = y[keep.tolist()].to_numpy().ravel()
            y_pred = est.predict(X[keep.tolist()])
            r2.append(float(r2_score(y_true, y_pred)))
            rmse.append(float(root_mean_squared_error(y_true, y_pred)))
            mae.append(float(mean_absolute_error(y_true, y_pred)))
            mse.append(float(mean_squared_error(y_true, y_pred)))
        return {
            "r2": np.asarray(r2),
            "rmse": np.asarray(rmse),
            "mae": np.asarray(mae),
            "mse": np.asarray(mse),
        }

    def cv_fold_predictions(
        self, X: pl.DataFrame, y: pl.DataFrame, n_splits: int
    ) -> list[dict[str, Any]]:
        """Per-fold masked validation predictions for a fresh ``n_splits``
        ``TimeSeriesSplit``. Each entry holds ``time``, ``y_true`` and
        ``y_pred`` for that fold's real (unmasked) validation rows.
        """
        idx = np.arange(len(X))
        folds: list[dict[str, Any]] = []
        for tr, va in TimeSeriesSplit(n_splits=n_splits).split(idx):
            keep = va[self.mask[va]]
            if keep.size == 0:
                continue
            est = cast(Pipeline, clone(self.pipeline))
            est.fit(X[tr.tolist()], y[tr.tolist()].to_numpy().ravel())
            folds.append(
                {
                    "time": X[keep.tolist()]["time"].to_list(),
                    "y_true": y[keep.tolist()].to_numpy().ravel(),
                    "y_pred": np.asarray(
                        est.predict(X[keep.tolist()]), dtype=float
                    ),
                }
            )
        return folds

    def run(self, X: pl.DataFrame, y: pl.DataFrame) -> float:
        results: dict[int, dict[str, NDArray[np.float64]]] = {
            n: self.masked_cv_metrics(X, y, n) for n in self.split_counts
        }
        primary = results[self.split_counts[0]]
        score = float(primary["r2"].mean())

        self.pipeline.fit(X, y.to_numpy().ravel())
        train_r2 = float(r2_score(y.to_numpy().ravel(), self.pipeline.predict(X)))

        def _safe(
            a: NDArray[np.float64], fn: Callable[[NDArray[np.float64]], Any]
        ) -> float:
            return float(fn(a)) if a.size else float("nan")

        metrics: dict[str, float] = {
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
        log_model(
            self.pipeline,
            name="model",
            skops_trusted_types=_SKOPS_TRUSTED_TYPES,
        )
        return score
