import os
from typing import Any

import mlflow
import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline

from data_wrangler import DataWrangler, Granularity
from pipeline_runner import PipelineRunner
from plots import plot_cv_folds, plot_predictions_vs_actual

CV_FOLDS_ARTIFACT = "cv_folds_pred_vs_actual.png"
FUTURE_CSV_ARTIFACT = "future_predictions.csv"
FUTURE_PLOT_ARTIFACT = "future_predictions.png"


def load_training_frame(
    power_csv: str, weather_csv: str, granularity: Granularity
) -> pl.DataFrame:
    power_df = pl.read_csv(power_csv, try_parse_dates=True).select(
        ["time", "Total"]
    )
    wind_df = pl.read_csv(weather_csv, try_parse_dates=True).select(
        ["time", "Direction", "Speed"]
    )
    return DataWrangler(granularity=granularity).resample(
        wind_df=wind_df, power_df=power_df
    )


def resolve_experiment(experiment_name: str, tracking_uri: str) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    # ``mlflow run`` reserves the run/experiment through env vars; only pick the
    # experiment ourselves for a plain ``python <entrypoint>.py`` invocation.
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(experiment_name)


def run_experiment(
    pipeline: Pipeline,
    *,
    run_name: str,
    joined_df: pl.DataFrame,
    granularity: Granularity,
    log_params: dict[str, Any],
    future_csv: str,
    predict_future: bool,
    registered_model_name: str,
) -> float:
    X = joined_df.select(["time", "Speed", "Direction"])
    y = joined_df.select("Total")
    mask_full = joined_df["is_real"].to_numpy()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(log_params)
        mlflow.log_param("granularity", granularity)

        runner = PipelineRunner(pipeline, mask_full, split_counts=(3, 5, 8))
        score = runner.run(X, y)

        plot_cv_folds(runner.cv_fold_predictions(X, y, 3), CV_FOLDS_ARTIFACT)
        mlflow.log_artifact(CV_FOLDS_ARTIFACT)

        if registered_model_name:
            mlflow.register_model(
                f"runs:/{run.info.run_id}/model", registered_model_name
            )

        if predict_future:
            _log_future(runner, joined_df, future_csv)

    return score


def _log_future(
    runner: PipelineRunner, joined_df: pl.DataFrame, future_csv: str
) -> None:
    future_df = pl.read_csv(future_csv, try_parse_dates=True).select(
        ["time", "Speed", "Direction"]
    )
    preds = np.asarray(runner.pipeline.predict(future_df), dtype=float)
    out = future_df.with_columns(pl.Series("predicted_Total", preds))
    out.write_csv(FUTURE_CSV_ARTIFACT)
    mlflow.log_artifact(FUTURE_CSV_ARTIFACT)

    real = joined_df.filter(pl.col("is_real"))
    hist = real.select(["time", "Total"]).tail(2000)
    train_actual = real.select("Total").to_numpy().ravel()
    train_fitted = np.asarray(
        runner.pipeline.predict(real.select(["time", "Speed", "Direction"])),
        dtype=float,
    )
    if train_actual.shape[0] > 5000:
        sel = np.random.default_rng(0).choice(
            train_actual.shape[0], 5000, replace=False
        )
        train_actual, train_fitted = train_actual[sel], train_fitted[sel]

    plot_predictions_vs_actual(
        hist["time"].to_list(),
        hist["Total"].to_numpy().ravel(),
        out["time"].to_list(),
        preds,
        train_actual,
        train_fitted,
        FUTURE_PLOT_ARTIFACT,
    )
    mlflow.log_artifact(FUTURE_PLOT_ARTIFACT)
