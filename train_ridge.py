from typing import Any

import click
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_wrangler import Granularity
from experiment import load_training_frame, resolve_experiment, run_experiment
from transforms import (
    DirectionSinCos,
    PolynomialColumns,
    RollingMean,
    TimeFeatures,
    numeric_columns,
)


def build_pipeline(
    alpha_min: float, alpha_max: float, alpha_count: int, cv_splits: int
) -> Pipeline:
    num_branch = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    roll_branch = Pipeline(
        [
            ("roll", RollingMean(window=12, alpha=0.3)),
            (
                "impute",
                SimpleImputer(strategy="median"),
            ),  # added fill_null in RollingMean, this should be unneeded keeping for safety
            ("scale", StandardScaler()),
        ]
    )
    return Pipeline(
        [
            ("time", TimeFeatures(col="time")),
            ("poly", PolynomialColumns(col="Speed", degrees=(2, 3))),
            (
                "prep",
                ColumnTransformer(
                    [
                        ("dir", DirectionSinCos(), ["Direction"]),
                        ("roll", roll_branch, ["Speed"]),
                        ("num", num_branch, numeric_columns),
                    ]
                ),
            ),
            (
                "pred",
                RidgeCV(
                    alphas=np.logspace(alpha_min, alpha_max, alpha_count),
                    cv=TimeSeriesSplit(n_splits=cv_splits),
                ),
            ),
        ]
    )


@click.command(help=__doc__)
@click.option("--power-csv", default="data/power.csv", show_default=True)
@click.option("--weather-csv", default="data/weather.csv", show_default=True)
@click.option("--future-csv", default="data/future.csv", show_default=True)
@click.option(
    "--granularity",
    type=click.Choice(("1h", "1m")),
    default="1m",
    show_default=True,
)
@click.option("--experiment-name", default="wind-power-ridge", show_default=True)
@click.option(
    "--tracking-uri", default="", help="Overrides MLFLOW_TRACKING_URI when set."
)
@click.option(
    "--registered-model-name",
    default="",
    help="Register the logged model under this name when set.",
)
@click.option("--predict-future", type=bool, default=True, show_default=True)
@click.option("--alpha-min", type=float, default=-3.0, show_default=True)
@click.option("--alpha-max", type=float, default=3.0, show_default=True)
@click.option("--alpha-count", type=int, default=25, show_default=True)
@click.option("--cv-splits", type=int, default=3, show_default=True)
def main(
    power_csv: str,
    weather_csv: str,
    future_csv: str,
    granularity: Granularity,
    experiment_name: str,
    tracking_uri: str,
    registered_model_name: str,
    predict_future: bool,
    alpha_min: float,
    alpha_max: float,
    alpha_count: int,
    cv_splits: int,
) -> None:
    resolve_experiment(experiment_name, tracking_uri)

    params: dict[str, Any] = {
        "model": "RidgeCV",
        "alphas": f"logspace({alpha_min}, {alpha_max}, {alpha_count})",
        "ridge_cv_splits": cv_splits,
    }

    joined_df = load_training_frame(power_csv, weather_csv, granularity)
    score = run_experiment(
        build_pipeline(alpha_min, alpha_max, alpha_count, cv_splits),
        run_name="linear_pipeline_ridge",
        joined_df=joined_df,
        granularity=granularity,
        log_params=params,
        future_csv=future_csv,
        predict_future=predict_future,
        registered_model_name=registered_model_name,
    )
    click.echo(f"primary CV R2: {score:.4f}")


if __name__ == "__main__":
    main()
