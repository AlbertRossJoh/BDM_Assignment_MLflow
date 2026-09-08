from typing import Any

import click
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from data_wrangler import Granularity
from experiment import load_training_frame, resolve_experiment, run_experiment
from transforms import (
    DirectionSinCos,
    PolynomialColumns,
    RollingMean,
    TimeFeatures,
    numeric_columns,
)

BASELINE_PARAMS: dict[str, str | int | float | bool] = {
    "loss": "absolute_error",
    "learning_rate": 0.05503031368187496,
    "max_iter": 154,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 7,
    "l2_regularization": 4.59640147691376,
    "max_bins": 212,
    "max_features": 0.5004554464353821,
    "early_stopping": False,
    "validation_fraction": 0.15314258840296469,
    "n_iter_no_change": 28,
}


def build_pipeline(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("time", TimeFeatures(col="time")),
            ("poly", PolynomialColumns(col="Speed", degrees=(2, 3))),
            (
                "prep",
                ColumnTransformer(
                    [
                        ("dir", DirectionSinCos(), ["Direction"]),
                        ("roll", RollingMean(window=12, alpha=0.3), ["Speed"]),
                        ("num", "passthrough", numeric_columns),
                    ]
                ),
            ),
            ("pred", HistGradientBoostingRegressor(random_state=42, **params)),
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
@click.option("--experiment-name", default="wind-power-hgbr", show_default=True)
@click.option(
    "--tracking-uri", default="", help="Overrides MLFLOW_TRACKING_URI when set."
)
@click.option(
    "--registered-model-name",
    default="",
    help="Register the logged model under this name when set.",
)
@click.option("--predict-future", type=bool, default=True, show_default=True)
@click.option("--loss", default=BASELINE_PARAMS["loss"], show_default=True)
@click.option(
    "--learning-rate",
    type=float,
    default=BASELINE_PARAMS["learning_rate"],
    show_default=True,
)
@click.option(
    "--max-iter", type=int, default=BASELINE_PARAMS["max_iter"], show_default=True
)
@click.option(
    "--max-leaf-nodes",
    type=int,
    default=BASELINE_PARAMS["max_leaf_nodes"],
    show_default=True,
)
@click.option(
    "--min-samples-leaf",
    type=int,
    default=BASELINE_PARAMS["min_samples_leaf"],
    show_default=True,
)
@click.option(
    "--l2-regularization",
    type=float,
    default=BASELINE_PARAMS["l2_regularization"],
    show_default=True,
)
@click.option(
    "--max-bins", type=int, default=BASELINE_PARAMS["max_bins"], show_default=True
)
@click.option(
    "--max-features",
    type=float,
    default=BASELINE_PARAMS["max_features"],
    show_default=True,
)
@click.option(
    "--early-stopping",
    type=bool,
    default=BASELINE_PARAMS["early_stopping"],
    show_default=True,
)
@click.option(
    "--validation-fraction",
    type=float,
    default=BASELINE_PARAMS["validation_fraction"],
    show_default=True,
)
@click.option(
    "--n-iter-no-change",
    type=int,
    default=BASELINE_PARAMS["n_iter_no_change"],
    show_default=True,
)
def main(
    power_csv: str,
    weather_csv: str,
    future_csv: str,
    granularity: Granularity,
    experiment_name: str,
    tracking_uri: str,
    registered_model_name: str,
    predict_future: bool,
    loss: str,
    learning_rate: float,
    max_iter: int,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    max_bins: int,
    max_features: float,
    early_stopping: bool,
    validation_fraction: float,
    n_iter_no_change: int,
) -> None:
    resolve_experiment(experiment_name, tracking_uri)

    params: dict[str, Any] = {
        "loss": loss,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_regularization,
        "max_bins": max_bins,
        "max_features": max_features,
        "early_stopping": early_stopping,
        "validation_fraction": validation_fraction,
        "n_iter_no_change": n_iter_no_change,
    }

    joined_df = load_training_frame(power_csv, weather_csv, granularity)
    score = run_experiment(
        build_pipeline(params),
        run_name="hgbr_baseline",
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
