import click
from typing import Literal

# Cell tags: imports
import mlflow

# You will probably need these
import pandas as pd
from sklearn.pipeline import Pipeline

# This are for example purposes. You may discard them if you don't use them.
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor


import polars as pl


def add_cyclic_direction(df: pl.DataFrame, col="Direction"):
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
    DIR_TO_DEG = {d: i * 22.5 for i, d in enumerate(COMPASS_16)}

    deg = df[col].replace(DIR_TO_DEG).cast(pl.Float64)
    rad = deg.radians()
    return df.with_columns(
        direction_sin=rad.sin(),
        direction_cos=rad.cos(),
    ).drop(col)


Loss = Literal["squared_error", "absolute_error", "gamma", "poisson", "quantile"]


@click.command()
@click.argument("training_data")
@click.option("--granularity", default="1h", help="Time bucket size, e.g. '1h', '15m'.")
@click.option(
    "--n-splits", default=5, type=int, help="Number of TimeSeriesSplit folds."
)
@click.option(
    "--loss",
    default="squared_error",
    type=click.Choice(
        ["squared_error", "absolute_error", "gamma", "poisson", "quantile"]
    ),
    help="HistGradientBoostingRegressor loss function.",
)
@click.option(
    "--learning-rate", default=0.1, type=float, help="Boosting learning rate."
)
@click.option(
    "--max-iter", default=100, type=int, help="Number of boosting iterations."
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    help="Max depth of each tree. None = unlimited.",
)
@click.option(
    "--max-leaf-nodes",
    default=31,
    type=int,
    help="Max leaves per tree. 0/None = unlimited.",
)
@click.option(
    "--min-samples-leaf", default=20, type=int, help="Min samples per leaf node."
)
@click.option(
    "--l2-regularization", default=0.0, type=float, help="L2 regularization penalty."
)
@click.option(
    "--max-bins",
    default=255,
    type=int,
    help="Max bins for feature discretization (<=255).",
)
@click.option(
    "--max-features",
    default=1.0,
    type=float,
    help="Fraction of features sampled per split.",
)
@click.option(
    "--early-stopping/--no-early-stopping",
    default=True,
    help="Enable early stopping on a validation set.",
)
@click.option(
    "--validation-fraction",
    default=0.1,
    type=float,
    help="Fraction held out for early-stopping validation.",
)
@click.option(
    "--n-iter-no-change",
    default=10,
    type=int,
    help="Iterations without improvement before stopping.",
)
@click.option(
    "--random-state", default=None, type=int, help="Seed for reproducibility."
)
def run(
    training_data: str,
    granularity: str,
    n_splits: int,
    loss: Loss,
    learning_rate: float,
    max_iter: int,
    max_depth: int | None,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
    max_bins: int,
    max_features: float,
    early_stopping: bool,
    validation_fraction: float,
    n_iter_no_change: int,
    random_state: int | None,
):
    power_df = (
        pl.from_pandas(
            pd.read_csv(training_data, parse_dates=["time"]), include_index=False
        )
        .select(["time", "Total"])
        .with_columns(pl.col("time").dt.round(granularity))
    )
    wind_df = (
        pl.from_pandas(
            pd.read_csv(training_data, parse_dates=["time"]), include_index=False
        )
        .select(["time", "Direction", "Speed"])
        .upsample(time_column="time", every=granularity)
        .with_columns(
            pl.col("Direction").fill_null(strategy="forward"),
            pl.col("Speed").fill_null(strategy="forward"),
        )
    )

    joined_df = power_df.join(wind_df, how="inner", on="time")
    x = joined_df.select(["Speed", "Direction"])
    x = add_cyclic_direction(x)
    y = joined_df.select("Total")

    pipeline = Pipeline(
        [
            ("Scaling", StandardScaler()),
            (
                "Gradient boosting",
                HistGradientBoostingRegressor(
                    loss=loss,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=l2_regularization,
                    max_bins=max_bins,
                    max_features=max_features,
                    early_stopping=early_stopping,
                    validation_fraction=validation_fraction,
                    n_iter_no_change=n_iter_no_change,
                    random_state=random_state,
                ),
            ),
        ]
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    with mlflow.start_run() as parent_run:
        mlflow.log_param("n_splits", n_splits)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(x)):
            X_train, y_train = x[train_idx], y[train_idx]
            X_test, y_test = x[test_idx], y[test_idx]
            fold_pipeline = clone(pipeline)

    return
