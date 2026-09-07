import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import (
        train_test_split,
        cross_val_score,
        TimeSeriesSplit,
    )
    from sklearn.metrics import (
        mean_absolute_error,
        r2_score,
        mean_squared_error,
        root_mean_squared_error,
    )

    return (
        HistGradientBoostingRegressor,
        LinearRegression,
        OneHotEncoder,
        Pipeline,
        TimeSeriesSplit,
        cross_val_score,
        mean_absolute_error,
        mean_squared_error,
        np,
        pl,
        r2_score,
        root_mean_squared_error,
        train_test_split,
    )


@app.cell
def _(pl):
    # Loading
    power_df = pl.read_csv("data/power.csv", try_parse_dates=True).select(
        ["time", "Total"]
    )
    wind_df = pl.read_csv("data/weather.csv", try_parse_dates=True).select(
        ["time", "Direction", "Speed"]
    )
    return power_df, wind_df


@app.cell
def _(pl, power_df, wind_df):
    # Wrangling
    def resample_old(
        wind_df: pl.DataFrame, power_df: pl.DataFrame, granularity="1h"
    ) -> pl.DataFrame:
        wind_df_upsampled = wind_df.upsample(
            time_column="time", every=granularity
        ).with_columns(
            pl.col("Direction").fill_null(strategy="forward"),
            pl.col("Speed").fill_null(strategy="forward"),
        )
        power_df_downsampled = power_df.with_columns(
            pl.col("time").dt.round(granularity)
        )
        return power_df_downsampled.join(
            wind_df_upsampled, how="inner", on="time"
        )

    def resample(
        wind_df: pl.DataFrame, power_df: pl.DataFrame, granularity="1h"
    ) -> pl.DataFrame:
        _power_df = power_df.group_by_dynamic(
            "time", every=granularity
        ).agg(pl.col("Total").mean())
        _wind_df = (
            wind_df.sort("time")
            .upsample(time_column="time", every=granularity)
            .with_columns(
                pl.col("Speed").interpolate(),
                pl.col("Direction").fill_null(strategy="forward"),
                (pl.int_range(pl.len()) * 0).alias("_"),
            )
            .with_columns(
                pl.col("Speed").is_not_null().cum_sum().alias("obs_id")
            )
        )
        return _power_df.join(_wind_df, how="inner", on="time")

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

    def add_cyclic_direction(df, col="Direction"):
        deg = df[col].replace(DIR_TO_DEG).cast(pl.Float64)
        rad = deg.radians()
        return df.with_columns(
            direction_sin=rad.sin(),
            direction_cos=rad.cos(),
        )

    joined_df = (
        resample(wind_df=wind_df, power_df=power_df)
        .pipe(add_cyclic_direction)
        .drop_nulls()
    )
    return add_cyclic_direction, joined_df, resample


@app.cell
def _(add_cyclic_direction, power_df, resample, wind_df):
    resample(wind_df=wind_df, power_df=power_df).pipe(add_cyclic_direction)
    return


@app.cell
def _(HistGradientBoostingRegressor, Pipeline):
    # Pipeline
    def mk_pipeline(
        loss,
        learning_rate,
        max_iter,
        max_leaf_nodes,
        min_samples_leaf,
        l2_regularization,
        max_bins,
        max_features,
        early_stopping,
        validation_fraction,
        n_iter_no_change,
    ):
        return Pipeline(
            [
                # ("quantile", QuantileTransformer()),
                (
                    "pred",
                    HistGradientBoostingRegressor(
                        loss=loss,
                        learning_rate=learning_rate,
                        max_iter=max_iter,
                        max_leaf_nodes=max_leaf_nodes,
                        min_samples_leaf=min_samples_leaf,
                        l2_regularization=l2_regularization,
                        max_bins=max_bins,
                        max_features=max_features,
                        early_stopping=early_stopping,
                        validation_fraction=validation_fraction,
                        n_iter_no_change=n_iter_no_change,
                        random_state=42,
                        categorical_features=["Direction"],
                    ),
                ),
            ]
        )

    return (mk_pipeline,)


@app.cell
def _():
    import mlflow
    import optuna

    mlflow.sklearn.autolog()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Hyperparameter Tuning")
    return mlflow, optuna


@app.cell
def _(np):
    # kernel
    def get_window(n, alpha=0.3):
        n = 12
        alpha = 0.3

        raw_weights = np.exp(-alpha * np.arange(n))

        weights = list(reversed(raw_weights / np.sum(raw_weights)))
        return weights

    return (get_window,)


@app.cell
def _(get_window, joined_df, pl):
    X = (
        joined_df.select(["time", "Speed", "Direction"])
        .with_columns(
            pl.col("time").dt.hour().alias("hour"),
            pl.col("time").dt.ordinal_day().alias("doy"),
            pl.col("Speed")
            .rolling_mean(window_size=12, weights=get_window(12))
            .alias("rolling_mean"),
        )
        .drop("time")
    )
    y = joined_df.select("Total")
    return X, y


@app.cell
def _(TimeSeriesSplit, X, train_test_split, y):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    cv = TimeSeriesSplit(n_splits=3)
    return X_test, X_train, cv, y_test, y_train


@app.cell
def _(
    X,
    X_test,
    X_train,
    cross_val_score,
    cv,
    mean_absolute_error,
    mean_squared_error,
    mlflow,
    r2_score,
    root_mean_squared_error,
    y,
    y_test,
    y_train,
):
    def run_with_pipeline(pipeline):
        score = cross_val_score(pipeline, X, y, cv=cv, scoring="r2").mean()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mlflow.log_metrics(
            {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": score}
        )
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=[
                "sklearn._loss.link.IdentityLink",
                "sklearn._loss.link.Interval",
                "sklearn._loss.loss.AbsoluteError",
                "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                "sklearn._loss.loss.HalfSquaredError",
                "functools.partial",
                "sklearn.compose._column_transformer._RemainderColsList",
                "sklearn.utils.validation.check_array",
            ],
        )
        return score

    return (run_with_pipeline,)


@app.cell
def _(mk_pipeline, mlflow, optuna, run_with_pipeline):
    # objective function
    def objective(trial: optuna.trial.Trial):
        with mlflow.start_run(
            nested=True, run_name=f"trail_{trial.number}"
        ) as child_run:
            loss = trial.suggest_categorical(
                "loss",
                [
                    "squared_error",
                    "absolute_error",
                    # "gamma",
                    # "poisson",
                    # "quantile",
                ],
            )
            params = {
                "loss": "absolute_error",
                # "learning_rate": trial.suggest_float(
                #    "learning_rate", 0.1, 0.3
                # ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.1,  # lower learning rate
                ),
                # "max_iter": trial.suggest_int("max_iter", 50, 200),
                "max_iter": 100,  # set max iter to default for now
                # "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 20, 62),
                "max_leaf_nodes": trial.suggest_int(
                    "max_leaf_nodes", 8, 31
                ),  # high leaf nodes could contribute to overfitting
                # "min_samples_leaf": trial.suggest_int(
                #    "min_samples_leaf", 50, 300
                # ),
                # "min_samples_leaf": trial.suggest_int(
                #    "min_samples_leaf", 100, 400 # up sample size
                # ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    300,
                    2000,  # up sample size even more
                ),
                # "l2_regularization": trial.suggest_float(
                #    "l2_regularization", 1.0, 10.0
                # ),
                "l2_regularization": trial.suggest_float(
                    "l2_regularization",
                    8.0,
                    16.0,  # increase l2 for more pruning
                ),
                "max_bins": trial.suggest_int("max_bins", 16, 32),
                # "max_features": trial.suggest_float("max_features", 0.0, 1.0),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
                # "early_stopping": trial.suggest_categorical(
                #    "early_stopping", ["auto", False, True]
                # ),
                "early_stopping": True,  # fix overfitting
                # "validation_fraction": trial.suggest_float(
                #    "validation_fraction", 1e-3, 0.5
                # ),
                "validation_fraction": trial.suggest_float(
                    "validation_fraction",
                    0.1,
                    0.5,  # raise validation fraction floor, could be a false flag
                ),
                "n_iter_no_change": trial.suggest_int(
                    "n_iter_no_change", 5, 30
                ),
            }
            mlflow.log_params(params)
            pipeline = mk_pipeline(**params)
            score = run_with_pipeline(pipeline)
            trial.set_user_attr("run_id", child_run.info.run_id)
            return score

    return (objective,)


@app.cell
def _(mlflow, objective, optuna):
    # Run tuning
    with mlflow.start_run(run_name="Tune model") as run:
        n_trails = 50
        mlflow.log_param("n_trails", n_trails)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trails)

        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics({"best_r2": study.best_value})
        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)
    return


@app.cell
def _(HistGradientBoostingRegressor, Pipeline, mlflow, run_with_pipeline):
    pipeline = Pipeline(
        [
            # ("quantile", QuantileTransformer()),
            (
                "pred",
                HistGradientBoostingRegressor(
                    categorical_features=["Direction"]
                ),
            ),
        ]
    )
    with mlflow.start_run(
        nested=True, run_name=f"default_init"
    ) as child_run_defualt:
        run_with_pipeline(pipeline)
        # trial.set_user_attr("run_id", child_run_defualt.info.run_id)
    return


@app.cell
def _(LinearRegression, OneHotEncoder, Pipeline, mlflow, run_with_pipeline):
    pipeline_linear = Pipeline(
        [
            ("encoding", OneHotEncoder(handle_unknown="ignore")),
            ("pred", LinearRegression()),
        ]
    )
    with mlflow.start_run(
        nested=True, run_name=f"linear_pipeline"
    ) as child_run_linear:
        run_with_pipeline(pipeline_linear)
        # trial.set_user_attr("run_id", child_run_linear.info.run_id)
    return


@app.cell
def _(mk_pipeline, mlflow, optuna, run_with_pipeline):
    # objective v2: search space rescaled for the ~2k-row hourly dataset.
    # The old ranges (min_samples_leaf 300-2000, max_iter fixed at 100,
    # learning_rate down to 0.01) forced severe underfitting once resample()
    # collapsed the data from ~109k rows to ~2k, which is why default_init
    # beat every tuned trial. These ranges are sized for the current dataset.
    def objective_v2(trial: optuna.trial.Trial):
        with mlflow.start_run(
            nested=True, run_name=f"v2_trial_{trial.number}"
        ) as child_run:
            params = {
                "loss": "absolute_error",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.02, 0.2, log=True
                ),
                "max_iter": trial.suggest_int("max_iter", 100, 500),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 5, 50
                ),
                "l2_regularization": trial.suggest_float(
                    "l2_regularization", 0.0, 5.0
                ),
                "max_bins": trial.suggest_int("max_bins", 32, 255),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
                "early_stopping": trial.suggest_categorical(
                    "early_stopping", [False, True]
                ),
                "validation_fraction": trial.suggest_float(
                    "validation_fraction", 0.1, 0.2
                ),
                "n_iter_no_change": trial.suggest_int(
                    "n_iter_no_change", 15, 30
                ),
            }
            mlflow.log_params(params)
            pipeline = mk_pipeline(**params)
            score = run_with_pipeline(pipeline)
            trial.set_user_attr("run_id", child_run.info.run_id)
            return score

    return (objective_v2,)


@app.cell
def _(
    HistGradientBoostingRegressor,
    Pipeline,
    mlflow,
    objective_v2,
    optuna,
    run_with_pipeline,
):
    mlflow.set_experiment("Hyperparameter Tuning v2")

    with mlflow.start_run(run_name="default_init_v2"):
        run_with_pipeline(
            Pipeline(
                [
                    (
                        "pred",
                        HistGradientBoostingRegressor(
                            categorical_features=["Direction"]
                        ),
                    )
                ]
            )
        )

    with mlflow.start_run(run_name="Tune model v2") as run_v2:
        n_trials_v2 = 50
        mlflow.log_param("n_trials", n_trials_v2)

        study_v2 = optuna.create_study(direction="maximize")
        study_v2.enqueue_trial(
            {
                "learning_rate": 0.1,
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.0,
                "max_bins": 255,
                "max_features": 1.0,
                "early_stopping": False,
                "validation_fraction": 0.1,
                "n_iter_no_change": 15,
            }
        )
        study_v2.optimize(objective_v2, n_trials=n_trials_v2)

        mlflow.log_params(study_v2.best_trial.params)
        mlflow.log_metrics({"best_r2": study_v2.best_value})
        if best_run_id_v2 := study_v2.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id_v2)
    return


@app.cell
def _(mk_pipeline, mlflow, optuna, run_with_pipeline):

    BASELINE_PARAMS = {
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

    def objective_v3(trial: optuna.trial.Trial):
        with mlflow.start_run(
            nested=True, run_name=f"v3_trial_{trial.number}"
        ) as child_run:
            params = {
                "loss": "absolute_error",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.03, 0.09, log=True
                ),
                "max_iter": trial.suggest_int("max_iter", 120, 220),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 24),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 3, 20
                ),
                "l2_regularization": trial.suggest_float(
                    "l2_regularization", 2.0, 8.0
                ),
                "max_bins": trial.suggest_int("max_bins", 128, 255),
                "max_features": trial.suggest_float("max_features", 0.4, 0.8),
                "early_stopping": trial.suggest_categorical(
                    "early_stopping", [False, True]
                ),
                "validation_fraction": trial.suggest_float(
                    "validation_fraction", 0.1, 0.2
                ),
                "n_iter_no_change": trial.suggest_int(
                    "n_iter_no_change", 20, 30
                ),
            }
            mlflow.log_params(params)
            pipeline = mk_pipeline(**params)
            score = run_with_pipeline(pipeline)
            trial.set_user_attr("run_id", child_run.info.run_id)
            return score

    return BASELINE_PARAMS, objective_v3


@app.cell
def _(
    BASELINE_PARAMS,
    mk_pipeline,
    mlflow,
    objective_v3,
    optuna,
    run_with_pipeline,
):
    mlflow.set_experiment("Hyperparameter Tuning v3")

    with mlflow.start_run(run_name="baseline_v2_best"):
        run_with_pipeline(mk_pipeline(**BASELINE_PARAMS))

    with mlflow.start_run(run_name="Tune model v3") as run_v3:
        n_trials_v3 = 20
        mlflow.log_param("n_trials", n_trials_v3)

        study_v3 = optuna.create_study(direction="maximize")
        study_v3.enqueue_trial(
            {k: v for k, v in BASELINE_PARAMS.items() if k != "loss"}
        )
        study_v3.optimize(objective_v3, n_trials=n_trials_v3)

        mlflow.log_params(study_v3.best_trial.params)
        mlflow.log_metrics({"best_r2": study_v3.best_value})
        if best_run_id_v3 := study_v3.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id_v3)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
