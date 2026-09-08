import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import HistGradientBoostingRegressor
    from transforms import (
        DirectionSinCos,
        PolynomialColumns,
        RollingMean,
        TimeFeatures,
        numeric_columns,
    )
    from data_wrangler import DataWrangler
    from pipeline_runner import PipelineRunner

    return (
        ColumnTransformer,
        DataWrangler,
        DirectionSinCos,
        LinearRegression,
        Pipeline,
        PipelineRunner,
        PolynomialColumns,
        RollingMean,
        SimpleImputer,
        StandardScaler,
        TimeFeatures,
        numeric_columns,
        pl,
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
def _(pl, wind_df):
    wind_df.filter(pl.any_horizontal(pl.all().is_null()))
    return


@app.cell
def _(wind_df):
    len(wind_df)
    return


@app.cell
def _(DataWrangler, power_df, wind_df):
    joined_df = DataWrangler(granularity="1m").resample(
        wind_df=wind_df, power_df=power_df
    )
    return (joined_df,)


@app.cell
def _(power_df, wind_df):
    wind_df["Speed"].std() + power_df["Total"].std()
    return


@app.cell
def _(joined_df):
    {
        "rows": joined_df.height,
        "real_rows": int(joined_df["is_real"].sum()),
        "real_frac": joined_df["is_real"].mean(),
    }
    return


app._unparsable_cell(
    r"""
    def direction_prep():
        return ColumnTransformer(
            [
                ("dir", DirectionSinCos(), ["Direction"]),
                ("roll", RollingMean(window=12, alpha=0.3), ["Speed"]),
                ("num", "passthrough", numeric_columns),
            ]
        )

    def feature_prep():
        return [
            ("time", TimeFeatures(col="time")),
            ("poly", PolynomialColumns(col="Speed", degrees=(2, 3))),
            ("prep", direction_prep()),
    

    def default_pipeline():
        return Pipeline(
            [*feature_prep(), ("pred", HistGradientBoostingRegressor())]
        )

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
                *feature_prep(),
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
                    ),
                ),
            ]
        )
    """,
    name="_"
)


@app.cell
def _():
    import mlflow
    import optuna

    mlflow.sklearn.autolog()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Hyperparameter Tuning")
    return mlflow, optuna


@app.cell
def _(joined_df):
    X = joined_df.select(["time", "Speed", "Direction"])
    y = joined_df.select("Total")
    mask_full = joined_df["is_real"].to_numpy()
    return X, mask_full, y


@app.cell
def _(PipelineRunner, X, mask_full, mk_pipeline, mlflow, optuna, y):
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
            score = PipelineRunner(
                pipeline, mask_full, split_counts=(3, 5, 8)
            ).run(X, y)
            trial.set_user_attr("run_id", child_run.info.run_id)
            return score

    return (objective,)


@app.cell(disabled=True)
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


@app.cell(disabled=True)
def _(PipelineRunner, X, default_pipeline, mask_full, mlflow, y):
    with mlflow.start_run(
        nested=True, run_name="default_init"
    ) as child_run_defualt:
        PipelineRunner(
            default_pipeline(), mask_full, split_counts=(3, 5, 8)
        ).run(X, y)
    return


@app.cell
def _(
    ColumnTransformer,
    DirectionSinCos,
    LinearRegression,
    Pipeline,
    PipelineRunner,
    PolynomialColumns,
    RollingMean,
    SimpleImputer,
    StandardScaler,
    TimeFeatures,
    X,
    mask_full,
    mlflow,
    numeric_columns,
    y,
):
    _num_branch = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    _roll_branch = Pipeline(
        [
            ("roll", RollingMean(window=12, alpha=0.3)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    pipeline_linear = Pipeline(
        [
            ("time", TimeFeatures(col="time")),
            ("poly", PolynomialColumns(col="Speed", degrees=(2, 3))),
            (
                "prep",
                ColumnTransformer(
                    [
                        ("dir", DirectionSinCos(), ["Direction"]),
                        ("roll", _roll_branch, ["Speed"]),
                        ("num", _num_branch, numeric_columns),
                    ]
                ),
            ),
            ("pred", LinearRegression()),
        ]
    )
    with mlflow.start_run(
        nested=True, run_name="linear_pipeline"
    ) as child_run_linear:
        PipelineRunner(
            pipeline_linear, mask_full, split_counts=(3, 5, 8)
        ).run(X, y)
    return


@app.cell
def _(PipelineRunner, X, mask_full, mk_pipeline, mlflow, optuna, y):
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
            score = PipelineRunner(
                pipeline, mask_full, split_counts=(3, 5, 8)
            ).run(X, y)
            trial.set_user_attr("run_id", child_run.info.run_id)
            return score

    return (objective_v2,)


@app.cell
def _(
    PipelineRunner,
    X,
    default_pipeline,
    mask_full,
    mlflow,
    objective_v2,
    optuna,
    y,
):
    mlflow.set_experiment("Hyperparameter Tuning v2")

    with mlflow.start_run(run_name="default_init_v2"):
        PipelineRunner(
            default_pipeline(), mask_full, split_counts=(3, 5, 8)
        ).run(X, y)

    with mlflow.start_run(run_name="Tune model v2") as run_v2:
        n_trials_v2 = 20
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
def _(PipelineRunner, X, mask_full, mk_pipeline, mlflow, optuna, y):
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

            score = PipelineRunner(
                pipeline, mask_full, split_counts=(3, 5, 8)
            ).run(X, y)
            trial.set_user_attr("run_id", child_run.info.run_id)
            return score

    return BASELINE_PARAMS, objective_v3


@app.cell
def _(
    BASELINE_PARAMS,
    PipelineRunner,
    X,
    mask_full,
    mk_pipeline,
    mlflow,
    objective_v3,
    optuna,
    y,
):
    mlflow.set_experiment("Hyperparameter Tuning v3")

    with mlflow.start_run(run_name="baseline_v2_best"):
        PipelineRunner(
            mk_pipeline(**BASELINE_PARAMS), mask_full, split_counts=(3, 5, 8)
        ).run(X, y)

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
