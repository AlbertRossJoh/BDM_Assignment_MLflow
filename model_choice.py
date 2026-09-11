import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        StandardScaler,
        FunctionTransformer,
        MinMaxScaler,
        OneHotEncoder,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR, LinearSVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.base import clone
    from transforms import (
        DirectionSinCos,
        PolynomialColumns,
        RollingMean,
        TimeFeatures,
        numeric_columns,
    )
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        root_mean_squared_error,
    )
    from data_wrangler import DataWrangler
    from pipeline_runner import PipelineRunner
    from experiment_builder import ExperimentBuilder, ColumnTransformerBuilder
    import mlflow

    return (
        ColumnTransformerBuilder,
        ExperimentBuilder,
        FunctionTransformer,
        HistGradientBoostingRegressor,
        LinearRegression,
        LinearSVR,
        MLPRegressor,
        OneHotEncoder,
        SVR,
        SimpleImputer,
        StandardScaler,
        TimeFeatures,
        mlflow,
        numeric_columns,
        pl,
    )


@app.cell
def _(mlflow):
    mlflow.sklearn.autolog(log_models=False)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    return


@app.cell
def _():
    dirs = [
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
    degs = [22.5 * i for i in range(1, len(dirs) + 1)]
    dir_2_deg = {dir: deg for dir, deg in zip(dirs, degs)}
    dir_2_deg
    return dir_2_deg, dirs


@app.cell
def _(pl):
    every = "1m"
    power_df = (
        pl.read_csv("data/power.csv", try_parse_dates=True)
        .select(["time", "Total"])
        .sort("time")
        .group_by_dynamic("time", every=every)
        .agg(
            pl.col("Total").mean(),
            pl.col("Total").first().alias("real_obs_power"),
        )
    )
    wind_df = (
        pl.read_csv("data/weather.csv", try_parse_dates=True)
        .select(["time", "Speed", "Direction"])
        .sort("time")
        .with_columns(real_obs_weather=True)
        .upsample("time", every=every)
        .with_columns(
            pl.col("Speed").interpolate(),
            pl.col("real_obs_weather").fill_null(False),
        )
    )
    joined_df = wind_df.join(power_df, how="inner", on=["time"])
    return (joined_df,)


@app.cell
def _(joined_df):
    joined_df
    return


@app.cell
def _(ExperimentBuilder, TimeFeatures):
    builder = (
        ExperimentBuilder(label="Total")
        .ignore("real_obs_weather")
        .with_features("time", "Speed", "Direction")
        .with_pipeline(
            lambda p: p.with_transformer(
                name="time feature engineering",
                transformer=TimeFeatures(),
            )
            # .with_transformer(
            #    name="direction encoding",
            #    transformer=OneHotEncoder(
            #        categories=[dirs],
            #        sparse_output=False, handle_unknown="ignore"
            #    ),
            #    select="Direction",
            # )
            # .with_transformer(
            #    name="Speed trans",
            #    transformer=FunctionTransformer(
            #        func=lambda x: x.with_columns(
            #            speed_sq=pl.col("Speed") ** 2,
            #            speed_cu=pl.col("Speed") ** 3,
            #        )
            #    ),
            # )
        )
    )
    return (builder,)


@app.cell
def _(builder, joined_df):
    builder.build_transformers().fit_transform(
        joined_df.select(builder.features)
    )
    return


@app.cell
def _(HistGradientBoostingRegressor, LinearRegression, MLPRegressor, SVR):
    regressors = [
        LinearRegression(),
        HistGradientBoostingRegressor(),
        SVR(),
        MLPRegressor(),
    ]
    # pipelines = list(map(build_pipeline, regressors))
    return (regressors,)


@app.cell
def _(builder, joined_df, regressors):
    (
        builder.using_regressors(*regressors)
        .drop("Direction")
        .with_experiment_name("Compare feature engineering")
        .with_run_name("Default")
        .run_experiment(joined_df)
    )
    return


@app.cell
def _(OneHotEncoder, builder, dirs, joined_df, regressors):
    (
        builder.using_regressors(*regressors)
        .with_pipeline(
            lambda p: p.with_transformer(
                name="direction encoding",
                transformer=OneHotEncoder(
                    categories=[dirs], sparse_output=False, handle_unknown="ignore"
                ),
                select="Direction",
            )
        )
        .with_experiment_name("Compare feature engineering")
        .with_run_name("Using OneHotEncoder")
        .run_experiment(joined_df)
    )
    return


@app.cell
def _(FunctionTransformer, builder, joined_df, pl, regressors):
    (
        builder.using_regressors(*regressors)
        .drop("Direction")
        .with_pipeline(
            lambda p: p.with_transformer(
                name="Speed trans",
                transformer=FunctionTransformer(
                    func=lambda x: x.with_columns(
                        speed_sq=pl.col("Speed") ** 2,
                        speed_cu=pl.col("Speed") ** 3,
                    )
                ),
            )
        )
        .with_experiment_name("Compare feature engineering")
        .with_run_name("with speed trans")
        .run_experiment(joined_df)
    )
    return


@app.cell
def _(
    FunctionTransformer,
    OneHotEncoder,
    builder,
    dirs,
    joined_df,
    pl,
    regressors,
):
    (
        builder.using_regressors(*regressors)
        .with_pipeline(
            lambda p: p.with_transformer(
                name="direction encoding",
                transformer=OneHotEncoder(
                    categories=[dirs], sparse_output=False, handle_unknown="ignore"
                ),
                select="Direction",
            ).with_transformer(
                name="Speed trans",
                transformer=FunctionTransformer(
                    func=lambda x: x.with_columns(
                        speed_sq=pl.col("Speed") ** 2,
                        speed_cu=pl.col("Speed") ** 3,
                    )
                ),
            )
        )
        .with_experiment_name("Compare feature engineering")
        .with_run_name("Using OneHotEncoder and speed trans")
        .run_experiment(joined_df)
    )
    return


@app.cell
def _(
    ColumnTransformerBuilder,
    ExperimentBuilder,
    FunctionTransformer,
    HistGradientBoostingRegressor,
    LinearRegression,
    LinearSVR,
    MLPRegressor,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
    TimeFeatures,
    dir_2_deg,
    dirs,
    joined_df,
    numeric_columns,
    pl,
):
    def build_rolling_mean_transformer(
        builder: ColumnTransformerBuilder,
    ) -> ColumnTransformerBuilder:
        return builder.with_transformer(
            chain_name="rolling mean",
            name="roll",
            select="Speed",
            transformer=FunctionTransformer(
                func=lambda x: x.with_columns(
                    pl.col("Speed")
                    .rolling_mean(window_size=12)
                    .alias("speed 12h mean")
                )
            ),
        ).with_chained(name="impute", transformer=SimpleImputer())

    def build_sin_cos_encoding_transformer(
        builder: ColumnTransformerBuilder,
    ) -> ColumnTransformerBuilder:
        return builder.with_transformer(
            chain_name="sin cos",
            name="trans",
            select="Direction",
            transformer=FunctionTransformer(
                func=lambda x: (
                    x.with_columns(
                        pl.col("Direction")
                        .replace_strict(dir_2_deg, return_dtype=pl.Float64)
                        .radians()
                        .alias("rad"),
                    )
                    .with_columns(
                        pl.col("rad").sin().alias("dir_sin"),
                        pl.col("rad").cos().alias("dir_cos"),
                    )
                    .select(["dir_sin", "dir_cos"])
                )
            ),
        ).with_chained(name="impute", transformer=SimpleImputer())

    (
        ExperimentBuilder(label="Total")
        .ignore("real_obs_weather")
        .with_features("time", "Speed", "Direction")
        .with_transformer(
            name="time feature engineering",
            transformer=TimeFeatures(),
        )
        .with_column_transformer(
            build_rolling_mean_transformer, name="rolling mean"
        )
        .with_transformer(
            name="direction encoding",
            transformer=OneHotEncoder(
                categories=[dirs], sparse_output=False, handle_unknown="ignore"
            ),
            select="Direction",
        )
        .with_transformer(
            name="Speed trans",
            transformer=FunctionTransformer(
                func=lambda x: x.with_columns(
                    speed_sq=pl.col("Speed") ** 2,
                    speed_cu=pl.col("Speed") ** 3,
                )
            ),
        )
        .with_column_transformer(
            build_sin_cos_encoding_transformer, name="sin cos encoding"
        )
        .with_transformer(
            name="standard scaler",
            transformer=StandardScaler(),
            select=numeric_columns,
        )
        .using_regressors(
            LinearRegression(),
            HistGradientBoostingRegressor(),
            LinearSVR(),  # cannot use standard svr due to dataset size when resampled to 1m
            MLPRegressor(),
        )
        .with_experiment_name(
            "Exhaustive search 1 minute resampling",
        )
        .exhaustive(
            joined_df,
            conflicting={
                "direction encoding": "sin cos encoding",
            },
            select="Speed",
            keep_on_present={
                "direction encoding": "Direction",
                "sin cos encoding": "Direction",
                "time feature engineering": "time",
            },
        )
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
