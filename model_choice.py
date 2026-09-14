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

    # from transforms import (
    #    DirectionSinCos,
    #    PolynomialColumns,
    #    RollingMean,
    #    TimeFeatures,
    #    numeric_columns,
    # )
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        root_mean_squared_error,
    )

    # from data_wrangler import DataWrangler
    # from pipeline_runner import PipelineRunner
    from experiment_builder import ExperimentBuilder, ColumnTransformerBuilder
    import mlflow

    return (
        ColumnTransformerBuilder,
        ExperimentBuilder,
        FunctionTransformer,
        HistGradientBoostingRegressor,
        LinearRegression,
        MLPRegressor,
        OneHotEncoder,
        SVR,
        SimpleImputer,
        StandardScaler,
        mlflow,
        np,
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
    power_df_raw = pl.read_csv("data/power.csv", try_parse_dates=True)
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
    wind_df_raw = pl.read_csv("data/weather.csv", try_parse_dates=True)
    wind_df = (
        pl.read_csv("data/weather.csv", try_parse_dates=True)
        .select(["time", "Speed", "Direction"])
        .sort("time")
        .with_columns(real_obs_weather=True)
        .upsample("time", every=every)
        .with_columns(
            pl.col("Speed").interpolate(),
            pl.col("Direction").fill_null(strategy="forward"),
            pl.col("real_obs_weather").fill_null(False),
        )
    )
    joined_df = wind_df.join(power_df, how="inner", on=["time"])
    return joined_df, power_df, power_df_raw, wind_df, wind_df_raw


@app.cell
def _(np, pl):
    def print_dataset_metrics(power_df: pl.DataFrame, wind_df: pl.DataFrame):
        power_metrics = power_df.select(
            [
                pl.col("Total").mean().alias("mean"),
                pl.col("Total").var().alias("variance"),
                pl.col("Total").std().alias("std"),
                pl.col("Total").skew().alias("skew"),
                pl.col("Total").kurtosis().alias("kurtosis"),
            ]
        )
        wind_metrics = wind_df.select(
            [
                pl.col("Speed").mean().alias("mean"),
                pl.col("Speed").var().alias("variance"),
                pl.col("Speed").std().alias("std"),
                pl.col("Speed").skew().alias("skew"),
                pl.col("Speed").kurtosis().alias("kurtosis"),
            ]
        )

        def shannon_entropy(df: pl.DataFrame, col: str) -> float:
            pcts = (
                df.group_by(col)
                .agg((pl.len() / df.height).alias("pct"))["pct"]
                .to_numpy()
            )
            return -np.sum(pcts * np.log2(pcts))

        entropy = shannon_entropy(wind_df, "Direction")
        return power_metrics, wind_metrics, entropy

    return (print_dataset_metrics,)


@app.cell
def _(power_df_raw):
    power_df_raw.sort("Total", descending=True)
    return


@app.cell
def _(power_df_raw, print_dataset_metrics, wind_df_raw):
    power_df_raw_metrics, wind_df_raw_metrics, power_df_raw_entropy = print_dataset_metrics(power_df_raw, wind_df_raw)
    return (power_df_raw_metrics,)


@app.cell
def _(power_df_raw_metrics):
    power_df_raw_metrics
    return


@app.cell
def _(power_df, print_dataset_metrics, wind_df):
    print_dataset_metrics(power_df, wind_df)
    return


@app.cell
def _(joined_df):
    joined_df
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
    return


@app.cell(disabled=True)
def _(
    ColumnTransformerBuilder,
    ExperimentBuilder,
    FunctionTransformer,
    HistGradientBoostingRegressor,
    LinearRegression,
    MLPRegressor,
    OneHotEncoder,
    SVR,
    SimpleImputer,
    StandardScaler,
    dir_2_deg,
    dirs,
    joined_df,
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
        .with_pipeline(
            lambda builder: (
                builder.with_transformer(
                    name="time feature engineering",
                    transformer=FunctionTransformer(
                        func=lambda x: x.with_columns(
                            pl.col("time").dt.hour().alias("hour"),
                            pl.col("time").dt.ordinal_day().alias("doy"),
                        ).drop("time")
                    ),
                )
                .with_column_transformer(
                    build_rolling_mean_transformer, name="rolling mean"
                )
                .with_transformer(
                    name="direction encoding",
                    transformer=OneHotEncoder(
                        categories=[dirs],
                        sparse_output=False,
                        handle_unknown="ignore",
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
                )
            )
        )
        .using_regressors(
            LinearRegression(),
            HistGradientBoostingRegressor(),
            SVR(),  # cannot use standard svr due to dataset size when resampled to 1m
            MLPRegressor(),
        )
        .with_experiment_name(
            "Exhaustive search 1 hour resampling, direction FF",
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
