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
    from sklearn.model_selection import (
        TimeSeriesSplit,
        ShuffleSplit,
        train_test_split,
        GroupShuffleSplit,
    )
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
    from experiment_builder import (
        ExperimentBuilder,
        ColumnTransformerBuilder,
        PipelineBuilder,
    )
    import mlflow

    return (
        ColumnTransformerBuilder,
        ExperimentBuilder,
        FunctionTransformer,
        GroupShuffleSplit,
        HistGradientBoostingRegressor,
        LinearRegression,
        MLPRegressor,
        MinMaxScaler,
        OneHotEncoder,
        Pipeline,
        PipelineBuilder,
        SVR,
        SimpleImputer,
        StandardScaler,
        TimeSeriesSplit,
        mlflow,
        np,
        pl,
        r2_score,
        train_test_split,
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
    every = "1h"
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
    return joined_df, power_df_raw, wind_df_raw


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
    power_df_raw_metrics, wind_df_raw_metrics, power_df_raw_entropy = (
        print_dataset_metrics(power_df_raw, wind_df_raw)
    )
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


@app.cell
def _(
    ColumnTransformerBuilder,
    ExperimentBuilder,
    FunctionTransformer,
    GroupShuffleSplit,
    HistGradientBoostingRegressor,
    LinearRegression,
    MLPRegressor,
    MinMaxScaler,
    OneHotEncoder,
    SVR,
    SimpleImputer,
    StandardScaler,
    TimeSeriesSplit,
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

    gss = GroupShuffleSplit()
    tscv = TimeSeriesSplit()

    def group_shuffle_split(df: pl.DataFrame):
        groups = joined_df["real_obs_weather"].cum_sum()
        return gss.split(df, groups=groups)

    def tss(df: pl.DataFrame):
        groups = joined_df["real_obs_weather"].cum_sum()
        for train_idx, test_idx in tscv.split(df):
            last_train_group = groups[train_idx].max()
            keep = (
                groups[train_idx] != last_train_group
            )  # avoids interpolation leak, drop last group
            yield train_idx[keep], test_idx

    (
        ExperimentBuilder(label="Total")
        .ignore("real_obs_weather")
        .with_features("time", "Speed", "Direction")
        .with_pipeline(
            lambda builder: (
                builder
                # .with_transformer(
                #    name="time feature engineering",
                #    transformer=FunctionTransformer(
                #        func=lambda x: x.with_columns(
                #            pl.col("time").dt.hour().alias("hour"),
                #            pl.col("time").dt.ordinal_day().alias("doy"),
                #        ).drop("time")
                #    ),
                # )
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
                .with_transformer(
                    name="min max scaler",
                    transformer=MinMaxScaler(),
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
            "Exhaustive search test scaling",
        )
        .exhaustive(
            joined_df,
            conflicting={
                "direction encoding": "sin cos encoding",
                "standard scaler": "min max scaler",
            },
            select="Speed",
            keep_on_present={
                "direction encoding": "Direction",
                "sin cos encoding": "Direction",
                #"time feature engineering": "time",
            },
            split=tss,
        )
    )
    return


@app.cell
def _(
    OneHotEncoder,
    PipelineBuilder,
    SVR,
    dirs,
    joined_df,
    r2_score,
    train_test_split,
):
    svr_pipeline = (
        PipelineBuilder()
        .with_transformer(
            name="direction encoding",
            transformer=OneHotEncoder(
                categories=[dirs],
                sparse_output=False,
                handle_unknown="ignore",
            ),
            select="Direction",
        )
        .with_regressor(SVR())
        .build()
    )
    X = joined_df.select(["Speed", "Direction"])
    y = joined_df.select(["Total"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    svr_pipeline.fit(X_train, y_train)
    y_pred = svr_pipeline.predict(X_test)
    r2_score(y_test, y_pred)
    return (svr_pipeline,)


@app.cell
def _(pl, svr_pipeline):
    future_df = pl.read_csv("./data/future.csv", try_parse_dates=True)
    future_X = future_df.select(["Speed", "Direction"])
    y_fut = svr_pipeline.predict(future_X)
    return future_df, y_fut


@app.cell
def _(future_df, pl, y_fut):
    import altair as alt

    data_future = pl.concat(
        [future_df, pl.Series("Total", y_fut).to_frame()], how="horizontal"
    ).select(["time", "Speed", "Direction", "Total"])
    data_future
    return alt, data_future


@app.cell
def _(joined_df):
    joined_df.select(["time", "Speed", "Direction", "Total"])
    return


@app.cell
def _(alt, data_future, joined_df, pl):
    total_data = pl.concat(
        [
            joined_df.select(
                ["time", "Speed", "Direction", "Total"]
            ).with_columns(pl.lit("Historical").alias("Period")),
            data_future.with_columns(pl.lit("Forecast").alias("Period")),
        ]
    )

    alt.Chart(total_data).mark_line(point=False).encode(
        x=alt.X("time:T"),
        y=alt.Y("Total:Q"),
        color=alt.Color(
            "Period:N",
            scale=alt.Scale(
                domain=["Historical", "Forecast"], range=["#4C78A8", "#E45756"]
            ),
        ),
    ).properties(width=800)
    return


@app.cell
def _(
    FunctionTransformer,
    GridSearchCV,
    HistGradientBoostingRegressor,
    LinearRegression,
    MLPRegressor,
    MinMaxScaler,
    OneHotEncoder,
    Pipeline,
    SVR,
    SimpleImputer,
    StandardScaler,
    TimeSeriesSplit,
    dir_2_deg,
    dirs,
    pl,
):
    class CustomTSCV:
        def __init__(self):
            self.tscv = TimeSeriesSplit()

        def split(self, X, y=None, groups=None):
            g = X["real_obs_weather"].cum_sum()
            for train_idx, test_idx in self.tscv.split(X):
                last_train_group = g.gather(train_idx).max()
                keep = g.gather(train_idx) != last_train_group
                yield train_idx[keep], test_idx

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.tscv.get_n_splits()

    base_pipeline = Pipeline(
        [
            ("rolling mean", "passthrough"),
            ("time features", "passthrough"),
            ("direction encoding", "passthrough"),
            ("scaler", "passthrough"),
            ("pred", LinearRegression()),
        ]
    )

    param_grid = {
        "rolling mean": [
            Pipeline(
                [
                    (
                        "mean",
                        FunctionTransformer(
                            func=lambda x: x.with_columns(
                                pl.col("Speed")
                                .rolling_mean(window_size=12)
                                .alias("speed 12h mean")
                            )
                        ),
                    ),
                    ("imputing", SimpleImputer()),
                ]
            )
        ],
        "time features": [
            FunctionTransformer(
                func=lambda x: x.with_columns(
                    pl.col("time").dt.hour().alias("hour"),
                    pl.col("time").dt.ordinal_day().alias("doy"),
                ).drop("time")
            )
        ],
        "direction encoding": [
            OneHotEncoder(
                categories=[dirs],
                sparse_output=False,
                handle_unknown="ignore",
            ),
            Pipeline(
                [
                    (
                        "sin cos",
                        FunctionTransformer(
                            func=lambda x: (
                                x.with_columns(
                                    pl.col("Direction")
                                    .replace_strict(
                                        dir_2_deg, return_dtype=pl.Float64
                                    )
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
                    ),
                    ("imputing", SimpleImputer()),
                ]
            ),
        ],
        "scaler": [StandardScaler(), MinMaxScaler()],
        "pred": [SVR(), HistGradientBoostingRegressor(), MLPRegressor()],
    }

    grid = GridSearchCV(base_pipeline, param_grid, cv=CustomTSCV())
    return


if __name__ == "__main__":
    app.run()
