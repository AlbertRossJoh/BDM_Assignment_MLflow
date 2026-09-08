import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import mlflow
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
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
        Pipeline,
        PipelineRunner,
        PolynomialColumns,
        RidgeCV,
        RollingMean,
        SimpleImputer,
        StandardScaler,
        TimeFeatures,
        TimeSeriesSplit,
        mlflow,
        np,
        numeric_columns,
        pl,
    )


@app.cell
def _(pl):
    power_df = pl.read_csv("data/power.csv", try_parse_dates=True).select(
        ["time", "Total"]
    )
    wind_df = pl.read_csv("data/weather.csv", try_parse_dates=True).select(
        ["time", "Direction", "Speed"]
    )
    return power_df, wind_df


@app.cell
def _(DataWrangler, power_df, wind_df):
    joined_df = DataWrangler(granularity="1m").resample(
        wind_df=wind_df, power_df=power_df
    )
    return (joined_df,)


@app.cell
def _(joined_df):
    X = joined_df.select(["time", "Speed", "Direction"])
    y = joined_df.select("Total")
    mask_full = joined_df["is_real"].to_numpy()
    return X, mask_full, y


@app.cell
def _(mlflow):
    mlflow.sklearn.autolog()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Ridge CV")
    return


@app.cell
def _(TimeSeriesSplit):
    tscv = TimeSeriesSplit(n_splits=3)
    return (tscv,)


@app.cell
def _(
    ColumnTransformer,
    DirectionSinCos,
    Pipeline,
    PipelineRunner,
    PolynomialColumns,
    RidgeCV,
    RollingMean,
    SimpleImputer,
    StandardScaler,
    TimeFeatures,
    X,
    mask_full,
    mlflow,
    np,
    numeric_columns,
    tscv,
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
    pipeline_linear_ridge = Pipeline(
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
            ("pred", RidgeCV(alphas=np.logspace(-3, 3, 25), cv=tscv)),
        ]
    )
    with mlflow.start_run(
        run_name="linear_pipeline_ridge"
    ) as child_run_linear_ridge:
        PipelineRunner(
            pipeline_linear_ridge, mask_full, split_counts=(3, 5, 8)
        ).run(X, y)
    return


if __name__ == "__main__":
    app.run()
