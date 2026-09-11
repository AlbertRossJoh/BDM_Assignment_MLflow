import mlflow
import click
import polars as pl
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


from experiment_builder import ExperimentBuilder

COMPASS = [
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


@click.command(help=__doc__)
@click.option("--power-csv", default="data/power.csv", show_default=True)
@click.option("--weather-csv", default="data/weather.csv", show_default=True)
def main(power_csv: str, weather_csv: str) -> None:
    mlflow.sklearn.autolog(log_models=True)
    every = "1h"
    power_df = (
        pl.read_csv(power_csv, try_parse_dates=True)
        .select(["time", "Total"])
        .sort("time")
        .group_by_dynamic("time", every=every)
        .agg(
            pl.col("Total").mean().alias("Total"),
            pl.col("Total").first().alias("real_obs_power"),
        )
    )
    wind_df = (
        pl.read_csv(weather_csv, try_parse_dates=True)
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

    (
        ExperimentBuilder(label="Total")
        .ignore("real_obs_weather")
        .with_features("Speed", "Direction")
        .with_pipeline(
            lambda p: p.with_transformer(
                name="direction encoding",
                transformer=OneHotEncoder(
                    categories=[COMPASS], sparse_output=False, handle_unknown="ignore"
                ),
                select="Direction",
            )
        )
        .run_experiment(joined_df, regressor=SVR(), validation="train_test")
    )


if __name__ == "__main__":
    main()
