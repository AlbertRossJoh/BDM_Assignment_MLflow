import click
import matplotlib.pyplot as plt
import polars as pl
import requests


def load_history(power_csv: str, weather_csv: str, every: str = "1h") -> pl.DataFrame:
    power_df = (
        pl.read_csv(power_csv, try_parse_dates=True)
        .select(["time", "Total"])
        .sort("time")
        .group_by_dynamic("time", every=every)
        .agg(pl.col("Total").mean().alias("Total"))
    )
    wind_df = (
        pl.read_csv(weather_csv, try_parse_dates=True)
        .select(["time", "Speed", "Direction"])
        .sort("time")
        .upsample("time", every=every)
        .with_columns(pl.col("Speed").interpolate())
    )
    return wind_df.join(power_df, how="inner", on=["time"]).sort("time")


def predict(df: pl.DataFrame, url: str) -> list[float]:
    features = df.select(["Speed", "Direction"])
    payload = {
        "dataframe_split": {
            "columns": features.columns,
            "data": features.rows(),
        }
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["predictions"]


@click.command()
@click.option("--power-csv", default="data/power.csv", show_default=True)
@click.option("--weather-csv", default="data/weather.csv", show_default=True)
@click.option("--future-csv", default="data/future.csv", show_default=True)
@click.option("--url", default="http://127.0.0.1:5001/invocations", show_default=True)
@click.option("--out", default="forecast.png", show_default=True)
def main(power_csv: str, weather_csv: str, future_csv: str, url: str, out: str) -> None:
    history = load_history(power_csv, weather_csv)
    history = history.with_columns(pl.Series("predicted", predict(history, url)))

    future = pl.read_csv(future_csv, try_parse_dates=True).sort("time")
    future = future.select(["time"]).with_columns(
        pl.Series("predicted", predict(future, url))
    )

    _history = history.select(["time", "predicted"]).tail(1)
    future = pl.concat([_history, future])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history["time"], history["Total"], color="black", label="actual")
    ax.plot(
        history["time"],
        history["predicted"],
        color="blue",
        alpha=0.6,
        label="predicted",
    )
    ax.plot(future["time"], future["predicted"], color="red", label="forecast")
    ax.set_xlabel("time")
    ax.set_ylabel("Total power")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved plot to {out}")


if __name__ == "__main__":
    main()
