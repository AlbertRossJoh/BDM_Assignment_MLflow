import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt
    from scipy.stats import gaussian_kde

    power_df_raw = pl.read_csv("data/power.csv", try_parse_dates=True)
    wind_df_raw = pl.read_csv("data/weather.csv", try_parse_dates=True)

    every_selector = mo.ui.dropdown(
        options=["1m", "5m", "15m", "30m", "1h", "2h", "3h"],
        value="1h",
        label="resampling interval",
    )
    return (
        alt,
        every_selector,
        gaussian_kde,
        mo,
        np,
        pl,
        power_df_raw,
        wind_df_raw,
    )


@app.cell
def _(
    alt,
    every_selector,
    gaussian_kde,
    mo,
    np,
    pl,
    power_df_raw,
    wind_df_raw,
):
    every = every_selector.value

    power_df = (
        power_df_raw.select(["time", "Total"])
        .sort("time")
        .group_by_dynamic("time", every=every)
        .agg(
            pl.col("Total").mean(),
            pl.col("Total").first().alias("real_obs_power"),
        )
    )
    wind_df = (
        wind_df_raw.select(["time", "Speed", "Direction"])
        .sort("time")
        .with_columns(real_obs_weather=True)
        .upsample("time", every=every)
        .with_columns(
            pl.col("Speed").interpolate(),
            pl.col("Direction").fill_null(strategy="forward"),
            pl.col("real_obs_weather").fill_null(False),
        )
    )

    def kde_df(vals: np.ndarray, grid: np.ndarray, bw_method: float, stage: str) -> pl.DataFrame:
        density = gaussian_kde(vals, bw_method=bw_method)(grid)
        return pl.DataFrame({"value": grid, "density": density, "stage": stage})

    def resampling_overlap_chart(
        raw: pl.DataFrame, resampled: pl.DataFrame, col: str, title: str, n_points: int = 200
    ):
        raw_vals = raw[col].drop_nulls().to_numpy()
        resampled_vals = resampled[col].drop_nulls().to_numpy()
        lo = float(min(raw_vals.min(), resampled_vals.min()))
        hi = float(max(raw_vals.max(), resampled_vals.max()))
        grid = np.linspace(lo, hi, n_points)
        # Scott's rule bandwidth shrinks with n, so fitting each stage
        # independently over-smooths the (much smaller) resampled series
        # relative to the raw one. Share one bandwidth, from the larger raw
        # sample, so any visual difference reflects the distribution, not
        # the sample-size-dependent smoothing.
        shared_bw = gaussian_kde(raw_vals).factor
        data = pl.concat(
            [
                kde_df(raw_vals, grid, shared_bw, "before resampling"),
                kde_df(resampled_vals, grid, shared_bw, "after resampling"),
            ]
        )
        stage_selection = alt.selection_point(fields=["stage"], bind="legend", empty=True)
        return (
            alt.Chart(data)
            .mark_area(interpolate="monotone", line=True)
            .encode(
                x=alt.X("value:Q", title=col),
                y=alt.Y("density:Q", stack=None, title="density"),
                color=alt.Color("stage:N", title="stage"),
                opacity=alt.condition(stage_selection, alt.value(0.45), alt.value(0.05)),
            )
            .add_params(stage_selection)
            .properties(title=title, width=380, height=300)
        )

    def categorical_class_size_chart(
        raw: pl.DataFrame, resampled: pl.DataFrame, col: str, title: str, order: list[str]
    ):
        def counts_df(df: pl.DataFrame, stage: str) -> pl.DataFrame:
            counts = df.group_by(col).agg(pl.len().alias("count"))
            total = counts["count"].sum()
            return counts.with_columns(
                share=pl.col("count") / total,
                stage=pl.lit(stage),
            )

        data = pl.concat(
            [
                counts_df(raw, "before resampling"),
                counts_df(resampled, "after resampling"),
            ]
        )
        stage_selection = alt.selection_point(fields=["stage"], bind="legend", empty=True)
        return (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X(f"{col}:N", title=col, sort=order),
                y=alt.Y("share:Q", stack=None, title="share of observations", axis=alt.Axis(format="%")),
                color=alt.Color("stage:N", title="stage"),
                opacity=alt.condition(stage_selection, alt.value(0.55), alt.value(0.05)),
            )
            .add_params(stage_selection)
            .properties(title=title, width=380, height=300)
        )

    direction_order = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]

    power_chart = resampling_overlap_chart(
        power_df_raw, power_df, "Total", f"Power (Total) distribution — {every} resampling"
    )
    wind_chart = resampling_overlap_chart(
        wind_df_raw, wind_df, "Speed", f"Wind Speed distribution — {every} resampling"
    )
    direction_chart = categorical_class_size_chart(
        wind_df_raw,
        wind_df,
        "Direction",
        f"Direction class sizes — {every} resampling",
        direction_order,
    )
    mo.hstack(
        [every_selector, alt.hconcat(power_chart, wind_chart, direction_chart)],
        align="center",
    )
    return


if __name__ == "__main__":
    app.run()
