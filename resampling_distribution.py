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
    from sklearn.model_selection import TimeSeriesSplit

    power_df_raw = pl.read_csv("data/power.csv", try_parse_dates=True)
    wind_df_raw = pl.read_csv("data/weather.csv", try_parse_dates=True)

    every_selector = mo.ui.dropdown(
        options=["1m", "5m", "15m", "30m", "1h", "2h", "3h"],
        value="1h",
        label="resampling interval",
    )
    return (
        TimeSeriesSplit,
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
            pl.col("Total").median(),
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

    def kde_df(
        vals: np.ndarray, grid: np.ndarray, bw_method: float, stage: str
    ) -> pl.DataFrame:
        density = gaussian_kde(vals, bw_method=bw_method)(grid)
        return pl.DataFrame(
            {"value": grid, "density": density, "stage": stage}
        )

    def resampling_overlap_chart(
        raw: pl.DataFrame,
        resampled: pl.DataFrame,
        col: str,
        title: str,
        n_points: int = 200,
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
        stage_selection = alt.selection_point(
            fields=["stage"], bind="legend", empty=True
        )
        return (
            alt.Chart(data)
            .mark_area(interpolate="monotone", line=True)
            .encode(
                x=alt.X("value:Q", title=col),
                y=alt.Y("density:Q", stack=None, title="density"),
                color=alt.Color("stage:N", title="stage"),
                opacity=alt.condition(
                    stage_selection, alt.value(0.45), alt.value(0.05)
                ),
            )
            .add_params(stage_selection)
            .properties(title=title, width=380, height=300)
        )

    def categorical_class_size_chart(
        raw: pl.DataFrame,
        resampled: pl.DataFrame,
        col: str,
        title: str,
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
        stage_selection = alt.selection_point(
            fields=["stage"], bind="legend", empty=True
        )
        return (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X(
                    f"{col}:N",
                    title=col,
                    sort=alt.EncodingSortField(
                        field="share", op="sum", order="descending"
                    ),
                ),
                y=alt.Y(
                    "share:Q",
                    stack=None,
                    title="share of observations",
                    axis=alt.Axis(format="%"),
                ),
                color=alt.Color("stage:N", title="stage"),
                opacity=alt.condition(
                    stage_selection, alt.value(0.55), alt.value(0.05)
                ),
            )
            .add_params(stage_selection)
            .properties(title=title, width=380, height=300)
        )

    power_chart = resampling_overlap_chart(
        power_df_raw,
        power_df,
        "Total",
        f"Power (Total) distribution — {every} resampling",
    )
    wind_chart = resampling_overlap_chart(
        wind_df_raw,
        wind_df,
        "Speed",
        f"Wind Speed distribution — {every} resampling",
    )
    direction_chart = categorical_class_size_chart(
        wind_df_raw,
        wind_df,
        "Direction",
        f"Direction class sizes — {every} resampling",
    )
    mo.hstack(
        [
            every_selector,
            alt.hconcat(power_chart, wind_chart, direction_chart),
        ],
        align="center",
    )
    return direction_chart, power_chart, power_df, wind_chart, wind_df


@app.cell
def _(power_chart):
    power_chart
    return


@app.cell
def _(wind_chart):
    wind_chart
    return


@app.cell
def _(direction_chart):
    direction_chart
    return


@app.cell
def _(alt, pl, power_df_raw, wind_df_raw):
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
    dir_2_deg = {d: 22.5 * i for i, d in enumerate(dirs)}

    corr_cols = [
        "Speed",
        "speed_sq",
        "speed_cu",
        "dir_sin",
        "dir_cos",
        "doy",
        "hour",
        "Total",
    ]
    corr_df = (
        wind_df_raw.join(power_df_raw, on="time", how="inner")
        .select(
            pl.col("Speed"),
            (pl.col("Speed") ** 2).alias("speed_sq"),
            (pl.col("Speed") ** 3).alias("speed_cu"),
            pl.col("Direction")
            .replace_strict(dir_2_deg, return_dtype=pl.Float64)
            .radians()
            .sin()
            .alias("dir_sin"),
            pl.col("Direction")
            .replace_strict(dir_2_deg, return_dtype=pl.Float64)
            .radians()
            .cos()
            .alias("dir_cos"),
            pl.col("time").dt.ordinal_day().alias("doy"),
            pl.col("time").dt.hour().alias("hour"),
            pl.col("Total"),
        )
        .corr()
        .with_columns(row=pl.Series(corr_cols))
        .unpivot(index="row", variable_name="col", value_name="corr")
    )

    heatmap = (
        alt.Chart(corr_df)
        .mark_rect()
        .encode(
            x=alt.X("col:N", sort=corr_cols, title=None),
            y=alt.Y("row:N", sort=corr_cols, title=None),
            color=alt.Color(
                "corr:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                title="correlation",
            ),
        )
    )
    labels = heatmap.mark_text(baseline="middle").encode(
        text=alt.Text("corr:Q", format=".2f"),
        color=alt.condition(
            "abs(datum.corr) > 0.5", alt.value("white"), alt.value("black")
        ),
    )
    (heatmap + labels).properties(
        title="Speed / Direction / Total correlation", width=300, height=300
    )
    return


@app.cell
def _(pl, wind_df_raw):
    (
        wind_df_raw.with_columns(next_timestamp=pl.col("time").shift(-1))
        .with_columns(time_diff=pl.col("next_timestamp") - pl.col("time"))
        .select(pl.col("time_diff").value_counts())
        .unnest("time_diff")
        .sort("time_diff")
        .filter(pl.col("time_diff").is_not_null())
    )
    return


@app.cell
def _(TimeSeriesSplit, pl, power_df, wind_df):
    joined_df = wind_df.join(power_df, how="inner", on=["time"])

    def tss():
        groups = joined_df["real_obs_weather"].cum_sum()
        for train_idx, test_idx in TimeSeriesSplit().split(joined_df):
            last_train_group = groups[train_idx].max()
            keep = (
                groups[train_idx] != last_train_group
            )  # avoids interpolation leak, drop last group
            yield train_idx[keep], test_idx


    folds = list(tss())
    fold_2_train, fold_2_test = folds[2]
    joined_df[fold_2_test].select([
        pl.col("time").min().alias("min_date"),
        pl.col("time").max().alias("max_date")
    ])

    return (joined_df,)


@app.cell
def _(joined_df):
    joined_df
    return


@app.cell
def _(wind_df):
    wind_df
    return


@app.cell
def _(alt, wind_df):
    _chart = (
        alt.Chart(wind_df) # <-- replace with data
        .transform_filter(f"datum.time != null")
        .transform_timeunit(as_="_time", field="time", timeUnit="yearmonthdate")
        .mark_area()
        .encode(
            x=alt.X("_time:T", title="time"),
            y=alt.Y("count():Q", title="Number of records"),
            tooltip=[
                alt.Tooltip("_time:T", title="time", timeUnit="yearmonthdate"),
                alt.Tooltip("count():Q", title="Number of records", format=",.0f")
            ]
        ).properties(width="container").configure_view(stroke=None)
    )
    _chart
    return


@app.cell
def _(alt, power_df):
    _chart = (
        alt.Chart(power_df) # <-- replace with data
        .transform_filter(f"datum.time != null")
        .transform_timeunit(as_="_time", field="time", timeUnit="yearmonthdate")
        .mark_area()
        .encode(
            x=alt.X("_time:T", title="time"),
            y=alt.Y("count():Q", title="Number of records"),
            tooltip=[
                alt.Tooltip("_time:T", title="time", timeUnit="yearmonthdate"),
                alt.Tooltip("count():Q", title="Number of records", format=",.0f")
            ]
        ).properties(width="container").configure_view(stroke=None)
    )
    _chart
    return


@app.cell
def _(power_df):
    power_df
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


if __name__ == "__main__":
    app.run()
