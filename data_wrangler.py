from typing import Literal
import polars as pl

from transforms.direction import DIR_TO_IDX, IDX_TO_DIR

Granularity = Literal["1h", "1m"]


class DataWrangler:
    def __init__(self, granularity: Granularity = "1h"):
        self.granularity: Granularity = granularity

    def _power_grid(self, power_df: pl.DataFrame) -> pl.DataFrame:
        return (
            power_df.sort("time")
            .group_by_dynamic("time", every=self.granularity)
            .agg(pl.col("Total").mean())
        )

    def _interpolate_direction(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(
                _grp=pl.col("is_real").cum_sum(),
                _idx=pl.col("Direction").replace_strict(
                    DIR_TO_IDX, return_dtype=pl.Int64
                ),
            )
            .with_columns(
                _prev=pl.col("_idx").forward_fill(),
                _next=pl.col("_idx").backward_fill(),
                _t=pl.int_range(pl.len()).over("_grp") / pl.len().over("_grp"),
            )
            .with_columns(_d=((pl.col("_next") - pl.col("_prev") + 8) % 16) - 8)
            .with_columns(
                Direction=(
                    pl.col("_prev")
                    + pl.col("_d").sign() * (pl.col("_d").abs() * pl.col("_t")).floor()
                )
                .cast(pl.Int64)
                .mod(16)
                .replace_strict(IDX_TO_DIR, return_dtype=pl.Utf8)
            )
            .drop("_idx", "_grp", "_prev", "_next", "_t", "_d")
        )

    def _wind_grid(self, wind_df: pl.DataFrame) -> pl.DataFrame:
        return (
            wind_df.sort("time")
            .select(["time", "Direction", "Speed"])
            .with_columns(
                is_real=True,
            )
            .upsample(time_column="time", every=self.granularity)
            .with_columns(
                pl.col("is_real").fill_null(False),
                pl.col("Speed").interpolate(),
            )
            .pipe(self._interpolate_direction)
            .filter(pl.col("Speed").is_not_null())
        )

    def resample(self, wind_df: pl.DataFrame, power_df: pl.DataFrame) -> pl.DataFrame:
        return self._power_grid(power_df).join(
            self._wind_grid(wind_df), how="inner", on="time"
        )
