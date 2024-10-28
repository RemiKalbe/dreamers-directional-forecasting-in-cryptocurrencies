from typing import Iterator
from src.pipeline.feature import FeatureStep, Columns
import polars as pl


class DirectionalMovementSystemStep(FeatureStep):
    """Computes Directional Movement System (ADX, +DI, -DI)"""

    def __init__(self, window: int):
        self.window = window

    def compute_expressions(self, interval) -> Iterator[list[pl.Expr]]:
        prefix = f"{interval}_" if interval else ""

        # Calculate True Range
        high_low = pl.col(Columns.HIGH) - pl.col(Columns.LOW)
        high_close = (pl.col(Columns.HIGH) - pl.col(Columns.CLOSE).shift(1)).abs()
        low_close = (pl.col(Columns.LOW) - pl.col(Columns.CLOSE).shift(1)).abs()

        tr = pl.max_horizontal([high_low, high_close, low_close]).alias(f"{prefix}TR")

        # Calculate +DM and -DM
        up_move = pl.col(Columns.HIGH) - pl.col(Columns.HIGH).shift(1)
        down_move = pl.col(Columns.LOW).shift(1) - pl.col(Columns.LOW)

        plus_dm = (
            pl.when((up_move > down_move) & (up_move > 0))
            .then(up_move)
            .otherwise(0)
            .alias(f"{prefix}+DM")
        )

        minus_dm = (
            pl.when((down_move > up_move) & (down_move > 0))
            .then(down_move)
            .otherwise(0)
            .alias(f"{prefix}-DM")
        )

        # Calculate smoothed values
        tr_window = tr.rolling_mean(window_size=self.window)
        plus_dm_window = plus_dm.rolling_mean(window_size=self.window)
        minus_dm_window = minus_dm.rolling_mean(window_size=self.window)

        # Calculate +DI and -DI
        plus_di = (plus_dm_window / tr_window * 100).alias(
            f"{prefix}{Columns.plus_di(self.window)}"
        )
        minus_di = (minus_dm_window / tr_window * 100).alias(
            f"{prefix}{Columns.minus_di(self.window)}"
        )

        # Calculate ADX
        dx = (
            ((plus_di - minus_di).abs() / (plus_di + minus_di) * 100)
            .rolling_mean(window_size=self.window)
            .alias(f"{prefix}{Columns.adx(self.window)}")
        )

        yield [plus_di, minus_di, dx]

    @property
    def required_columns(self) -> list[str]:
        return [Columns.CLOSE, Columns.HIGH, Columns.LOW]

    @property
    def generated_columns(self) -> list[str]:
        return [
            Columns.plus_di(self.window),
            Columns.minus_di(self.window),
            Columns.adx(self.window),
        ]

    @property
    def required_windows(self) -> list[int]:
        return [self.window]
