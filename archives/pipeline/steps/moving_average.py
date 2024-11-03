from typing import Iterator
from src.pipeline.feature import FeatureStep, Columns
import polars as pl


class MovingAverageStep(FeatureStep):
    """Computes SMA and EMA for specified windows"""

    def __init__(self, windows: list[int]):
        self.windows = windows

    def compute_expressions(self, interval) -> Iterator[list[pl.Expr]]:
        prefix = f"{interval}_" if interval else ""
        expressions = []

        # Simple Moving Averages
        expressions.extend(
            [
                pl.col(Columns.CLOSE)
                .rolling_mean(window_size=window)
                .alias(f"{prefix}{Columns.sma(window)}")
                for window in self.windows
            ]
        )

        # Exponential Moving Averages
        expressions.extend(
            [
                pl.col(Columns.CLOSE)
                .ewm_mean(span=window)
                .alias(f"{prefix}{Columns.ema(window)}")
                for window in self.windows
            ]
        )

        yield expressions

    @property
    def required_columns(self) -> list[str]:
        return [Columns.CLOSE]

    @property
    def required_columns_in_agg(self) -> list[pl.Expr]:
        return [
            pl.col(Columns.CLOSE).last().alias(Columns.CLOSE),
        ]

    @property
    def generated_columns(self) -> list[str]:
        return [
            *[Columns.sma(w) for w in self.windows],
            *[Columns.ema(w) for w in self.windows],
        ]
