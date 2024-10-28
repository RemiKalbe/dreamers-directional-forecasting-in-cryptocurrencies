from typing import Iterator
from src.pipeline.feature import FeatureStep, Columns
import polars as pl


class RSIStep(FeatureStep):
    """Computes Relative Strength Index"""

    def __init__(self, window: int):
        self.window = window

    def compute_expressions(self, interval) -> Iterator[list[pl.Expr]]:
        prefix = f"{interval}_" if interval else ""

        # Calculate price changes
        delta = pl.col(Columns.CLOSE) - pl.col(Columns.CLOSE).shift(1)

        # Separate gains and losses
        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)

        # Calculate average gain and loss
        avg_gain = gain.rolling_mean(window_size=self.window)
        avg_loss = loss.rolling_mean(window_size=self.window)

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        yield [rsi.alias(f"{prefix}{Columns.rsi(self.window)}")]

    @property
    def required_columns(self) -> list[str]:
        return [Columns.CLOSE]

    @property
    def generated_columns(self) -> list[str]:
        return [Columns.rsi(self.window)]

    @property
    def required_windows(self) -> list[int]:
        return [self.window]
