from typing import Iterator
from src.pipeline.feature import FeatureStep, Columns
import polars as pl


class StochasticOscillatorStep(FeatureStep):
    """Computes Stochastic Oscillator with %K and %D lines"""

    def __init__(self, k_window: int, d_window: int):
        self.k_window = k_window
        self.d_window = d_window

    def compute_expressions(self, interval) -> Iterator[list[pl.Expr]]:
        prefix = f"{interval}_" if interval else ""

        # Calculate %K
        lowest_low = pl.col(Columns.LOW).rolling_min(window_size=self.k_window)
        highest_high = pl.col(Columns.HIGH).rolling_max(window_size=self.k_window)

        k = (
            (pl.col(Columns.CLOSE) - lowest_low) / (highest_high - lowest_low) * 100
        ).alias(f"{prefix}{Columns.stoch_k(self.k_window)}")

        # Calculate %D (SMA of %K)
        d = k.rolling_mean(window_size=self.d_window).alias(
            f"{prefix}{Columns.stoch_d(self.k_window, self.d_window)}"
        )

        yield [k, d]

    @property
    def required_columns(self) -> list[str]:
        return [Columns.CLOSE, Columns.HIGH, Columns.LOW]

    @property
    def generated_columns(self) -> list[str]:
        return [
            Columns.stoch_k(self.k_window),
            Columns.stoch_d(self.k_window, self.d_window),
        ]

    @property
    def required_windows(self) -> list[int]:
        return [self.k_window, self.d_window]
