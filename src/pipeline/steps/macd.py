from typing import Iterator
from src.pipeline.feature import FeatureStep, Columns
import polars as pl


class MACDStep(FeatureStep):
    """Computes MACD, Signal Line, and Histogram"""

    def __init__(self, fast_window: int, slow_window: int, signal_window: int):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_window = signal_window

    def compute_expressions(self, interval) -> Iterator[list[pl.Expr]]:
        prefix = f"{interval}_" if interval else ""

        # Calculate MACD line
        fast_ema = pl.col(Columns.CLOSE).ewm_mean(span=self.fast_window)
        slow_ema = pl.col(Columns.CLOSE).ewm_mean(span=self.slow_window)
        macd_line = (fast_ema - slow_ema).alias(
            f"{prefix}{Columns.macd(self.fast_window, self.slow_window)}"
        )

        # Calculate Signal line
        signal_line = macd_line.ewm_mean(span=self.signal_window).alias(
            f"{prefix}{Columns.macd_signal(self.fast_window, self.slow_window, self.signal_window)}"
        )

        # Calculate Histogram
        histogram = (macd_line - signal_line).alias(
            f"{prefix}{Columns.macd_histogram(
                self.fast_window, self.slow_window, self.signal_window
            )}"
        )

        yield [macd_line, signal_line, histogram]

    @property
    def required_columns(self) -> list[str]:
        return [Columns.CLOSE]

    @property
    def generated_columns(self) -> list[str]:
        return [
            Columns.macd(self.fast_window, self.slow_window),
            Columns.macd_signal(self.fast_window, self.slow_window, self.signal_window),
            Columns.macd_histogram(
                self.fast_window, self.slow_window, self.signal_window
            ),
        ]

    @property
    def required_windows(self) -> list[int]:
        return [
            self.fast_window,
            self.slow_window,
            self.signal_window,
        ]
