from typing import Iterator
from src.pipeline.feature import FeatureStep, Columns
import polars as pl


class VolumeMovingAverageStep(FeatureStep):
    """Computes Volume Moving Average"""

    def __init__(self, windows: list[int]):
        self.windows = windows

    def compute_expressions(self, interval) -> Iterator[list[pl.Expr]]:
        prefix = f"{interval}_" if interval else ""

        yield [
            pl.col(Columns.VOLUME)
            .rolling_mean(window_size=window)
            .alias(f"{prefix}{Columns.vma(window)}")
            for window in self.windows
        ]

        # Add relative volume (current volume / average volume)
        yield [
            (pl.col(Columns.VOLUME) / pl.col(f"{prefix}{Columns.vma(window)}")).alias(
                f"{prefix}rel_{Columns.vma(window)}"
            )
            for window in self.windows
        ]

    @property
    def required_columns(self) -> list[str]:
        return [Columns.VOLUME]

    @property
    def generated_columns(self) -> list[str]:
        return [
            *[Columns.vma(w) for w in self.windows],
            *[f"rel_{Columns.vma(w)}" for w in self.windows],
        ]

    @property
    def required_windows(self) -> list[int]:
        return self.windows
