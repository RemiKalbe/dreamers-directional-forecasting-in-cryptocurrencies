from typing import Iterator
from src.pipeline.feature import FeatureStep, Columns
import polars as pl


class RateOfChangeStep(FeatureStep):
    """Computes price rate of change for specified windows"""

    def __init__(self, windows: list[int]):
        self.windows = windows

    def compute_expressions(self, interval) -> Iterator[list[pl.Expr]]:
        prefix = f"{interval}_" if interval else ""
        yield [
            (
                (pl.col(Columns.CLOSE) - pl.col(Columns.CLOSE).shift(window))
                / pl.col(Columns.CLOSE).shift(window)
                * 100
            ).alias(f"{prefix}{Columns.roc(window)}")
            for window in self.windows
        ]

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
        return [Columns.roc(w) for w in self.windows]

    @property
    def required_windows(self) -> list[int]:
        return self.windows
