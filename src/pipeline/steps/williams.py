from typing import Iterator
from src.pipeline.feature import FeatureStep, Columns
import polars as pl


class WilliamsRStep(FeatureStep):
    """Computes Williams %R"""

    def __init__(self, window: int):
        self.window = window

    def compute_expressions(self, interval) -> Iterator[list[pl.Expr]]:
        prefix = f"{interval}_" if interval else ""

        highest_high = pl.col(Columns.HIGH).rolling_max(window_size=self.window)
        lowest_low = pl.col(Columns.LOW).rolling_min(window_size=self.window)

        williams_r = (
            (highest_high - pl.col(Columns.CLOSE)) / (highest_high - lowest_low) * -100
        ).alias(f"{prefix}{Columns.williams_r(self.window)}")

        yield [williams_r]

    @property
    def required_columns(self) -> list[str]:
        return [Columns.CLOSE, Columns.HIGH, Columns.LOW]

    @property
    def generated_columns(self) -> list[str]:
        return [Columns.williams_r(self.window)]

    @property
    def required_windows(self) -> list[int]:
        return [self.window]
