from datetime import timedelta
import logging
import polars as pl
from tqdm import tqdm

from .feature import FeatureStep, Columns
from .utils import group_by_dynamic_right_aligned


class FeatureWorker:
    """Worker class responsible for processing chunks."""

    def __init__(
        self,
        id: int,
        steps: list[FeatureStep],
        intervals: list[str],
        required_history: timedelta,
        enable_logging: bool,
    ):
        self.id = id
        self.steps = steps
        self.intervals = intervals
        self.required_history = required_history
        self.enable_logging = enable_logging
        self.logger = logging.getLogger(__name__)

    def process_chunk(self, chunk: pl.DataFrame) -> list[dict[str, float]]:
        """Processes a single chunk of data."""
        result_rows = []
        for i in tqdm(range(len(chunk)), desc=f"Worker {self.id}"):
            current_time = chunk[i, "datetime"]
            history_start = current_time - self.required_history

            df_slice = chunk.filter(
                (pl.col("datetime") >= history_start)
                & (pl.col("datetime") <= current_time)
            )

            if len(df_slice) < 2:
                continue

            row_features = self.compute_features_for_intervals(i, df_slice)
            result_rows.append(row_features)

        return result_rows

    def compute_features_for_intervals(
        self, i: int, df_slice: pl.DataFrame
    ) -> dict[str, float]:
        """Compute features for each interval."""
        row_features = {}
        for interval in self.intervals:
            if self.enable_logging and i == 0:
                self.logger.info(f"Computing features for interval: {interval}")

            df_slice_interval = group_by_dynamic_right_aligned(
                df=df_slice, index_column="datetime", every=interval
            ).agg(
                [
                    pl.col(Columns.OPEN).first().alias(Columns.OPEN),
                    pl.col(Columns.HIGH).max().alias(Columns.HIGH),
                    pl.col(Columns.LOW).min().alias(Columns.LOW),
                    pl.col(Columns.CLOSE).last().alias(Columns.CLOSE),
                    pl.col(Columns.VOLUME).sum().alias(Columns.VOLUME),
                    pl.col(Columns.QUOTE_ASSET_VOLUME)
                    .sum()
                    .alias(Columns.QUOTE_ASSET_VOLUME),
                    pl.col(Columns.NUMBER_OF_TRADES)
                    .sum()
                    .alias(Columns.NUMBER_OF_TRADES),
                    pl.col(Columns.TAKER_BUY_BASE_VOLUME)
                    .sum()
                    .alias(Columns.TAKER_BUY_BASE_VOLUME),
                    pl.col(Columns.TAKER_BUY_QUOTE_VOLUME)
                    .sum()
                    .alias(Columns.TAKER_BUY_QUOTE_VOLUME),
                    pl.col(Columns.TARGET).last().alias(Columns.TARGET),
                ]
            )

            # Apply feature expressions
            features = df_slice_interval
            for expr_group in self._compute_feature_expressions(interval):
                features = features.with_columns(expr_group)

            latest_features = features.tail(1).row(index=0, named=True)
            row_features.update(latest_features)

        return row_features

    def _compute_feature_expressions(self, interval: str | None) -> list[list[pl.Expr]]:
        """Compute feature expressions for each step."""
        expressions = []
        for step in self.steps:
            step_expressions = step.compute_expressions(interval)
            expressions.extend(step_expressions)
        return expressions
