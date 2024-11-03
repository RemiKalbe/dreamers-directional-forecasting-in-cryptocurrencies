from email.policy import strict
from pathlib import Path
from typing import Tuple, Iterator
import polars as pl
import logging
from tqdm.notebook import tqdm

from .feature import FeatureStep, Columns, Intervals

SCHEMA = {
    "timestamp": pl.UInt64,
    "open": pl.Float32,
    "high": pl.Float32,
    "low": pl.Float32,
    "close": pl.Float32,
    "volume": pl.Float32,
    "quote_asset_volume": pl.Float32,
    "number_of_trades": pl.Float32,
    "taker_buy_base_volume": pl.Float32,
    "taker_buy_quote_volume": pl.Float32,
    "target": pl.Float32,
}


class DataFrameChunkIterator:
    def __init__(self, df: pl.DataFrame, chunk_size: int, padding: int):
        self.df = df
        self.chunk_size = chunk_size
        self.padding = padding
        self.num_rows = df.shape[0]
        self.start = 0

    def __iter__(self) -> Iterator[pl.DataFrame]:
        return self

    def __next__(self) -> pl.DataFrame:
        if self.start >= self.num_rows:
            raise StopIteration

        # Compute the end index for the current chunk
        end = min(self.start + self.chunk_size, self.num_rows)

        # Add padding to the start if it's not the first chunk
        start_with_padding = max(self.start - self.padding, 0)

        # Extract the chunk with padding
        chunk = self.df[start_with_padding:end].clone()

        # Update the start index for the next chunk
        self.start = end

        return chunk

    def __len__(self) -> int:
        """Return the total number of chunks needed."""
        # Calculate how many chunks are required to iterate over the DataFrame
        total_chunks = (self.num_rows + self.chunk_size - 1) // self.chunk_size
        return total_chunks


class FeaturePipeline:
    def __init__(self, steps: list[FeatureStep]):
        self.steps = steps
        self._validate_steps()
        self.logger = logging.getLogger(__name__)

    def _validate_steps(self) -> None:
        """Validate step dependencies."""
        available_columns = set(Columns.get_base_columns())
        for step in self.steps:
            missing_columns = [
                col for col in step.required_columns if col not in available_columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Step {step.__class__.__name__} requires columns: {missing_columns}"
                )
            available_columns.update(step.generated_columns)

    def _get_lookback_requirements(
        self, ordered_intervals: list[str]
    ) -> Tuple[str, int]:
        max_interval = ordered_intervals[0]
        max_window = max(
            window
            for step in self.steps
            for window in step.required_windows
            if window > 0
        )
        return max_interval, max_window

    def _convert_interval_to_int_minutes(self, interval: str) -> int:
        interval_minutes = {
            "1w": 7 * 24 * 60,
            "1d": 24 * 60,
            "4h": 4 * 60,
            "1h": 60,
            "30m": 30,
            "15m": 15,
            "5m": 5,
            "1m": 1,
        }
        return interval_minutes[interval]

    def _compute_required_history(self, max_interval: str, max_window: int) -> int:
        interval_minutes = self._convert_interval_to_int_minutes(max_interval)
        return interval_minutes * max_window

    def _base_agg(self) -> list[pl.Expr]:
        return [
            pl.col(Columns.OPEN).first().alias(Columns.OPEN),
            pl.col(Columns.HIGH).max().alias(Columns.HIGH),
            pl.col(Columns.LOW).min().alias(Columns.LOW),
            pl.col(Columns.CLOSE).last().alias(Columns.CLOSE),
            pl.col(Columns.VOLUME).sum().alias(Columns.VOLUME),
            pl.col(Columns.QUOTE_ASSET_VOLUME).sum().alias(Columns.QUOTE_ASSET_VOLUME),
            pl.col(Columns.NUMBER_OF_TRADES).sum().alias(Columns.NUMBER_OF_TRADES),
            pl.col(Columns.TAKER_BUY_BASE_VOLUME)
            .sum()
            .alias(Columns.TAKER_BUY_BASE_VOLUME),
            pl.col(Columns.TAKER_BUY_QUOTE_VOLUME)
            .sum()
            .alias(Columns.TAKER_BUY_QUOTE_VOLUME),
            pl.col(Columns.TARGET).last().alias(Columns.TARGET),
        ]

    def process_and_save_chunk(
        self,
        chunk: pl.DataFrame,
        padding: int,
        output_file: Path,
        is_first_chunk: bool,
        interval: str,
    ) -> None:
        """Process a chunk, trim the padding, and save to disk."""
        processed_chunk = self.compute_features_for_interval(interval, chunk)

        # Trim padding from the beginning (if not the first chunk)
        if not is_first_chunk:
            processed_chunk = processed_chunk[padding:]

        # Save the chunk to disk
        processed_chunk.write_parquet(output_file)

    def compute_features_for_interval(
        self, interval: str, df: pl.DataFrame
    ) -> pl.DataFrame:
        """Compute features for a single interval."""
        rolling_df = df

        # Apply feature expressions
        for step in tqdm(self.steps, desc="Processing step", position=2, leave=None):
            expr_groups = list(step.compute_expressions(interval))
            for expr_group in tqdm(
                expr_groups,
                desc="Processing feature expression group",
                position=3,
                total=len(expr_groups),
                leave=None,
            ):
                for expr in tqdm(
                    expr_group,
                    desc="Processing feature expression",
                    total=len(expr_group),
                    leave=None,
                    position=4,
                ):
                    merged_exprs = step.required_columns_in_agg + [expr]
                    result_df = rolling_df.rolling(
                        period=interval, index_column=Columns.DATETIME, closed="right"
                    ).agg(merged_exprs)
                    # Extract the last element of each generated column
                    result_df = result_df.select(
                        [
                            *[
                                pl.col(c).arr.last().alias(c)
                                for c in set(step.generated_columns)
                                if c in result_df.columns
                            ],
                            Columns.DATETIME,
                        ]
                    )
                    # Merge the result back to the original DataFrame
                    rolling_df = rolling_df.join(
                        result_df.drop(
                            filter(
                                lambda c: c != Columns.DATETIME,
                                set(Columns.get_base_columns() + step.required_columns),
                            ),
                            strict=False,
                        ),
                        on=Columns.DATETIME,
                    )

        return rolling_df

    def execute(
        self,
        input: Path,
        output_dir: Path,
        base_interval: str = Intervals.MINUTE_1,
        ordered_intervals: list[str] = Intervals.get_ordered(),
        enable_logging: bool = False,
    ) -> pl.DataFrame:
        df = pl.read_csv(input, schema=SCHEMA)
        df = df.with_columns(
            [
                pl.from_epoch("timestamp").alias("datetime"),
                pl.col("timestamp").cast(pl.UInt64),
                pl.col("number_of_trades").cast(pl.UInt32),
                pl.col("target").cast(pl.UInt8).cast(pl.Boolean),
            ]
        )

        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger.info("Starting pipeline execution")

        # Set the chunk size to the maximal window required
        max_interval, max_window = self._get_lookback_requirements(ordered_intervals)
        padding = self._compute_required_history(max_interval, max_window)
        chunk_size = padding

        if enable_logging:
            self.logger.info(
                f"Padding for interval {max_interval} window {max_window} is {padding}"
            )

        # Create the chunk iterator
        chunks = DataFrameChunkIterator(df, chunk_size, padding)

        # Compute features for each interval
        for interval in tqdm(ordered_intervals, desc="Processing interval", position=0):
            for i, chunk in tqdm(
                enumerate(chunks),
                desc="Processing chunk",
                position=1,
                total=len(chunks),
                leave=None,
            ):
                self.process_and_save_chunk(
                    chunk,
                    padding,
                    output_dir / f"temp-{interval}-{i}.parquet",
                    i == 0,
                    interval,
                )

        dfs = []
        for interval in ordered_intervals:
            chunked_dfs = []
            for i in range(len(chunks)):
                chunk_df = pl.read_parquet(output_dir / f"temp-{interval}-{i}.parquet")
                chunked_dfs.append(chunk_df)
            # Concatenate all chunks
            df = pl.concat(chunked_dfs, how="vertical")
            # Sort by datetime
            df = df.sort(Columns.DATETIME)
            if interval != base_interval:
                df = df.drop(Columns.get_base_columns(), strict=False)
            dfs.append(df)

        # Concatenate all dataframes
        result_df: pl.DataFrame = pl.concat(dfs, how="horizontal")

        return result_df
