from typing import Tuple
import polars as pl
import logging
from datetime import timedelta
from multiprocessing import Pool, cpu_count

from .feature import FeatureStep, Columns, Intervals
from .worker import FeatureWorker


# Helper function to call the worker's process_chunk method
def process_worker_chunk(worker: FeatureWorker, chunk: pl.DataFrame):
    """Helper function to process a chunk using a worker."""
    return worker.process_chunk(chunk)


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

    def _get_lookback_requirements(self) -> Tuple[str, int]:
        max_interval = Intervals.get_ordered()[0]
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

    def _compute_required_history(
        self, max_interval: str, max_window: int
    ) -> timedelta:
        interval_minutes = self._convert_interval_to_int_minutes(max_interval)
        return timedelta(minutes=interval_minutes * max_window)

    def execute(
        self,
        df: pl.DataFrame,
        intervals: list[str] | None = None,
        enable_logging: bool = False,
    ) -> pl.DataFrame:
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger.info("Starting pipeline execution")

        if intervals is None:
            intervals = Intervals.get_ordered()

        max_interval, max_window = self._get_lookback_requirements()
        required_history = self._compute_required_history(max_interval, max_window)

        num_workers = cpu_count()
        chunk_size = len(df) // num_workers
        chunks = [
            df[i * chunk_size : (i + 1) * chunk_size].clone()
            for i in range(num_workers)
        ]

        # Initialize the worker with necessary data
        workers = [
            FeatureWorker(i, self.steps, intervals, required_history, enable_logging)
            for i in range(num_workers)
        ]

        with Pool(num_workers) as pool:
            results = pool.starmap(process_worker_chunk, zip(workers, chunks))

        flat_results = [row for result in results for row in result]
        result_df = pl.DataFrame(flat_results)

        if enable_logging:
            self.logger.info(f"Pipeline complete. Output shape: {result_df.shape}")

        return result_df
