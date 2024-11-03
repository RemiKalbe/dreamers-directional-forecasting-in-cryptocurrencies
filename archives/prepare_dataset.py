import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from typing import List
from pyspark.sql import SparkSession, DataFrame, Column, Row
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from dataclasses import dataclass

# ---- Configuration Settings ----


@dataclass(frozen=True)
class TechnicalWindows:
    """Defines the window sizes for each technical indicator."""

    MA: tuple[int, int] = (12, 26)  # Moving Averages
    VOLATILITY: int = 20
    ROC: tuple[int, int] = (12, 25)  # Rate of Change
    RSI: int = 14
    STOCHASTIC: tuple[int, int] = (14, 3)  # (%K, %D)
    VMA: int = 20
    MACD: tuple[int, int, int] = (12, 26, 9)
    WILLIAMS: int = 14
    ADX: int = 14

    def get_max_window(self) -> int:
        """Compute the maximum window size across all technical indicators."""
        all_windows: list[int] = [
            *self.MA,
            self.VOLATILITY,
            *self.ROC,
            self.RSI,
            *self.STOCHASTIC,
            self.VMA,
            *self.MACD,
            self.WILLIAMS,
            self.ADX,
        ]
        return max(all_windows)


@dataclass(frozen=True)
class Interval:
    """A single time interval for resampling."""

    as_str: str
    as_minutes: int


@dataclass(frozen=True)
class Intervals:
    """Time intervals for resampling."""

    MINUTE_1: Interval = Interval("1 minute", 1)
    MINUTE_5: Interval = Interval("5 minutes", 5)
    MINUTE_30: Interval = Interval("30 minutes", 30)
    HOUR_1: Interval = Interval("1 hour", 60)
    DAY_1: Interval = Interval("1 day", 60 * 24)
    WEEK_1: Interval = Interval("1 week", 60 * 24 * 7)

    @classmethod
    def get_all(cls):
        return [
            cls.MINUTE_1,
            cls.MINUTE_5,
            cls.MINUTE_30,
            cls.HOUR_1,
            cls.DAY_1,
            cls.WEEK_1,
        ]


# ---- Argument Parsing for EMR ----


def parse_args():
    parser = argparse.ArgumentParser(
        description="PySpark Job for Dynamic Technical Indicators"
    )
    parser.add_argument("--input", required=True, help="Input S3 path (CSV)")
    parser.add_argument("--output", required=True, help="Output S3 path (Parquet)")
    return parser.parse_args()


# ---- Initialize Spark Session ----


def init_spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()  # type: ignore


# ---- Feature Computation ----


def compute_ema(col_name: str, window_size: int) -> Column:
    """Compute Exponential Moving Average (EMA) using PySpark."""
    alpha = 2 / (window_size + 1)  # Smoothing factor

    # Use a cumulative sum to simulate EMA calculation iteratively
    ema = F.sum(
        (1 - alpha) ** F.row_number().over(Window.orderBy("datetime")) * F.col(col_name)
    ) / F.sum((1 - alpha) ** F.row_number().over(Window.orderBy("datetime")))
    return ema


def compute_technical_indicators(
    df: DataFrame, interval: str, windows: TechnicalWindows
) -> Row:
    """Compute all technical indicators for the given interval."""

    indicators_df = df.select(
        #
        # --- True Range (TR) ---
        #
        F.greatest(
            F.col("high") - F.col("low"),
            F.abs(F.col("high") - F.lag("close")),
            F.abs(F.col("low") - F.lag("close")),
        ).alias(f"tr_{interval}"),
        #
        # --- Moving Averages (SMA, EMA) ---
        #
        F.avg("close").alias(f"sma_{interval}_{windows.MA[0]}"),
        compute_ema("close", windows.MA[1]).alias(f"ema_{interval}_{windows.MA[1]}"),
        #
        # --- Rate of Change (ROC) ---
        #
        (
            (F.col("close") - F.lag("close", windows.ROC[0]))
            / F.lag("close", windows.ROC[0])
            * 100
        ).alias(f"roc_{interval}_{windows.ROC[0]}"),
        #
        # --- Relative Strength Index (RSI) ---
        #
        (
            100
            - (
                100
                / (
                    1
                    + (
                        F.avg(
                            F.when(
                                F.col("close") > F.lag("close"),
                                F.col("close") - F.lag("close"),
                            ).otherwise(0)
                        )
                        / F.avg(
                            F.when(
                                F.col("close") < F.lag("close"),
                                F.lag("close") - F.col("close"),
                            ).otherwise(0)
                        )
                    )
                )
            )
        ).alias(f"rsi_{interval}_{windows.RSI}"),
        #
        # --- Volatility (Rolling Standard Deviation) ---
        #
        F.stddev("close").alias(f"volatility_{interval}_{windows.VOLATILITY}"),
        #
        # --- MACD ---
        #
        (
            compute_ema("close", windows.MACD[0])
            - compute_ema("close", windows.MACD[1])
        ).alias(f"macd_{interval}"),
        #
        # --- Stochastic Oscillator (%K) ---
        #
        ((F.col("close") - F.min("low")) / (F.max("high") - F.min("low")) * 100).alias(
            f"stoch_k_{interval}"
        ),
        #
        # --- Williams %R ---
        #
        (
            (F.max("high") - F.col("close")) / (F.max("high") - F.min("low")) * -100
        ).alias(f"williams_r_{interval}"),
        #
        # --- Volume Moving Average (VMA) ---
        #
        F.avg("volume").alias(f"vma_{interval}_{windows.VMA}"),
        #
        # --- Directional Indicators (+DM, -DM) ---
        #
        F.when(
            F.col("high") > F.lag("high"),
            F.col("high") - F.lag("high"),
        )
        .otherwise(0)
        .alias(f"plus_dm_{interval}"),
        F.when(
            F.col("low") < F.lag("low"),
            F.lag("low") - F.col("low"),
        )
        .otherwise(0)
        .alias(f"minus_dm_{interval}"),
        #
        # --- +DI and -DI (Directional Indicators) ---
        #
        (
            F.when(F.col(f"tr_{interval}") == 0, F.lit(0)).otherwise(
                (F.col(f"plus_dm_{interval}") / F.col(f"tr_{interval}")) * 100
            )
        ).alias(f"plus_di_{interval}"),
        (
            F.when(F.col(f"tr_{interval}") == 0, F.lit(0)).otherwise(
                (F.col(f"minus_dm_{interval}") / F.col(f"tr_{interval}")) * 100
            )
        ).alias(f"minus_di_{interval}"),
        #
        # --- ADX (Average Directional Index) ---
        #
        (
            F.avg(
                F.when(
                    (F.col(f"plus_di_{interval}") + F.col(f"minus_di_{interval}")) != 0,
                    F.abs(F.col(f"plus_di_{interval}") - F.col(f"minus_di_{interval}"))
                    / (F.col(f"plus_di_{interval}") + F.col(f"minus_di_{interval}"))
                    * 100,
                ).otherwise(0)
            )
        ).alias(f"adx_{interval}"),
    )

    # Return only the **last row** of the resampled window as a DataFrame
    return indicators_df.orderBy(F.col("datetime").desc()).limit(1).collect()[0]


# ---- Compute Features for Each Minute ----


def compute_features_for_minutes(
    spark: SparkSession, df: DataFrame, windows: TechnicalWindows
) -> DataFrame:
    """Compute features for each minute dynamically and independently."""

    # Get the maximum window size required
    max_interval = max(Intervals.get_all(), key=lambda x: x.as_minutes)
    max_lookback: int = windows.get_max_window() * max_interval.as_minutes

    logging.info(f"The maximum lookback window is {max_lookback} minutes")

    # Ensure the DataFrame is ordered by datetime
    df = df.orderBy("datetime")

    # Extract unique minutes as a list of timestamps/integers
    unique_minutes: list[datetime] = [
        row["datetime"]
        for row in df.select("datetime").distinct().orderBy("datetime").collect()
    ]

    def create_partition_for_minute(minute: datetime) -> List[Row]:
        """Create a partition containing only the relevant rows for a minute."""

        # Filter all rows up to the current minute
        partition_data = df.filter(F.col("datetime") <= minute)

        # Keep only the last `max_lookback` rows + the current minute
        partition_data = partition_data.tail(max_lookback + 1)

        return partition_data

    def process_minute(df_chunk: DataFrame, windows: TechnicalWindows) -> Row:
        """Process features for a single minute."""
        current_minute: datetime = (
            df_chunk.select("datetime").orderBy("datetime").tail(1)[0]["datetime"]
        )
        logging.info(f"Processing minute: {current_minute}")

        # List to store the features for all intervals for this minute
        interval_features: list[Row] = []

        # Iterate over intervals and compute resampled features
        for interval in Intervals.get_all():
            # Resample data up to the current minute for the interval
            resampled = df_chunk.groupBy(
                F.window("datetime", interval.as_str).alias("window")
            ).agg(
                F.first("open").alias("open"),
                F.max("high").alias("high"),
                F.min("low").alias("low"),
                F.last("close").alias("close"),
                F.sum("volume").alias("volume"),
                F.sum("quote_asset_volume").alias("quote_asset_volume"),
                F.sum("number_of_trades").alias("number_of_trades"),
                F.sum("taker_buy_base_volume").alias("taker_buy_base_volume"),
                F.sum("taker_buy_quote_volume").alias("taker_buy_quote_volume"),
                F.last("target").alias("target"),
            )

            # Compute technical indicators on the resampled data
            feature_row = compute_technical_indicators(
                resampled, interval.as_str, windows
            )
            interval_features.append(feature_row)

        # Concatenate all interval features horizontally into one row
        interval_features_as_dict = dict(
            *[feature.asDict().items() for feature in interval_features],
            datetime=current_minute,
        )
        interval_features_as_row = Row(**interval_features_as_dict)

        logging.info(
            f"Minute {current_minute} processed successfully -> {interval_features_as_row}"
        )

        return interval_features_as_row

    # Group partitions into batches
    def group_partitions(
        partitions: list[DataFrame], batch_size: int
    ) -> list[list[DataFrame]]:
        """Group partitions into batches for efficient processing on each worker."""
        return [
            partitions[i : i + batch_size]
            for i in range(0, len(partitions), batch_size)
        ]

    # Define the partition computation for a batch of partitions
    def compute_partition_group(batch: list[DataFrame]) -> list[Row]:
        """Compute features for a batch of partitions and return as a list of Rows."""
        result_rows: list[Row] = []
        # Use ThreadPoolExecutor to parallelize within the worker
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_minute, chunk, windows): chunk
                for chunk in batch
            }

            # Collect results as each future completes
            for future in as_completed(futures):
                result_rows.append(future.result())  # Each future returns a Row

        return result_rows

    partitions = [
        spark.createDataFrame(create_partition_for_minute(minute), schema=df.schema)
        for minute in unique_minutes
    ]

    # Each worker will process a batch of partitions
    batch_size = 100
    partition_batches = group_partitions(partitions, batch_size)

    logging.info(f"Processing {len(partitions)}, in {len(partition_batches)} batches")

    # Use Spark to distribute the task using `parallelize`
    result_rdd = (
        spark.sparkContext.parallelize(partition_batches)
        .flatMap(compute_partition_group)
        .collect()
    )

    # Convert the result RDD back into a DataFrame
    result_df = spark.createDataFrame(result_rdd)

    # Merge the computed features back onto the original DataFrame
    final_df = df.join(result_df, on="datetime", how="left")

    return final_df


# ---- Main Function ----


def main():
    args = parse_args()
    spark = init_spark()
    logging.basicConfig(level=logging.INFO)
    windows = TechnicalWindows()

    # Load data from S3
    logging.info(f"Reading data from {args.input}")
    df = spark.read.csv(args.input, header=True, inferSchema=True)
    df = df.withColumn("datetime", F.from_unixtime(F.col("timestamp")))

    # Compute features for all minutes
    result_df = compute_features_for_minutes(spark, df, windows)

    # Save the results to S3
    logging.info(f"Saving results to {args.output}")
    result_df.write.mode("overwrite").parquet(args.output)

    logging.info("Feature computation complete!")
    spark.stop()


if __name__ == "__main__":
    main()
