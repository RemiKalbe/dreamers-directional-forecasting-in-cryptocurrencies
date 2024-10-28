from datetime import timedelta
from typing import Iterable
from polars._typing import IntoExpr
import polars as pl


def group_by_dynamic_right_aligned(
    df: pl.DataFrame,
    index_column: IntoExpr,
    *,
    every: str | timedelta,
    period: str | timedelta | None = None,
    include_boundaries: bool = False,
    by: IntoExpr | Iterable[IntoExpr] | None = None,
    include_windows_ending_after_last_index: bool = False,
):
    """
    Wrapper for polars group_by_dynamic that aligns the windows such that the last window ends on the last date/datetime in the data.
    Consequently, the first window may have a shorter date range than the others.
    Windows are labelled by their right (end) date (inclusive).
    Set include_windows_ending_after_last_index=True to include windows that extend beyond the last date, therefore only contain subsets of the last full window.
    Refer to group_by_dynamic for other parameters.

    Example:

     .. code-block:: python
        df = pl.DataFrame(pl.date_range(date(2023,1,1), date(2023,1,12), interval='1d', eager=True))
        print(df)

        group_by_dynamic_right_aligned(df, 'date', every='3d', period='5d').agg(
            start=pl.col.date.min(),
            end=pl.col.date.max(),
            n=pl.col.date.count())
        )

    returns:

    ```
    ┌────────────┬────────────┬────────────┬─────┐
    │ date       ┆ start      ┆ end        ┆ n   │
    │ ---        ┆ ---        ┆ ---        ┆ --- │
    │ date       ┆ date       ┆ date       ┆ u32 │
    ╞════════════╪════════════╪════════════╪═════╡
    │ 2023-01-03 ┆ 2023-01-01 ┆ 2023-01-03 ┆ 3   │
    │ 2023-01-06 ┆ 2023-01-02 ┆ 2023-01-06 ┆ 5   │
    │ 2023-01-09 ┆ 2023-01-05 ┆ 2023-01-09 ┆ 5   │
    │ 2023-01-12 ┆ 2023-01-08 ┆ 2023-01-12 ┆ 5   │
    └────────────┴────────────┴────────────┴─────┘
    ```
    """
    # First pass at the groups, with no offset
    labels_series = (
        df.group_by_dynamic(
            index_column,
            every=every,
            period=period,
            include_boundaries=include_boundaries,
            by=by,  # type: ignore
            closed="right",
            label="right",
            start_by="window",
        )
        .agg()
        .select(index_column)
        .to_series(0)
    )

    max_date = df.select(index_column).to_series(0).max()
    end_of_first_window_extending_beyond_data = labels_series.filter(
        labels_series >= max_date
    )[0]
    # The negative offset to shift windows by such that the last window ends exactly on max_date
    offset = max_date - end_of_first_window_extending_beyond_data

    # Redo the group_by_dynamic with the offset, so we get a window ending exactly on max_date
    groups = df.group_by_dynamic(
        index_column,
        every=every,
        period=period,
        by=by,  # type: ignore
        closed="right",
        label="right",
        start_by="window",
        offset=offset,
    )

    if include_windows_ending_after_last_index:
        return groups

    # Monkey patch the agg function to filter out the groups that extend beyond the last date, which contain only subsets of the last full window
    def wrapped_agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> pl.DataFrame:
        return groups.__class__.agg(self, *aggs, **named_aggs).filter(
            str_to_col(index_column) <= max_date  # type: ignore
        )

    groups.agg = wrapped_agg.__get__(groups, groups.__class__)
    return groups


def str_to_col(column):
    return pl.col(column) if isinstance(column, str) else column
