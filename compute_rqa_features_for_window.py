import numpy as np
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation


def compute_rqa_features_for_window(
    signal: np.ndarray,
    embedding_dimension: int = 2,
    time_delay: int = 1,
    radius: np.floating | None = None,
) -> dict:
    """
    Compute RQA statistics for a given signal window using PyRQA.

    Parameters:
        signal (np.ndarray): Array of data for which to compute RQA statistics.
        embedding_dimension (int): Embedding dimension for phase space reconstruction.
        time_delay (int): Time delay for phase space reconstruction.
        radius (float): Threshold for defining recurrence points.

    Returns:
        dict: Contains RQA statistics.
    """
    # Ensure the signal is a 1D numpy array
    signal = signal.flatten()

    # Create a TimeSeries object
    time_series = TimeSeries(
        signal, embedding_dimension=embedding_dimension, time_delay=time_delay
    )

    # Define the neighborhood (radius)
    if radius is None:
        # Set radius as 10% of the standard deviation of the signal
        radius = 0.1 * np.std(signal)

    neighbourhood = FixedRadius(radius.item())

    # Create settings for RQA computation
    settings = Settings(
        time_series,
        neighbourhood=neighbourhood,
        similarity_measure=EuclideanMetric,
        theiler_corrector=1,
    )

    # Create and run the computation
    computation = RQAComputation.create(settings, verbose=False)
    result = computation.run()

    # Extract RQA measures
    features = {
        "RecurrenceRate": result.recurrence_rate,
        "Determinism": result.determinism,
        "AverageDiagonalLineLength": result.average_diagonal_line,
        "LongestDiagonalLineLength": result.longest_diagonal_line,
        "Divergence": result.divergence,
        "EntropyDiagonalLines": result.entropy_diagonal_lines,
        "Laminarity": result.laminarity,
        "TrappingTime": result.trapping_time,
        "LongestVerticalLineLength": result.longest_vertical_line,
        "EntropyVerticalLines": result.entropy_vertical_lines,
        "AverageWhiteVerticalLineLength": result.average_white_vertical_line,
        "LongestWhiteVerticalLineLength": result.longest_white_vertical_line,
        "EntropyWhiteVerticalLines": result.entropy_white_vertical_lines,
    }

    return features
