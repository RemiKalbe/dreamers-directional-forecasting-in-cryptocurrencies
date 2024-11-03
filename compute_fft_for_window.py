import numpy as np


def compute_fft_for_window(signal: np.ndarray, top_n_frequencies: int) -> dict:
    """
    Compute FFT for a given signal and return the top N dominant frequencies.

    Parameters:
        signal (np.ndarray): Array of data for the FFT calculation.
        top_n_frequencies (int): Number of dominant frequencies to return.

    Returns:
        dict: Contains dominant frequencies, magnitudes, and periods.
    """
    # Apply FFT to the signal
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal))
    magnitude = np.abs(fft_result)

    # Find the top N dominant frequencies by magnitude (ignoring the zero-frequency component)
    dominant_indices = np.argsort(magnitude[1:])[-top_n_frequencies:] + 1
    dominant_frequencies = frequencies[dominant_indices]
    dominant_magnitudes = magnitude[dominant_indices]
    dominant_periods = (1 / np.abs(dominant_frequencies)).astype(int)

    # Return results as a dictionary
    return {
        "frequencies": dominant_frequencies,
        "magnitudes": dominant_magnitudes,
        "periods": dominant_periods,
    }
