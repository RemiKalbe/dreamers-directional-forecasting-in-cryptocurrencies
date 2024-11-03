import numpy as np
from statsmodels.tsa.stattools import acf, pacf


def compute_acf_pacf_for_window(signal: np.ndarray, nlags: int = 10) -> dict:
    """
    Compute ACF and PACF for a given signal window.

    Parameters:
        signal (np.ndarray): Array of data for which to compute ACF and PACF.
        nlags (int): Number of lags to compute for ACF and PACF.

    Returns:
        dict: Contains ACF and PACF values up to specified lags.
    """
    # Ensure the signal is a 1D numpy array
    signal = np.asarray(signal).flatten()

    # Compute ACF and PACF
    acf_values = acf(signal, nlags=nlags, fft=False)
    pacf_values = pacf(signal, nlags=nlags)

    # Exclude lag 0 (autocorrelation with itself) if desired
    acf_values = acf_values[1:]  # Remove lag 0
    pacf_values = pacf_values[1:]  # Remove lag 0

    # Create a dictionary to store the results
    features = {}
    for lag in range(1, nlags + 1):
        features[f"ACF_lag_{lag}"] = acf_values[lag - 1]
        features[f"PACF_lag_{lag}"] = pacf_values[lag - 1]

    return features
