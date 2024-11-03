import numpy as np
import pywt


def compute_wavelet_features_for_window(
    signal: np.ndarray, wavelet: str = "db4", level: int | None = None
) -> dict:
    """
    Compute wavelet features for a given signal window.

    Parameters:
        signal (np.ndarray): Array of data for the wavelet calculation.
        wavelet (str): Name of the wavelet to use (e.g., 'db4', 'haar', 'sym5').
        level (int): Decomposition level. If None, it will be calculated based on the signal length.

    Returns:
        dict: Contains statistical features extracted from wavelet coefficients.
    """
    # Ensure the signal is a 1D numpy array
    signal = np.asarray(signal).flatten()

    # Determine the maximum level of decomposition if not specified
    if level is None:
        level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)  # type: ignore

    # Perform Discrete Wavelet Transform
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)

    # Initialize a dictionary to store features
    features = {}

    # Extract statistical features from approximation and detail coefficients
    for i, coeff in enumerate(coeffs):
        # Name the coefficients: 'A' for approximation, 'D1', 'D2', ..., for details
        coeff_name = "A" if i == 0 else f"D{i}"
        features[f"{coeff_name}_mean"] = np.mean(coeff)
        features[f"{coeff_name}_std"] = np.std(coeff)
        features[f"{coeff_name}_max"] = np.max(coeff)
        features[f"{coeff_name}_min"] = np.min(coeff)
        features[f"{coeff_name}_energy"] = np.sum(coeff**2)

    return features
