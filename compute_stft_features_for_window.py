import numpy as np
from scipy.signal import stft


def compute_stft_features_for_window(
    signal: np.ndarray,
    fs: float = 1.0,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
) -> dict:
    """
    Compute STFT features for a given signal window.

    Parameters:
        signal (np.ndarray): Array of data for which to compute STFT features.
        fs (float): Sampling frequency of the signal.
        window (str): Desired window to use. See scipy.signal.get_window for options.
        nperseg (int): Length of each segment for STFT.
        noverlap (int): Number of points to overlap between segments. If None, defaults to nperseg // 8.

    Returns:
        dict: Contains STFT features like spectral centroid and spectral entropy.
    """
    # Ensure the signal is a 1D numpy array
    signal = np.asarray(signal).flatten()

    # Compute STFT
    f, t, Zxx = stft(
        signal,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=None,
        detrend=False,
    )

    # Magnitude of STFT
    magnitude = np.abs(Zxx)

    # Normalize the magnitude spectrum for each time frame
    magnitude_sum = np.sum(magnitude, axis=0, keepdims=True)
    magnitude_norm = magnitude / (magnitude_sum + 1e-12)  # Avoid division by zero

    # Spectral Centroid
    spectral_centroid = np.sum(f[:, np.newaxis] * magnitude_norm, axis=0)

    # Spectral Bandwidth
    spectral_bandwidth = np.sqrt(
        np.sum(
            ((f[:, np.newaxis] - spectral_centroid[np.newaxis, :]) ** 2)
            * magnitude_norm,
            axis=0,
        )
    )

    # Spectral Entropy
    spectral_entropy = -np.sum(magnitude_norm * np.log2(magnitude_norm + 1e-12), axis=0)

    # Average over time frames
    features = {
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_centroid_std": np.std(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "spectral_bandwidth_std": np.std(spectral_bandwidth),
        "spectral_entropy_mean": np.mean(spectral_entropy),
        "spectral_entropy_std": np.std(spectral_entropy),
    }

    return features
