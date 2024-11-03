import numpy as np
from PyEMD import EMD


def compute_emd_features_for_window(signal: np.ndarray, max_imf: int = 5) -> dict:
    """
    Compute EMD features for a given signal window.

    Parameters:
        signal (np.ndarray): Array of data for which to compute EMD features.
        max_imf (int): Maximum number of Intrinsic Mode Functions (IMFs) to extract.

    Returns:
        dict: Contains statistical features extracted from the IMFs.
    """
    # Ensure the signal is a 1D numpy array
    signal = np.asarray(signal).flatten()

    # Perform Empirical Mode Decomposition
    emd = EMD()
    IMFs = emd.emd(signal, max_imf=max_imf)

    # Initialize a dictionary to store features
    features = {}

    # Extract statistical features from each IMF
    for i, imf in enumerate(IMFs):
        # Limit to max_imf
        if i >= max_imf:
            break
        features[f"IMF_{i+1}_mean"] = np.mean(imf)
        features[f"IMF_{i+1}_std"] = np.std(imf)
        features[f"IMF_{i+1}_energy"] = np.sum(imf**2)
        # You can add more features like skewness, kurtosis, etc.

    return features
