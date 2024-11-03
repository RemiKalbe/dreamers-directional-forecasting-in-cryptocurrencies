import numpy as np
import torch as t
from torch.utils.data import Dataset


class CandleSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int):
        """
        Memory-efficient Dataset for time series sequences.

        Parameters:
            X (np.ndarray): Input features array, shape (num_samples, num_features)
            y (np.ndarray): Target array, shape (num_samples,)
            seq_length (int): Number of timesteps in each sequence
        """
        self.X = X
        self.y = y
        self.seq_length = seq_length

        # Calculate valid indices once
        self.valid_indices = len(X) - seq_length

    def __len__(self):
        """Return the number of possible sequences"""
        return self.valid_indices

    def __getitem__(self, idx):
        """Generate sequence on the fly for given index"""
        if idx < 0 or idx >= self.valid_indices:
            raise IndexError("Index out of bounds")

        # Extract sequence
        start_idx = idx
        end_idx = idx + self.seq_length

        # Get sequence and target
        X_seq = self.X[start_idx:end_idx]
        y_seq = self.y[end_idx - 1]  # Target for last timestep

        # Convert to tensors without transposing
        X_seq = t.tensor(X_seq, dtype=t.float32)
        y_seq = t.tensor(y_seq, dtype=t.float32)

        return X_seq, y_seq
