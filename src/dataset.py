import numpy as np
from torch.utils.data import Dataset
import torch


class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        """
        Args:
            features (np.ndarray): The input features of shape (num_samples, num_features).
            targets (np.ndarray): The target values of shape (num_samples,).
            seq_length (int): The length of each input sequence.
        """
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

        self.num_samples = self.features.shape[0] - self.seq_length
        self.num_features = self.features.shape[1]

        # Create sequences using as_strided
        self.X_seq = self._create_sequences(self.features, self.seq_length)
        self.y_seq = self.targets[self.seq_length :]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        X_seq = self.X_seq[idx]
        y_target = self.y_seq[idx]

        # Make writable copies to avoid read-only warnings
        X_seq = np.array(X_seq, copy=True)
        y_target = np.array(y_target, copy=True)

        # Convert to torch.Tensor
        X_seq = torch.from_numpy(X_seq).float()
        y_target = torch.tensor(y_target).float()

        return X_seq, y_target

    def _create_sequences(self, data, seq_length):
        num_samples = data.shape[0] - seq_length + 1
        num_features = data.shape[1]

        shape = (num_samples, seq_length, num_features)
        strides = (data.strides[0], data.strides[0], data.strides[1])

        sequences = np.lib.stride_tricks.as_strided(
            data,
            shape=shape,
            strides=strides,
            writeable=False,
        )

        return sequences
