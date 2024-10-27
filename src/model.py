import math
import torch
import torch.nn as nn
import os


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as described in the Transformer architecture.
    This provides the model with information about the position of each element in the sequence.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of shape (1, max_len, d_model) to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the positional encodings using sine and cosine functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0)  # Add a batch dimension
        # Register pe as a buffer to prevent it from being considered a model parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + self.pe[:, : x.size(1), :]
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        feature_size: int,
        seq_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: int,
    ):
        super(TimeSeriesTransformer, self).__init__()

        # Project input features to the Transformer model dimension
        self.input_projection = nn.Linear(feature_size, d_model)

        # Positional encoding to provide sequence order information
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, max_len=seq_length + 1
        )

        # Classification token to aggregate sequence information
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,  # Use batch_first=True for compatibility
        )
        # Stack multiple Transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Final classifier that maps the Transformer output to a single prediction
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def initialize_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.size(0)

        # Project input features to the model dimension
        x = self.input_projection(x)

        # Expand and concatenate the classification token to the input sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # Shape: [batch_size, seq_length + 1, d_model]

        # Add positional encoding to the input embeddings
        x = self.positional_encoding(x)

        # Pass the input through the Transformer encoder
        x = self.transformer_encoder(x)

        # Extract the output corresponding to the classification token
        cls_output = x[:, 0, :]  # Shape: [batch_size, d_model]

        # Pass the classification token output through the classifier
        out = self.classifier(cls_output)

        # Return raw logits (do not apply sigmoid here)
        return out.squeeze()

    def get_params(self) -> dict[str, int | float]:
        return {
            "feature_size": self.feature_size,
            "seq_length": self.seq_length,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    def save(self, path: str) -> None:
        """Save the model parameters and state dict."""
        model_info = {"state_dict": self.state_dict(), "params": self.get_params()}
        torch.save(model_info, path)

    @classmethod
    def load(
        cls, path: str, device: torch.device | None = None
    ) -> "TimeSeriesTransformer":
        """Load a model from a file."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model info containing parameters and state dict
        model_info = torch.load(path, map_location=device)

        # Create new model with saved parameters
        model = cls(**model_info["params"])

        # Load the state dict
        model.load_state_dict(model_info["state_dict"])

        return model.to(device)

    @classmethod
    def load_from_dir(
        cls, model_dir: str, device: torch.device | None = None
    ) -> "TimeSeriesTransformer":
        """Load a model from a directory (for SageMaker inference)."""
        return cls.load(os.path.join(model_dir, "model.pth"), device)
