"""
SignLanguageLSTM — Bidirectional LSTM with attention for continuous sign language recognition.

Architecture:
    Input:  (batch_size, seq_len=30, input_dim=162)
    Output: (batch_size, num_classes)

Feature vector (162 dimensions):
    - Upper body pose (landmarks 11-22): 12 × 3 = 36
    - Left hand (21 landmarks):          21 × 3 = 63
    - Right hand (21 landmarks):         21 × 3 = 63
    Total: 36 + 63 + 63 = 162
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Scaled dot-product attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)
        """
        # (batch, seq_len, 1)
        scores = self.attention_weights(lstm_output)
        # (batch, seq_len, 1)
        alpha = F.softmax(scores, dim=1)
        # weighted sum → (batch, hidden_dim)
        context = torch.sum(alpha * lstm_output, dim=1)
        return context


class SignLanguageLSTM(nn.Module):
    """
    Bidirectional LSTM with attention for sign language gesture classification.

    Parameters:
        input_dim:   162 (MediaPipe keypoint features)
        hidden_dim:  128 (LSTM hidden size)
        num_layers:  2
        num_classes: variable (user-defined vocabulary size)
        dropout:     0.3
    """

    def __init__(
        self,
        input_dim: int = 162,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention over LSTM outputs (bidirectional → 2 * hidden_dim)
        self.attention = Attention(hidden_dim * 2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)  e.g. (B, 30, 162)
        Returns:
            logits: (batch_size, num_classes)
        """
        # Project input features
        x = self.input_proj(x)  # (B, 30, hidden_dim)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, 30, hidden_dim * 2)

        # Attention pooling
        context = self.attention(lstm_out)  # (B, hidden_dim * 2)

        # Classify
        logits = self.classifier(context)  # (B, num_classes)
        return logits


def create_model(num_classes: int, device: str = "cpu") -> SignLanguageLSTM:
    """Factory function to create and initialize a model."""
    model = SignLanguageLSTM(
        input_dim=162,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    )
    model.to(device)
    return model


def save_model(model: SignLanguageLSTM, path: str, vocab: dict):
    """Save model weights and vocabulary mapping."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": model.num_classes,
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "vocab": vocab,  # {label_str: index_int}
        },
        path,
    )


def load_model(path: str, device: str = "cpu"):
    """Load model weights and vocabulary mapping. Returns (model, vocab)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SignLanguageLSTM(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["vocab"]
