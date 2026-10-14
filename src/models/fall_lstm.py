import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Temporal attention mechanism over BiLSTM output sequence."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out shape: (B, T, hidden_dim)
        scores = self.attn(lstm_out).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1)    # (B, T)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden_dim)
        return context


class FallLSTM(nn.Module):
    """Bidirectional LSTM with Temporal Attention for human fall detection."""

    def __init__(
        self,
        input_size: int = 102,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.attention = TemporalAttention(hidden * 2)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        return self.head(context)
