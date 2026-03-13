import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network.
    Args:
        d_model: Dimension of the model.
        d_ff: Dimension of the feed-forward network.
        dropout: Dropout rate.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
