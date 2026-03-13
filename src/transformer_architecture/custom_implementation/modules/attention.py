import torch
import torch.nn as nn
import math
from typing import Optional

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None, dropout: Optional[nn.Dropout] = None) -> torch.Tensor:
    """Scaled dot-product attention.
    Args:
        q: Queries tensor of shape (batch, n_heads, seq_len_q, d_k).
        k: Keys tensor of shape (batch, n_heads, seq_len_k, d_k).
        v: Values tensor of shape (batch, n_heads, seq_len_v, d_k).
        mask: Mask tensor broadcastable to (batch, n_heads, seq_len_q, seq_len_k).
        dropout: Dropout layer.   
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, v)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module.
    Args:
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        batch = query.size(0)
        def reshape(x):
            return x.view(batch, -1, self.n_heads, self.d_k).transpose(1,2)
        q = reshape(self.w_q(query))
        k = reshape(self.w_k(key))
        v = reshape(self.w_v(value))
        attn_out = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        attn_out = attn_out.transpose(1,2).contiguous().view(batch, -1, self.d_model)
        return self.w_o(attn_out)
