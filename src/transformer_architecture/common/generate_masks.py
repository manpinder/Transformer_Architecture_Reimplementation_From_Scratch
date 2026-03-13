import torch

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates a lower triangular mask for decoder self-attention.
    Args:
        sz: Sequence length.
    """
    return torch.tril(torch.ones(sz, sz, dtype=torch.float)).view(1, 1, sz, sz)

def generate_padding_mask(seq: torch.Tensor, pad_token: int) -> torch.Tensor:
    """Generates a padding mask for attention.
    Args:
        seq: Input sequence, shape [batch_size, seq_len].
        pad_token: Token ID for padding.
    """
    return (seq != pad_token).unsqueeze(1).unsqueeze(2).float()