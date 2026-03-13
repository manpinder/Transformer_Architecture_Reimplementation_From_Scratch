import torch.nn as nn
import math
from common.positional_encoding import PositionalEncoding 

class TorchTransformer(nn.Module):
    """
    A PyTorch implementation of the Transformer architecture using nn.Transformer.
    Args:
        src_vocab_size: Size of the source vocabulary.
        tgt_vocab_size: Size of the target vocabulary.
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        n_layers: Number of encoder and decoder layers.
        d_ff: Dimension of the feedforward network.
        dropout_rate: Dropout rate.
        max_seq_len: Maximum sequence length for positional encoding.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout_rate: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=False
        )
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, tgt_mask=None, src_mask=None, tgt_padding_mask=None):
        """
        Args:
            src: Source sequences [Batch, Src_Seq_Len]
            tgt: Target sequences [Batch, Tgt_Seq_Len]
            tgt_mask: Causal mask [1, 1, Tgt_Len, Tgt_Len]
            src_padding_mask: Padding mask [Batch, 1, 1, Src_Len]
            tgt_padding_mask: Padding mask [Batch, 1, 1, Tgt_Len]
        """

        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        seq_len = tgt.size(1)
        tgt_subsequent_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tgt.device)

        official_src_key_padding = None
        if src_mask is not None:
            sq_mask = src_mask.view(src.size(0), -1)
            official_src_key_padding = (sq_mask == 0)

        official_tgt_key_padding = None
        if tgt_padding_mask is not None:
            sq_mask = tgt_padding_mask.view(tgt.size(0), -1)
            official_tgt_key_padding = (sq_mask == 0)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_subsequent_mask,
            src_key_padding_mask=official_src_key_padding,
            tgt_key_padding_mask=official_tgt_key_padding,
            memory_key_padding_mask=official_src_key_padding 
        )
        return self.output_linear(out)