import torch.nn as nn
import math
from .modules.attention import MultiHeadAttention
from .modules.feed_forward import PositionWiseFeedForward
from common.positional_encoding import PositionalEncoding

class EncoderLayer(nn.Module):
    """Encoder layer consisting of multi-head self-attention and position-wise feed-forward network.
    Args:
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        d_ff: Dimension of the feed-forward network.
        dropout: Dropout rate.
    """
    def __init__(self, d_model:int, n_heads:int, d_ff:int, dropout:float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x2 = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x3 = self.norm2(x2 + self.dropout(self.ffn(x2)))
        return x3

class DecoderLayer(nn.Module):
    """Decoder layer consisting of multi-head self-attention, cross-attention, and position-wise feed-forward network.
    Args:
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        d_ff: Dimension of the feed-forward network.
        dropout: Dropout rate.
    """
    def __init__(self, d_model:int, n_heads:int, d_ff:int, dropout:float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_output, enc_output, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x

class CustomTransformer(nn.Module):
    """Transformer model consisting of encoder and decoder stacks.
    Args:
        src_vocab_size: Size of the source vocabulary.
        tgt_vocab_size: Size of the target vocabulary.
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        n_layers: Number of encoder and decoder layers.
        d_ff: Dimension of the feed-forward network.
        dropout_rate: Dropout rate.
        max_seq_len: Maximum sequence length of the input sequences.
    """
    def __init__(self, src_vocab_size:int, tgt_vocab_size:int, d_model:int, n_heads:int,
                 n_layers:int, d_ff:int, dropout_rate:float, max_seq_len:int):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, tgt_padding_mask=None):
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim))
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, tgt_mask, src_mask)
        return self.output_linear(dec_output)
