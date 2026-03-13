import torch
import torch.nn as nn
from transformer_architecture.custom_implementation.architecture import (
    PositionalEncoding, 
    MultiHeadAttention, 
    CustomTransformer,
    EncoderLayer,
    DecoderLayer,
    PositionWiseFeedForward
)
from transformer_architecture.common.generate_masks import (
    generate_square_subsequent_mask, 
    generate_padding_mask
)

def test_generate_padding_mask():
    """Test padding mask creates correct tensor for attention."""
    pad_token = 0
    seq = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    mask = generate_padding_mask(seq, pad_token)

    assert mask.shape == (2, 1, 1, 4), "Padding mask shape mismatch"
    assert mask.dim() == 4, "Mask must be broadcastable to 4D"

def test_generate_square_subsequent_mask():
    """Test causal mask is lower triangular."""
    seq_len = 5
    mask = generate_square_subsequent_mask(seq_len)
    
    assert mask.shape == (1, 1, 5, 5), "Causal mask shape mismatch"
    assert torch.allclose(mask[0, 0, 0, 1], torch.tensor(0.0)), "Upper triangle should be masked"

def test_positional_encoding():
    """Test the PositionalEncoding module for correct shape and determinism."""
    d_model, max_seq_len = 512, 50
    pe = PositionalEncoding(d_model, max_seq_len)
    
    input_tensor = torch.zeros(2, 10, d_model)
    output = pe(input_tensor)
    
    assert output.shape == (2, 10, d_model), "PositionalEncoding shape mismatch"
    assert torch.allclose(output[0, 0, 0::2][:2], torch.tensor([0.0, 0.0]), atol=1e-4), "Sin values at pos 0 should be 0"
    assert not pe.pe.requires_grad, "Positional encodings should not be trainable parameters"

def test_position_wise_feed_forward():
    """Test FeedForward network shape and gradient readiness."""
    d_model, d_ff, dropout = 512, 2048, 0.1
    ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
    x = torch.randn(2, 10, d_model)
    output = ffn(x)
    assert output.shape == x.shape, "FFN output shape must match input shape"

def test_multi_head_attention():
    """Test the MultiHeadAttention module for correct output shape with/without mask."""
    d_model, n_heads, dropout_rate = 512, 8, 0.1
    mha = MultiHeadAttention(d_model, n_heads, dropout_rate)
    query = torch.randn(2, 10, d_model)
    key = torch.randn(2, 10, d_model)
    value = torch.randn(2, 10, d_model)
    
    out_no_mask = mha(query, key, value, mask=None)
    assert out_no_mask.shape == query.shape, "MHA shape mismatch without mask"
    
    mask = torch.ones(2, 1, 1, 10)
    out_with_mask = mha(query, key, value, mask=mask)
    assert out_with_mask.shape == query.shape, "MHA shape mismatch with mask"
    
def test_encoder_layer():
    """Test a single EncoderLayer forward pass."""
    d_model, n_heads, d_ff, dropout = 512, 8, 2048, 0.1
    encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
    x = torch.randn(2, 10, d_model)
    mask = torch.ones(2, 1, 1, 10)
    
    output = encoder_layer(x, src_mask=mask)
    assert output.shape == (2, 10, d_model), "EncoderLayer shape mismatch"

def test_decoder_layer():
    """Test a single DecoderLayer forward pass including cross-attention."""
    d_model, n_heads, d_ff, dropout = 512, 8, 2048, 0.1
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
    
    tgt = torch.randn(2, 8, d_model)
    enc_output = torch.randn(2, 10, d_model)
    
    tgt_mask = generate_square_subsequent_mask(8)
    src_mask = torch.ones(2, 1, 1, 10)
    
    output = decoder_layer(tgt, enc_output, tgt_mask=tgt_mask, src_mask=src_mask)
    assert output.shape == (2, 8, d_model), "DecoderLayer shape mismatch"

def test_transformer_forward():
    """Test the full Transformer model forward pass."""
    vocab_size, d_model, n_heads, n_layers, d_ff, dropout_rate, max_seq_len = 1000, 256, 4, 2, 512, 0.1, 128
    model = CustomTransformer(vocab_size, vocab_size, d_model, n_heads, n_layers, d_ff, dropout_rate, max_seq_len)
    
    src = torch.randint(1, vocab_size, (2, 10))
    tgt = torch.randint(1, vocab_size, (2, 8))
    
    src_mask = generate_padding_mask(src, 0)
    tgt_subsequent_mask = generate_square_subsequent_mask(tgt.size(1))
    tgt_padding_mask = generate_padding_mask(tgt, 0)
    tgt_mask = tgt_subsequent_mask * tgt_padding_mask
    
    output = model(src, tgt, src_mask, tgt_mask)
    assert output.shape == (2, 8, vocab_size), "Full Transformer shape mismatch"

def test_transformer_backward_pass():
    """Test that gradients flow backward without detaching."""
    vocab_size, d_model = 1000, 256
    model = CustomTransformer(vocab_size, vocab_size, d_model, n_heads=4, n_layers=2, d_ff=512, dropout_rate=0.1, max_seq_len=64)
    
    src = torch.randint(1, vocab_size, (2, 10))
    tgt = torch.randint(1, vocab_size, (2, 8))

    output = model(src, tgt)
    target_labels = torch.randint(0, vocab_size, (2, 8))
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output.view(-1, vocab_size), target_labels.view(-1))
    loss.backward()
    
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            break
            
    assert has_gradients, "Gradients did not flow during backward pass! Check for disconnected tensors or argmax operations."

def test_attention_mask_integrity():
    """Test that masked positions do not influence the output at all."""
    d_model, n_heads = 128, 4
    mha = MultiHeadAttention(d_model, n_heads, dropout=0.0)
    mha.eval()

    q = torch.randn(1, 5, d_model)
    k = torch.randn(1, 5, d_model)
    v = torch.randn(1, 5, d_model)

    mask = torch.tensor([[[[1, 1, 1, 0, 0]]]])
    out1 = mha(q, k, v, mask)
    
    k[:, 3:, :] = torch.randn(1, 2, d_model)
    v[:, 3:, :] = torch.randn(1, 2, d_model)
    out2 = mha(q, k, v, mask)
    
    assert torch.allclose(out1, out2, atol=1e-7), "Masking Leak: Masked tokens influenced the output!"

def test_residual_and_norm():
    """Test that LayerNorm and Residual connections maintain signal variance."""
    d_model = 128
    encoder_layer = EncoderLayer(d_model, n_heads=4, d_ff=512, dropout=0.0)

    x = torch.randn(2, 10, d_model) * 100 + 50
    output = encoder_layer(x, src_mask=None)
    
    assert output.shape == x.shape
    assert abs(output.mean()) < 1.0
    assert 0.5 < output.std() < 1.5

def test_transformer_causality():
    """Test that model cannot look ahead in the target sequence."""
    vocab_size, d_model = 100, 128
    model = CustomTransformer(vocab_size, vocab_size, d_model, n_heads=4, n_layers=2, d_ff=256, dropout_rate=0.0, max_seq_len=20)
    model.eval()

    src = torch.randint(1, vocab_size, (1, 10))
    tgt1 = torch.tensor([[1, 2, 3, 4, 5]])
    tgt2 = torch.tensor([[1, 2, 3, 99, 99]])
    tgt_mask = generate_square_subsequent_mask(5)

    with torch.no_grad():
        out1 = model(src, tgt1, None, tgt_mask)
        out2 = model(src, tgt2, None, tgt_mask)

    assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-7), "Causality Violation: Future tokens influenced past predictions!"

def test_model_overfit_small_data():
    """Test that the model can memorize a single batch?"""
    torch.manual_seed(42)
    vocab_size, d_model = 50, 64
    
    model = CustomTransformer(
        src_vocab_size=vocab_size, 
        tgt_vocab_size=vocab_size, 
        d_model=d_model, 
        n_heads=2, 
        n_layers=1, 
        d_ff=128, 
        dropout_rate=0.0, 
        max_seq_len=20
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = torch.nn.CrossEntropyLoss()

    src = torch.randint(1, vocab_size, (1, 5))
    tgt = torch.randint(1, vocab_size, (1, 5))
    tgt_input = tgt[:, :-1]
    tgt_expected = tgt[:, 1:]

    model.train()
    final_loss = 0
    for _ in range(100):
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.view(-1, vocab_size), tgt_expected.reshape(-1))
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    predictions = output.argmax(dim=-1)
    accuracy = (predictions == tgt_expected).float().mean().item()
    
    assert accuracy >= 0.95, f"Model failed to memorize. Accuracy: {accuracy * 100}%, Loss: {final_loss}"