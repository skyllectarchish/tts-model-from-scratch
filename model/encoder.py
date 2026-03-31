"""
model/encoder.py — FastSpeech 2 Encoder

Implements:
  - PositionalEncoding:        sinusoidal positional embeddings
  - MultiHeadAttention:        standard scaled dot-product self-attention
  - PositionwiseFeedForward:   2-layer Conv1d FFN (FastSpeech 2 style)
  - FFTBlock:                  one Feed-Forward Transformer block
  - Encoder:                   character embedding + positional encoding + N FFT blocks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    VOCAB_SIZE, PAD_ID,
    ENCODER_HIDDEN_DIM, ENCODER_N_LAYERS, ENCODER_N_HEADS,
    ENCODER_CONV_FILTER_SIZE, ENCODER_CONV_KERNEL_SIZE, ENCODER_DROPOUT,
    MAX_SEQ_LEN,
)


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from 'Attention Is All You Need'.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Shape: (1, max_len, d_model) — broadcast over batch.
    """

    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product self-attention.

    Args:
        d_model:  Model dimension
        n_heads:  Number of attention heads
        dropout:  Dropout on attention weights
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Self-attention: query = key = value = x.

        Args:
            x:    (B, T, d_model)
            mask: (B, 1, 1, T) bool — True where positions should be ignored

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        # Project → (B, n_heads, T, d_head)
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                                  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)
        return self.W_o(out)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward using two 1D convolutions.

    FastSpeech 2 uses Conv1d instead of Linear so each position
    benefits from local context (kernel_size > 1).

    kernel_sizes: (k1, k2), typically (9, 1) per the paper.
    """

    def __init__(
        self,
        d_model:      int,
        d_inner:      int,
        kernel_sizes: tuple,
        dropout:      float = 0.1,
    ):
        super().__init__()
        k1, k2 = kernel_sizes
        self.conv1   = nn.Conv1d(d_model, d_inner, kernel_size=k1, padding=k1 // 2)
        self.conv2   = nn.Conv1d(d_inner, d_model, kernel_size=k2, padding=k2 // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        out = x.transpose(1, 2)              # → (B, d_model, T)
        out = F.relu(self.conv1(out))        # → (B, d_inner, T)
        out = self.conv2(out)                # → (B, d_model, T)
        out = out.transpose(1, 2)            # → (B, T, d_model)
        return self.dropout(out)


class FFTBlock(nn.Module):
    """
    One Feed-Forward Transformer (FFT) block:

        x → LayerNorm → MultiHeadAttention → + residual
          → LayerNorm → PositionwiseFeedForward → + residual

    Shared by both encoder and decoder.
    """

    def __init__(
        self,
        d_model:      int,
        n_heads:      int,
        d_inner:      int,
        kernel_sizes: tuple,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm  = nn.LayerNorm(d_model)
        self.attn      = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_inner, kernel_sizes, dropout)
        self.dropout   = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, d_model)
            mask: (B, 1, 1, T) or None
        Returns:
            (B, T, d_model)
        """
        # Self-attention sub-layer
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, mask=mask)
        x = self.dropout(x) + residual

        # FFN sub-layer
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual

        return x


class Encoder(nn.Module):
    """
    FastSpeech 2 Encoder:
      1. Character embedding  (vocab_size → d_model)
      2. Positional encoding  (sinusoidal, fixed)
      3. N × FFTBlock

    Outputs a hidden state sequence — one vector per input character.

    Args:
        vocab_size:   Size of the Gujarati character vocabulary
        d_model:      Hidden/embedding dimension
        n_layers:     Number of FFT blocks
        n_heads:      Attention heads per block
        d_inner:      FFN inner dimension
        kernel_sizes: FFN conv kernel sizes, e.g. (9, 1)
        dropout:      Dropout rate
        pad_id:       Padding token ID — masked in attention
    """

    def __init__(
        self,
        vocab_size:   int   = VOCAB_SIZE,
        d_model:      int   = ENCODER_HIDDEN_DIM,
        n_layers:     int   = ENCODER_N_LAYERS,
        n_heads:      int   = ENCODER_N_HEADS,
        d_inner:      int   = ENCODER_CONV_FILTER_SIZE,
        kernel_sizes: tuple = ENCODER_CONV_KERNEL_SIZE,
        dropout:      float = ENCODER_DROPOUT,
        pad_id:       int   = PAD_ID,
    ):
        super().__init__()
        self.pad_id  = pad_id
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)
        self.layers    = nn.ModuleList([
            FFTBlock(d_model, n_heads, d_inner, kernel_sizes, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        text_ids: torch.Tensor,
    ) -> tuple:
        """
        Encode a batch of character sequences.

        Args:
            text_ids: (B, T_text) LongTensor of character IDs

        Returns:
            enc_out:  (B, T_text, d_model) — encoder hidden states
            src_mask: (B, 1, 1, T_text)   — True where padding (for attention masking)
        """
        # Build padding mask: True where text_ids == pad_id
        src_mask = (text_ids == self.pad_id).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)

        # Scale embedding by sqrt(d_model) as in the original Transformer
        x = self.embedding(text_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, mask=src_mask)

        x = self.norm(x)
        return x, src_mask


if __name__ == "__main__":
    B, T = 2, 15
    ids = torch.randint(2, 80, (B, T))
    ids[0, 12:] = PAD_ID

    enc = Encoder()
    out, mask = enc(ids)
    print(f"Encoder output : {out.shape}")   # (2, 15, 256)
    print(f"Padding mask   : {mask.shape}")  # (2, 1, 1, 15)
    print("Encoder self-test PASSED")
