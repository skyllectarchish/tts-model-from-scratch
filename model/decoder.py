"""
model/decoder.py — FastSpeech 2 Decoder

Same FFT block structure as the encoder, applied after the variance adaptor.
Positional encoding is re-added here (after length regulation changes lengths).

Architecture:
  1. Positional encoding
  2. N × FFTBlock  (reused from model/encoder.py)
  3. LayerNorm
  4. Linear(d_model → n_mels) — output mel spectrogram frames
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    DECODER_HIDDEN_DIM, DECODER_N_LAYERS, DECODER_N_HEADS,
    DECODER_CONV_FILTER_SIZE, DECODER_CONV_KERNEL_SIZE, DECODER_DROPOUT,
    N_MELS, MAX_MEL_LEN,
)
from model.encoder import FFTBlock, PositionalEncoding


class Decoder(nn.Module):
    """
    FastSpeech 2 Decoder.

    Takes variance-adapted hidden states and predicts mel spectrogram frames.

    Args:
        d_model:      Hidden dimension
        n_layers:     Number of FFT blocks
        n_heads:      Attention heads per block
        d_inner:      FFN inner dimension
        kernel_sizes: FFN conv kernel sizes, e.g. (9, 1)
        dropout:      Dropout rate
        n_mels:       Number of mel filter banks (output dimension)
    """

    def __init__(
        self,
        d_model:      int   = DECODER_HIDDEN_DIM,
        n_layers:     int   = DECODER_N_LAYERS,
        n_heads:      int   = DECODER_N_HEADS,
        d_inner:      int   = DECODER_CONV_FILTER_SIZE,
        kernel_sizes: tuple = DECODER_CONV_KERNEL_SIZE,
        dropout:      float = DECODER_DROPOUT,
        n_mels:       int   = N_MELS,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, max_len=MAX_MEL_LEN, dropout=dropout)
        self.layers  = nn.ModuleList([
            FFTBlock(d_model, n_heads, d_inner, kernel_sizes, dropout)
            for _ in range(n_layers)
        ])
        self.norm   = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, n_mels)

    def forward(
        self,
        x: torch.Tensor,
        mel_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        (B, T_mel, d_model) — variance-adapted encoder output
            mel_mask: (B, 1, 1, T_mel)   — True where mel padding

        Returns:
            mel_out: (B, T_mel, n_mels) — predicted mel spectrogram frames
        """
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, mask=mel_mask)

        x = self.norm(x)
        mel_out = self.linear(x)    # (B, T_mel, n_mels)

        return mel_out


if __name__ == "__main__":
    from config import DECODER_HIDDEN_DIM

    B, T_mel, d = 2, 120, DECODER_HIDDEN_DIM
    x = torch.randn(B, T_mel, d)

    dec = Decoder()
    out = dec(x)
    print(f"Decoder output: {out.shape}")   # (2, 120, 80)
    print("Decoder self-test PASSED")
