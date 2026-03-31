"""
model/variance_adaptor.py — FastSpeech 2 Variance Adaptor

Components:
  - VariancePredictor:  2-conv network predicting duration, pitch, or energy
  - LengthRegulator:    expands encoder output by per-character durations
  - VarianceAdaptor:    combines all three predictors + length regulator

Training:  ground-truth durations/pitch/energy are passed in (teacher-forced)
Inference: all three are predicted by the model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    ENCODER_HIDDEN_DIM,
    VARIANCE_PREDICTOR_FILTER_SIZE,
    VARIANCE_PREDICTOR_KERNEL_SIZE,
    VARIANCE_PREDICTOR_DROPOUT,
    PITCH_EMBEDDING_DIM,
    ENERGY_EMBEDDING_DIM,
    N_PITCH_BINS,
    N_ENERGY_BINS,
)


class VariancePredictor(nn.Module):
    """
    Small 2-layer Conv1d network that predicts a scalar value per frame.

    Used for duration, pitch, and energy prediction.

    Architecture:
        Conv1d(d_model, filter_size, k) → ReLU → LayerNorm → Dropout
        Conv1d(filter_size, filter_size, k) → ReLU → LayerNorm → Dropout
        Linear(filter_size, 1) → scalar per frame
    """

    def __init__(
        self,
        d_model:     int   = ENCODER_HIDDEN_DIM,
        filter_size: int   = VARIANCE_PREDICTOR_FILTER_SIZE,
        kernel_size: int   = VARIANCE_PREDICTOR_KERNEL_SIZE,
        dropout:     float = VARIANCE_PREDICTOR_DROPOUT,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.conv1   = nn.Conv1d(d_model,     filter_size, kernel_size, padding=pad)
        self.conv2   = nn.Conv1d(filter_size, filter_size, kernel_size, padding=pad)
        self.norm1   = nn.LayerNorm(filter_size)
        self.norm2   = nn.LayerNorm(filter_size)
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(filter_size, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    (B, T, d_model)
            mask: (B, T) bool — True where padding (output zeroed)

        Returns:
            (B, T) scalar predictions per frame
        """
        out = x.transpose(1, 2)                       # (B, d_model, T)

        out = F.relu(self.conv1(out))                 # (B, filter_size, T)
        out = self.norm1(out.transpose(1, 2))         # (B, T, filter_size)
        out = self.dropout(out)

        out = F.relu(self.conv2(out.transpose(1, 2))) # (B, filter_size, T)
        out = self.norm2(out.transpose(1, 2))         # (B, T, filter_size)
        out = self.dropout(out)

        out = self.linear(out).squeeze(-1)            # (B, T)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class LengthRegulator(nn.Module):
    """
    Expand encoder hidden states by per-character durations.

    For character i, its hidden state is repeated duration[i] times.
    This maps text-length sequences → mel-length sequences.

    Training:  use ground-truth durations
    Inference: use rounded predicted durations
    """

    def forward(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
        max_len: int = None,
    ) -> tuple:
        """
        Args:
            x:         (B, T_text, d_model) — encoder output
            durations: (B, T_text) int tensor — frames per character
            max_len:   Pad output to this length (optional)

        Returns:
            output:   (B, T_mel, d_model) — expanded hidden states
            mel_lens: (B,) — actual mel lengths per sample
        """
        outputs  = []
        mel_lens = []

        for i in range(x.size(0)):
            dur = durations[i].clamp(min=0)
            expanded = torch.repeat_interleave(x[i], dur, dim=0)  # (T_mel_i, d_model)
            outputs.append(expanded)
            mel_lens.append(expanded.size(0))

        mel_lens   = torch.LongTensor(mel_lens).to(x.device)
        target_len = max_len if max_len is not None else int(mel_lens.max().item())

        padded = torch.zeros(x.size(0), target_len, x.size(2), device=x.device)
        for i, out in enumerate(outputs):
            length = min(out.size(0), target_len)
            padded[i, :length] = out[:length]

        return padded, mel_lens


class VarianceAdaptor(nn.Module):
    """
    FastSpeech 2 Variance Adaptor.

    Processing pipeline:
      1. Duration predictor  → teacher-forced (train) or predicted (infer) durations
      2. Length regulator    → expand encoder output to mel length
      3. Pitch predictor     → quantize → pitch embedding → add to output
      4. Energy predictor    → quantize → energy embedding → add to output

    Pitch and energy are quantized into N uniform bins before embedding,
    following the original FastSpeech 2 paper.
    """

    def __init__(
        self,
        d_model:          int   = ENCODER_HIDDEN_DIM,
        filter_size:      int   = VARIANCE_PREDICTOR_FILTER_SIZE,
        kernel_size:      int   = VARIANCE_PREDICTOR_KERNEL_SIZE,
        dropout:          float = VARIANCE_PREDICTOR_DROPOUT,
        n_pitch_bins:     int   = N_PITCH_BINS,
        n_energy_bins:    int   = N_ENERGY_BINS,
        pitch_embed_dim:  int   = PITCH_EMBEDDING_DIM,
        energy_embed_dim: int   = ENERGY_EMBEDDING_DIM,
    ):
        super().__init__()

        self.duration_predictor = VariancePredictor(d_model, filter_size, kernel_size, dropout)
        self.pitch_predictor    = VariancePredictor(d_model, filter_size, kernel_size, dropout)
        self.energy_predictor   = VariancePredictor(d_model, filter_size, kernel_size, dropout)

        self.length_regulator = LengthRegulator()

        # Pitch quantization boundaries (normalized F0 range)
        self.register_buffer(
            "pitch_bins",
            torch.linspace(-3.0, 3.0, n_pitch_bins - 1),
        )
        self.pitch_embed = nn.Embedding(n_pitch_bins, pitch_embed_dim)

        # Energy quantization boundaries
        self.register_buffer(
            "energy_bins",
            torch.linspace(-3.0, 3.0, n_energy_bins - 1),
        )
        self.energy_embed = nn.Embedding(n_energy_bins, energy_embed_dim)

    def _quantize(self, values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """Quantize continuous values into discrete bin indices."""
        return torch.bucketize(values.contiguous(), bins)

    def forward(
        self,
        enc_out:          torch.Tensor,
        src_mask:         torch.Tensor,
        mel_mask:         torch.Tensor  = None,
        durations:        torch.Tensor  = None,
        pitch_target:     torch.Tensor  = None,
        energy_target:    torch.Tensor  = None,
        max_mel_len:      int           = None,
        duration_control: float         = 1.0,
        pitch_control:    float         = 1.0,
        energy_control:   float         = 1.0,
    ) -> tuple:
        """
        Args:
            enc_out:       (B, T_text, d_model) — encoder hidden states
            src_mask:      (B, 1, 1, T_text)   — True where text padding
            mel_mask:      (B, T_mel)           — True where mel padding
            durations:     (B, T_text) int      — ground-truth durations (train) or None
            pitch_target:  (B, T_mel) float     — ground-truth pitch (train) or None
            energy_target: (B, T_mel) float     — ground-truth energy (train) or None
            max_mel_len:   Pad mel output to this length
            duration/pitch/energy_control: Inference scaling (1.0 = natural)

        Returns:
            output:      (B, T_mel, d_model)
            mel_lens:    (B,)
            dur_pred:    (B, T_text) — log-duration predictions (for loss)
            pitch_pred:  (B, T_mel) — pitch predictions (for loss)
            energy_pred: (B, T_mel) — energy predictions (for loss)
        """
        # Flatten src_mask → (B, T_text) for predictor masking
        text_mask = src_mask.squeeze(1).squeeze(1)

        # 1. Duration prediction (log domain)
        dur_pred = self.duration_predictor(enc_out, mask=text_mask)   # (B, T_text)

        # 2. Length regulation
        if durations is not None:
            # Training: use ground-truth durations
            reg_out, mel_lens = self.length_regulator(enc_out, durations, max_mel_len)
        else:
            # Inference: exp(pred) → round to int
            pred_dur = (torch.exp(dur_pred) - 1.0).clamp(min=0)
            pred_dur = (pred_dur * duration_control).round().long()
            reg_out, mel_lens = self.length_regulator(enc_out, pred_dur, max_mel_len)

        # 3. Pitch prediction + embedding
        pitch_pred = self.pitch_predictor(reg_out, mask=mel_mask)     # (B, T_mel)

        if pitch_target is not None:
            pitch_quant = self._quantize(pitch_target, self.pitch_bins)
        else:
            pitch_pred  = pitch_pred * pitch_control
            pitch_quant = self._quantize(pitch_pred, self.pitch_bins)

        reg_out = reg_out + self.pitch_embed(pitch_quant)

        # 4. Energy prediction + embedding
        energy_pred = self.energy_predictor(reg_out, mask=mel_mask)   # (B, T_mel)

        if energy_target is not None:
            energy_quant = self._quantize(energy_target, self.energy_bins)
        else:
            energy_pred  = energy_pred * energy_control
            energy_quant = self._quantize(energy_pred, self.energy_bins)

        reg_out = reg_out + self.energy_embed(energy_quant)

        return reg_out, mel_lens, dur_pred, pitch_pred, energy_pred


if __name__ == "__main__":
    from config import ENCODER_HIDDEN_DIM

    B, T_text, d = 2, 15, ENCODER_HIDDEN_DIM
    enc_out   = torch.randn(B, T_text, d)
    src_mask  = torch.zeros(B, 1, 1, T_text, dtype=torch.bool)
    durations = torch.randint(5, 10, (B, T_text))
    T_mel     = int(durations.sum(dim=1).max().item())
    pitch     = torch.randn(B, T_mel)
    energy    = torch.randn(B, T_mel)

    va = VarianceAdaptor()
    out, mel_lens, dur_pred, pitch_pred, energy_pred = va(
        enc_out, src_mask,
        durations=durations, pitch_target=pitch, energy_target=energy,
        max_mel_len=T_mel,
    )
    print(f"VA output    : {out.shape}")          # (2, T_mel, 256)
    print(f"mel_lens     : {mel_lens}")
    print(f"dur_pred     : {dur_pred.shape}")     # (2, 15)
    print(f"pitch_pred   : {pitch_pred.shape}")   # (2, T_mel)
    print(f"energy_pred  : {energy_pred.shape}")  # (2, T_mel)
    print("VarianceAdaptor self-test PASSED")
