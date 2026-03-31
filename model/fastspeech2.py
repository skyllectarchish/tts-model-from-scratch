"""
model/fastspeech2.py — Full FastSpeech 2 Model

Assembles:   Encoder → VarianceAdaptor → Decoder

Input:  character IDs  (B, T_text)
Output: mel spectrogram (B, n_mels, T_mel)

During training:  ground-truth durations / pitch / energy are passed in
During inference: all variance signals are predicted by the model
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    DURATION_CONTROL, PITCH_CONTROL, ENERGY_CONTROL,
)
from model.encoder import Encoder
from model.variance_adaptor import VarianceAdaptor
from model.decoder import Decoder


class FastSpeech2(nn.Module):
    """
    FastSpeech 2: Non-Autoregressive Text-to-Speech.

    Data flow:
        text_ids (B, T_text)
            ↓  Encoder
        enc_out (B, T_text, d_model)
            ↓  VarianceAdaptor  [duration → length regulate → pitch + energy]
        va_out  (B, T_mel, d_model)
            ↓  Decoder
        mel_out (B, n_mels, T_mel)
    """

    def __init__(self):
        super().__init__()
        self.encoder          = Encoder()
        self.variance_adaptor = VarianceAdaptor()
        self.decoder          = Decoder()

    def _make_mel_mask(
        self,
        mel_lens: torch.Tensor,
        max_len: int,
    ) -> tuple:
        """
        Build mel-frame padding masks from lengths.

        Returns:
            mel_mask_2d: (B, T_mel)       — True where padding
            mel_mask_4d: (B, 1, 1, T_mel) — for attention masking
        """
        B   = mel_lens.size(0)
        ids = torch.arange(max_len, device=mel_lens.device).unsqueeze(0).expand(B, -1)
        mel_mask_2d = ids >= mel_lens.unsqueeze(1)
        mel_mask_4d = mel_mask_2d.unsqueeze(1).unsqueeze(2)
        return mel_mask_2d, mel_mask_4d

    def forward(
        self,
        text_ids:         torch.Tensor,
        durations:        torch.Tensor = None,
        pitch_target:     torch.Tensor = None,
        energy_target:    torch.Tensor = None,
        mel_lens:         torch.Tensor = None,
        max_mel_len:      int          = None,
        duration_control: float        = 1.0,
        pitch_control:    float        = 1.0,
        energy_control:   float        = 1.0,
    ) -> dict:
        """
        Forward pass.

        Training call:
            out = model(text_ids, durations=d, pitch_target=p,
                        energy_target=e, mel_lens=ml, max_mel_len=T)

        Inference call:
            out = model(text_ids)

        Returns dict:
            mel_out:     (B, n_mels, T_mel) — predicted mel spectrogram
            dur_pred:    (B, T_text)        — predicted log-durations
            pitch_pred:  (B, T_mel)         — predicted pitch
            energy_pred: (B, T_mel)         — predicted energy
            mel_lens:    (B,)               — actual mel lengths
            mel_mask:    (B, T_mel)         — True where mel padding
        """
        # ---- Encoder ----
        enc_out, src_mask = self.encoder(text_ids)       # (B, T, d), (B,1,1,T)

        # ---- Build mel mask for variance adaptor (training) ----
        mel_mask_2d = None
        if mel_lens is not None and max_mel_len is not None:
            mel_mask_2d, _ = self._make_mel_mask(mel_lens, max_mel_len)

        # ---- Variance Adaptor ----
        va_out, pred_mel_lens, dur_pred, pitch_pred, energy_pred = self.variance_adaptor(
            enc_out,
            src_mask,
            mel_mask       = mel_mask_2d,
            durations      = durations,
            pitch_target   = pitch_target,
            energy_target  = energy_target,
            max_mel_len    = max_mel_len,
            duration_control = duration_control,
            pitch_control    = pitch_control,
            energy_control   = energy_control,
        )

        # ---- Build mel mask for decoder ----
        T_mel = va_out.size(1)
        mel_mask_2d_dec, mel_mask_4d_dec = self._make_mel_mask(pred_mel_lens, T_mel)

        # ---- Decoder ----
        dec_out = self.decoder(va_out, mel_mask=mel_mask_4d_dec)   # (B, T_mel, n_mels)

        # Zero out padded frames
        dec_out = dec_out.masked_fill(mel_mask_2d_dec.unsqueeze(-1), 0.0)

        # Transpose to (B, n_mels, T_mel) — standard mel convention
        mel_out = dec_out.transpose(1, 2)

        return {
            "mel_out"     : mel_out,           # (B, n_mels, T_mel)
            "dur_pred"    : dur_pred,           # (B, T_text)
            "pitch_pred"  : pitch_pred,         # (B, T_mel)
            "energy_pred" : energy_pred,        # (B, T_mel)
            "mel_lens"    : pred_mel_lens,      # (B,)
            "mel_mask"    : mel_mask_2d_dec,    # (B, T_mel) True=padding
        }

    @torch.no_grad()
    def infer(
        self,
        text_ids:         torch.Tensor,
        duration_control: float = DURATION_CONTROL,
        pitch_control:    float = PITCH_CONTROL,
        energy_control:   float = ENERGY_CONTROL,
    ) -> dict:
        """
        Convenience inference method.
        Switches to eval mode, disables gradients, uses predictor outputs.
        """
        self.eval()
        return self.forward(
            text_ids,
            duration_control = duration_control,
            pitch_control    = pitch_control,
            energy_control   = energy_control,
        )


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from config import PAD_ID

    B, T_text = 2, 20
    text_ids  = torch.randint(2, 80, (B, T_text))
    text_ids[0, 16:] = PAD_ID

    durations = torch.randint(4, 8, (B, T_text))
    mel_lens  = durations.sum(dim=1)
    T_mel_act = int(mel_lens.max().item())
    pitch     = torch.randn(B, T_mel_act)
    energy    = torch.randn(B, T_mel_act)

    model = FastSpeech2()
    out   = model(
        text_ids,
        durations     = durations,
        pitch_target  = pitch,
        energy_target = energy,
        mel_lens      = mel_lens,
        max_mel_len   = T_mel_act,
    )

    print(f"mel_out      : {out['mel_out'].shape}")      # (B, 80, T_mel)
    print(f"dur_pred     : {out['dur_pred'].shape}")     # (B, T_text)
    print(f"pitch_pred   : {out['pitch_pred'].shape}")   # (B, T_mel)
    print(f"energy_pred  : {out['energy_pred'].shape}")  # (B, T_mel)
    print(f"mel_lens     : {out['mel_lens']}")
    print(f"\nTotal parameters: {count_parameters(model):,}")
    print("FastSpeech2 self-test PASSED")
