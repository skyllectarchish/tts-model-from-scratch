"""
inference.py — Generate Gujarati speech from text using trained FastSpeech 2

Usage:
    python inference.py --text "આ ગુજરાતી ટેક્સ્ટ છે."
    python inference.py --text "નમસ્તે" --checkpoint ./checkpoints/checkpoint_epoch1000.pt
    python inference.py --text "ગુજરાત" --duration_control 0.8 --pitch_control 1.1

Prerequisites:
    1. A trained FastSpeech 2 checkpoint  (train.py)
    2. HiFi-GAN vocoder checkpoint + config  (see VOCODER_CHECKPOINT in config.py)
       Download: https://github.com/jik876/hifi-gan  →  LJ_FT_T2_V1 checkpoint
       Place at: ./vocoder/hifigan_checkpoint  and  ./vocoder/hifigan_config.json

Output:
    WAV file saved to OUTPUT_DIR/  (or path given via --output)
"""

import os
import sys
import argparse
import torch
import numpy as np
import soundfile as sf

from config import (
    DEVICE, OUTPUT_DIR, CHECKPOINT_DIR,
    VOCODER_CHECKPOINT, VOCODER_CONFIG,
    SAMPLE_RATE, DURATION_CONTROL, PITCH_CONTROL, ENERGY_CONTROL,
)
from data.text import text_to_ids
from model.fastspeech2 import FastSpeech2


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_fastspeech2(checkpoint_path: str) -> FastSpeech2:
    """Load trained FastSpeech 2 from a checkpoint file."""
    model = FastSpeech2().to(DEVICE)
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded FastSpeech2: {checkpoint_path}")
    return model


def load_hifigan(checkpoint_path: str, config_path: str):
    """
    Load HiFi-GAN vocoder from checkpoint.

    HiFi-GAN converts mel spectrograms → audio waveforms.

    Setup:
        git clone https://github.com/jik876/hifi-gan ./hifi-gan
        # Download the LJ_FT_T2_V1 weights from the repo releases
        cp generator_v1 ./vocoder/hifigan_checkpoint
        cp config_v1.json ./vocoder/hifigan_config.json

    If the hifi-gan repo is not on sys.path, add it below.
    """
    import json

    # Uncomment and set this path if hifi-gan is not installed as a package:
    # sys.path.insert(0, "./hifi-gan")

    try:
        from models import Generator
        from env import AttrDict
    except ImportError:
        raise ImportError(
            "\nHiFi-GAN not found on sys.path.\n"
            "Clone it: git clone https://github.com/jik876/hifi-gan\n"
            "Then either:\n"
            "  a) Run inference.py from inside the hifi-gan directory, or\n"
            "  b) Uncomment the sys.path.insert line near the top of load_hifigan()"
        )

    with open(config_path) as f:
        h = AttrDict(json.load(f))

    vocoder = Generator(h).to(DEVICE)
    ckpt    = torch.load(checkpoint_path, map_location=DEVICE)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    print(f"Loaded HiFi-GAN:   {checkpoint_path}")
    return vocoder


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(
    text:             str,
    model:            FastSpeech2,
    vocoder           = None,
    output_path:      str   = None,
    duration_control: float = DURATION_CONTROL,
    pitch_control:    float = PITCH_CONTROL,
    energy_control:   float = ENERGY_CONTROL,
) -> np.ndarray:
    """
    Convert Gujarati text → mel spectrogram → waveform.

    Args:
        text:             Input Gujarati text (will be cleaned automatically)
        model:            Trained FastSpeech2 model
        vocoder:          HiFi-GAN generator (or None → return mel only)
        output_path:      Where to save WAV; None = don't save
        duration_control: Speed  — < 1.0 faster, > 1.0 slower
        pitch_control:    Pitch  — < 1.0 lower,  > 1.0 higher
        energy_control:   Volume — < 1.0 quieter, > 1.0 louder

    Returns:
        If vocoder provided: 1D float32 audio waveform
        If no vocoder:       2D float32 mel spectrogram (n_mels, T)
    """
    print(f"\nText: '{text}'")

    # Text → token IDs
    ids = text_to_ids(text)
    if not ids:
        raise ValueError(f"Empty token sequence after cleaning: '{text}'")

    text_ids = torch.LongTensor(ids).unsqueeze(0).to(DEVICE)   # (1, T_text)
    print(f"Tokens: {len(ids)}")

    # FastSpeech2 → mel spectrogram
    with torch.no_grad():
        outputs = model.infer(
            text_ids,
            duration_control = duration_control,
            pitch_control    = pitch_control,
            energy_control   = energy_control,
        )

    mel     = outputs["mel_out"]                    # (1, n_mels, T_mel)
    mel_len = outputs["mel_lens"][0].item()
    mel     = mel[:, :, :mel_len]                   # trim padding
    print(f"Mel: {mel.shape}  ({mel_len} frames)")

    # Mel → waveform via HiFi-GAN
    if vocoder is not None:
        with torch.no_grad():
            wav = vocoder(mel).squeeze().cpu().numpy().astype(np.float32)

        duration_sec = len(wav) / SAMPLE_RATE
        print(f"Audio: {len(wav)} samples ({duration_sec:.2f}s)")

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            sf.write(output_path, wav, SAMPLE_RATE)
            print(f"Saved: {output_path}")

        return wav

    # No vocoder — return raw mel
    mel_np = mel.squeeze(0).cpu().numpy()           # (n_mels, T_mel)
    if output_path:
        npy_path = output_path.replace(".wav", "_mel.npy")
        np.save(npy_path, mel_np)
        print(f"Saved mel: {npy_path}")
    return mel_np


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint() -> str | None:
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    ckpts = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt"))
    return os.path.join(CHECKPOINT_DIR, ckpts[-1]) if ckpts else None


def _safe_filename(text: str, max_chars: int = 24) -> str:
    """Turn text into a safe filename stem."""
    safe = "".join(c for c in text[:max_chars] if c.isalnum() or c == " ").strip()
    return safe.replace(" ", "_") or "output"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gujarati TTS — FastSpeech 2 inference")
    parser.add_argument("--text",             type=str,   required=True,
                        help="Gujarati text to synthesize")
    parser.add_argument("--checkpoint",       type=str,   default=None,
                        help="Path to .pt checkpoint (default: latest in CHECKPOINT_DIR)")
    parser.add_argument("--output",           type=str,   default=None,
                        help="Output WAV path (default: OUTPUT_DIR/<text>.wav)")
    parser.add_argument("--duration_control", type=float, default=DURATION_CONTROL,
                        help="Speed control: <1 faster, >1 slower (default: 1.0)")
    parser.add_argument("--pitch_control",    type=float, default=PITCH_CONTROL,
                        help="Pitch control: <1 lower, >1 higher (default: 1.0)")
    parser.add_argument("--energy_control",   type=float, default=ENERGY_CONTROL,
                        help="Volume control: <1 quieter, >1 louder (default: 1.0)")
    parser.add_argument("--no_vocoder",       action="store_true",
                        help="Skip vocoder — save mel as .npy instead of .wav")
    args = parser.parse_args()

    # ---- Resolve checkpoint ----
    ckpt_path = args.checkpoint or _find_latest_checkpoint()
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("ERROR: No checkpoint found. Train the model first:\n  python train.py")
        sys.exit(1)

    # ---- Resolve output path ----
    output_path = args.output
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"{_safe_filename(args.text)}.wav")

    # ---- Load FastSpeech2 ----
    model = load_fastspeech2(ckpt_path)

    # ---- Load HiFi-GAN ----
    vocoder = None
    if not args.no_vocoder:
        if os.path.exists(VOCODER_CHECKPOINT) and os.path.exists(VOCODER_CONFIG):
            vocoder = load_hifigan(VOCODER_CHECKPOINT, VOCODER_CONFIG)
        else:
            print(
                "WARNING: HiFi-GAN not found.\n"
                f"  Expected checkpoint : {VOCODER_CHECKPOINT}\n"
                f"  Expected config     : {VOCODER_CONFIG}\n"
                "Saving mel spectrogram as .npy instead.\n"
                "Download HiFi-GAN from: https://github.com/jik876/hifi-gan"
            )

    # ---- Synthesize ----
    synthesize(
        args.text, model, vocoder,
        output_path      = output_path,
        duration_control = args.duration_control,
        pitch_control    = args.pitch_control,
        energy_control   = args.energy_control,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
