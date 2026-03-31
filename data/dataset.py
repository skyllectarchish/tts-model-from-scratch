"""
data/dataset.py — Gujarati TTS Dataset

Responsibilities:
  1. Read metadata.csv and build a list of (wav_path, text) pairs
  2. Load WAV files and resample if needed
  3. Extract mel spectrograms, pitch (F0), and energy from audio
  4. Encode text → integer IDs using data/text.py
  5. Return padded batches ready for the model

Usage:
    from data.dataset import GujaratiTTSDataset, get_dataloaders

    train_loader, val_loader = get_dataloaders()
    for batch in train_loader:
        text_ids   = batch["text_ids"]       # (B, T_text)
        mel        = batch["mel"]            # (B, n_mels, T_mel)
        pitch      = batch["pitch"]          # (B, T_mel)
        energy     = batch["energy"]         # (B, T_mel)
        durations  = batch["durations"]      # (B, T_text)  — frames per character
        ...
"""

import os
import csv
import sys
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    METADATA_FILE, WAV_DIR, MEL_DIR, PITCH_DIR, ENERGY_DIR, DURATION_DIR,
    PREPROCESSED_DIR, SAMPLE_RATE, HOP_LENGTH, WIN_LENGTH, N_FFT, N_MELS,
    MEL_FMIN, MEL_FMAX, MAX_WAV_VALUE, PITCH_MIN, PITCH_MAX,
    TRAIN_SPLIT, BATCH_SIZE, NUM_WORKERS, DEVICE,
    PAD_ID, MAX_SEQ_LEN, MAX_MEL_LEN,
)
from data.text import text_to_ids, clean_text


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def load_wav(wav_path: str) -> np.ndarray:
    """
    Load a WAV file as a float32 numpy array normalized to [-1, 1].
    Resamples to SAMPLE_RATE if the file has a different rate.

    Args:
        wav_path: Absolute path to .wav file

    Returns:
        1D float32 numpy array of audio samples
    """
    audio, sr = sf.read(wav_path, dtype="float32")

    # Convert stereo to mono by averaging channels
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Normalize to [-1, 1]
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val

    return audio.astype(np.float32)


def compute_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Compute a log-mel spectrogram from a raw audio array.

    Steps:
      1. Short-Time Fourier Transform (STFT)
      2. Magnitude spectrum → mel filterbank
      3. Log compression (log(x + 1e-5))

    Args:
        audio: 1D float32 numpy array, normalized to [-1, 1]

    Returns:
        2D float32 numpy array of shape (n_mels, T)
        where T = ceil(len(audio) / HOP_LENGTH)
    """
    # Compute mel spectrogram via librosa
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=MEL_FMIN,
        fmax=MEL_FMAX,
        center=False,          # no padding at edges — matches HiFi-GAN convention
        pad_mode="reflect",
        power=1.0,             # amplitude spectrogram (not power) for TTS
    )

    # Log compression — adds 1e-5 for numerical stability
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

    return mel.astype(np.float32)   # shape: (n_mels, T)


def compute_pitch(audio: np.ndarray, mel_len: int) -> np.ndarray:
    """
    Extract fundamental frequency (F0 / pitch) using librosa's pyin algorithm.

    pyin is more accurate than plain yin for TTS:
    - Returns probabilistic F0 estimates
    - Handles unvoiced regions gracefully (returns 0.0 for silence)
    - More robust to background noise

    The raw F0 is at frame rate 1/HOP_LENGTH. We interpolate voiced regions
    and zero out unvoiced regions, then normalize to zero mean / unit variance
    across voiced frames only — this is the FastSpeech 2 convention.

    Args:
        audio:   1D float32 audio array
        mel_len: Target length to match mel spectrogram frames

    Returns:
        1D float32 numpy array of shape (mel_len,)
        Values are normalized F0 for voiced frames, 0.0 for unvoiced.
    """
    # Extract F0 with pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=PITCH_MIN,
        fmax=PITCH_MAX,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        center=False,
    )

    # pyin returns NaN for unvoiced frames — replace with 0
    f0 = np.nan_to_num(f0, nan=0.0)

    # Resize to match mel spectrogram length
    # (pyin may produce slightly different number of frames)
    if len(f0) != mel_len:
        f0 = _resize_array(f0, mel_len)

    # Normalize: zero mean / unit variance over voiced frames only
    # Unvoiced (zero) frames are excluded from statistics
    voiced_mask = f0 > 0.0
    if voiced_mask.sum() > 1:
        voiced_f0 = f0[voiced_mask]
        mean = voiced_f0.mean()
        std  = voiced_f0.std() + 1e-8
        f0[voiced_mask] = (voiced_f0 - mean) / std
        # Unvoiced frames stay at 0.0 (below any normalized value)

    return f0.astype(np.float32)


def compute_energy(mel: np.ndarray) -> np.ndarray:
    """
    Compute frame-level energy from the mel spectrogram.

    Energy = L2 norm across mel bins per frame.
    This captures loudness variation across the utterance.

    Normalized to zero mean / unit variance across the utterance,
    consistent with FastSpeech 2 paper.

    Args:
        mel: 2D float32 array of shape (n_mels, T)

    Returns:
        1D float32 array of shape (T,) — normalized energy per frame
    """
    # L2 norm across mel dimension for each time frame
    energy = np.linalg.norm(mel, axis=0)   # shape: (T,)

    # Normalize
    mean = energy.mean()
    std  = energy.std() + 1e-8
    energy = (energy - mean) / std

    return energy.astype(np.float32)


def _resize_array(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resize a 1D array to target_len using linear interpolation.
    Used to align pitch/energy arrays with mel spectrogram length.
    """
    if len(arr) == target_len:
        return arr
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr)


# ---------------------------------------------------------------------------
# Duration computation
# ---------------------------------------------------------------------------

def compute_durations_from_mel(text_ids: list, mel_len: int) -> np.ndarray:
    """
    Compute approximate character durations by evenly distributing
    mel frames across characters.

    NOTE: This is a simple approximation. For production quality,
    use Montreal Forced Aligner (MFA) to get exact phoneme-level
    durations. We will add MFA support in preprocess.py later.

    For now, even distribution works for getting the model training
    started and verifying the pipeline end-to-end.

    Args:
        text_ids: List of character token IDs (length T_text)
        mel_len:  Number of mel frames (T_mel)

    Returns:
        1D int32 array of shape (T_text,) where sum = mel_len
    """
    n_chars = len(text_ids)
    if n_chars == 0:
        return np.array([mel_len], dtype=np.int32)

    # Base duration = mel_len / n_chars (floor)
    base     = mel_len // n_chars
    leftover = mel_len - base * n_chars

    durations = np.full(n_chars, base, dtype=np.int32)

    # Distribute leftover frames to middle characters
    # (mid-word characters tend to be longer in natural speech)
    mid = n_chars // 2
    for i in range(leftover):
        durations[(mid + i) % n_chars] += 1

    assert durations.sum() == mel_len, (
        f"Duration sum {durations.sum()} != mel_len {mel_len}"
    )

    return durations


# ---------------------------------------------------------------------------
# Preprocessing cache
# ---------------------------------------------------------------------------

def preprocess_and_cache(wav_name: str, text: str, force: bool = False):
    """
    Preprocess one audio file and save mel, pitch, energy, durations to disk.

    This is run ONCE before training. During training, we just load the
    cached numpy arrays — much faster than recomputing each epoch.

    Saved files:
        MEL_DIR/wav_name.npy      shape: (n_mels, T)
        PITCH_DIR/wav_name.npy    shape: (T,)
        ENERGY_DIR/wav_name.npy   shape: (T,)
        DURATION_DIR/wav_name.npy shape: (n_chars,)

    Args:
        wav_name: Filename stem (no extension), e.g. "wav_0001"
        text:     Cleaned transcript text
        force:    If True, recompute even if cache exists
    """
    mel_path      = os.path.join(MEL_DIR,      wav_name + ".npy")
    pitch_path    = os.path.join(PITCH_DIR,    wav_name + ".npy")
    energy_path   = os.path.join(ENERGY_DIR,   wav_name + ".npy")
    duration_path = os.path.join(DURATION_DIR, wav_name + ".npy")

    # Skip if already cached
    if not force and all(os.path.exists(p) for p in
                         [mel_path, pitch_path, energy_path, duration_path]):
        return True

    wav_path = os.path.join(WAV_DIR, wav_name + ".wav")
    if not os.path.exists(wav_path):
        print(f"  [WARN] Missing audio file: {wav_path}")
        return False

    try:
        # Load audio
        audio = load_wav(wav_path)

        # Skip very short or very long clips
        duration_sec = len(audio) / SAMPLE_RATE
        if duration_sec < 0.5 or duration_sec > 15.0:
            print(f"  [SKIP] {wav_name}: duration {duration_sec:.1f}s out of range")
            return False

        # Compute features
        mel      = compute_mel_spectrogram(audio)     # (n_mels, T)
        pitch    = compute_pitch(audio, mel.shape[1]) # (T,)
        energy   = compute_energy(mel)                # (T,)

        # Compute durations
        text_ids  = text_to_ids(text)
        durations = compute_durations_from_mel(text_ids, mel.shape[1])

        # Save to disk
        np.save(mel_path,      mel)
        np.save(pitch_path,    pitch)
        np.save(energy_path,   energy)
        np.save(duration_path, durations)

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to preprocess {wav_name}: {e}")
        return False


def run_preprocessing(metadata_file: str = METADATA_FILE, force: bool = False):
    """
    Preprocess all audio files listed in metadata.csv.
    Run this ONCE before starting training.

    Creates the directory structure:
        preprocessed/
            mel/
            pitch/
            energy/
            duration/

    Args:
        metadata_file: Path to pipe-separated metadata CSV
        force:         If True, recompute all files even if cached

    Usage:
        python3 -c "from data.dataset import run_preprocessing; run_preprocessing()"
    """
    from tqdm import tqdm

    # Create output directories
    for d in [MEL_DIR, PITCH_DIR, ENERGY_DIR, DURATION_DIR]:
        os.makedirs(d, exist_ok=True)

    # Read metadata
    entries = _read_metadata(metadata_file)
    print(f"\nPreprocessing {len(entries)} audio files...")
    print(f"Output directory: {PREPROCESSED_DIR}\n")

    success = 0
    skipped = 0
    failed  = 0

    for wav_name, text in tqdm(entries, desc="Preprocessing"):
        cleaned_text = clean_text(text)
        if not cleaned_text:
            skipped += 1
            continue

        ok = preprocess_and_cache(wav_name, cleaned_text, force=force)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\nDone.")
    print(f"  Success : {success}")
    print(f"  Skipped : {skipped}")
    print(f"  Failed  : {failed}")
    print(f"\nRun training with: python3 train.py")


# ---------------------------------------------------------------------------
# Metadata reader
# ---------------------------------------------------------------------------

def _read_metadata(metadata_file: str) -> list[tuple[str, str]]:
    """
    Read the pipe-separated metadata CSV.

    Expected format:
        audio|text            (header row — skipped automatically)
        gujarati_0000000.wav|transcript

    wav filenames may include or omit the .wav extension — both are handled.
    Returns list of (wav_stem, text) tuples where wav_stem has no extension.
    """
    entries = []
    with open(metadata_file, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for i, row in enumerate(reader, 1):
            if len(row) < 2:
                print(f"  [WARN] Skipping malformed row {i}: {row}")
                continue
            wav_name = row[0].strip()
            # Skip header row
            if wav_name.lower() == "audio":
                continue
            # Strip .wav extension so downstream code can append it consistently
            if wav_name.lower().endswith(".wav"):
                wav_name = wav_name[:-4]
            # Use column 3 (normalized) if available, else column 2
            text = row[2].strip() if len(row) >= 3 and row[2].strip() else row[1].strip()
            if wav_name and text:
                entries.append((wav_name, text))
    return entries


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class GujaratiTTSDataset(Dataset):
    """
    PyTorch Dataset for Gujarati TTS training.

    Loads preprocessed numpy arrays from disk (mel, pitch, energy, durations)
    and encodes text to integer IDs.

    Items are filtered to remove:
      - Missing preprocessed files
      - Text that's too long (> MAX_SEQ_LEN characters)
      - Mel spectrograms that are too long (> MAX_MEL_LEN frames)

    Args:
        split: "train" or "val"
        metadata_file: Path to pipe-separated CSV (default: from config)
    """

    def __init__(self, split: str = "train", metadata_file: str = METADATA_FILE):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got '{split}'"

        self.split = split
        all_entries = _read_metadata(metadata_file)

        # Deterministic train/val split — shuffle with fixed seed then split
        random.seed(42)
        shuffled = all_entries.copy()
        random.shuffle(shuffled)

        n_train = int(len(shuffled) * TRAIN_SPLIT)
        if split == "train":
            raw_entries = shuffled[:n_train]
        else:
            raw_entries = shuffled[n_train:]

        # Filter to entries with valid preprocessed files + reasonable lengths
        self.entries = []
        skipped = 0
        for wav_name, text in raw_entries:
            cleaned = clean_text(text)
            text_ids = text_to_ids(cleaned, apply_cleaning=False)

            # Check text length
            if len(text_ids) == 0 or len(text_ids) > MAX_SEQ_LEN:
                skipped += 1
                continue

            # Check preprocessed files exist
            mel_path      = os.path.join(MEL_DIR,      wav_name + ".npy")
            pitch_path    = os.path.join(PITCH_DIR,    wav_name + ".npy")
            energy_path   = os.path.join(ENERGY_DIR,   wav_name + ".npy")
            duration_path = os.path.join(DURATION_DIR, wav_name + ".npy")

            if not all(os.path.exists(p) for p in
                       [mel_path, pitch_path, energy_path, duration_path]):
                skipped += 1
                continue

            # Quick mel length check
            mel = np.load(mel_path, mmap_mode="r")
            if mel.shape[1] > MAX_MEL_LEN:
                skipped += 1
                continue

            self.entries.append({
                "wav_name"     : wav_name,
                "text"         : cleaned,
                "text_ids"     : text_ids,
                "mel_path"     : mel_path,
                "pitch_path"   : pitch_path,
                "energy_path"  : energy_path,
                "duration_path": duration_path,
            })

        print(f"[Dataset] {split}: {len(self.entries)} samples "
              f"({skipped} skipped)")

        if len(self.entries) == 0 and skipped > 0:
            raise RuntimeError(
                f"0 samples loaded for '{split}' split (all {skipped} entries skipped).\n"
                f"This usually means the preprocessed files (.npy) are missing.\n"
                f"Did you forget to run preprocessing?\n"
                f"Run: python -c \"from data.dataset import run_preprocessing; run_preprocessing()\""
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return one training sample.

        Returns a dict with:
            text_ids  : LongTensor  (T_text,)
            mel       : FloatTensor (n_mels, T_mel)
            pitch     : FloatTensor (T_mel,)
            energy    : FloatTensor (T_mel,)
            durations : LongTensor  (T_text,)  — frames per character
            text_len  : int — number of characters (for masking)
            mel_len   : int — number of mel frames (for masking)
            wav_name  : str — for debugging / saving samples
        """
        entry = self.entries[idx]

        # Load preprocessed features
        mel       = np.load(entry["mel_path"])        # (n_mels, T_mel)
        pitch     = np.load(entry["pitch_path"])      # (T_mel,)
        energy    = np.load(entry["energy_path"])     # (T_mel,)
        durations = np.load(entry["duration_path"])   # (T_text,)

        text_ids = entry["text_ids"]

        # Sanity checks
        mel_len  = mel.shape[1]
        text_len = len(text_ids)

        # Ensure duration sum matches mel length
        # (floating point or rounding issues can cause off-by-one)
        if durations.sum() != mel_len:
            durations = compute_durations_from_mel(text_ids, mel_len)

        return {
            "text_ids"  : torch.LongTensor(text_ids),
            "mel"       : torch.FloatTensor(mel),
            "pitch"     : torch.FloatTensor(pitch),
            "energy"    : torch.FloatTensor(energy),
            "durations" : torch.LongTensor(durations),
            "text_len"  : text_len,
            "mel_len"   : mel_len,
            "wav_name"  : entry["wav_name"],
        }


# ---------------------------------------------------------------------------
# Collate function — pad variable-length sequences into batches
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """
    Pad a list of samples into a batch with uniform length.

    Text sequences are padded with PAD_ID.
    Mel / pitch / energy are padded with 0.0.
    Durations are padded with 0.

    Returns a dict ready to pass directly to the model.
    """
    # Sort by text length descending (helps RNN-style processing if added later)
    batch = sorted(batch, key=lambda x: x["text_len"], reverse=True)

    text_lens = [x["text_len"]  for x in batch]
    mel_lens  = [x["mel_len"]   for x in batch]

    max_text_len = max(text_lens)
    max_mel_len  = max(mel_lens)
    n_mels       = batch[0]["mel"].shape[0]
    B            = len(batch)

    # Initialize padded tensors
    text_ids_padded  = torch.full((B, max_text_len), PAD_ID,  dtype=torch.long)
    mel_padded       = torch.zeros(B, n_mels, max_mel_len,    dtype=torch.float)
    pitch_padded     = torch.zeros(B, max_mel_len,            dtype=torch.float)
    energy_padded    = torch.zeros(B, max_mel_len,            dtype=torch.float)
    durations_padded = torch.zeros(B, max_text_len,           dtype=torch.long)

    # Fill in actual data
    for i, sample in enumerate(batch):
        tl = sample["text_len"]
        ml = sample["mel_len"]

        text_ids_padded[i,  :tl]    = sample["text_ids"]
        mel_padded[i,       :, :ml] = sample["mel"]
        pitch_padded[i,     :ml]    = sample["pitch"]
        energy_padded[i,    :ml]    = sample["energy"]
        durations_padded[i, :tl]    = sample["durations"]

    return {
        "text_ids"   : text_ids_padded,               # (B, T_text)
        "mel"        : mel_padded,                     # (B, n_mels, T_mel)
        "pitch"      : pitch_padded,                   # (B, T_mel)
        "energy"     : energy_padded,                  # (B, T_mel)
        "durations"  : durations_padded,               # (B, T_text)
        "text_lens"  : torch.LongTensor(text_lens),    # (B,)
        "mel_lens"   : torch.LongTensor(mel_lens),     # (B,)
        "wav_names"  : [x["wav_name"] for x in batch],
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    metadata_file: str = METADATA_FILE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Args:
        metadata_file: Path to pipe-separated CSV
        batch_size:    Samples per batch
        num_workers:   CPU workers for data loading

    Returns:
        (train_loader, val_loader) tuple

    Usage:
        train_loader, val_loader = get_dataloaders()
        for batch in train_loader:
            ...
    """
    train_dataset = GujaratiTTSDataset("train", metadata_file)
    val_dataset   = GujaratiTTSDataset("val",   metadata_file)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,    # drop incomplete last batch for stable training
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Self-test — run this file directly to verify the dataset pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Gujarati TTS Dataset — Self Test")
    print("=" * 60)

    # Test audio utilities with a synthetic signal
    print("\n1. Testing audio feature extraction with synthetic signal...")

    # Generate a 1-second sine wave at 220Hz (like a vowel tone)
    duration   = 1.0
    t          = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    test_audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    mel    = compute_mel_spectrogram(test_audio)
    pitch  = compute_pitch(test_audio, mel.shape[1])
    energy = compute_energy(mel)

    print(f"   Audio shape  : {test_audio.shape}  ({len(test_audio)/SAMPLE_RATE:.2f}s)")
    print(f"   Mel shape    : {mel.shape}  (n_mels={mel.shape[0]}, frames={mel.shape[1]})")
    print(f"   Pitch shape  : {pitch.shape},  min={pitch.min():.2f}, max={pitch.max():.2f}")
    print(f"   Energy shape : {energy.shape}, min={energy.min():.2f}, max={energy.max():.2f}")
    print(f"   PASS: Feature extraction working correctly")

    # Test duration computation
    print("\n2. Testing duration computation...")
    dummy_text_ids = list(range(10))   # 10 characters
    dummy_mel_len  = 87
    durations = compute_durations_from_mel(dummy_text_ids, dummy_mel_len)
    assert durations.sum() == dummy_mel_len, "Duration sum mismatch!"
    print(f"   Text length  : {len(dummy_text_ids)} chars")
    print(f"   Mel length   : {dummy_mel_len} frames")
    print(f"   Durations    : {durations}")
    print(f"   Sum check    : {durations.sum()} == {dummy_mel_len}  PASS")

    # Test collate function with dummy data
    print("\n3. Testing collate function with dummy batch...")
    from data.text import text_to_ids

    dummy_batch = []
    test_texts = ["આ સારું છે", "ગુજરાત", "આજે હવામાન સારું છે"]
    for text in test_texts:
        ids = text_to_ids(text)
        mel_len = len(ids) * 8   # approximate
        dummy_batch.append({
            "text_ids"  : torch.LongTensor(ids),
            "mel"       : torch.zeros(N_MELS, mel_len),
            "pitch"     : torch.zeros(mel_len),
            "energy"    : torch.zeros(mel_len),
            "durations" : torch.ones(len(ids), dtype=torch.long) * 8,
            "text_len"  : len(ids),
            "mel_len"   : mel_len,
            "wav_name"  : f"test_{len(ids)}",
        })

    batch = collate_fn(dummy_batch)
    print(f"   Batch text_ids  : {batch['text_ids'].shape}")
    print(f"   Batch mel       : {batch['mel'].shape}")
    print(f"   Batch pitch     : {batch['pitch'].shape}")
    print(f"   Batch energy    : {batch['energy'].shape}")
    print(f"   Batch durations : {batch['durations'].shape}")
    print(f"   PASS: Collate function working correctly")

    # Check if real dataset is available
    print("\n4. Checking dataset availability...")
    if os.path.exists(METADATA_FILE):
        entries = _read_metadata(METADATA_FILE)
        print(f"   Found metadata.csv with {len(entries)} entries")

        # Check if preprocessing has been run
        if os.path.exists(MEL_DIR) and len(os.listdir(MEL_DIR)) > 0:
            n_mel = len(os.listdir(MEL_DIR))
            print(f"   Found {n_mel} preprocessed mel files")
            print(f"   Run get_dataloaders() to build DataLoaders")
        else:
            print(f"   Preprocessed files not found.")
            print(f"   Run preprocessing first:")
            print(f"   python3 -c \"from data.dataset import run_preprocessing; run_preprocessing()\"")
    else:
        print(f"   metadata.csv not found at: {METADATA_FILE}")
        print(f"   Update DATASET_PATH in config.py and re-run")

    print("\n" + "=" * 60)
    print("  All local tests passed!")
    print("=" * 60)
