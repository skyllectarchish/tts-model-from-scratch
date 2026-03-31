"""
config.py — Central configuration for Gujarati FastSpeech 2 TTS
All hyperparameters, paths, and settings live here.
Change values here; everything else in the project reads from this file.
"""

import os

# ---------------------------------------------------------------------------
# Paths — update DATASET_PATH to point to your dataset folder
# ---------------------------------------------------------------------------
DATASET_PATH    = "/root/.cache/kagglehub/datasets/archishtesting/rasa-gujarati/versions/1/rasa_gujarati"       # root of your dataset
WAV_DIR         = os.path.join(DATASET_PATH, "wavs")      # folder containing .wav files
METADATA_FILE   = os.path.join(DATASET_PATH, "metadata.csv")  # pipe-separated CSV

PREPROCESSED_DIR = os.path.join(DATASET_PATH, "preprocessed")  # created during preprocessing
MEL_DIR          = os.path.join(PREPROCESSED_DIR, "mel")        # saved mel spectrograms
PITCH_DIR        = os.path.join(PREPROCESSED_DIR, "pitch")      # saved pitch (F0) values
ENERGY_DIR       = os.path.join(PREPROCESSED_DIR, "energy")     # saved energy values
DURATION_DIR     = os.path.join(PREPROCESSED_DIR, "duration")   # saved phoneme durations

CHECKPOINT_DIR  = "./checkpoints"   # model checkpoints saved here
LOG_DIR         = "./logs"          # TensorBoard logs
OUTPUT_DIR      = "./output"        # generated audio during inference


# ---------------------------------------------------------------------------
# Audio settings
# ---------------------------------------------------------------------------
SAMPLE_RATE     = 22050    # Hz — standard for TTS; resample your audio if different
HOP_LENGTH      = 256      # STFT hop size in samples (~11.6ms at 22050Hz)
WIN_LENGTH      = 1024     # STFT window size in samples
N_FFT           = 1024     # FFT size — same as WIN_LENGTH
N_MELS          = 80       # number of mel filter banks — 80 is standard for TTS
MEL_FMIN        = 0.0      # minimum frequency for mel filterbank (Hz)
MEL_FMAX        = 8000.0   # maximum frequency for mel filterbank (Hz)
MAX_WAV_VALUE   = 32768.0  # for normalizing 16-bit PCM audio

# Pitch (F0) extraction settings
PITCH_MIN       = 50.0     # Hz — lowest expected fundamental frequency
PITCH_MAX       = 600.0    # Hz — highest expected fundamental frequency
# Gujarati speakers typically fall in 80–400Hz range


# ---------------------------------------------------------------------------
# Gujarati character set
# ---------------------------------------------------------------------------
# All characters the model can handle as input.
# Gujarati Unicode block: U+0A80 to U+0AFF
# We include the most common characters + punctuation + space.

PAD_TOKEN   = "<pad>"   # padding token (index 0)
UNK_TOKEN   = "<unk>"   # unknown character token

GUJARATI_CHARS = [
    # Vowels (સ્વર)
    'અ', 'આ', 'ઇ', 'ઈ', 'ઉ', 'ઊ', 'ઋ', 'એ', 'ઐ', 'ઓ', 'ઔ', 'અં', 'અઃ',
    # Vowel diacritics (માત્રા)
    'ા', 'િ', 'ી', 'ુ', 'ૂ', 'ૃ', 'ે', 'ૈ', 'ો', 'ૌ', 'ં', 'ઃ', '્',
    # Consonants (વ્યંજન)
    'ક', 'ખ', 'ગ', 'ઘ', 'ઙ',
    'ચ', 'છ', 'જ', 'ઝ', 'ઞ',
    'ટ', 'ઠ', 'ડ', 'ઢ', 'ણ',
    'ત', 'થ', 'દ', 'ધ', 'ન',
    'પ', 'ફ', 'બ', 'ભ', 'મ',
    'ય', 'ર', 'લ', 'વ', 'શ',
    'ષ', 'સ', 'હ', 'ળ', 'ક્ષ', 'જ્ઞ',
    # Digits
    '૦', '૧', '૨', '૩', '૪', '૫', '૬', '૭', '૮', '૯',
    # Punctuation and space
    ' ', ',', '.', '!', '?', '-', '–', '।', '॥', '"', "'",
]

# Build lookup tables
VOCAB = [PAD_TOKEN, UNK_TOKEN] + GUJARATI_CHARS
CHAR_TO_ID = {ch: i for i, ch in enumerate(VOCAB)}
ID_TO_CHAR = {i: ch for i, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

PAD_ID = CHAR_TO_ID[PAD_TOKEN]
UNK_ID = CHAR_TO_ID[UNK_TOKEN]


# ---------------------------------------------------------------------------
# Model architecture — FastSpeech 2
# ---------------------------------------------------------------------------

# Encoder
ENCODER_HIDDEN_DIM      = 256    # hidden dimension throughout the model
ENCODER_N_LAYERS        = 4      # number of Transformer encoder layers
ENCODER_N_HEADS         = 2      # number of attention heads
ENCODER_CONV_FILTER_SIZE = 1024  # FFN inner dimension in each Transformer layer
ENCODER_CONV_KERNEL_SIZE = (9, 1) # kernel sizes for FFN convolutions
ENCODER_DROPOUT         = 0.2    # dropout rate

# Decoder (same structure as encoder)
DECODER_HIDDEN_DIM      = 256
DECODER_N_LAYERS        = 4
DECODER_N_HEADS         = 2
DECODER_CONV_FILTER_SIZE = 1024
DECODER_CONV_KERNEL_SIZE = (9, 1)
DECODER_DROPOUT         = 0.2

# Variance adaptor (duration / pitch / energy predictors)
# Each predictor is a small 2-layer conv network
VARIANCE_PREDICTOR_FILTER_SIZE  = 256
VARIANCE_PREDICTOR_KERNEL_SIZE  = 3
VARIANCE_PREDICTOR_DROPOUT      = 0.5

# Pitch and energy are embedded back into the hidden dim
PITCH_EMBEDDING_DIM  = 256
ENERGY_EMBEDDING_DIM = 256

# Number of buckets for pitch/energy quantization
# Pitch and energy are binned into N buckets, then looked up as embeddings
N_PITCH_BINS  = 256
N_ENERGY_BINS = 256

# Output linear layer: maps hidden_dim → n_mels
MEL_LINEAR_DIM = N_MELS


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 1000   # maximum input text length (characters)
MAX_MEL_LEN = 1500   # maximum output mel spectrogram length (frames)


# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
BATCH_SIZE          = 16      # reduce to 8 if you get OOM errors
NUM_WORKERS         = 4       # DataLoader workers (CPU cores for data loading)
EPOCHS              = 1000    # total training epochs
SAVE_EVERY               = 50   # save checkpoint every N epochs
KEEP_LAST_N_CHECKPOINTS  = 5    # how many checkpoints to keep on disk:
                                 #   -1 → keep ALL checkpoints (unlimited)
                                 #    0 → don't save any checkpoints at all
                                 #   >0 → keep only the last N (older ones are deleted)
LOG_EVERY           = 10      # log to TensorBoard every N steps
VAL_EVERY           = 50      # run validation every N epochs

# Optimizer (Adam with warm-up + decay)
LEARNING_RATE       = 1e-3
BETAS               = (0.9, 0.98)   # Adam beta values (standard Transformer values)
EPS                 = 1e-9
WEIGHT_DECAY        = 0.0

# Learning rate scheduler — Noam scheduler (same as original Transformer paper)
WARMUP_STEPS        = 4000    # linearly ramp LR for this many steps, then decay

# Gradient clipping — prevents exploding gradients
GRAD_CLIP_THRESH    = 1.0

# Mixed precision training — uses bfloat16 on your RTX 5060 Ti → ~2x faster
USE_AMP             = True
AMP_DTYPE           = "bfloat16"   # bfloat16 is more stable than float16 on Blackwell


# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------
# FastSpeech 2 optimizes 4 losses simultaneously.
# These weights balance their relative contribution.

MEL_LOSS_WEIGHT      = 1.0   # main mel spectrogram reconstruction loss
DURATION_LOSS_WEIGHT = 1.0   # duration predictor loss
PITCH_LOSS_WEIGHT    = 1.0   # pitch predictor loss
ENERGY_LOSS_WEIGHT   = 1.0   # energy predictor loss


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
TRAIN_SPLIT = 0.95   # 95% of data used for training
VAL_SPLIT   = 0.05   # 5% used for validation


# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------
# These control how the model generates audio at inference time.

# Scale factors — 1.0 = natural speed/pitch/volume
# Change these to control output style:
DURATION_CONTROL = 1.0   # < 1.0 = faster speech, > 1.0 = slower
PITCH_CONTROL    = 1.0   # < 1.0 = lower pitch, > 1.0 = higher pitch
ENERGY_CONTROL   = 1.0   # < 1.0 = quieter, > 1.0 = louder

# Vocoder — we use a pretrained HiFi-GAN
# Download from: https://github.com/jik876/hifi-gan (LJ_FT_T2_V1 checkpoint)
VOCODER_CHECKPOINT = "./vocoder/hifigan_checkpoint"
VOCODER_CONFIG     = "./vocoder/hifigan_config.json"


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Sanity check — print config summary when this file is run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  Gujarati FastSpeech 2 — Configuration Summary")
    print("=" * 55)
    print(f"  Device          : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM            : {vram:.1f} GB")
    print(f"  Vocab size      : {VOCAB_SIZE} characters")
    print(f"  Sample rate     : {SAMPLE_RATE} Hz")
    print(f"  Mel bins        : {N_MELS}")
    print(f"  Encoder layers  : {ENCODER_N_LAYERS}")
    print(f"  Decoder layers  : {DECODER_N_LAYERS}")
    print(f"  Hidden dim      : {ENCODER_HIDDEN_DIM}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Mixed precision : {USE_AMP} ({AMP_DTYPE})")
    print(f"  Dataset path    : {DATASET_PATH}")
    print("=" * 55)
    print(f"\n  Sample vocab entries:")
    for i, ch in enumerate(VOCAB[:10]):
        print(f"    [{i:3d}] '{ch}'")
    print(f"    ... ({VOCAB_SIZE} total)")
