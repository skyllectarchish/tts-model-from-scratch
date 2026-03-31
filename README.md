# Gujarati FastSpeech 2 — Text-to-Speech from Scratch

A complete, fully hand-coded implementation of the **FastSpeech 2** text-to-speech model
trained exclusively on the **Gujarati language**. Every component — architecture, training loop,
data pipeline, text processor, and inference engine — is implemented from scratch in pure PyTorch.

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Why FastSpeech 2](#2-why-fastspeech-2)
3. [Project Structure](#3-project-structure)
4. [Architecture Overview](#4-architecture-overview)
   - 4.1 [High-Level Data Flow](#41-high-level-data-flow)
   - 4.2 [Encoder](#42-encoder)
   - 4.3 [Variance Adaptor](#43-variance-adaptor)
   - 4.4 [Decoder](#44-decoder)
   - 4.5 [Vocoder — HiFi-GAN](#45-vocoder--hifi-gan)
5. [Gujarati Language Handling](#5-gujarati-language-handling)
   - 5.1 [Character Vocabulary](#51-character-vocabulary)
   - 5.2 [Text Cleaning Pipeline](#52-text-cleaning-pipeline)
   - 5.3 [Number Expansion](#53-number-expansion)
   - 5.4 [Abbreviation Expansion](#54-abbreviation-expansion)
6. [Dataset Format](#6-dataset-format)
   - 6.1 [metadata.csv](#61-metadatacsv)
   - 6.2 [WAV Files](#62-wav-files)
7. [Audio Feature Extraction](#7-audio-feature-extraction)
   - 7.1 [Mel Spectrogram](#71-mel-spectrogram)
   - 7.2 [Pitch (F0) Extraction](#72-pitch-f0-extraction)
   - 7.3 [Energy Extraction](#73-energy-extraction)
   - 7.4 [Duration Computation](#74-duration-computation)
8. [Configuration Reference](#8-configuration-reference)
   - 8.1 [Paths](#81-paths)
   - 8.2 [Audio Settings](#82-audio-settings)
   - 8.3 [Character Vocabulary Settings](#83-character-vocabulary-settings)
   - 8.4 [Model Architecture Settings](#84-model-architecture-settings)
   - 8.5 [Training Settings](#85-training-settings)
   - 8.6 [Loss Weights](#86-loss-weights)
   - 8.7 [Data Split](#87-data-split)
   - 8.8 [Inference Settings](#88-inference-settings)
9. [Installation](#9-installation)
   - 9.1 [System Requirements](#91-system-requirements)
   - 9.2 [Python Dependencies](#92-python-dependencies)
   - 9.3 [PyTorch with CUDA](#93-pytorch-with-cuda)
10. [Step-by-Step Usage Guide](#10-step-by-step-usage-guide)
    - 10.1 [Step 1 — Prepare Your Dataset](#101-step-1--prepare-your-dataset)
    - 10.2 [Step 2 — Update config.py](#102-step-2--update-configpy)
    - 10.3 [Step 3 — Verify Text Pipeline](#103-step-3--verify-text-pipeline)
    - 10.4 [Step 4 — Preprocess Audio](#104-step-4--preprocess-audio)
    - 10.5 [Step 5 — Verify Config](#105-step-5--verify-config)
    - 10.6 [Step 6 — Train the Model](#106-step-6--train-the-model)
    - 10.7 [Step 7 — Monitor Training](#107-step-7--monitor-training)
    - 10.8 [Step 8 — Set Up HiFi-GAN Vocoder](#108-step-8--set-up-hifi-gan-vocoder)
    - 10.9 [Step 9 — Run Inference](#109-step-9--run-inference)
11. [File-by-File Code Reference](#11-file-by-file-code-reference)
    - 11.1 [config.py](#111-configpy)
    - 11.2 [data/text.py](#112-datatextpy)
    - 11.3 [data/dataset.py](#113-datadatasetpy)
    - 11.4 [model/encoder.py](#114-modelencoderpy)
    - 11.5 [model/variance_adaptor.py](#115-modelvariance_adaptorpy)
    - 11.6 [model/decoder.py](#116-modeldecoderpy)
    - 11.7 [model/fastspeech2.py](#117-modelfastspeech2py)
    - 11.8 [train.py](#118-trainpy)
    - 11.9 [inference.py](#119-inferencepy)
12. [Training Deep Dive](#12-training-deep-dive)
    - 12.1 [Loss Functions](#121-loss-functions)
    - 12.2 [Noam Learning Rate Scheduler](#122-noam-learning-rate-scheduler)
    - 12.3 [Mixed Precision Training](#123-mixed-precision-training)
    - 12.4 [Gradient Clipping](#124-gradient-clipping)
    - 12.5 [Checkpointing and Resuming](#125-checkpointing-and-resuming)
    - 12.6 [TensorBoard Logging](#126-tensorboard-logging)
13. [Inference Deep Dive](#13-inference-deep-dive)
    - 13.1 [Duration Control](#131-duration-control)
    - 13.2 [Pitch Control](#132-pitch-control)
    - 13.3 [Energy Control](#133-energy-control)
    - 13.4 [Running Without a Vocoder](#134-running-without-a-vocoder)
14. [Self-Tests](#14-self-tests)
15. [Troubleshooting](#15-troubleshooting)
    - 15.1 [CUDA / GPU Issues](#151-cuda--gpu-issues)
    - 15.2 [OOM (Out of Memory) Errors](#152-oom-out-of-memory-errors)
    - 15.3 [Preprocessing Failures](#153-preprocessing-failures)
    - 15.4 [Training Divergence](#154-training-divergence)
    - 15.5 [Bad Audio Quality](#155-bad-audio-quality)
    - 15.6 [Import Errors](#156-import-errors)
16. [Extending the Project](#16-extending-the-project)
    - 16.1 [Adding MFA Alignments](#161-adding-mfa-alignments)
    - 16.2 [Multi-Speaker Support](#162-multi-speaker-support)
    - 16.3 [Emotion / Style Control](#163-emotion--style-control)
    - 16.4 [Fine-Tuning on New Data](#164-fine-tuning-on-new-data)
    - 16.5 [Training Your Own Vocoder](#165-training-your-own-vocoder)
17. [Theoretical Background](#17-theoretical-background)
    - 17.1 [Transformer Architecture](#171-transformer-architecture)
    - 17.2 [FastSpeech 2 vs Tacotron 2](#172-fastspeech-2-vs-tacotron-2)
    - 17.3 [The Mel Spectrogram](#173-the-mel-spectrogram)
    - 17.4 [Pitch and Energy as Variance Signals](#174-pitch-and-energy-as-variance-signals)
    - 17.5 [Non-Autoregressive Generation](#175-non-autoregressive-generation)
18. [Hardware Recommendations](#18-hardware-recommendations)
19. [Project Roadmap](#19-project-roadmap)
20. [References](#20-references)

---

## 1. What This Project Is

This repository contains a **complete, from-scratch implementation** of a Gujarati text-to-speech
system built on the FastSpeech 2 architecture. "From scratch" means:

- The neural network is implemented in raw PyTorch — no `espnet`, no `fairseq`, no pre-built TTS
  libraries. Every layer, every forward pass, every loss function is written explicitly.
- The data pipeline reads raw WAV files and computes mel spectrograms, pitch, and energy entirely
  with `librosa` and `numpy`.
- The text processing pipeline handles Gujarati Unicode, number expansion, abbreviations, and
  character encoding with no third-party NLP tools.
- The training loop handles checkpointing, mixed-precision training, gradient clipping, the Noam
  scheduler, and TensorBoard logging — all hand-coded.

The goal is both a **working Gujarati TTS model** and a **learning resource** — every design
decision is explained, every function is documented, and every number has a comment explaining why
it was chosen.

---

## 2. Why FastSpeech 2

FastSpeech 2 was chosen over alternatives for several reasons:

### vs. Tacotron 2
| Property | Tacotron 2 | FastSpeech 2 |
|----------|-----------|-------------|
| Architecture | RNN (LSTM) | Transformer (FFT blocks) |
| Generation | Autoregressive (frame-by-frame) | Parallel (all frames at once) |
| Inference speed | Slow | **~50–100x faster** |
| Training stability | Can diverge (attention failures) | Stable (no attention alignment to learn) |
| Duration control | None | **Explicit duration predictor** |
| Pitch control | None | **Explicit pitch predictor** |
| Energy control | None | **Explicit energy predictor** |
| Emotion/style extension | Complex | Natural fit |

### vs. VITS
VITS produces slightly more natural audio but is significantly more complex (variational
autoencoder + normalizing flows + adversarial training). FastSpeech 2 is the right starting point
for building understanding and a clean codebase.

### vs. YourTTS / Coqui-TTS
Pre-built libraries hide the mechanics. This project exists precisely to expose them.

### Why FastSpeech 2 specifically suits Gujarati
- Gujarati is a low-resource language — explicit duration modeling helps when you have fewer than
  10,000 audio samples.
- Gujarati has complex consonant clusters and vowel diacritics. The character-level approach
  (rather than phoneme-level) lets the model learn grapheme-to-phoneme patterns from data.
- The explicit pitch, energy, and duration controls will be critical for future emotion and prosody
  work in Gujarati, where sentence-final intonation patterns are particularly distinctive.

---

## 3. Project Structure

```
tts-model-from-scratch/
│
├── config.py                   ← ALL hyperparameters and paths (start here)
│
├── data/
│   ├── text.py                 ← Gujarati text cleaning + character encoding
│   └── dataset.py              ← PyTorch Dataset, audio feature extraction, DataLoaders
│
├── model/
│   ├── __init__.py
│   ├── encoder.py              ← PositionalEncoding, MultiHeadAttention, FFTBlock, Encoder
│   ├── variance_adaptor.py     ← VariancePredictor, LengthRegulator, VarianceAdaptor
│   ├── decoder.py              ← Decoder (reuses FFTBlock from encoder)
│   └── fastspeech2.py          ← Full model, FastSpeech2, count_parameters()
│
├── train.py                    ← Training loop, losses, scheduler, checkpoints
├── inference.py                ← Text → mel → WAV, CLI interface
│
├── metadata.csv                ← Dataset manifest: wav_filename|transcript
├── requirements.txt            ← Python dependencies
├── fastspeech2_components.svg  ← Architecture diagram
│
├── wavs/                       ← Audio files (gujarati_XXXXXXX.wav)
│
├── preprocessed/               ← Created by preprocessing (not committed to git)
│   ├── mel/                    ← Saved mel spectrograms (.npy)
│   ├── pitch/                  ← Saved pitch arrays (.npy)
│   ├── energy/                 ← Saved energy arrays (.npy)
│   └── duration/               ← Saved duration arrays (.npy)
│
├── checkpoints/                ← Saved model checkpoints (.pt)
├── logs/                       ← TensorBoard logs
├── output/                     ← Generated audio at inference
└── vocoder/                    ← HiFi-GAN weights (you provide these)
    ├── hifigan_checkpoint
    └── hifigan_config.json
```

### Key relationships between files

```
config.py
    ↑ imported by everything

data/text.py
    ← uses CHAR_TO_ID, ID_TO_CHAR, VOCAB from config.py
    → provides text_to_ids(), clean_text(), ids_to_text()

data/dataset.py
    ← uses config.py paths and audio settings
    ← uses data/text.py for text encoding
    → provides GujaratiTTSDataset, get_dataloaders(), run_preprocessing()

model/encoder.py
    ← uses config.py encoder settings
    → provides Encoder, FFTBlock, PositionalEncoding

model/variance_adaptor.py
    ← uses config.py variance settings
    → provides VarianceAdaptor, VariancePredictor, LengthRegulator

model/decoder.py
    ← uses config.py decoder settings
    ← imports FFTBlock, PositionalEncoding from model/encoder.py
    → provides Decoder

model/fastspeech2.py
    ← imports Encoder, VarianceAdaptor, Decoder
    → provides FastSpeech2, count_parameters()

train.py
    ← imports FastSpeech2, get_dataloaders, config settings
    → saves checkpoints, writes TensorBoard logs

inference.py
    ← imports FastSpeech2, text_to_ids, config settings
    → loads checkpoint, generates audio
```

---

## 4. Architecture Overview

### 4.1 High-Level Data Flow

```
INPUT TEXT (Gujarati Unicode string)
        │
        ▼
  [data/text.py]
  Unicode normalization → abbreviation expansion → number expansion
  → punctuation cleanup → OOV removal → character encoding
        │
        ▼
  text_ids: LongTensor (B, T_text)   ← integer ID per character
        │
        ▼
╔══════════════════════════════════════╗
║            FASTSPEECH 2              ║
║                                      ║
║  ┌─────────────────────────────┐     ║
║  │         ENCODER             │     ║
║  │  Embedding (vocab → 256)    │     ║
║  │  + Positional Encoding      │     ║
║  │  × 4 FFT Blocks             │     ║
║  │  → enc_out (B, T_text, 256) │     ║
║  └─────────────┬───────────────┘     ║
║                │                     ║
║  ┌─────────────▼───────────────┐     ║
║  │      VARIANCE ADAPTOR       │     ║
║  │  Duration Predictor         │     ║
║  │  → Length Regulator         │     ║
║  │  Pitch Predictor → Embed    │     ║
║  │  Energy Predictor → Embed   │     ║
║  │  → va_out (B, T_mel, 256)   │     ║
║  └─────────────┬───────────────┘     ║
║                │                     ║
║  ┌─────────────▼───────────────┐     ║
║  │         DECODER             │     ║
║  │  + Positional Encoding      │     ║
║  │  × 4 FFT Blocks             │     ║
║  │  Linear(256 → 80)           │     ║
║  │  → mel_out (B, 80, T_mel)   │     ║
║  └─────────────────────────────┘     ║
╚══════════════════════════════════════╝
        │
        ▼
  mel spectrogram (B, 80, T_mel)
        │
        ▼
  [HiFi-GAN Vocoder]
  mel → waveform
        │
        ▼
  WAV FILE (audio)
```

---

### 4.2 Encoder

**File:** `model/encoder.py`

The encoder converts a sequence of Gujarati character IDs into a sequence of hidden state vectors,
one vector per character.

#### Character Embedding

```python
self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
```

- Each character ID is looked up in a learned embedding table.
- Embedding dimension = `ENCODER_HIDDEN_DIM` = 256.
- The `padding_idx` ensures PAD token gradients are zeroed.
- Embeddings are scaled by `sqrt(d_model)` = `sqrt(256)` = 16, following the original Transformer
  paper. This prevents the positional encoding from overwhelming the semantic content of the
  embedding at initialization.

#### Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- Fixed (not learned). The same encoding is used at every training step.
- Stored as a buffer — not a parameter — so it's not updated by the optimizer.
- Maximum sequence length = `MAX_SEQ_LEN` = 1000 characters.
- Added to the embedding output before passing to FFT blocks.

**Why sinusoidal over learned?**
- Generalizes to sequence lengths not seen during training.
- Requires no additional parameters.
- Works well for TTS where sentence length varies significantly.

#### FFT Block (Feed-Forward Transformer Block)

Each FFT block has two sub-layers, each with a pre-norm residual connection:

```
Sub-layer 1 (Self-Attention):
    residual = x
    x = LayerNorm(x)
    x = MultiHeadAttention(x, x, x)
    x = Dropout(x) + residual

Sub-layer 2 (Feed-Forward Network):
    residual = x
    x = LayerNorm(x)
    x = PositionwiseFeedForward(x)
    x = Dropout(x) + residual
```

**Multi-Head Self-Attention:**
- `n_heads` = 2 attention heads.
- Head dimension = `256 / 2` = 128.
- Scaled dot-product: `softmax(QK^T / sqrt(d_head)) V`
- Masking: padding positions are filled with `-inf` before softmax, so they contribute zero
  attention weight.

**Positionwise Feed-Forward Network (FFN):**
- FastSpeech 2 uses two **Conv1d layers** instead of Linear layers. This gives each position
  awareness of its local context (kernel size > 1).
- Conv1d(256 → 1024, kernel=9, padding=4) → ReLU → Conv1d(1024 → 256, kernel=1, padding=0)
- kernel_size=(9, 1): The first layer aggregates 9 adjacent frames; the second projects back.
- This is different from standard Transformer which uses two Linear layers.

**Encoder configuration:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `ENCODER_HIDDEN_DIM` | 256 | Hidden dimension throughout |
| `ENCODER_N_LAYERS` | 4 | Number of FFT blocks |
| `ENCODER_N_HEADS` | 2 | Attention heads |
| `ENCODER_CONV_FILTER_SIZE` | 1024 | FFN inner dimension |
| `ENCODER_CONV_KERNEL_SIZE` | (9, 1) | FFN conv kernel sizes |
| `ENCODER_DROPOUT` | 0.2 | Dropout rate |

**Encoder output:**
- `enc_out`: shape `(B, T_text, 256)` — one 256-dim vector per character.
- `src_mask`: shape `(B, 1, 1, T_text)` — boolean mask for attention (True = padding).

---

### 4.3 Variance Adaptor

**File:** `model/variance_adaptor.py`

The variance adaptor is the heart of FastSpeech 2's controllability. It bridges the encoder
(text-length) and decoder (mel-length) by:
1. Predicting how long each character should last (duration).
2. Expanding the encoder output to match the mel spectrogram length (length regulation).
3. Predicting and embedding pitch (F0) per mel frame.
4. Predicting and embedding energy per mel frame.

#### Variance Predictor (shared architecture)

The same 2-layer Conv1d architecture is used for all three predictors (duration, pitch, energy):

```
Input: (B, T, d_model=256)
    │
    Conv1d(256 → 256, kernel=3, padding=1)
    ReLU
    LayerNorm
    Dropout(0.5)
    │
    Conv1d(256 → 256, kernel=3, padding=1)
    ReLU
    LayerNorm
    Dropout(0.5)
    │
    Linear(256 → 1)
    squeeze(-1)
    │
Output: (B, T) scalar per frame
```

Configuration:
| Parameter | Value |
|-----------|-------|
| `VARIANCE_PREDICTOR_FILTER_SIZE` | 256 |
| `VARIANCE_PREDICTOR_KERNEL_SIZE` | 3 |
| `VARIANCE_PREDICTOR_DROPOUT` | 0.5 |

**Why higher dropout (0.5) for variance predictors?**
These predictors are small networks making a hard prediction task (predicting exact F0 values,
exact durations). Higher dropout prevents them from memorizing training utterances.

#### Duration Predictor

- Predicts log-duration: `log(dur + 1)` for each character in the encoder output.
- Log domain prevents the predictor from being penalized more for errors on long durations than
  short ones.
- **At training time:** Ground-truth durations from preprocessing are used (teacher forcing). The
  predicted log-durations are only used for computing the duration MSE loss.
- **At inference time:** `round(exp(predicted_log_dur) - 1)` gives integer frame counts per
  character, which are passed to the length regulator.

#### Length Regulator

Expands each encoder hidden state by its corresponding duration:

```
enc_out: [h_char1, h_char2, h_char3, ...]  lengths: [3, 5, 2, ...]
           ↓
mel_out: [h_char1, h_char1, h_char1,        # repeated 3 times
          h_char2, h_char2, h_char2, h_char2, h_char2,  # repeated 5 times
          h_char3, h_char3,                  # repeated 2 times
          ...]
```

Implementation uses `torch.repeat_interleave(x[i], dur, dim=0)` per sample. The output is then
zero-padded to the maximum mel length in the batch.

**Why not upsample?**
`repeat_interleave` is more faithful to the acoustic reality — a character held for N frames
genuinely has that hidden state repeated. Bilinear upsampling would interpolate between characters,
which is less meaningful phonetically.

#### Pitch Predictor + Embedding

1. The pitch predictor runs on the length-regulated output `(B, T_mel, 256)` and predicts a
   normalized F0 value per mel frame.
2. **Training:** The ground-truth pitch array (from preprocessing) is used.
3. **Inference:** The predicted pitch is used (scaled by `pitch_control`).
4. The pitch value (whether ground-truth or predicted) is **quantized** into one of
   `N_PITCH_BINS = 256` uniform buckets spanning [-3.0, 3.0] (normalized F0 range).
5. The bin index is looked up in a `nn.Embedding(256, 256)` table.
6. The resulting 256-dim embedding is **added** to the length-regulated hidden states.

**Why quantize instead of directly adding the scalar F0?**
The scalar F0 has a very different magnitude from the 256-dim hidden states. Quantization +
embedding maps the continuous F0 into the same 256-dimensional space as the hidden states, making
the addition semantically meaningful. The model learns which pitch regions map to which embedding
vectors.

#### Energy Predictor + Embedding

Same structure as pitch but for frame-level energy (L2 norm of mel bins).

1. Energy predictor on length-regulated output → scalar per frame.
2. Quantized into `N_ENERGY_BINS = 256` buckets spanning [-3.0, 3.0].
3. Looked up in `nn.Embedding(256, 256)`.
4. Added to the output (which already has pitch embedding added).

**Processing order matters:**
Pitch is added first, then energy is predicted on the pitch-conditioned output. This means the
energy predictor can condition on pitch — which makes sense acoustically (louder frames often
correlate with higher pitch in Gujarati).

---

### 4.4 Decoder

**File:** `model/decoder.py`

The decoder takes the variance-adapted output `(B, T_mel, 256)` and predicts the mel spectrogram
`(B, 80, T_mel)`.

```
va_out (B, T_mel, 256)
    │
    + Positional Encoding    ← re-added after length regulation changes sequence length
    │
    × 4 FFT Blocks           ← same architecture as encoder
    │
    LayerNorm
    │
    Linear(256 → 80)
    │
mel_out (B, T_mel, 80)       ← transposed to (B, 80, T_mel) for standard convention
```

**Why re-add positional encoding?**
After length regulation, the sequence has a new length (`T_mel` instead of `T_text`). The
positions are different, so positional encoding must be re-applied. The decoder's positional
encoding table is sized for `MAX_MEL_LEN = 1500` frames.

**Why same structure as encoder?**
FastSpeech 2 uses symmetric encoder and decoder. The decoder doesn't need to attend to the encoder
output (like in Tacotron) because the length regulator has already aligned the sequence. So it's
simply self-attention over mel-length positions.

**Decoder configuration:**
| Parameter | Value |
|-----------|-------|
| `DECODER_HIDDEN_DIM` | 256 |
| `DECODER_N_LAYERS` | 4 |
| `DECODER_N_HEADS` | 2 |
| `DECODER_CONV_FILTER_SIZE` | 1024 |
| `DECODER_CONV_KERNEL_SIZE` | (9, 1) |
| `DECODER_DROPOUT` | 0.2 |

---

### 4.5 Vocoder — HiFi-GAN

The model outputs a **mel spectrogram**, not audio directly. A vocoder converts mel → waveform.

This project uses **HiFi-GAN** (Kong et al., 2020):
- GAN-based neural vocoder.
- ~1000x faster than WaveNet.
- Produces near-studio quality audio.
- A pretrained checkpoint (LJ_FT_T2_V1) is available for English that generalizes reasonably well
  to other languages including Gujarati due to the universal nature of mel spectrograms.

**Vocoder is NOT trained in this repo.** You download a pretrained HiFi-GAN checkpoint.
See [Step 8](#108-step-8--set-up-hifi-gan-vocoder) for setup instructions.

---

## 5. Gujarati Language Handling

### 5.1 Character Vocabulary

**File:** `config.py` (GUJARATI_CHARS, VOCAB, CHAR_TO_ID, ID_TO_CHAR)

The model operates at the **character level** — each Gujarati Unicode character (or compound) is
one token. The full vocabulary is defined in `config.py`:

```python
PAD_TOKEN = "<pad>"   # index 0 — padding
UNK_TOKEN = "<unk>"   # index 1 — unknown character
```

Then all Gujarati characters in order:

**Vowels (સ્વર):**
```
અ આ ઇ ઈ ઉ ઊ ઋ એ ઐ ઓ ઔ અં અઃ
```

**Vowel Diacritics / Matras (માત્રા):**
```
ા િ ી ુ ૂ ૃ ે ૈ ો ૌ ં ઃ ્
```
These are combining characters that modify the preceding consonant. They are treated as separate
tokens in this vocabulary.

**Consonants (વ્યંજન) — 5×5 grid + extras:**
```
ક ખ ગ ઘ ઙ
ચ છ જ ઝ ઞ
ટ ઠ ડ ઢ ણ
ત થ દ ધ ન
પ ફ બ ભ મ
ય ર લ વ શ
ષ સ હ ળ ક્ષ જ્ઞ
```

**Gujarati Digits:**
```
૦ ૧ ૨ ૩ ૪ ૫ ૬ ૭ ૮ ૯
```
Note: Gujarati digits in text are first converted to ASCII digits by the text cleaner, then the
ASCII digits are expanded to Gujarati words. The Gujarati digit characters in the vocabulary are
retained for cases where digit expansion is not applied.

**Punctuation and space:**
```
(space)  ,  .  !  ?  -  –  ।  ॥  "  '
```

The Devanagari danda (`।`) and double danda (`॥`) are included because Gujarati text sometimes
uses them as sentence terminators.

**Total vocabulary size:** approximately 100 tokens (exact count printed by running `config.py`).

#### Token IDs

The vocabulary is built as:
```python
VOCAB = [PAD_TOKEN, UNK_TOKEN] + GUJARATI_CHARS
CHAR_TO_ID = {ch: i for i, ch in enumerate(VOCAB)}
ID_TO_CHAR = {i: ch for i, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PAD_ID = 0
UNK_ID = 1
```

**Adding new characters:**
Simply add them to the `GUJARATI_CHARS` list in `config.py`. This changes `VOCAB_SIZE`, so you
must start training from scratch (not from an existing checkpoint).

---

### 5.2 Text Cleaning Pipeline

**File:** `data/text.py` — `clean_text()`

All input text passes through a 5-stage pipeline in this exact order:

```
Stage 1: Unicode Normalization (NFC)
    "a\u0300" → "à"  (combining char → precomposed)
    Ensures consistent byte sequences for the same visual character.
    Critical for Gujarati where the same glyph can be encoded multiple ways.

Stage 2: Abbreviation Expansion
    "ડૉ." → "ડોક્ટર"
    "કિ.મી." → "કિલોમીટર"
    (see full list in section 5.4)

Stage 3: Number Expansion
    "25" → "પચ્ચીસ"
    "3.14" → "ત્રણ દશાંશ એક ચાર"
    (see section 5.3 for full rules)

Stage 4: Punctuation Cleanup
    - Curly quotes → straight quotes
    - Em/en dashes → hyphen
    - Zero-width characters removed (U+200B, U+200C, U+200D, U+FEFF)
    - Multiple spaces → single space
    - Strip leading/trailing whitespace

Stage 5: Out-of-Vocabulary (OOV) Removal
    Any character not in VOCAB is silently dropped.
    This prevents UNK flooding for texts with stray Latin characters.
```

**Usage:**
```python
from data.text import clean_text, text_to_ids, ids_to_text

cleaned = clean_text("ડૉ. સ્મિત 25 વર્ષ")
# → "ડોક્ટર સ્મિત પચ્ચીસ વર્ષ"

ids = text_to_ids("આ ગુજરાતી ટેક્સ્ટ છે.")
# → [4, 1, 22, 4, 10, ...]

back = ids_to_text(ids)
# → "આ ગુજરાતી ટેક્સ્ટ છે."
```

---

### 5.3 Number Expansion

**File:** `data/text.py` — `expand_numbers()`, `_number_to_gujarati_words()`

Numbers are expanded to their spoken Gujarati word equivalents before encoding.

#### Gujarati digit characters → ASCII first

Gujarati numeral characters (૦–૯) are first converted to ASCII digits (0–9) using
`str.maketrans("૦૧૨૩૪૫૬૭૮૯", "0123456789")`.

#### Decimal numbers

Handled before integers to avoid double-replacement:
```
3.14 → "ત્રણ દશાંશ એક ચાર"
       ("three point one four" — digits read individually after decimal point)
```

#### Integer expansion table

| Range | Rule | Example |
|-------|------|---------|
| 0 | "શૂન્ય" | 0 → "શૂન્ય" |
| 1–19 | Direct lookup | 5 → "પાંચ", 15 → "પંદર" |
| 20–99 | Tens + ones | 25 → "વીસ" + "પાંચ" = "વીસપાંચ" |
| 100–999 | `n_hundreds` + "સો" + remainder | 250 → "બે સો પચ્ચીસ" |
| 1,000–99,999 | `n_thousands` + "હજાર" + remainder | 1500 → "એક હજાર પાંચ સો" |
| 1,00,000–99,99,999 | `n_lakhs` + "લાખ" + remainder | 5,00,000 → "પાંચ લાખ" |
| 1,00,00,000–9,99,99,999 | `n_crores` + "કરોડ" + remainder | |

Note: The Indian numbering system is used (lakhs and crores, not millions/billions).

#### Ones lookup table (1–19)
```python
["", "એક", "બે", "ત્રણ", "ચાર", "પાંચ", "છ", "સાત", "આઠ", "નવ", "દસ",
 "અગિયાર", "બાર", "તેર", "ચૌદ", "પંદર", "સોળ", "સત્તર", "અઢાર", "ઓગણીસ"]
```

#### Tens lookup table
```python
["", "", "વીસ", "ત્રીસ", "ચાલીસ", "પચાસ", "સાઈઠ", "સિત્તેર", "એંસી", "નેવું"]
```

---

### 5.4 Abbreviation Expansion

**File:** `data/text.py` — `ABBREVIATIONS` dict, `expand_abbreviations()`

| Abbreviation | Expansion |
|-------------|-----------|
| `ડૉ.` / `ડૉ` | ડોક્ટર |
| `શ્રી.` | શ્રી |
| `સ્ત્રી.` | સ્ત્રી |
| `કિ.મી.` / `કિ.મી` | કિલોમીટર |
| `કિ.ગ્રા.` | કિલોગ્રામ |
| `રૂ.` / `રૂ` | રૂપિયા |
| `કલા.` | કલાક |
| `મિ.` | મિનિટ |
| `સે.` | સેકન્ડ |
| `વિ.સ.` | વિક્રમ સંવત |
| `ઈ.સ.` | ઈસ્વીસન |
| `પ્રા.` | પ્રાથમિક |
| `માધ.` | માધ્યમિક |

**Adding new abbreviations:** Simply add to the `ABBREVIATIONS` dict in `data/text.py`. Longer
abbreviations are matched first to prevent partial replacements.

---

## 6. Dataset Format

### 6.1 metadata.csv

**Location:** `metadata.csv` in the project root (path controlled by `METADATA_FILE` in config).

**Format:** Pipe-separated CSV with a header row.

```
audio|text
gujarati_0000000.wav|મને લાગ્યું કે મહામારીની બીજી લહેર આવી રહી છે.
gujarati_0000001.wav|હું દુકાનમાંથી પાણી ખરીદવા માટે નર્વસ હતો.
gujarati_0000002.wav|હરિયાણા
```

**Rules:**
- Column 1: WAV filename. May include or omit the `.wav` extension — both are handled by
  `_read_metadata()`.
- Column 2: Transcript text in Gujarati Unicode.
- Column 3 (optional): Normalized transcript. If present and non-empty, it is used instead of
  column 2. This allows you to pre-clean text separately.
- The header row (`audio|text`) is automatically skipped.
- Rows with fewer than 2 columns are skipped with a warning.

**Current dataset stats:**
- ~25,154 audio-text pairs (header excluded).
- Mix of single words, short phrases, and full sentences.
- Some entries contain transliterated words and proper nouns (city names, etc.).

**Creating your own metadata.csv:**
```python
import csv

entries = [
    ("my_wav_001", "આ ગુજરાતી વાક્ય છે."),
    ("my_wav_002", "નમસ્તે, કેમ છો?"),
]

with open("metadata.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="|")
    writer.writerow(["audio", "text"])
    for wav_name, text in entries:
        writer.writerow([wav_name + ".wav", text])
```

---

### 6.2 WAV Files

**Location:** `wavs/` directory (path controlled by `WAV_DIR` in config).

**Expected format:**
- Mono or stereo (stereo is averaged to mono during loading).
- Any sample rate — will be resampled to `SAMPLE_RATE = 22050` Hz if different.
- 16-bit PCM is most common; float32 is also supported (via `soundfile`).
- Duration: filtered to **0.5–15.0 seconds** during preprocessing. Files outside this range are
  skipped.

**Current files:** Named `gujarati_XXXXXXX.wav` (zero-padded 7-digit index).

**Audio quality recommendations:**
- Record in a quiet room with minimal background noise.
- Consistent microphone position and distance.
- Avoid clipping (peaks should not reach 0 dBFS).
- Sample rate of 22050 Hz or higher is preferred (lower will be upsampled).
- Mono is preferred (stereo wastes memory and adds no information for TTS).

---

## 7. Audio Feature Extraction

**File:** `data/dataset.py`

All audio features are computed once during preprocessing and saved as `.npy` files. This avoids
recomputing them on every training epoch.

### 7.1 Mel Spectrogram

**Function:** `compute_mel_spectrogram(audio)` → `(n_mels, T)`

A mel spectrogram is a 2D time-frequency representation of audio where the frequency axis is
warped to the mel scale (which approximates human hearing perception).

**Pipeline:**
```
audio (float32, normalized to [-1, 1])
    │
    Short-Time Fourier Transform (STFT)
    n_fft=1024, hop_length=256, win_length=1024, center=False
    │
    Magnitude spectrum |X(t, f)|
    │
    Mel filterbank (80 triangular filters, 0 Hz – 8000 Hz)
    │
    Log compression: log(x + 1e-5)
    │
mel: float32 (80, T)
```

**Why `center=False`?**
HiFi-GAN was trained without center padding. Using `center=False` in both preprocessing and the
vocoder ensures the time alignment is consistent.

**Why amplitude spectrum (power=1.0) instead of power spectrum (power=2.0)?**
Several TTS systems including Tacotron 2 and HiFi-GAN's original training used amplitude
spectrograms. Using power=1.0 matches those systems, making the pretrained vocoder more compatible.

**Configuration:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `SAMPLE_RATE` | 22050 | Audio sample rate in Hz |
| `HOP_LENGTH` | 256 | STFT hop: ~11.6ms per frame |
| `WIN_LENGTH` | 1024 | STFT window: ~46.4ms |
| `N_FFT` | 1024 | FFT size (= WIN_LENGTH) |
| `N_MELS` | 80 | Mel filter count |
| `MEL_FMIN` | 0.0 | Min frequency |
| `MEL_FMAX` | 8000.0 | Max frequency |

**Frame rate:** 22050 / 256 ≈ 86 frames per second. A 3-second utterance ≈ 258 mel frames.

---

### 7.2 Pitch (F0) Extraction

**Function:** `compute_pitch(audio, mel_len)` → `(mel_len,)`

Fundamental frequency (F0) is extracted using **librosa's pyin algorithm** — a probabilistic
variant of the YIN algorithm.

**Why pyin over plain yin?**
- pyin returns probabilistic F0 estimates, reducing octave errors.
- Handles unvoiced regions (silence, fricatives) gracefully by returning `NaN`.
- More robust to noise than plain YIN.

**Pipeline:**
```
audio
    │
    librosa.pyin(fmin=50 Hz, fmax=600 Hz, sr=22050, hop_length=256, center=False)
    │
    f0: float array with NaN for unvoiced frames
    │
    NaN → 0.0   (unvoiced frames become zero)
    │
    Resize to mel_len if needed (linear interpolation)
    │
    Normalize: zero mean / unit variance over voiced frames only
    (unvoiced frames remain 0.0)
    │
pitch: float32 (mel_len,)
```

**Why normalize only over voiced frames?**
Including silence (F0 = 0) in the mean/std calculation would distort the statistics. Zero is
reserved to mean "unvoiced", so it must stay at zero after normalization.

**Pitch range for Gujarati speakers:**
- `PITCH_MIN = 50 Hz` — below the lowest human voice
- `PITCH_MAX = 600 Hz` — above the highest expected F0
- Gujarati speakers typically fall in **80–400 Hz** depending on speaker gender

---

### 7.3 Energy Extraction

**Function:** `compute_energy(mel)` → `(T,)`

Energy is the L2 norm (magnitude) of the mel spectrum at each time frame.

```
energy[t] = sqrt(sum(mel[:, t]^2))   for each frame t
```

Then normalized to zero mean / unit variance across the entire utterance.

**Why L2 norm of mel instead of raw waveform energy?**
Mel-domain energy is smoother and better represents perceptual loudness. Raw waveform energy is
noisy and affected by high-frequency components that are less perceptually relevant.

---

### 7.4 Duration Computation

**Function:** `compute_durations_from_mel(text_ids, mel_len)` → `(n_chars,)` int array

**Current implementation: uniform distribution (approximation)**

The simplest possible duration assignment:
```
base_duration = mel_len // n_chars
leftover = mel_len % n_chars
```
Each character gets `base_duration` frames, and any leftover frames are distributed to
middle characters (middle characters tend to be longer in natural speech).

**Guarantee:** `durations.sum() == mel_len` always. This assertion is checked.

**Limitation:** This is a poor approximation of real phoneme durations. Short vowels get the same
duration as long ones. Consonant clusters are treated the same as single consonants.

**Better approach: Montreal Forced Aligner (MFA)**
The standard approach for production TTS is to use an automatic speech aligner that finds exact
phoneme boundaries:
```
WAV file + transcript → MFA → TextGrid files → exact phoneme durations
```
See [Section 16.1](#161-adding-mfa-alignments) for MFA integration instructions.

**Why start with approximate durations?**
The approximate durations allow you to verify the full pipeline end-to-end quickly. The model
will learn something reasonable even with approximate durations — it just won't produce as natural
speech as a model trained with exact MFA durations.

---

## 8. Configuration Reference

**File:** `config.py` — single source of truth for all settings.

**Never hardcode settings in other files.** All tunable values must live in `config.py`.

---

### 8.1 Paths

```python
DATASET_PATH    = "/home/user/gujarati_tts_dataset"
WAV_DIR         = os.path.join(DATASET_PATH, "wavs")
METADATA_FILE   = os.path.join(DATASET_PATH, "metadata.csv")
PREPROCESSED_DIR = os.path.join(DATASET_PATH, "preprocessed")
MEL_DIR          = os.path.join(PREPROCESSED_DIR, "mel")
PITCH_DIR        = os.path.join(PREPROCESSED_DIR, "pitch")
ENERGY_DIR       = os.path.join(PREPROCESSED_DIR, "energy")
DURATION_DIR     = os.path.join(PREPROCESSED_DIR, "duration")
CHECKPOINT_DIR  = "./checkpoints"
LOG_DIR         = "./logs"
OUTPUT_DIR      = "./output"
```

**IMPORTANT:** Update `DATASET_PATH` to your actual dataset location before running anything.

On Windows (if running from WSL or Git Bash):
```python
DATASET_PATH = "C:/laragon/www/AI-ML/SkyllectDemos/Fine-tuning-gujarati/tts-model-from-scratch"
```

On Linux:
```python
DATASET_PATH = "/home/username/gujarati_tts"
```

---

### 8.2 Audio Settings

```python
SAMPLE_RATE   = 22050    # Hz — standard for TTS
HOP_LENGTH    = 256      # STFT hop (~11.6ms per frame)
WIN_LENGTH    = 1024     # STFT window (~46.4ms)
N_FFT         = 1024     # FFT size
N_MELS        = 80       # mel filter banks
MEL_FMIN      = 0.0      # Hz
MEL_FMAX      = 8000.0   # Hz
MAX_WAV_VALUE = 32768.0  # 16-bit PCM normalization constant
PITCH_MIN     = 50.0     # Hz — minimum expected F0
PITCH_MAX     = 600.0    # Hz — maximum expected F0
```

**When to change these:**
- `SAMPLE_RATE`: Only change if your dataset has a fixed different sample rate AND you want to
  avoid resampling. Your vocoder must be retrained to match.
- `N_MELS`: 80 is standard for HiFi-GAN. Changing this requires retraining the vocoder.
- `MEL_FMAX`: Some voices benefit from 11025 Hz (Nyquist for 22050 Hz). HiFi-GAN was trained at
  8000 Hz upper limit.
- `PITCH_MIN/MAX`: Adjust if your speaker has an unusual range. Children: raise PITCH_MAX to 800.
  Very deep voice: lower PITCH_MIN to 40.

---

### 8.3 Character Vocabulary Settings

```python
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
GUJARATI_CHARS = [...]   # full list in config.py
VOCAB = [PAD_TOKEN, UNK_TOKEN] + GUJARATI_CHARS
CHAR_TO_ID = {ch: i for i, ch in enumerate(VOCAB)}
ID_TO_CHAR = {i: ch for i, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PAD_ID = 0
UNK_ID = 1
```

These are computed automatically from `GUJARATI_CHARS`. Do not edit `VOCAB`, `CHAR_TO_ID`, etc.
directly — just edit `GUJARATI_CHARS`.

---

### 8.4 Model Architecture Settings

```python
# Encoder
ENCODER_HIDDEN_DIM       = 256
ENCODER_N_LAYERS         = 4
ENCODER_N_HEADS          = 2
ENCODER_CONV_FILTER_SIZE = 1024
ENCODER_CONV_KERNEL_SIZE = (9, 1)
ENCODER_DROPOUT          = 0.2

# Decoder (same structure)
DECODER_HIDDEN_DIM       = 256
DECODER_N_LAYERS         = 4
DECODER_N_HEADS          = 2
DECODER_CONV_FILTER_SIZE = 1024
DECODER_CONV_KERNEL_SIZE = (9, 1)
DECODER_DROPOUT          = 0.2

# Variance predictor (shared by duration/pitch/energy)
VARIANCE_PREDICTOR_FILTER_SIZE  = 256
VARIANCE_PREDICTOR_KERNEL_SIZE  = 3
VARIANCE_PREDICTOR_DROPOUT      = 0.5

# Embeddings for pitch and energy
PITCH_EMBEDDING_DIM  = 256
ENERGY_EMBEDDING_DIM = 256
N_PITCH_BINS         = 256
N_ENERGY_BINS        = 256

# Sequence length limits
MAX_SEQ_LEN = 1000   # max input characters
MAX_MEL_LEN = 1500   # max output mel frames (~17.5 seconds)

# Output
MEL_LINEAR_DIM = N_MELS  # = 80
```

**Scaling the model up or down:**

For faster training / smaller GPU:
```python
ENCODER_HIDDEN_DIM = 128
ENCODER_N_LAYERS   = 2
```

For better quality / large GPU:
```python
ENCODER_HIDDEN_DIM = 384
ENCODER_N_LAYERS   = 6
ENCODER_N_HEADS    = 4
```

**Note:** `ENCODER_HIDDEN_DIM` must be divisible by `ENCODER_N_HEADS`.

---

### 8.5 Training Settings

```python
BATCH_SIZE   = 16      # reduce to 8 if OOM
NUM_WORKERS  = 4       # DataLoader CPU workers
EPOCHS       = 1000
SAVE_EVERY   = 50      # checkpoint interval
LOG_EVERY    = 10      # TensorBoard log interval (steps)
VAL_EVERY    = 50      # validation interval (epochs)

LEARNING_RATE = 1e-3
BETAS         = (0.9, 0.98)
EPS           = 1e-9
WEIGHT_DECAY  = 0.0
WARMUP_STEPS  = 4000

GRAD_CLIP_THRESH = 1.0

USE_AMP    = True
AMP_DTYPE  = "bfloat16"   # or "float16"
```

**Batch size guidance:**
| GPU VRAM | Recommended BATCH_SIZE |
|----------|----------------------|
| 4 GB | 4 |
| 8 GB | 8 |
| 12 GB | 12 |
| 16 GB | 16 |
| 24 GB | 24–32 |

**Mixed precision:**
- `bfloat16`: Better numerical stability. Recommended for Ampere+ (RTX 30xx, 40xx) and Blackwell
  (RTX 50xx). No GradScaler needed.
- `float16`: Slightly faster on some hardware. Requires GradScaler for stability. Use on older
  GPUs (RTX 20xx, GTX 16xx).
- Set `USE_AMP = False` if you encounter NaN losses.

**Warmup steps:**
4000 steps is standard for Transformer TTS models. At `BATCH_SIZE = 16` with ~24,000 training
samples, one epoch ≈ 1500 steps, so warmup completes in about 3 epochs.

---

### 8.6 Loss Weights

```python
MEL_LOSS_WEIGHT      = 1.0
DURATION_LOSS_WEIGHT = 1.0
PITCH_LOSS_WEIGHT    = 1.0
ENERGY_LOSS_WEIGHT   = 1.0
```

All four losses contribute equally. You can adjust these to prioritize certain aspects:
- Increase `MEL_LOSS_WEIGHT` to prioritize mel reconstruction quality.
- Increase `DURATION_LOSS_WEIGHT` if speech rhythm sounds wrong.
- Increase `PITCH_LOSS_WEIGHT` if intonation sounds monotone.

---

### 8.7 Data Split

```python
TRAIN_SPLIT = 0.95   # 95% for training
VAL_SPLIT   = 0.05   # 5% for validation
```

The split is deterministic: the dataset is shuffled with `random.seed(42)` then split. The same
5% will always be used for validation regardless of how many times you run preprocessing.

---

### 8.8 Inference Settings

```python
DURATION_CONTROL = 1.0   # 1.0 = natural speed
PITCH_CONTROL    = 1.0   # 1.0 = natural pitch
ENERGY_CONTROL   = 1.0   # 1.0 = natural volume

VOCODER_CHECKPOINT = "./vocoder/hifigan_checkpoint"
VOCODER_CONFIG     = "./vocoder/hifigan_config.json"
```

These are the **default** values used when no `--duration_control` etc. flags are passed to
`inference.py`. Override them on the command line without changing config.

---

## 9. Installation

### 9.1 System Requirements

- **OS:** Linux, macOS, or Windows (WSL2 recommended for Windows)
- **Python:** 3.10 or higher (uses `list[tuple[str, str]]` type hints requiring Python 3.9+)
- **GPU:** NVIDIA GPU strongly recommended for training.
  - Minimum: 8 GB VRAM (reduce BATCH_SIZE if needed)
  - Recommended: 16 GB+ VRAM
  - CPU training is possible but impractically slow for 1000 epochs
- **Disk:** ~10 GB for preprocessed features (mel, pitch, energy, duration) for ~25k samples

---

### 9.2 Python Dependencies

**`requirements.txt`:**
```
librosa         # audio loading, mel spectrogram, pyin pitch extraction
soundfile       # WAV reading/writing (faster than scipy.io.wavfile)
numpy           # array operations
pandas          # (available for future CSV handling)
matplotlib      # plotting mel spectrograms for debugging
tqdm            # progress bars during preprocessing and training
scipy           # signal processing utilities
Unidecode       # (available for future transliteration)
inflect         # (available for future number normalization)
tensorboard     # training visualization
```

Install all at once:
```bash
pip install librosa soundfile numpy pandas matplotlib tqdm scipy Unidecode inflect tensorboard
```

---

### 9.3 PyTorch with CUDA

PyTorch is not in `requirements.txt` because the correct version depends on your CUDA version.

**Step 1: Check your CUDA version**
```bash
nvidia-smi
# Look for "CUDA Version: X.X" in the top right
```

**Step 2: Install PyTorch**

For CUDA 11.8:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 12.8+ (Blackwell / RTX 5060 Ti):
```bash
# Use PyTorch nightly for latest Blackwell support
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

For CPU only (no GPU):
```bash
pip install torch torchaudio
```

**Verify installation:**
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 10. Step-by-Step Usage Guide

### 10.1 Step 1 — Prepare Your Dataset

Your dataset needs two things:
1. A `wavs/` folder with `.wav` audio files.
2. A `metadata.csv` with filename-transcript pairs.

**If you're using the existing dataset** (already in this repo):
- `wavs/` already exists with ~25k files.
- `metadata.csv` already exists.
- Skip to Step 2.

**If you're adding your own data:**
```bash
mkdir wavs
# Copy your WAV files to wavs/
# Create metadata.csv:
echo "audio|text" > metadata.csv
echo "my_file_001.wav|તમારો ટ્રાન્સ્ક્રિપ્ટ અહીં" >> metadata.csv
```

---

### 10.2 Step 2 — Update config.py

Open `config.py` and change `DATASET_PATH` to point to this project's directory:

```python
# Example for Windows
DATASET_PATH = "C:/laragon/www/AI-ML/SkyllectDemos/Fine-tuning-gujarati/tts-model-from-scratch"

# Example for Linux
DATASET_PATH = "/home/yourname/tts-model-from-scratch"
```

Everything else (`WAV_DIR`, `MEL_DIR`, etc.) is derived from `DATASET_PATH` automatically.

**Optional:** Adjust `BATCH_SIZE` based on your GPU VRAM (see Section 8.5).

---

### 10.3 Step 3 — Verify Text Pipeline

Before preprocessing audio (which takes time), verify the text pipeline works:

```bash
python data/text.py
```

Expected output:
```
============================================================
  Gujarati Text Pipeline — Self Test
============================================================

  [PASS]
  Original : આ સારું છે.
  Cleaned  : આ સારું છે.
  IDs      : [4, 1, 22, ...]
  Decoded  : આ સારું છે.

  [PASS]
  Original : મારી ઉંમર 25 વર્ષ છે.
  Cleaned  : મારી ઉંમર પચ્ચીસ વર્ષ છે.
  ...

  Number expansion examples:
           0 → શૂન્ય
           1 → એક
          15 → પંદર
          25 → વીસ...
```

---

### 10.4 Step 4 — Preprocess Audio

This step reads every WAV file, computes mel spectrograms, pitch, energy, and durations, and
saves them as `.npy` files. **This only needs to be run once.**

```bash
python -c "from data.dataset import run_preprocessing; run_preprocessing()"
```

**What it does:**
1. Creates `preprocessed/mel/`, `preprocessed/pitch/`, `preprocessed/energy/`,
   `preprocessed/duration/` directories.
2. Reads `metadata.csv` and iterates over all entries.
3. For each entry:
   - Loads the WAV file via `soundfile`.
   - Resamples to 22050 Hz if needed.
   - Skips files shorter than 0.5s or longer than 15s.
   - Computes mel spectrogram `(80, T)`, pitch `(T,)`, energy `(T,)`, durations `(n_chars,)`.
   - Saves four `.npy` files per sample.
4. Prints a summary: success / skipped / failed counts.

**Expected runtime:**
- ~25,000 samples: 30–90 minutes on CPU (pyin pitch extraction is the bottleneck).
- With a fast CPU or parallelization: faster.

**Progress looks like:**
```
Preprocessing 25154 audio files...
Output directory: /path/to/preprocessed

Preprocessing: 100%|████████████| 25154/25154 [45:23<00:00,  9.24it/s]

Done.
  Success : 24891
  Skipped : 12
  Failed  : 251
```

**After preprocessing, verify dataset loading:**
```bash
python data/dataset.py
```

---

### 10.5 Step 5 — Verify Config

```bash
python config.py
```

Expected output:
```
=======================================================
  Gujarati FastSpeech 2 — Configuration Summary
=======================================================
  Device          : cuda
  GPU             : NVIDIA GeForce RTX 5060 Ti
  VRAM            : 16.0 GB
  Vocab size      : 97 characters
  Sample rate     : 22050 Hz
  Mel bins        : 80
  Encoder layers  : 4
  Decoder layers  : 4
  Hidden dim      : 256
  Batch size      : 16
  Mixed precision : True (bfloat16)
  Dataset path    : /path/to/tts-model-from-scratch
=======================================================
```

If Device shows `cpu` when you expect `cuda`, check your PyTorch CUDA installation.

---

### 10.6 Step 6 — Train the Model

```bash
python train.py
```

**What happens at startup:**
1. Prints device info and GPU name.
2. Loads train and val DataLoaders (reads preprocessed `.npy` files, filters invalid samples).
3. Instantiates FastSpeech2 model and prints parameter count.
4. Checks `checkpoints/` for any existing checkpoint and resumes from the latest if found.
5. Starts training loop.

**Training output (one line per epoch):**
```
Epoch 0001 | loss=8.4231 | mel=6.2311 | dur=0.8431 | pitch=0.7234 | energy=0.6255 | lr=2.50e-06 | 34.2s
Epoch 0002 | loss=7.8923 | mel=5.8122 | dur=0.7891 | pitch=0.6834 | energy=0.6076 | lr=5.00e-06 | 33.8s
...
  Saved: ./checkpoints/checkpoint_epoch0050.pt
...
Epoch 0050 | loss=4.1231 | mel=2.8311 | dur=0.4231 | pitch=0.4534 | energy=0.4155 | lr=1.58e-04 | 33.1s
  Val loss: 4.3892
```

**Resuming after interruption:**
```bash
python train.py   # automatically finds and loads the latest checkpoint
```

**Training duration estimates:**
| Hardware | Epochs/hour | Time for 1000 epochs |
|---------|------------|---------------------|
| RTX 3060 (12 GB) | ~40 | ~25 hours |
| RTX 3090 (24 GB) | ~80 | ~12 hours |
| RTX 4090 (24 GB) | ~120 | ~8 hours |
| RTX 5060 Ti (16 GB) | ~70 | ~14 hours |

---

### 10.7 Step 7 — Monitor Training

```bash
tensorboard --logdir ./logs
```

Then open `http://localhost:6006` in your browser.

**Metrics logged:**
- `train/total_loss` — overall loss per step
- `train/mel_loss` — mel reconstruction loss
- `train/duration_loss` — duration predictor loss
- `train/pitch_loss` — pitch predictor loss
- `train/energy_loss` — energy predictor loss
- `train/lr` — current learning rate
- `val/total_loss` — validation loss (every `VAL_EVERY` epochs)

**What healthy training looks like:**
- All losses decrease steadily for the first few hundred epochs.
- Learning rate rises linearly for ~3 epochs (warmup), then slowly decays.
- Val loss tracks train loss closely (small gap = no overfitting).
- After ~200 epochs: `total_loss` around 3–4. After ~500 epochs: around 1–2.

**Signs of trouble:**
- `NaN` losses → lower learning rate, disable AMP (`USE_AMP = False`), or switch to `float16`
  + GradScaler.
- Loss not decreasing after warmup → check dataset (are preprocessed files correct?).
- Val loss much higher than train loss → overfitting (need more data or stronger dropout).

---

### 10.8 Step 8 — Set Up HiFi-GAN Vocoder

HiFi-GAN converts mel spectrograms to audio waveforms. A pretrained checkpoint is required.

**Step 8a: Clone HiFi-GAN**
```bash
git clone https://github.com/jik876/hifi-gan
cd hifi-gan
```

**Step 8b: Download pretrained checkpoint**

Go to the HiFi-GAN releases page and download `LJ_FT_T2_V1.zip`. This is the checkpoint fine-tuned
on LJ Speech, which works reasonably for Gujarati as a universal vocoder.

Unzip and place files:
```bash
# Inside this project (tts-model-from-scratch/):
mkdir -p vocoder
cp /path/to/LJ_FT_T2_V1/generator_v1 vocoder/hifigan_checkpoint
cp /path/to/LJ_FT_T2_V1/config.json vocoder/hifigan_config.json
```

**Step 8c: Add HiFi-GAN to Python path**

In `inference.py`, find the load_hifigan function and uncomment:
```python
sys.path.insert(0, "./hifi-gan")   # or the absolute path to your hifi-gan clone
```

**Step 8d: Verify vocoder**
```bash
python -c "
from inference import load_hifigan
v = load_hifigan('./vocoder/hifigan_checkpoint', './vocoder/hifigan_config.json')
print('HiFi-GAN loaded successfully')
"
```

**Alternative: Use a different vocoder**
Any neural vocoder that accepts mel spectrograms with these parameters
(80 mel bins, 22050 Hz, hop_length=256) will work. Options include:
- WaveGlow (NVIDIA, slightly lower quality)
- BigVGAN (NVIDIA, higher quality than HiFi-GAN)
- EnvVGAN
- Vocos

---

### 10.9 Step 9 — Run Inference

**Basic usage:**
```bash
python inference.py --text "નમસ્તે, ગુજરાત!"
```

**With a specific checkpoint:**
```bash
python inference.py --text "આ ગુજરાતી ભાષણ છે." \
    --checkpoint ./checkpoints/checkpoint_epoch1000.pt
```

**Speed control (0.8 = 20% faster):**
```bash
python inference.py --text "ધીમે ધીમે બોલો." --duration_control 1.3
```

**Pitch control (higher pitch):**
```bash
python inference.py --text "ઊંચો અવાજ." --pitch_control 1.2
```

**Custom output path:**
```bash
python inference.py --text "ગુજરાત" --output ./my_audio/test.wav
```

**Without vocoder (saves mel as .npy):**
```bash
python inference.py --text "ટેસ્ટ" --no_vocoder
```

**Full options:**
```
usage: inference.py [-h] --text TEXT
                    [--checkpoint CHECKPOINT]
                    [--output OUTPUT]
                    [--duration_control DURATION_CONTROL]
                    [--pitch_control PITCH_CONTROL]
                    [--energy_control ENERGY_CONTROL]
                    [--no_vocoder]

options:
  --text TEXT                     Gujarati text to synthesize (required)
  --checkpoint CHECKPOINT         Path to .pt checkpoint file
                                  (default: latest in checkpoints/)
  --output OUTPUT                 Output WAV path
                                  (default: output/<text>.wav)
  --duration_control FLOAT        Speed: <1.0 faster, >1.0 slower (default: 1.0)
  --pitch_control FLOAT           Pitch: <1.0 lower, >1.0 higher (default: 1.0)
  --energy_control FLOAT          Volume: <1.0 quieter, >1.0 louder (default: 1.0)
  --no_vocoder                    Skip vocoder, save mel as .npy
```

---

## 11. File-by-File Code Reference

### 11.1 config.py

**Purpose:** Central configuration hub. Every tunable parameter, path, and constant lives here.
All other files import from `config.py` — never the other way around.

**Key exports:**
```python
# Paths
DATASET_PATH, WAV_DIR, METADATA_FILE, PREPROCESSED_DIR
MEL_DIR, PITCH_DIR, ENERGY_DIR, DURATION_DIR
CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR

# Audio
SAMPLE_RATE, HOP_LENGTH, WIN_LENGTH, N_FFT, N_MELS
MEL_FMIN, MEL_FMAX, MAX_WAV_VALUE, PITCH_MIN, PITCH_MAX

# Vocabulary
GUJARATI_CHARS, VOCAB, CHAR_TO_ID, ID_TO_CHAR, VOCAB_SIZE
PAD_TOKEN, UNK_TOKEN, PAD_ID, UNK_ID

# Model
ENCODER_HIDDEN_DIM, ENCODER_N_LAYERS, ENCODER_N_HEADS
ENCODER_CONV_FILTER_SIZE, ENCODER_CONV_KERNEL_SIZE, ENCODER_DROPOUT
DECODER_HIDDEN_DIM, DECODER_N_LAYERS, DECODER_N_HEADS
DECODER_CONV_FILTER_SIZE, DECODER_CONV_KERNEL_SIZE, DECODER_DROPOUT
VARIANCE_PREDICTOR_FILTER_SIZE, VARIANCE_PREDICTOR_KERNEL_SIZE, VARIANCE_PREDICTOR_DROPOUT
PITCH_EMBEDDING_DIM, ENERGY_EMBEDDING_DIM, N_PITCH_BINS, N_ENERGY_BINS
MAX_SEQ_LEN, MAX_MEL_LEN, MEL_LINEAR_DIM

# Training
BATCH_SIZE, NUM_WORKERS, EPOCHS, SAVE_EVERY, LOG_EVERY, VAL_EVERY
LEARNING_RATE, BETAS, EPS, WEIGHT_DECAY, WARMUP_STEPS, GRAD_CLIP_THRESH
USE_AMP, AMP_DTYPE
MEL_LOSS_WEIGHT, DURATION_LOSS_WEIGHT, PITCH_LOSS_WEIGHT, ENERGY_LOSS_WEIGHT
TRAIN_SPLIT, VAL_SPLIT

# Inference
DURATION_CONTROL, PITCH_CONTROL, ENERGY_CONTROL
VOCODER_CHECKPOINT, VOCODER_CONFIG

# Device
DEVICE   # "cuda" or "cpu"
```

**Self-test:** `python config.py` prints a configuration summary.

---

### 11.2 data/text.py

**Purpose:** Gujarati text cleaning, normalization, and character-level encoding.

**Public API:**
```python
clean_text(text: str) -> str
    # Full 5-stage cleaning pipeline. Always use this before encoding.

text_to_ids(text: str, apply_cleaning: bool = True) -> list[int]
    # Convert text to integer token IDs.
    # apply_cleaning=True (default): runs clean_text() first
    # apply_cleaning=False: assumes text is already cleaned

ids_to_text(ids: list[int]) -> str
    # Convert integer IDs back to text. Skips PAD tokens.
    # Useful for debugging round-trip consistency.

text_to_sequence_and_back(text: str) -> tuple[list[int], str]
    # Convenience: clean → encode → decode in one call.

expand_numbers(text: str) -> str
    # Replace digit sequences with Gujarati words.

expand_abbreviations(text: str) -> str
    # Replace known abbreviations with full forms.
```

**Internal functions:**
```python
_number_to_gujarati_words(n: int) -> str
    # Recursive integer → Gujarati word conversion.
    # Handles 0 to 9,99,99,999.

normalize_unicode(text: str) -> str
    # Apply NFC Unicode normalization.

clean_punctuation(text: str) -> str
    # Normalize quotes, dashes, remove zero-width chars, collapse spaces.

remove_out_of_vocab(text: str) -> str
    # Drop any character not in CHAR_TO_ID.
```

**Self-test:** `python data/text.py` runs 6 test cases and prints number expansion examples.

---

### 11.3 data/dataset.py

**Purpose:** Audio preprocessing, feature extraction, PyTorch Dataset, and DataLoaders.

**Public API:**
```python
run_preprocessing(metadata_file=METADATA_FILE, force=False)
    # Preprocess all audio files. Run once before training.
    # force=True: recompute even if cached .npy files exist.

get_dataloaders(metadata_file, batch_size, num_workers)
    -> (train_loader, val_loader)
    # Build train and validation DataLoaders.

class GujaratiTTSDataset(Dataset)
    # PyTorch Dataset loading from preprocessed .npy files.
    # __getitem__ returns: {text_ids, mel, pitch, energy, durations,
    #                       text_len, mel_len, wav_name}

collate_fn(batch: list[dict]) -> dict
    # Pad variable-length sequences into uniform batches.
    # Returns: {text_ids, mel, pitch, energy, durations,
    #           text_lens, mel_lens, wav_names}
```

**Feature extraction functions:**
```python
load_wav(wav_path: str) -> np.ndarray
    # Load WAV → float32 normalized to [-1, 1].

compute_mel_spectrogram(audio) -> (n_mels, T)
compute_pitch(audio, mel_len) -> (T,)
compute_energy(mel) -> (T,)
compute_durations_from_mel(text_ids, mel_len) -> (n_chars,)
    # Approximate duration from even distribution.

preprocess_and_cache(wav_name, text, force=False) -> bool
    # Preprocess one file and save 4 .npy files. Returns True on success.
```

**Internal functions:**
```python
_read_metadata(metadata_file) -> list[tuple[str, str]]
    # Read pipe-separated CSV. Skips header row, strips .wav extension.

_resize_array(arr, target_len) -> np.ndarray
    # Linear interpolation to resize 1D array (for pitch alignment).
```

**Self-test:** `python data/dataset.py` tests feature extraction with synthetic audio and the
collate function with dummy batches, then checks if the real dataset is available.

---

### 11.4 model/encoder.py

**Purpose:** Implements all building blocks of the Transformer encoder.

**Classes:**
```python
class PositionalEncoding(nn.Module)
    # Fixed sinusoidal positional encoding.
    # forward(x: (B,T,d)) -> (B,T,d)

class MultiHeadAttention(nn.Module)
    # Scaled dot-product multi-head self-attention.
    # forward(x: (B,T,d), mask: (B,1,1,T)) -> (B,T,d)

class PositionwiseFeedForward(nn.Module)
    # Two Conv1d layers (kernel sizes (9,1)).
    # forward(x: (B,T,d)) -> (B,T,d)

class FFTBlock(nn.Module)
    # Pre-norm attention + FFN block with residuals.
    # forward(x: (B,T,d), mask: (B,1,1,T)) -> (B,T,d)
    # Reused by both Encoder and Decoder.

class Encoder(nn.Module)
    # Full encoder: embedding + pos_enc + N FFTBlocks + LayerNorm.
    # forward(text_ids: (B,T)) -> (enc_out: (B,T,d), src_mask: (B,1,1,T))
```

**Self-test:** `python model/encoder.py`

---

### 11.5 model/variance_adaptor.py

**Purpose:** Duration prediction, length regulation, pitch embedding, energy embedding.

**Classes:**
```python
class VariancePredictor(nn.Module)
    # 2-layer Conv1d → Linear → scalar per frame.
    # forward(x: (B,T,d), mask: (B,T)) -> (B,T)

class LengthRegulator(nn.Module)
    # Expand hidden states by integer durations.
    # forward(x: (B,T_text,d), durations: (B,T_text), max_len)
    #   -> (output: (B,T_mel,d), mel_lens: (B,))

class VarianceAdaptor(nn.Module)
    # Combines all three predictors + length regulator.
    # forward(enc_out, src_mask, mel_mask, durations, pitch_target,
    #         energy_target, max_mel_len, *_control)
    #   -> (output, mel_lens, dur_pred, pitch_pred, energy_pred)
```

**Self-test:** `python model/variance_adaptor.py`

---

### 11.6 model/decoder.py

**Purpose:** Mel spectrogram prediction from variance-adapted hidden states.

**Classes:**
```python
class Decoder(nn.Module)
    # Positional encoding + N FFTBlocks + LayerNorm + Linear(d→n_mels).
    # forward(x: (B,T_mel,d), mel_mask: (B,1,1,T_mel)) -> (B,T_mel,n_mels)
```

**Note:** `Decoder` imports `FFTBlock` and `PositionalEncoding` from `model/encoder.py` directly.
There is no duplicate implementation.

**Self-test:** `python model/decoder.py`

---

### 11.7 model/fastspeech2.py

**Purpose:** Assembles the full model and provides the inference interface.

**Classes:**
```python
class FastSpeech2(nn.Module)
    # Full model: Encoder + VarianceAdaptor + Decoder.
    #
    # forward(text_ids, durations=None, pitch_target=None,
    #         energy_target=None, mel_lens=None, max_mel_len=None,
    #         duration_control=1.0, pitch_control=1.0, energy_control=1.0)
    #   -> dict {mel_out, dur_pred, pitch_pred, energy_pred, mel_lens, mel_mask}
    #
    # infer(text_ids, duration_control, pitch_control, energy_control)
    #   -> dict  (convenience method: eval mode + no_grad)

def count_parameters(model: nn.Module) -> int
    # Count trainable parameters.
```

**Forward pass input/output shapes:**

Training:
```
text_ids:      (B, T_text)   — character IDs
durations:     (B, T_text)   — int, ground-truth mel frames per char
pitch_target:  (B, T_mel)    — normalized F0 per mel frame
energy_target: (B, T_mel)    — normalized energy per mel frame
mel_lens:      (B,)          — actual mel lengths (for mask)
max_mel_len:   int           — = mel_target.size(2)

→ mel_out:     (B, n_mels, T_mel)
→ dur_pred:    (B, T_text)
→ pitch_pred:  (B, T_mel)
→ energy_pred: (B, T_mel)
→ mel_lens:    (B,)
→ mel_mask:    (B, T_mel)
```

Inference:
```
text_ids: (1, T_text)   — single utterance
→ mel_out: (1, n_mels, T_mel)  — trim to mel_lens[0]
```

**Self-test:** `python model/fastspeech2.py` prints all output shapes and total parameter count.

---

### 11.8 train.py

**Purpose:** Complete training loop with all production features.

**Classes:**
```python
class NoamScheduler
    # Noam (warm-up + inverse sqrt decay) learning rate scheduler.
    # lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    # step()        — advance one optimizer step, update LR
    # zero_grad()   — delegate to optimizer
    # state_dict(), load_state_dict()  — for checkpointing
    # current_lr    — property: current learning rate value
```

**Functions:**
```python
compute_loss(outputs, mel_target, dur_target, pitch_target,
             energy_target, mel_lens, text_lens)
    -> (total_loss, loss_dict)
    # 4 components: mel L1 + duration MSE + pitch MSE + energy MSE
    # All masked to exclude padding frames.

save_checkpoint(model, optimizer, scheduler, epoch, step, loss) -> str
load_checkpoint(model, optimizer, scheduler, path) -> (epoch, step)
find_latest_checkpoint() -> str | None

validate(model, val_loader, amp_enabled, amp_dtype) -> float
    # One validation pass, returns average total loss.

train()
    # Main training loop. Handles:
    #   - DataLoader creation
    #   - Model instantiation
    #   - Optimizer + Noam scheduler
    #   - Mixed precision (bfloat16/float16)
    #   - GradScaler (float16 only)
    #   - Checkpoint resume
    #   - TensorBoard logging
    #   - Epoch summaries
    #   - Periodic validation
    #   - Periodic checkpointing
```

---

### 11.9 inference.py

**Purpose:** Load trained model and generate audio from Gujarati text.

**Functions:**
```python
load_fastspeech2(checkpoint_path: str) -> FastSpeech2
    # Load model weights from checkpoint.

load_hifigan(checkpoint_path, config_path) -> vocoder
    # Load pretrained HiFi-GAN vocoder.
    # Requires hifi-gan repo on sys.path.

synthesize(text, model, vocoder, output_path,
           duration_control, pitch_control, energy_control)
    -> np.ndarray
    # Full pipeline: text → IDs → mel → WAV → save.

main()
    # CLI entry point. Parses --text, --checkpoint, --output,
    # --duration_control, --pitch_control, --energy_control, --no_vocoder.
```

---

## 12. Training Deep Dive

### 12.1 Loss Functions

The total loss is a weighted sum of four individual losses:

```
L_total = w_mel * L_mel + w_dur * L_dur + w_pitch * L_pitch + w_energy * L_energy
```

All losses are masked: padding frames (where `mel_mask = True`) contribute zero loss.

#### Mel Loss (L1 / MAE)

```python
L_mel = mean(|mel_pred[t] - mel_target[t]|)  for all non-padding frames t
```

- **Why L1 (MAE) instead of L2 (MSE)?**
  L1 is less sensitive to outliers. Mel spectrograms have occasional extreme values in silence
  regions. L2 would heavily penalize these, causing the model to "play it safe" and predict
  overly smooth spectrograms (leading to muffled audio).

#### Duration Loss (MSE in log domain)

```python
L_dur = mean((dur_pred[i] - log(dur_target[i] + 1))^2)  for all non-padding chars i
```

- Predicted values are log-durations.
- Ground-truth values are converted to log domain: `log(dur + 1)` (the +1 prevents log(0)).
- **Why log domain?** A 1-frame error on a 2-frame phoneme is much more perceptible than a 1-frame
  error on a 20-frame phoneme. Log domain makes the loss proportionally scale with duration.

#### Pitch Loss (MSE)

```python
L_pitch = mean((pitch_pred[t] - pitch_target[t])^2)  for all non-padding frames t
```

#### Energy Loss (MSE)

```python
L_energy = mean((energy_pred[t] - energy_target[t])^2)  for all non-padding frames t
```

Both pitch and energy are in normalized form (zero mean, unit variance over voiced/non-silent
frames), so MSE measures normalized deviations.

---

### 12.2 Noam Learning Rate Scheduler

```
lr(step) = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

This implements:
- **Warmup phase** (step ≤ warmup_steps): LR increases linearly from 0 to peak.
  `lr ≈ step * d_model^(-0.5) * warmup_steps^(-1.5)`
- **Decay phase** (step > warmup_steps): LR decays as inverse square root.
  `lr ≈ d_model^(-0.5) * step^(-0.5)`

With `d_model = 256`, `warmup_steps = 4000`:
- Peak LR ≈ `256^(-0.5) * 4000^(-0.5)` × `4000^0.5 * warmup^(-1.5) / warmup^(-0.5)` ≈ 3.9e-4

**Why Noam?**
Standard fixed LR or step-decay schedules often cause early divergence in Transformer training.
Noam's warm-up prevents large gradient steps before the model's parameters have settled, and the
slow decay ensures the model continues learning without oscillating.

---

### 12.3 Mixed Precision Training

```python
USE_AMP   = True
AMP_DTYPE = "bfloat16"
```

PyTorch's `torch.autocast` is used:
```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
    outputs = model(...)
    loss, loss_dict = compute_loss(...)
```

**bfloat16 vs float16:**

| Property | float16 | bfloat16 |
|----------|---------|----------|
| Mantissa bits | 10 | 7 |
| Exponent bits | 5 | 8 |
| Dynamic range | Limited | Same as float32 |
| Overflow risk | High | Very low |
| GradScaler needed? | Yes | No |
| Available on | Most NVIDIA GPUs | Ampere+ (RTX 30xx, 40xx, 50xx) |

For Blackwell (RTX 50xx) and Ampere (RTX 30xx/40xx) GPUs, `bfloat16` is the clear choice.
For older GPUs (RTX 20xx, GTX 16xx), use `float16` and set the GradScaler enable flag.

**Speedup:** Approximately 1.5–2x training throughput with AMP.

---

### 12.4 Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_THRESH)
```

Clips the global L2 norm of all gradients to `GRAD_CLIP_THRESH = 1.0`.

**Why gradient clipping?**
Transformer models can produce large gradient spikes, especially early in training. Without
clipping, a single large gradient can corrupt all model weights. Clipping to 1.0 is standard for
Transformer TTS models.

---

### 12.5 Checkpointing and Resuming

**Checkpoint format:**
```python
{
    "epoch"    : int,        # epoch number when saved
    "step"     : int,        # global step count when saved
    "model"    : state_dict, # all model weights
    "optimizer": state_dict, # Adam optimizer state (momentum, variance)
    "scheduler": state_dict, # Noam scheduler step and rate
    "loss"     : float,      # average training loss for the epoch
}
```

**Checkpoint files:** `checkpoints/checkpoint_epoch{N:04d}.pt`
Examples: `checkpoint_epoch0050.pt`, `checkpoint_epoch0100.pt`

**Auto-resume:** When you run `python train.py`, it automatically finds and loads the file with the
highest epoch number in `checkpoints/`. No arguments needed.

**Loading a specific checkpoint for inference:**
```bash
python inference.py --text "..." --checkpoint ./checkpoints/checkpoint_epoch0500.pt
```

**Checkpoint frequency:** Every `SAVE_EVERY = 50` epochs. You can reduce this to 10 to have finer
recovery points (at the cost of more disk space).

---

### 12.6 TensorBoard Logging

TensorBoard is used for training visualization.

**Start TensorBoard:**
```bash
tensorboard --logdir ./logs
# Open http://localhost:6006
```

**Logged metrics (all per global step):**
| Tag | Logged when | Description |
|-----|-------------|-------------|
| `train/total_loss` | Every `LOG_EVERY` steps | Weighted sum of all 4 losses |
| `train/mel_loss` | Every `LOG_EVERY` steps | Mel L1 loss |
| `train/duration_loss` | Every `LOG_EVERY` steps | Duration MSE loss |
| `train/pitch_loss` | Every `LOG_EVERY` steps | Pitch MSE loss |
| `train/energy_loss` | Every `LOG_EVERY` steps | Energy MSE loss |
| `train/lr` | Every `LOG_EVERY` steps | Current learning rate |
| `val/total_loss` | Every `VAL_EVERY` epochs | Validation total loss |

---

## 13. Inference Deep Dive

### 13.1 Duration Control

```bash
python inference.py --text "..." --duration_control 0.8   # 20% faster
python inference.py --text "..." --duration_control 1.5   # 50% slower
```

**How it works:**
At inference, the duration predictor outputs `log_dur` for each character. This is converted to an
integer frame count:
```python
pred_dur = round(exp(log_dur) * duration_control).clamp(min=0)
```

The `duration_control` multiplier uniformly scales all predicted durations. Values < 1.0 produce
faster speech; > 1.0 produces slower speech. The ratio is applied before rounding.

**Practical range:** 0.5 (2x faster) to 2.0 (2x slower). Outside this range, audio quality
degrades.

---

### 13.2 Pitch Control

```bash
python inference.py --text "..." --pitch_control 1.2   # 20% higher pitch
python inference.py --text "..." --pitch_control 0.8   # 20% lower pitch
```

**How it works:**
The pitch predictor outputs normalized F0 values per mel frame. The `pitch_control` multiplier
scales these before they are quantized and looked up in the pitch embedding:
```python
pitch_pred = pitch_pred * pitch_control
```

Since pitch is in normalized (zero mean, unit variance) space, a `pitch_control` of 1.2 shifts
the F0 by 0.2 standard deviations — a noticeable but not extreme pitch change.

**Practical range:** 0.5 to 2.0.

---

### 13.3 Energy Control

```bash
python inference.py --text "..." --energy_control 1.3   # louder
python inference.py --text "..." --energy_control 0.7   # quieter
```

Works the same way as pitch control — the predicted energy values are scaled before embedding.

---

### 13.4 Running Without a Vocoder

If HiFi-GAN is not available, you can still generate mel spectrograms:

```bash
python inference.py --text "ગુજરાત" --no_vocoder
# Saves: output/ગુજ_mel.npy
```

To visualize the mel spectrogram:
```python
import numpy as np
import matplotlib.pyplot as plt

mel = np.load("output/ગુજ_mel.npy")   # (n_mels, T)
plt.figure(figsize=(12, 4))
plt.imshow(mel, aspect="auto", origin="lower")
plt.colorbar()
plt.title("Predicted Mel Spectrogram")
plt.xlabel("Frames")
plt.ylabel("Mel bins")
plt.savefig("mel_plot.png")
plt.show()
```

---

## 14. Self-Tests

Each module has a self-test you can run independently to verify it works:

```bash
# Verify vocabulary and text pipeline
python config.py
python data/text.py

# Verify audio feature extraction and dataset loading
python data/dataset.py

# Verify individual model components
python model/encoder.py
python model/variance_adaptor.py
python model/decoder.py
python model/fastspeech2.py   # also prints total parameter count
```

**Expected parameter count:**
Running `python model/fastspeech2.py` should print approximately:
```
Total parameters: ~28,000,000   (28 million)
```

The exact count depends on `VOCAB_SIZE`, which varies based on how many characters are in
`GUJARATI_CHARS`.

**Run all tests in sequence:**
```bash
for f in config.py data/text.py data/dataset.py \
          model/encoder.py model/variance_adaptor.py \
          model/decoder.py model/fastspeech2.py; do
    echo "=== Testing $f ==="
    python "$f"
    echo ""
done
```

---

## 15. Troubleshooting

### 15.1 CUDA / GPU Issues

**Problem:** `DEVICE = "cpu"` even though you have a GPU.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`:
- Reinstall PyTorch with the correct CUDA index URL (see Section 9.3).
- Check that `nvidia-smi` works in your terminal.
- On Windows: make sure you're not in a virtual environment that shadows the system CUDA.

**Problem:** `CUDA error: device-side assert triggered`

This usually means invalid tensor values (negative indices, out-of-bounds embeddings).
Common cause: a character ID outside `[0, VOCAB_SIZE)` being passed to the embedding layer.
Check that `CHAR_TO_ID` covers all characters in your text after cleaning.

Run with:
```bash
CUDA_LAUNCH_BLOCKING=1 python train.py
```
to get a traceback pointing to the exact line.

---

### 15.2 OOM (Out of Memory) Errors

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions in order of severity:**

1. Reduce `BATCH_SIZE` in `config.py` (try 8, then 4).
2. Reduce `MAX_MEL_LEN` to filter out very long utterances (try 1000).
3. Set `NUM_WORKERS = 0` (reduces memory used by DataLoader workers).
4. Reduce model size: `ENCODER_HIDDEN_DIM = 128`, `ENCODER_N_LAYERS = 2`.
5. Ensure no other processes are using your GPU: `nvidia-smi`.
6. Add `torch.cuda.empty_cache()` after each epoch (not a permanent fix but helps debugging).

---

### 15.3 Preprocessing Failures

**Problem:** Many "Failed" entries during preprocessing.

Common causes:
- WAV file path mismatch: Check that `WAV_DIR` in `config.py` points to your `wavs/` folder.
- Missing WAV files: Some entries in `metadata.csv` may reference files that don't exist.
- WAV files too short/long: Audio outside 0.5–15s is skipped (this is expected for short words
  like single-character entries).
- Librosa errors: Very corrupted WAV files will throw exceptions and be counted as "Failed".

**Debug a specific file:**
```python
from data.dataset import load_wav, compute_mel_spectrogram, compute_pitch, compute_energy
audio = load_wav("wavs/gujarati_0000100.wav")
mel   = compute_mel_spectrogram(audio)
print(f"Audio: {audio.shape}, Mel: {mel.shape}")
```

**Problem:** `[WARN] Skipping malformed row` for many rows.

Check that `metadata.csv` uses `|` as the delimiter, not `,` or `\t`. Also verify there are no
extra blank lines.

---

### 15.4 Training Divergence

**Problem:** Loss becomes `NaN` or spikes upward suddenly.

**Solutions:**
1. Disable AMP first: set `USE_AMP = False` in `config.py`.
2. Switch from `bfloat16` to `float16` (or CPU fp32).
3. Reduce `LEARNING_RATE` to `5e-4`.
4. Increase `WARMUP_STEPS` to `8000`.
5. Reduce `GRAD_CLIP_THRESH` to `0.5`.
6. Check for bad data: a single corrupt preprocessed file can cause NaN. Try running preprocessing
   again with `force=True`.

**Problem:** Loss plateaus very early (e.g., stops improving after epoch 50).

- This often means the warmup period was too short. Check that `WARMUP_STEPS` matches your
  dataset size.
- Verify that preprocessed durations are not all identical (this would mean text encoding is broken).
- Try increasing `ENCODER_N_LAYERS` and `DECODER_N_LAYERS`.

---

### 15.5 Bad Audio Quality

**Problem:** Generated audio is muffled, robotic, or has artifacts.

**Causes and fixes:**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Muffled, quiet audio | Not enough training | Train for more epochs (500+) |
| Rhythmically wrong | Bad durations | Use MFA for accurate durations |
| Buzzy / metallic tone | Vocoder mismatch | Retrain HiFi-GAN on your mel parameters |
| Monotone pitch | Pitch predictor not learning | Increase `PITCH_LOSS_WEIGHT` to 2.0 |
| Clipping/distortion | Vocoder input scaling | Normalize mel to [-4, 4] range |
| Wrong phonemes | Text pipeline dropping chars | Check `clean_text()` output |
| Speech too fast/slow | Duration predictor issue | Use `--duration_control` to compensate |

**Problem:** English / non-Gujarati words sound wrong.

This is expected — the model only knows the Gujarati characters in its vocabulary. Any Latin-script
words are dropped by `remove_out_of_vocab()`. You need to either pre-transliterate them or add
Latin characters to the vocabulary.

---

### 15.6 Import Errors

**Problem:** `ModuleNotFoundError: No module named 'data'`

Run scripts from the project root:
```bash
cd /path/to/tts-model-from-scratch
python train.py      # correct
python data/train.py # wrong
```

**Problem:** `ModuleNotFoundError: No module named 'models'` (during inference with HiFi-GAN)

Add the HiFi-GAN directory to sys.path in `inference.py`:
```python
sys.path.insert(0, "/absolute/path/to/hifi-gan")
```

**Problem:** `ModuleNotFoundError: No module named 'librosa'`

```bash
pip install librosa soundfile
```

---

## 16. Extending the Project

### 16.1 Adding MFA Alignments

Montreal Forced Aligner (MFA) produces exact phoneme/character-level timing from audio + transcript.
This will significantly improve speech naturalness compared to uniform duration distribution.

**Installation:**
```bash
conda install -c conda-forge montreal-forced-aligner
```

**Prepare pronunciation dictionary:**
MFA needs a pronunciation dictionary mapping words to phonemes. For Gujarati:
```
ગુજરાત  ɡ u dʒ ə r ɑː t
```

You can create a character-based dictionary where each character is its own "phoneme":
```python
# generate_dict.py
from config import CHAR_TO_ID

with open("gujarati_dict.txt", "w", encoding="utf-8") as f:
    for char in CHAR_TO_ID:
        if char not in ("<pad>", "<unk>"):
            f.write(f"{char}\t{char}\n")
```

**Run MFA:**
```bash
mfa align wavs/ gujarati_dict.txt gujarati_acoustic_model alignments/
```

**Parse TextGrid output:**
MFA produces `.TextGrid` files with exact character timing. Replace
`compute_durations_from_mel()` in `dataset.py` with a function that reads these files:

```python
import textgrid

def compute_durations_from_textgrid(tg_path, text_ids, mel_len):
    tg = textgrid.TextGrid.fromFile(tg_path)
    tier = tg[0]   # first tier = character alignments
    durations = []
    for interval in tier:
        n_frames = round(interval.duration() * SAMPLE_RATE / HOP_LENGTH)
        durations.append(max(1, n_frames))
    # Adjust to match exact mel_len
    diff = sum(durations) - mel_len
    # distribute adjustment ...
    return np.array(durations, dtype=np.int32)
```

---

### 16.2 Multi-Speaker Support

To extend for multiple Gujarati speakers:

1. **Add a speaker embedding** to the model:
```python
# in fastspeech2.py
self.speaker_embed = nn.Embedding(n_speakers, d_model)

# in forward()
speaker_emb = self.speaker_embed(speaker_ids)   # (B, d_model)
enc_out = enc_out + speaker_emb.unsqueeze(1)    # broadcast over T_text
```

2. **Update `metadata.csv`** to include a speaker ID column:
```
audio|text|speaker_id
gujarati_0000000.wav|transcript|0
```

3. **Update `dataset.py`** to load `speaker_id` and pass it to the model.

4. **Update `inference.py`** to accept `--speaker_id` argument.

---

### 16.3 Emotion / Style Control

FastSpeech 2's explicit variance controls (duration, pitch, energy) are already the foundation for
emotional speech. To add explicit emotion labels:

1. **Categorical emotion embedding:**
```python
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]
self.emotion_embed = nn.Embedding(len(EMOTIONS), d_model)
```

2. **Reference encoder** (for zero-shot style transfer):
A CNN-based reference encoder extracts style from a reference audio clip and injects it as a style
embedding. This allows "synthesize in the style of this reference clip" without explicit labels.

3. **Global Style Tokens (GST):**
A learned bank of style tokens where the model learns to select a combination for each utterance
from a reference audio.

---

### 16.4 Fine-Tuning on New Data

To continue training from an existing checkpoint on new Gujarati data:

1. **Add new audio** to `wavs/` and new entries to `metadata.csv`.
2. **Run preprocessing** on just the new files:
```python
from data.dataset import preprocess_and_cache
from data.text import clean_text

preprocess_and_cache("new_file_001", clean_text("new transcript"))
```
3. **Lower the learning rate** for fine-tuning:
```python
LEARNING_RATE = 1e-4   # 10x smaller than initial
WARMUP_STEPS  = 0      # no warmup needed
```
4. **Load the checkpoint** and train:
```bash
python train.py   # auto-resumes from latest checkpoint
```

---

### 16.5 Training Your Own Vocoder

For best audio quality with Gujarati speech, train HiFi-GAN on your Gujarati data:

1. **Prepare training data** — the same mel spectrograms computed by `data/dataset.py`.
2. **Clone HiFi-GAN:** `git clone https://github.com/jik876/hifi-gan`
3. **Configure** `config_v1.json` to match this project's mel parameters:
```json
{
    "num_mels": 80,
    "sampling_rate": 22050,
    "fft_size": 1024,
    "win_size": 1024,
    "hop_size": 256,
    "fmin": 0,
    "fmax": 8000
}
```
4. **Train HiFi-GAN** on your Gujarati mel spectrograms.
5. **Replace** `VOCODER_CHECKPOINT` in `config.py` with your new checkpoint.

Training a vocoder on in-domain data (Gujarati speech) will noticeably improve naturalness
compared to using the English-trained LJ Speech checkpoint.

---

## 17. Theoretical Background

### 17.1 Transformer Architecture

The core building block is the multi-head attention mechanism:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Where:
- Q (Query), K (Key), V (Value) are linear projections of the input.
- `d_k` is the head dimension (used for scaling to prevent vanishing gradients).
- Multiple attention heads (`n_heads`) allow the model to attend to different aspects
  simultaneously (e.g., one head for local context, another for long-range dependencies).

The FFT block in FastSpeech 2 is a simplified Transformer block:
- No cross-attention (encoder attends only to itself, decoder attends only to itself).
- Positional encoding is fixed (sinusoidal), not learned.
- FFN uses Conv1d instead of Linear for local context.

---

### 17.2 FastSpeech 2 vs Tacotron 2

**Tacotron 2** generates mel spectrograms autoregressively:
- Each output frame depends on the previous frame (RNN-based decoder).
- Must learn an attention alignment between text and audio from data.
- Attention failures (misalignment) are a common training instability.
- Inference is sequential: cannot parallelize.

**FastSpeech 2** generates all frames in parallel:
- Duration predictor explicitly tells the model how long each character lasts.
- Length regulator maps text → mel alignment deterministically.
- No attention to learn between text and audio.
- Inference is parallel: all frames computed simultaneously.
- Training requires ground-truth durations (from preprocessing or MFA).

The key insight of FastSpeech 2: if you know durations externally, you don't need attention at all.
This eliminates the hardest learning problem in Tacotron 2.

---

### 17.3 The Mel Spectrogram

A mel spectrogram represents audio as a 2D image where:
- X axis = time (frames, one per `HOP_LENGTH` samples)
- Y axis = frequency (mel scale, not linear Hz)
- Pixel intensity = log energy at that time-frequency point

**Why mel scale?**
The mel scale approximates human pitch perception — equal distances on the mel scale sound like
equal pitch differences to humans. We perceive frequency logarithmically, so 100 Hz → 200 Hz
sounds like the same interval as 1000 Hz → 2000 Hz.

**Why 80 bins?**
80 mel bins give enough resolution for TTS while remaining computationally tractable. HiFi-GAN was
designed and trained with 80 bins.

**Why log compression?**
Speech energy spans a huge dynamic range (quiet consonants to loud vowels). Log compression
brings these into a comparable range that neural networks can learn from.

---

### 17.4 Pitch and Energy as Variance Signals

**Pitch (F0)** is the fundamental frequency of the voice:
- Determines whether speech sounds high or low.
- Varies within a sentence to convey questions, statements, emotions.
- In Gujarati, sentence-final rising intonation signals questions.
- Unvoiced frames (silence, fricatives like /s/, /sh/) have no F0.

**Energy** is the loudness of speech per frame:
- Stressed syllables have higher energy.
- Sentence-final syllables often reduce in energy.
- Energy correlates with both emotional intensity and clarity.

FastSpeech 2 explicitly models these as separate predictors so they can be independently
controlled at inference time — the key enabler for emotional speech synthesis.

---

### 17.5 Non-Autoregressive Generation

In autoregressive models (Tacotron, WaveNet), generating frame t requires all frames 0..t-1:
```
x_0 → x_1 → x_2 → ... → x_T   (sequential, cannot parallelize)
```

FastSpeech 2 is **non-autoregressive**: all output frames are computed in a single forward pass:
```
[x_0, x_1, x_2, ..., x_T]  ← all generated simultaneously
```

This means:
- **Inference speed** scales with model size, not sequence length.
- A 100-frame sequence takes the same time as a 1000-frame sequence.
- This is the primary reason FastSpeech 2 is practical for real-time TTS.

The trade-off: the model must receive all alignment information upfront (durations from the
duration predictor), rather than learning alignment through attention.

---

## 18. Hardware Recommendations

### Minimum (training possible)
- GPU: GTX 1080 Ti (11 GB) or RTX 2060 (6 GB, reduce BATCH_SIZE to 4)
- RAM: 16 GB
- Storage: 50 GB SSD for dataset + preprocessed features

### Recommended (comfortable training)
- GPU: RTX 3080 (10 GB) or RTX 3090 (24 GB)
- RAM: 32 GB
- Storage: 100 GB NVMe SSD

### Optimal (fast training)
- GPU: RTX 4090 (24 GB) or RTX 5060 Ti (16 GB, Blackwell bfloat16)
- RAM: 64 GB
- Storage: 200 GB NVMe SSD

### Cloud alternatives
If you don't have a suitable GPU:
- **Google Colab Pro+**: A100 (40 GB) — fastest, but session time limits
- **Lambda Labs**: On-demand A100/H100 — pay per hour
- **Vast.ai**: Cheap community GPU rentals
- **RunPod**: Good balance of cost and availability

**Estimated training cost on cloud (1000 epochs, ~25k samples):**
- A100 @ $1.50/hr × 8 hours ≈ $12
- RTX 4090 @ $0.70/hr × 12 hours ≈ $8

---

## 19. Project Roadmap

### Phase 1: Core TTS (Current)
- [x] FastSpeech 2 architecture
- [x] Gujarati character vocabulary
- [x] Text cleaning pipeline (numbers, abbreviations, Unicode)
- [x] Audio preprocessing (mel, pitch, energy, duration)
- [x] Training loop (Noam scheduler, AMP, checkpointing, TensorBoard)
- [x] Inference with HiFi-GAN vocoder
- [x] Duration / pitch / energy control

### Phase 2: Quality Improvements (Next)
- [ ] Montreal Forced Aligner (MFA) integration for accurate durations
- [ ] Phoneme-level processing instead of character-level
- [ ] HiFi-GAN fine-tuning on Gujarati data
- [ ] Better number expansion (compound forms like "બેતાળીસ")
- [ ] More complete abbreviation dictionary

### Phase 3: Advanced Features
- [ ] Multi-speaker support (speaker embeddings)
- [ ] Emotion / style control (emotion embeddings or GST)
- [ ] Voice conversion (transfer style between speakers)
- [ ] Streaming inference (real-time audio generation)

### Phase 4: Production
- [ ] ONNX / TorchScript export for deployment
- [ ] REST API server
- [ ] Web demo interface
- [ ] Model compression (knowledge distillation)
- [ ] Mobile-optimized inference

---

## 20. References

### Papers

1. **FastSpeech 2** — Yi Ren et al., 2020.
   *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech.*
   ICLR 2021. https://arxiv.org/abs/2006.04558

2. **FastSpeech** — Yi Ren et al., 2019 (original FastSpeech).
   *FastSpeech: Fast, Robust and Controllable Text to Speech.*
   NeurIPS 2019. https://arxiv.org/abs/1905.09263

3. **Attention Is All You Need** — Vaswani et al., 2017.
   The original Transformer paper. Defines the multi-head attention and positional encoding
   used in this project. https://arxiv.org/abs/1706.03762

4. **HiFi-GAN** — Jungil Kong et al., 2020.
   *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.*
   NeurIPS 2020. https://arxiv.org/abs/2010.05646

5. **pYIN** — Mauch and Dixon, 2014.
   *pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions.*
   ICASSP 2014. Used in librosa for pitch extraction.

6. **Tacotron 2** — Jonathan Shen et al., 2018.
   *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions.*
   ICASSP 2018. https://arxiv.org/abs/1712.05884

### Code Resources

- **HiFi-GAN official implementation:** https://github.com/jik876/hifi-gan
- **FastSpeech 2 reference implementation:** https://github.com/ming024/FastSpeech2
- **librosa documentation:** https://librosa.org/doc/latest/index.html
- **PyTorch documentation:** https://pytorch.org/docs/
- **Montreal Forced Aligner:** https://montreal-forced-aligner.readthedocs.io/

### Gujarati Language Resources

- **Gujarati Unicode block (U+0A80–U+0AFF):**
  https://unicode.org/charts/PDF/U0A80.pdf
- **Gujarati script overview:**
  https://en.wikipedia.org/wiki/Gujarati_alphabet
- **OpenSLR Gujarati datasets:**
  https://www.openslr.org/  (search "Gujarati")
- **AI4Bharat speech data:**
  https://ai4bharat.iitm.ac.in/  (includes Gujarati TTS data)

---

*This README is the single source of truth for the Gujarati FastSpeech 2 TTS project.*
*Last updated: 2026-03-30*
