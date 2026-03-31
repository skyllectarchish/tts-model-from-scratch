"""
train.py — Training loop for Gujarati FastSpeech 2

Usage:
    # Step 1 — preprocess audio (run once):
    python -c "from data.dataset import run_preprocessing; run_preprocessing()"

    # Step 2 — train:
    python train.py

Checkpoints → CHECKPOINT_DIR  (every SAVE_EVERY epochs)
TensorBoard logs → LOG_DIR
    tensorboard --logdir ./logs
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR, LOG_DIR,
    EPOCHS, SAVE_EVERY, KEEP_LAST_N_CHECKPOINTS, LOG_EVERY, VAL_EVERY,
    LEARNING_RATE, BETAS, EPS, WEIGHT_DECAY, WARMUP_STEPS,
    GRAD_CLIP_THRESH, USE_AMP, AMP_DTYPE,
    MEL_LOSS_WEIGHT, DURATION_LOSS_WEIGHT, PITCH_LOSS_WEIGHT, ENERGY_LOSS_WEIGHT,
    ENCODER_HIDDEN_DIM, DEVICE,
)
from data.dataset import get_dataloaders
from model.fastspeech2 import FastSpeech2, count_parameters


# ---------------------------------------------------------------------------
# Noam learning rate scheduler
# ---------------------------------------------------------------------------

class NoamScheduler:
    """
    Noam warm-up + inverse-sqrt decay scheduler from the original Transformer paper.

        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self._step        = 0
        self._rate        = 0.0

    def step(self):
        self._step += 1
        rate = self._compute_lr()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _compute_lr(self) -> float:
        s = self._step
        return (self.d_model ** -0.5) * min(s ** -0.5, s * self.warmup_steps ** -1.5)

    @property
    def current_lr(self) -> float:
        return self._rate

    def state_dict(self) -> dict:
        return {"step": self._step, "rate": self._rate}

    def load_state_dict(self, state: dict):
        self._step = state["step"]
        self._rate = state["rate"]


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(
    outputs:       dict,
    mel_target:    torch.Tensor,   # (B, n_mels, T_mel)
    dur_target:    torch.Tensor,   # (B, T_text) int
    pitch_target:  torch.Tensor,   # (B, T_mel)
    energy_target: torch.Tensor,   # (B, T_mel)
    mel_lens:      torch.Tensor,   # (B,)
    text_lens:     torch.Tensor,   # (B,)
) -> tuple:
    """
    FastSpeech 2 total loss = mel + duration + pitch + energy losses.

    All losses are mean-reduced over non-padding frames only.

    Returns:
        total_loss: scalar tensor (backpropagated)
        loss_dict:  {"total", "mel", "duration", "pitch", "energy"} — for logging
    """
    mel_pred    = outputs["mel_out"]      # (B, n_mels, T_mel)
    dur_pred    = outputs["dur_pred"]     # (B, T_text)
    pitch_pred  = outputs["pitch_pred"]   # (B, T_mel)
    energy_pred = outputs["energy_pred"]  # (B, T_mel)
    mel_mask    = outputs["mel_mask"]     # (B, T_mel)  True = padding

    B    = mel_target.size(0)
    T_t  = dur_target.size(1)
    device = mel_target.device

    # Text padding mask: True where text is padding
    ids       = torch.arange(T_t, device=device).unsqueeze(0).expand(B, -1)
    text_mask = ids >= text_lens.unsqueeze(1)   # (B, T_text)

    # ---- 1. Mel loss (L1, masked) ----
    mel_pred_t   = mel_pred.transpose(1, 2)      # (B, T_mel, n_mels)
    mel_target_t = mel_target.transpose(1, 2)
    mel_loss = F.l1_loss(mel_pred_t, mel_target_t, reduction="none")  # (B, T_mel, n_mels)
    mel_loss = mel_loss.mean(dim=-1)             # (B, T_mel)
    n_mel    = (~mel_mask).float().sum().clamp(min=1)
    mel_loss = mel_loss.masked_fill(mel_mask, 0.0).sum() / n_mel

    # ---- 2. Duration loss (MSE on log domain) ----
    log_dur_target = torch.log(dur_target.float() + 1.0)
    dur_loss = F.mse_loss(dur_pred, log_dur_target, reduction="none")
    n_text   = (~text_mask).float().sum().clamp(min=1)
    dur_loss = dur_loss.masked_fill(text_mask, 0.0).sum() / n_text

    # ---- 3. Pitch loss (MSE, masked) ----
    pitch_loss = F.mse_loss(pitch_pred, pitch_target, reduction="none")
    pitch_loss = pitch_loss.masked_fill(mel_mask, 0.0).sum() / n_mel

    # ---- 4. Energy loss (MSE, masked) ----
    energy_loss = F.mse_loss(energy_pred, energy_target, reduction="none")
    energy_loss = energy_loss.masked_fill(mel_mask, 0.0).sum() / n_mel

    # ---- Total ----
    total = (
        MEL_LOSS_WEIGHT      * mel_loss    +
        DURATION_LOSS_WEIGHT * dur_loss    +
        PITCH_LOSS_WEIGHT    * pitch_loss  +
        ENERGY_LOSS_WEIGHT   * energy_loss
    )

    loss_dict = {
        "total"   : total.item(),
        "mel"     : mel_loss.item(),
        "duration": dur_loss.item(),
        "pitch"   : pitch_loss.item(),
        "energy"  : energy_loss.item(),
    }

    return total, loss_dict


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model, optimizer, scheduler,
    epoch: int, step: int, loss: float,
) -> str | None:
    """
    Save a checkpoint and enforce the KEEP_LAST_N_CHECKPOINTS policy:
      KEEP_LAST_N_CHECKPOINTS == 0  →  skip saving entirely
      KEEP_LAST_N_CHECKPOINTS == -1 →  save, keep all (no deletion)
      KEEP_LAST_N_CHECKPOINTS >  0  →  save, then delete oldest so only N remain
    """
    if KEEP_LAST_N_CHECKPOINTS == 0:
        print("  [Checkpoint] Skipped (KEEP_LAST_N_CHECKPOINTS = 0)")
        return None

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch:04d}.pt")
    torch.save({
        "epoch"    : epoch,
        "step"     : step,
        "model"    : model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss"     : loss,
    }, path)
    print(f"  Saved: {path}")

    # Prune old checkpoints if a limit is set
    if KEEP_LAST_N_CHECKPOINTS > 0:
        all_ckpts: list[str] = sorted(
            [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
        )
        n_delete = max(0, len(all_ckpts) - KEEP_LAST_N_CHECKPOINTS)
        to_delete = all_ckpts[:n_delete]   # oldest checkpoints beyond the keep limit
        for fname in to_delete:
            old_path = os.path.join(CHECKPOINT_DIR, fname)
            os.remove(old_path)
            print(f"  Deleted old checkpoint: {old_path}")

    return path


def load_checkpoint(model, optimizer, scheduler, path: str) -> tuple:
    """Load checkpoint. Returns (epoch, global_step)."""
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  Resumed: {path}  (epoch {ckpt['epoch']}, step {ckpt['step']})")
    return ckpt["epoch"], ckpt["step"]


def find_latest_checkpoint() -> str | None:
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    ckpts = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt"))
    return os.path.join(CHECKPOINT_DIR, ckpts[-1]) if ckpts else None


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

def validate(model, val_loader, amp_enabled: bool, amp_dtype) -> float:
    model.eval()
    total = 0.0
    n     = 0

    with torch.no_grad():
        for batch in val_loader:
            text_ids      = batch["text_ids"].to(DEVICE)
            mel_target    = batch["mel"].to(DEVICE)
            pitch_target  = batch["pitch"].to(DEVICE)
            energy_target = batch["energy"].to(DEVICE)
            durations     = batch["durations"].to(DEVICE)
            mel_lens      = batch["mel_lens"].to(DEVICE)
            text_lens     = batch["text_lens"].to(DEVICE)
            max_mel_len   = mel_target.size(2)

            with torch.autocast(device_type=DEVICE, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    text_ids,
                    durations=durations, pitch_target=pitch_target,
                    energy_target=energy_target, mel_lens=mel_lens,
                    max_mel_len=max_mel_len,
                )
                _, loss_dict = compute_loss(
                    outputs,
                    mel_target=mel_target, dur_target=durations,
                    pitch_target=pitch_target, energy_target=energy_target,
                    mel_lens=mel_lens, text_lens=text_lens,
                )

            total += loss_dict["total"]
            n     += 1

    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    print("=" * 60)
    print("  Gujarati FastSpeech 2 — Training")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  VRAM   : {vram:.1f} GB")

    # ---- Data ----
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders()
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # ---- Model ----
    model = FastSpeech2().to(DEVICE)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY,
    )
    scheduler = NoamScheduler(optimizer, d_model=ENCODER_HIDDEN_DIM, warmup_steps=WARMUP_STEPS)

    # ---- Mixed precision ----
    amp_enabled = USE_AMP and DEVICE == "cuda"
    amp_dtype   = torch.bfloat16 if AMP_DTYPE == "bfloat16" else torch.float16
    # GradScaler is only useful for float16; bfloat16 doesn't need it
    use_scaler  = amp_enabled and amp_dtype == torch.float16
    scaler      = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ---- Resume ----
    start_epoch = 1
    global_step = 0
    latest      = find_latest_checkpoint()
    if latest:
        epoch_done, global_step = load_checkpoint(model, optimizer, scheduler, latest)
        start_epoch = epoch_done + 1

    # ---- TensorBoard ----
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    print(f"\nStarting from epoch {start_epoch} / {EPOCHS}")
    print("-" * 60)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_losses = {k: 0.0 for k in ("total", "mel", "duration", "pitch", "energy")}
        n_batches = 0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:04d}", leave=False):
            text_ids      = batch["text_ids"].to(DEVICE)
            mel_target    = batch["mel"].to(DEVICE)
            pitch_target  = batch["pitch"].to(DEVICE)
            energy_target = batch["energy"].to(DEVICE)
            durations     = batch["durations"].to(DEVICE)
            mel_lens      = batch["mel_lens"].to(DEVICE)
            text_lens     = batch["text_lens"].to(DEVICE)
            max_mel_len   = mel_target.size(2)

            scheduler.zero_grad()

            with torch.autocast(device_type=DEVICE, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    text_ids,
                    durations=durations, pitch_target=pitch_target,
                    energy_target=energy_target, mel_lens=mel_lens,
                    max_mel_len=max_mel_len,
                )
                loss, loss_dict = compute_loss(
                    outputs,
                    mel_target=mel_target, dur_target=durations,
                    pitch_target=pitch_target, energy_target=energy_target,
                    mel_lens=mel_lens, text_lens=text_lens,
                )

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_THRESH)
                scaler.step(optimizer)
                scaler.update()
                # Manually advance scheduler step counter without calling optimizer.step() again
                scheduler._step += 1
                scheduler._rate  = scheduler._compute_lr()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_THRESH)
                scheduler.step()   # calls optimizer.step() internally

            global_step += 1

            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1

            if global_step % LOG_EVERY == 0:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}_loss", v, global_step)
                writer.add_scalar("train/lr", scheduler.current_lr, global_step)

        # ---- Epoch summary ----
        elapsed = time.time() - t0
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        print(
            f"Epoch {epoch:04d} | "
            f"loss={avg['total']:.4f} | "
            f"mel={avg['mel']:.4f} | "
            f"dur={avg['duration']:.4f} | "
            f"pitch={avg['pitch']:.4f} | "
            f"energy={avg['energy']:.4f} | "
            f"lr={scheduler.current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # ---- Validation ----
        if epoch % VAL_EVERY == 0:
            val_loss = validate(model, val_loader, amp_enabled, amp_dtype)
            writer.add_scalar("val/total_loss", val_loss, global_step)
            print(f"  Val loss: {val_loss:.4f}")
            model.train()

        # ---- Checkpoint ----
        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, avg["total"])

    writer.close()
    print("\nTraining complete.")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"Run inference with:   python inference.py --text \"your text here\"")


if __name__ == "__main__":
    train()
