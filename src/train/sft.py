"""
Phase 5 — Supervised Fine-Tuning (SFT).

Loads pretrained checkpoint, fine-tunes on instruction pairs.
Loss computed ONLY on assistant response tokens (loss_mask).

Auto-resumes from the latest SFT checkpoint:
  - End-of-epoch:  checkpoints/sft/epoch_X/checkpoint.pt
  - Mid-epoch:     checkpoints/sft/epoch_X_step_Y/checkpoint.pt

⚠️  Switch Lightning AI to L4 (24GB) before running this.
Just run: python src/train/sft.py
"""

import os
import sys
import json
import math
import time
import random
import datetime
import shutil
from pathlib import Path

import numpy as np
import torch
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.config import ModelConfig
from src.model.model import SLM
from src.train.lr_schedule import get_lr

# ── CONFIG ────────────────────────────────────────────────────────────────────
PRETRAIN_CKPT_DIR = ROOT / "checkpoints" / "pretrain"
SFT_CKPT_DIR      = ROOT / "checkpoints" / "sft"
TRAIN_DATA        = ROOT / "data" / "sft" / "sft_train_tokenized.pt"
VAL_DATA          = ROOT / "data" / "sft" / "sft_val_tokenized.pt"
LOG_FILE          = ROOT / "logs" / "sft_loss.csv"
QUAL_LOG          = ROOT / "logs" / "sft_qualitative.txt"
STATUS_FILE       = ROOT / "status" / "phase5_sft_train.json"
FRESH_START       = os.environ.get("SFT_FRESH_START", "").lower() in {"1", "true", "yes"}

MAX_EPOCHS    = 5
LR            = 2e-5
MIN_LR        = 0.0
WARMUP_STEPS  = 100
WEIGHT_DECAY  = 0.01
GRAD_CLIP     = 1.0
MICRO_BATCH   = 16
GRAD_ACCUM    = 2
MAX_SEQ_LEN   = 4096
MID_EP_CKPT   = 500    # save mid-epoch checkpoint every N steps
PAD_ID        = 50259  # <|pad|>
LOSS_CHUNK_TOKENS = 256

PATIENCE      = 2      # early stopping patience
# ──────────────────────────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    def __init__(self, pt_file: Path, max_len: int = MAX_SEQ_LEN):
        self.samples = torch.load(pt_file, weights_only=False)
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        ids  = s["input_ids"][:self.max_len]
        mask = s["loss_mask"][:self.max_len]
        return ids, mask


def collate_fn(batch):
    """Pad to longest sequence in batch using PAD_ID."""
    ids_list, mask_list = zip(*batch)
    max_len = max(x.size(0) for x in ids_list)

    padded_ids  = torch.full((len(ids_list),  max_len), PAD_ID, dtype=torch.long)
    padded_mask = torch.zeros(len(mask_list), max_len,           dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
        n = ids.size(0)
        padded_ids[i,  :n] = ids
        padded_mask[i, :n] = mask

    return padded_ids, padded_mask


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def find_latest_pretrain_ckpt() -> Optional[Path]:
    ckpts = sorted(PRETRAIN_CKPT_DIR.glob("step_*"),
                   key=lambda p: int(p.name.split("_")[1]))
    return (ckpts[-1] / "checkpoint.pt") if ckpts else None


def find_latest_sft_ckpt() -> Tuple[Optional[Path], int, int]:
    """Returns (ckpt_path, epoch, step). -1 means no mid-epoch resume."""
    SFT_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Check mid-epoch checkpoints first (epoch_X_step_Y)
    mid_ckpts = sorted(
        SFT_CKPT_DIR.glob("epoch_*_step_*"),
        key=lambda p: (int(p.name.split("_")[1]), int(p.name.split("_")[3]))
    )
    if mid_ckpts:
        latest = mid_ckpts[-1]
        parts  = latest.name.split("_")
        return latest / "checkpoint.pt", int(parts[1]), int(parts[3])

    # Fall back to end-of-epoch checkpoints
    ep_ckpts = sorted(
        SFT_CKPT_DIR.glob("epoch_[0-9]*"),
        key=lambda p: int(p.name.split("_")[1])
    )
    if ep_ckpts:
        latest = ep_ckpts[-1]
        epoch  = int(latest.name.split("_")[1])
        return latest / "checkpoint.pt", epoch, -1

    return None, -1, -1


def save_sft_ckpt(model, optimizer, epoch, step, val_loss, best_epoch):
    if step >= 0:
        tag = f"epoch_{epoch}_step_{step}"
    else:
        tag = f"epoch_{epoch}"
    ckpt_dir = SFT_CKPT_DIR / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "epoch":            epoch,
        "step":             step,
        "val_response_loss": val_loss,
        "best_epoch":       best_epoch,
        "torch_rng":        torch.random.get_rng_state(),
        "numpy_rng":        np.random.get_state(),
        "python_rng":       random.getstate(),
    }, ckpt_dir / "checkpoint.pt")

    print(f"  ✓ SFT checkpoint → {tag}")


def update_best_model(model):
    """Copy current model weights to best_model.pt."""
    best_path = SFT_CKPT_DIR / "best_model.pt"
    torch.save({"model_state": model.state_dict()}, best_path)
    print(f"  ✓ Best model updated → {best_path}")


# ── Training helpers ──────────────────────────────────────────────────────────

def compute_response_loss(logits, labels, loss_mask):
    """Cross-entropy only on masked (response) tokens, computed in chunks."""
    bsz, seq_len, vocab = logits.shape
    denom = loss_mask.sum().clamp(min=1).float()
    loss_sum = logits.new_zeros(())

    for start in range(0, seq_len, LOSS_CHUNK_TOKENS):
        end = min(start + LOSS_CHUNK_TOKENS, seq_len)
        chunk_logits = logits[:, start:end, :].contiguous().view(-1, vocab)
        chunk_labels = labels[:, start:end].contiguous().view(-1)
        chunk_mask = loss_mask[:, start:end].contiguous().view(-1).float()

        chunk_loss = F.cross_entropy(
            chunk_logits,
            chunk_labels,
            reduction="none"
        )
        loss_sum = loss_sum + (chunk_loss * chunk_mask).sum()

    return loss_sum / denom


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    losses = []
    for ids, mask in loader:
        ids, mask = ids.to(device), mask.to(device)
        labels = ids[:, 1:].contiguous()
        inp    = ids[:, :-1].contiguous()
        m      = mask[:, 1:].contiguous()
        device_type = "cuda" if device == "cuda" else "cpu"
        with torch.autocast(device_type, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
            logits = model(inp)
            loss   = compute_response_loss(logits, labels, m)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    SFT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if FRESH_START:
        archive_root = ROOT / "checkpoints" / "sft_archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = archive_root / f"run_{ts}"
        if any(SFT_CKPT_DIR.iterdir()):
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(SFT_CKPT_DIR), str(archive_dir / "sft"))
            SFT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
            print(f"✓ Archived previous SFT checkpoints → {archive_dir / 'sft'}")
        if LOG_FILE.exists():
            shutil.move(str(LOG_FILE), str(ROOT / "logs" / f"sft_loss_{ts}.csv"))
        if QUAL_LOG.exists():
            shutil.move(str(QUAL_LOG), str(ROOT / "logs" / f"sft_qualitative_{ts}.txt"))
        if STATUS_FILE.exists():
            shutil.move(str(STATUS_FILE), str(ROOT / "status" / f"phase5_sft_train_{ts}.json"))
        print("✓ Starting a fresh SFT run from the latest pretrain checkpoint.")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU found. SFT on CPU will be slow.")
        print("Switch Lightning AI studio to L4 24GB before running.")
    elif device == "mps":
        print("✓ Using Apple Silicon MPS for training.")

    cfg   = ModelConfig()
    model = SLM(cfg, use_grad_ckpt=(device == "cuda" or device == "mps")).to(device)

    # ── Find checkpoint to start from ─────────────────────────────────────
    sft_ckpt, resume_epoch, resume_step = ((None, -1, -1) if FRESH_START else find_latest_sft_ckpt())

    if sft_ckpt and sft_ckpt.exists():
        ckpt = torch.load(sft_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = resume_epoch + (1 if resume_step < 0 else 0)
        start_step  = resume_step  if resume_step >= 0 else 0
        best_epoch  = ckpt.get("best_epoch", -1)
        best_val    = ckpt.get("val_response_loss", 999.0)
        optimizer   = model.configure_optimizers(WEIGHT_DECAY, LR, (0.9, 0.95), device)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        try:
            torch.random.set_rng_state(ckpt["torch_rng"])
            np.random.set_state(ckpt["numpy_rng"])
            random.setstate(ckpt["python_rng"])
        except Exception as e:
            print(f"WARNING: Could not restore RNG state: {e}")
        print(f"✓ Resuming SFT from epoch {start_epoch}, step {start_step}")
    else:
        # Load pretrain weights — fresh optimizer (do NOT copy pretrain optimizer)
        pretrain_ckpt = find_latest_pretrain_ckpt()
        if pretrain_ckpt and pretrain_ckpt.exists():
            ckpt = torch.load(pretrain_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"], strict=False)
            print(f"✓ Loaded pretrain weights from {pretrain_ckpt.parent.name}")
        else:
            print("WARNING: No pretrain checkpoint found. Starting SFT from random weights.")
        optimizer   = model.configure_optimizers(WEIGHT_DECAY, LR, (0.9, 0.95), device)
        start_epoch = 0
        start_step  = 0
        best_epoch  = -1
        best_val    = 999.0

    print(f"  Model params: {model.get_num_params()/1e6:.1f}M")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = SFTDataset(TRAIN_DATA)
    val_ds   = SFTDataset(VAL_DATA)
    train_loader = DataLoader(train_ds, batch_size=MICRO_BATCH, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=MICRO_BATCH, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    n_steps_per_epoch = math.ceil(len(train_ds) / (MICRO_BATCH * GRAD_ACCUM))
    max_steps         = MAX_EPOCHS * n_steps_per_epoch

    # ── CSV header ────────────────────────────────────────────────────────
    if not LOG_FILE.exists():
        LOG_FILE.write_text("epoch,step,response_loss,lr,split\n")

    # ── Training ──────────────────────────────────────────────────────────
    no_improve = 0
    global_step = start_epoch * n_steps_per_epoch + start_step

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")

        step_in_epoch = 0
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for batch_idx, (ids, mask) in enumerate(pbar):
            # Skip already-processed steps on resume
            if epoch == start_epoch and batch_idx < start_step:
                continue

            ids, mask = ids.to(device), mask.to(device)
            labels = ids[:, 1:].contiguous()
            inp    = ids[:, :-1].contiguous()
            m      = mask[:, 1:].contiguous()

            lr = get_lr(global_step, WARMUP_STEPS, max_steps, LR, MIN_LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            device_type = "cuda" if device == "cuda" else "cpu"
            with torch.autocast(device_type, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
                logits = model(inp)
                loss   = compute_response_loss(logits, labels, m) / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step     += 1
                step_in_epoch   += 1
                pbar.set_postfix({"loss": f"{accum_loss:.4f}", "lr": f"{lr:.1e}"})

                with open(LOG_FILE, "a") as lf:
                    lf.write(f"{epoch},{global_step},{accum_loss:.6f},{lr:.2e},train\n")
                accum_loss = 0.0

                # Mid-epoch checkpoint
                if step_in_epoch % MID_EP_CKPT == 0:
                    save_sft_ckpt(model, optimizer, epoch, step_in_epoch, best_val, best_epoch)
                    # Remove previous mid-epoch checkpoints to save space
                    for old in SFT_CKPT_DIR.glob(f"epoch_{epoch}_step_*"):
                        if old.name != f"epoch_{epoch}_step_{step_in_epoch}":
                            shutil.rmtree(old, ignore_errors=True)

        # ── End-of-epoch validation ────────────────────────────────────────
        val_loss = validate(model, val_loader, device)
        val_ppl  = math.exp(min(val_loss, 20))
        print(f"\nEpoch {epoch+1} | Train Loss: {accum_loss:.4f} | "
              f"Val Response Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}\n")
        with open(LOG_FILE, "a") as lf:
            lf.write(f"{epoch},{global_step},{val_loss:.6f},—,val\n")

        # Best model tracking
        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            no_improve = 0
            update_best_model(model)
        else:
            no_improve += 1

        save_sft_ckpt(model, optimizer, epoch, -1, val_loss, best_epoch)
        # Remove mid-epoch checkpoints for this completed epoch
        for old in SFT_CKPT_DIR.glob(f"epoch_{epoch}_step_*"):
            shutil.rmtree(old, ignore_errors=True)

        # Early stopping
        if no_improve >= PATIENCE:
            print(f"Early stopping: val loss did not improve for {PATIENCE} epochs.")
            break

    # ── Save status ────────────────────────────────────────────────────────
    status = {
        "phase": "sft_train", "done": True,
        "best_epoch": best_epoch, "best_val_loss": best_val,
        "timestamp": datetime.datetime.now().isoformat()
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    print(f"\n✓ SFT complete. Best epoch: {best_epoch+1} | Best val loss: {best_val:.4f}")
    print(f"  Best model saved to: {SFT_CKPT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
