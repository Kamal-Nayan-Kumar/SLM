"""
Phase 3 — Pretraining loop.

Auto-resumes from the latest checkpoint in checkpoints/pretrain/.
Just run:  python src/train/pretrain.py

⚠️  Requires A100 40GB GPU on Lightning AI.
"""

import os
import sys
import json
import math
import time
import random
import datetime
from pathlib import Path

import numpy as np
import torch
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.config import ModelConfig
from src.model.model import SLM
from src.train.lr_schedule import get_lr

# ── CONFIG ────────────────────────────────────────────────────────────────────
# (Edit these before launching — they live here, not in argparse)
TRAIN_BIN      = ROOT / "data" / "tokenized" / "train.bin"
VAL_BIN        = ROOT / "data" / "tokenized" / "val.bin"
CKPT_DIR       = ROOT / "checkpoints" / "pretrain"
LOG_FILE       = ROOT / "logs" / "pretrain_loss.csv"
STATUS_FILE    = ROOT / "status" / "phase3_pretrain.json"

MAX_TOKENS       = 3_000_000_000    # 3B — 1 epoch over 2.9B actual tokens (Chinchilla-optimal for 114M model)
MICRO_BATCH      = 10               # H100 80GB: 10 (A100 40GB: use 5)
GRAD_ACCUM       = 8                # effective batch = 10*8*4096 = 327,680 tokens (same as before)
SEQ_LEN          = 4096
GRAD_CLIP        = 1.0
CKPT_INTERVAL    = 1000             # save every N steps
VAL_INTERVAL     = 500
LOG_INTERVAL     = 10
KEEP_CKPTS       = 3                # keep only last N checkpoint dirs

MAX_LR       = 6e-4
MIN_LR       = 6e-5                 # 10% of max_lr
WARMUP_STEPS = 2_000
BETAS        = (0.9, 0.95)
WEIGHT_DECAY = 0.1

# Compute max_steps from token budget
MAX_STEPS = MAX_TOKENS // (MICRO_BATCH * GRAD_ACCUM * SEQ_LEN)
# ──────────────────────────────────────────────────────────────────────────────


def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU found. Training on CPU will be ~1000x slower.")
        print("Switch Lightning AI studio to A100 40GB before running this script.")
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def get_batch(data: np.memmap, seq_len: int, batch_size: int, device: str):
    """Sample batch_size random non-overlapping sequences."""
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x  = torch.stack([torch.from_numpy(data[i    : i + seq_len    ].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(data[i + 1: i + seq_len + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def find_latest_ckpt() -> Optional[Path]:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(CKPT_DIR.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return ckpts[-1] / "checkpoint.pt" if ckpts else None


def save_checkpoint(base_model, optimizer, global_step, tokens_processed):
    step_dir = CKPT_DIR / f"step_{global_step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state":     base_model.state_dict(),   # always raw module
        "optimizer_state": optimizer.state_dict(),
        "global_step":     global_step,
        "tokens_processed": tokens_processed,
        "torch_rng":       torch.random.get_rng_state(),
        "cuda_rng":        torch.cuda.random.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng":       np.random.get_state(),
        "python_rng":      random.getstate(),
    }, step_dir / "checkpoint.pt")

    # Prune old checkpoints — keep only KEEP_CKPTS most recent
    all_ckpts = sorted(CKPT_DIR.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    for old in all_ckpts[:-KEEP_CKPTS]:
        import shutil
        shutil.rmtree(old)

    print(f"  ✓ Checkpoint saved → step_{global_step:07d}")


@torch.no_grad()
def estimate_val_loss(model, val_data, device, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        xb, yb = get_batch(val_data, SEQ_LEN, MICRO_BATCH, device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(xb)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), yb.view(-1)
            )
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    device = setup_device()
    cfg = ModelConfig()

    # ── Build base model + optimizer BEFORE compile ────────────────────────
    # (always operate on the raw nn.Module for state_dict / optimizer params)
    latest_ckpt = find_latest_ckpt()
    base_model = SLM(cfg, use_grad_ckpt=True).to(device)
    optimizer  = base_model.configure_optimizers(WEIGHT_DECAY, MAX_LR, BETAS, device)

    global_step, tokens_processed = 0, 0

    if latest_ckpt and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        global_step      = ckpt["global_step"]
        tokens_processed = ckpt["tokens_processed"]
        torch.random.set_rng_state(ckpt["torch_rng"].cpu().byte())
        if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.random.set_rng_state(ckpt["cuda_rng"].cpu().byte())
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
        print(f"✓ Resuming from step {global_step} "
              f"({tokens_processed/1e9:.3f}B tokens processed)")
    else:
        print("Starting fresh pretraining.")

    # Compile AFTER checkpoint load — default mode avoids CUDA Graphs (safe with grad accum)
    if hasattr(torch, "compile") and device == "cuda":
        print("Compiling model with torch.compile()...")
        model = torch.compile(base_model, mode="default")
    else:
        model = base_model

    print(f"  Model params: {base_model.get_num_params()/1e6:.1f}M")
    print(f"  Max steps: {MAX_STEPS:,} | Max tokens: {MAX_TOKENS/1e9:.1f}B\n")

    # ── Load data ─────────────────────────────────────────────────────────
    train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_data   = np.memmap(VAL_BIN,   dtype=np.uint16, mode="r")

    # ── CSV header ────────────────────────────────────────────────────────
    if not LOG_FILE.exists():
        LOG_FILE.write_text("step,loss,lr,tokens_B,split\n")

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    t0 = time.time()

    pbar = tqdm(
        total=MAX_STEPS,
        initial=global_step,
        desc="Pretrain",
        unit="step",
        dynamic_ncols=True,
    )

    while tokens_processed < MAX_TOKENS:
        # Set LR for this step
        lr = get_lr(global_step, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(GRAD_ACCUM):
            xb, yb = get_batch(train_data, SEQ_LEN, MICRO_BATCH, device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(xb)
                loss   = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), yb.view(-1)
                ) / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()

        # Gradient clip then step
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        global_step      += 1
        tokens_processed += MICRO_BATCH * GRAD_ACCUM * SEQ_LEN
        pbar.update(1)
        pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr:.2e}",
                         tok=f"{tokens_processed/1e9:.3f}B")

        # ── Logging ───────────────────────────────────────────────────────
        if global_step % LOG_INTERVAL == 0:
            dt = (time.time() - t0) * 1000 / LOG_INTERVAL
            t0 = time.time()
            tqdm.write(f"step {global_step:6d} | loss {accum_loss:.4f} | "
                       f"lr {lr:.2e} | tokens {tokens_processed/1e9:.3f}B | "
                       f"{dt:.0f}ms/step")
            with open(LOG_FILE, "a") as lf:
                lf.write(f"{global_step},{accum_loss:.6f},{lr:.2e},"
                         f"{tokens_processed/1e9:.4f},train\n")

        # ── Validation ────────────────────────────────────────────────────
        if global_step % VAL_INTERVAL == 0:
            val_loss = estimate_val_loss(model, val_data, device)
            val_ppl  = math.exp(min(val_loss, 20))
            tqdm.write(f"  VAL step {global_step} | val_loss {val_loss:.4f} | "
                       f"val_ppl {val_ppl:.2f}")
            with open(LOG_FILE, "a") as lf:
                lf.write(f"{global_step},{val_loss:.6f},{lr:.2e},"
                         f"{tokens_processed/1e9:.4f},val\n")

        # ── Checkpoint ────────────────────────────────────────────────────
        if global_step % CKPT_INTERVAL == 0:
            save_checkpoint(base_model, optimizer, global_step, tokens_processed)

    # ── Done ──────────────────────────────────────────────────────────────
    pbar.close()
    save_checkpoint(base_model, optimizer, global_step, tokens_processed)
    print(f"\n✓ Pretraining complete. {tokens_processed/1e9:.2f}B tokens.")

    status = {
        "phase": "pretrain", "done": True,
        "global_step": global_step,
        "tokens_processed": tokens_processed,
        "timestamp": datetime.datetime.now().isoformat()
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
