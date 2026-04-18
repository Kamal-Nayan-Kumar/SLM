"""
Phase 6 — Inference module.

Functions:
  load_model(ckpt_path)             → (model, enc)
  generate(model, enc, prompt, ...) → str
  chunk_text(text, enc, ...)        → list[str]
  generate_from_file(path, ...)     → list[str]
  generate_from_text(text, ...)     → list[str]

Usage:
  python src/inference/generate.py --input my_passage.pdf --type mcq
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path

import torch
from typing import Optional, Tuple, Union
import tiktoken

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.config import ModelConfig
from src.model.model import SLM

# ── Special tokens ────────────────────────────────────────────────────────────
SPECIAL = {
    "<|im_start|>": 50257,
    "<|im_end|>":   50258,
    "<|pad|>":      50259,
}
IM_END_ID   = 50258
IM_START_ID = 50257
PAD_ID      = 50259


SYSTEM_PROMPT = (
    "You are an expert educational assessment generator. "
    "Given a passage of text, generate high-quality questions in the requested format."
)

INSTRUCTION_MAP = {
    "short_answer": "Generate exactly one question from this passage. Output only the question.",
    "complex":      "Generate exactly one complex, multi-hop reasoning question from this passage. Output only the question.",
}


# ── Tokenizer wrapper ─────────────────────────────────────────────────────────

def get_enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("r50k_base")


def encode_with_special(text: str, enc: tiktoken.Encoding) -> list[int]:
    """Encode text, replacing special token strings with their IDs."""
    parts = re.split(r'(<\|im_start\|>|<\|im_end\|>|<\|pad\|>)', text)
    ids = []
    for part in parts:
        if part in SPECIAL:
            ids.append(SPECIAL[part])
        elif part:
            ids.extend(enc.encode_ordinary(part))
    return ids


def decode_with_special(ids: list[int], enc: tiktoken.Encoding) -> str:
    """Decode, replacing special IDs with their string forms."""
    rev = {v: k for k, v in SPECIAL.items()}
    result, buf = [], []
    for token_id in ids:
        if token_id in rev:
            if buf:
                result.append(enc.decode(buf))
                buf = []
            result.append(rev[token_id])
        else:
            buf.append(token_id)
    if buf:
        result.append(enc.decode(buf))
    return "".join(result)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: Optional[Union[Path, str]] = None) -> tuple:
    """
    Load SFT model. If ckpt_path is None, looks for checkpoints/sft/best_model.pt.
    Returns (model, enc).
    """
    if ckpt_path is None:
        ckpt_path = ROOT / "checkpoints" / "sft" / "best_model.pt"
        if not ckpt_path.exists():
            # Fall back to latest epoch checkpoint
            sft_dir = ROOT / "checkpoints" / "sft"
            ep_ckpts = sorted(sft_dir.glob("epoch_[0-9]*"),
                              key=lambda p: int(p.name.split("_")[1]))
            if ep_ckpts:
                ckpt_path = ep_ckpts[-1] / "checkpoint.pt"
            else:
                raise FileNotFoundError(
                    "No SFT checkpoint found. Run Phase 5 (sft.py) first."
                )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg    = ModelConfig()
    model  = SLM(cfg, use_grad_ckpt=False).to(device)

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)  # support both full ckpt and weights-only
    model.load_state_dict(state, strict=False)
    model.eval()

    # Keep inference startup predictable on smaller GPUs. Re-enable explicitly
    # with USE_TORCH_COMPILE=1 if you want to experiment with torch.compile.
    if (
        hasattr(torch, "compile")
        and device == "cuda"
        and os.environ.get("USE_TORCH_COMPILE", "").lower() in {"1", "true", "yes"}
    ):
        model = torch.compile(model)

    enc = get_enc()
    print(f"Model loaded from {ckpt_path} | device: {device}")
    return model, enc, device


def _autocast_ctx(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def _sample_next_token(
    logits: torch.Tensor,
    generated: list[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> int:
    if repetition_penalty != 1.0 and generated:
        gen_ids = set(generated[-128:])
        for token_id in gen_ids:
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    remove = cumulative - sorted_probs > top_p
    sorted_probs[remove] = 0.0
    sorted_probs /= sorted_probs.sum()
    return int(sorted_idx[torch.multinomial(sorted_probs, 1)].item())


def _should_stop(text: str, question_type: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if "<|im_start|>" in text or "<|im_end|>" in text:
        return True
    if "\nAnswer:" in text:
        return True
    if question_type != "mcq" and stripped.endswith("?"):
        return True
    if question_type == "mcq" and all(opt in text for opt in ["\nA)", "\nB)", "\nC)", "\nD)"]):
        return True
    return False


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model,
    enc: tiktoken.Encoding,
    prompt: str,
    device: str,
    question_type: str = "short_answer",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.5,
) -> str:
    """
    Autoregressive generation with KV-cache, nucleus sampling, and repetition penalty.
    Stops at <|im_end|> or max_new_tokens.
    """
    ids = encode_with_special(prompt, enc)
    prompt_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    with _autocast_ctx(device):
        logits, past_kvs = model(prompt_ids, use_cache=True)

    next_token = _sample_next_token(
        logits[0, -1, :].float(),
        generated,
        temperature,
        top_p,
        repetition_penalty,
    )

    for _ in range(max_new_tokens):

        if next_token == IM_END_ID:
            break

        generated.append(next_token)
        decoded = enc.decode([t for t in generated if t not in {IM_START_ID, IM_END_ID, PAD_ID}])
        if _should_stop(decoded, question_type=question_type):
            break

        step_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
        with _autocast_ctx(device):
            logits, past_kvs = model(step_ids, past_kvs=past_kvs, use_cache=True)
        next_token = _sample_next_token(
            logits[0, -1, :].float(),
            generated,
            temperature,
            top_p,
            repetition_penalty,
        )

    # Strip special tokens (im_start/im_end/pad) before decoding
    SPECIAL = {IM_START_ID, IM_END_ID, PAD_ID}
    generated = [t for t in generated if t not in SPECIAL]
    return enc.decode(generated)


# ── Text chunking for long documents ─────────────────────────────────────────

def chunk_text(
    text: str,
    enc: tiktoken.Encoding,
    max_tokens: int = 3000,
    overlap: int = 500,
) -> list[str]:
    """Split text into overlapping token windows."""
    ids     = enc.encode_ordinary(text)
    chunks  = []
    start   = 0

    while start < len(ids):
        end    = min(start + max_tokens, len(ids))
        chunk  = enc.decode(ids[start:end])
        chunks.append(chunk)
        if end == len(ids):
            break
        start  = end - overlap   # overlap for context continuity

    return chunks


# ── High-level interfaces ─────────────────────────────────────────────────────

def build_prompt(passage: str, n: int, difficulty: str) -> str:
    """Construct a ChatML prompt for generation."""
    instruction = f"Generate {n} {difficulty} questions from the following text:\n\n{passage}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def post_process_output(text: str) -> str:
    """
    Extract Question(s) from raw generation output.
    """
    # Hard stop markers
    for stop in ["<|im_start|>", "<|im_end|>"]:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip()

def generate_from_text(
    text: str,
    n: int = 1,
    difficulty: str = "short",
    model=None, enc=None, device: str = "cpu",
    temperature: float = 0.7, top_p: float = 0.9,
    max_new_tokens: Optional[int] = None,
) -> list[str]:
    """Generate questions from a raw text string. Chunks if too long."""
    if max_new_tokens is None:
        max_new_tokens = 150 * n
    chunks  = chunk_text(text, enc)
    results = []
    for chunk in chunks:
        prompt = build_prompt(chunk, n, difficulty)
        output = generate(model, enc, prompt, device,
                          question_type=difficulty,
                          max_new_tokens=max_new_tokens,
                          temperature=temperature, top_p=top_p)
        results.append(post_process_output(output))
    return results

def generate_from_file(
    file_path: str,
    n: int = 1,
    difficulty: str = "short",
    model=None, enc=None, device: str = "cpu",
    temperature: float = 0.7, top_p: float = 0.9,
    max_new_tokens: Optional[int] = None,
) -> list[str]:
    """
    Generate questions from a .pdf or .txt file.
    PDF text is extracted page-by-page using pymupdf.
    """
    fpath = Path(file_path)
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if fpath.suffix.lower() == ".pdf":
        import fitz  # pymupdf
        doc  = fitz.open(str(fpath))
        text = "\n\n".join(page.get_text() for page in doc)
    else:
        text = fpath.read_text(encoding="utf-8")

    return generate_from_text(text, n, difficulty, model, enc, device,
                               temperature, top_p, max_new_tokens)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate questions from a document.")
    parser.add_argument("--input",  type=str, required=True,
                        help="Path to .pdf or .txt file, or raw text string.")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of questions to generate per chunk (e.g. 3 or 5)")
    parser.add_argument("--difficulty",   type=str, default="short",
                        choices=["short", "complex"],
                        help="Difficulty level to generate (short or complex).")
    parser.add_argument("--ckpt",   type=str, default=None,
                        help="Path to SFT checkpoint (default: best_model.pt).")
    parser.add_argument("--temp",   type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument("--top_p",  type=float, default=0.9,
                        help="Nucleus sampling top-p.")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Maximum number of new tokens to generate.")
    args = parser.parse_args()

    model, enc, device = load_model(args.ckpt)

    inp = args.input
    # If the input string is a file path, use file mode; else treat as raw text
    is_file = False
    try:
        if len(inp) < 255 and Path(inp).exists():
            is_file = True
    except OSError:
        pass

    if is_file:
        results = generate_from_file(inp, args.n, args.difficulty, model, enc, device,
                                     args.temp, args.top_p, args.max_new_tokens)
    else:
        results = generate_from_text(inp, args.n, args.difficulty, model, enc, device,
                                     args.temp, args.top_p, args.max_new_tokens)

    print("\n" + "="*60)
    for i, r in enumerate(results, 1):
        print(f"\n--- Chunk {i} ---\n{r}")

    # Write status
    status = {"phase": "inference", "done": True,
              "timestamp": __import__("datetime").datetime.now().isoformat()}
    (ROOT / "status" / "phase6_inference.json").write_text(
        json.dumps(status, indent=2)
    )


if __name__ == "__main__":
    main()
