# SLM Question Generator (114M)

A 114.1M-parameter small language model trained from scratch for educational question generation.

The project covers the full LLM pipeline:
- raw corpus collection
- preprocessing and tokenization
- decoder-only transformer training
- supervised fine-tuning for question generation
- inference, evaluation, and demo tooling

## Model

| Property | Value |
|---|---|
| Parameters | 114.1M |
| Architecture | Decoder-only Transformer |
| Layers | 12 |
| Hidden size | 768 |
| Attention | GQA (12 query heads / 4 KV heads) |
| Feed-forward | SwiGLU |
| Position encoding | RoPE |
| Vocabulary | `tiktoken r50k_base` + 3 special tokens |
| Context window | 4096 tokens |

## Training Summary

### Pretraining
- Data: Wikipedia + OpenWebText + Gutenberg + Medium
- Token budget used: about 3.0B tokens
- Completed steps: 9,156
- Final validation loss: about 2.71 to 2.77 in late checkpoints

### Supervised Fine-Tuning
- Data sources: SQuAD v2, HotpotQA
- Training samples: 83,861
- Validation samples: 4,414
- Supported question styles:
  - `short_answer`
  - `complex`

## Evaluation Metrics

Evaluated on SQuAD v2 validation set (50 samples):

| Metric | Score |
|--------|-------|
| BERTScore Precision | 0.8972 |
| BERTScore Recall | 0.8920 |
| **BERTScore F1** | **0.8945** |

## HuggingFace Links

- **HuggingFace Model (SFT)**: [nayan90k/slm-question-gen-sft](https://huggingface.co/nayan90k/slm-question-gen-sft)
- **HuggingFace Model (Pretrain)**: [nayan90k/slm-question-gen-pretrained](https://huggingface.co/nayan90k/slm-question-gen-pretrained)
- **HuggingFace Dataset (SFT)**: [nayan90k/slm-question-gen-sft-data](https://huggingface.co/datasets/nayan90k/slm-question-gen-sft-data)
- **HuggingFace Dataset (Pretrain)**: [nayan90k/slm-question-gen-pretrain-data](https://huggingface.co/datasets/nayan90k/slm-question-gen-pretrain-data)

## Current Project Status

- Pretraining is complete.
- The latest SFT run is complete and its `best_model.pt` is the current default inference checkpoint.
- Output quality is strongest for short-answer questions and more uneven for reasoning and MCQ.
- For presentations and live demos, use the latest `checkpoints/sft/best_model.pt` by default and keep an older archived checkpoint only as backup.

## Quick Start

```bash
pip install -r requirements.txt
python ask.py
```

Try different question types:

```bash
python ask.py --type short_answer
python ask.py --type complex
```

## Test an Older Stable Checkpoint

The default inference path now uses the latest completed SFT checkpoint:

```bash
python ask.py
```

To test an older archived checkpoint directly:

```bash
python ask.py --ckpt checkpoints/sft_archive/run_20260412_132638/sft/best_model.pt
```

A more deterministic demo setup with the latest model:

```bash
python ask.py \
  --type short_answer \
  --temp 0 \
  --max_new_tokens 48
```

Or with the older archived checkpoint:
 
```bash
python ask.py \
  --ckpt checkpoints/sft_archive/run_20260412_132638/sft/best_model.pt \
  --type short_answer \
  --temp 0 \
  --max_new_tokens 48
```

## Python Usage

```python
from src.inference.generate import load_model, generate_from_text

ckpt = "checkpoints/sft/best_model.pt"
model, enc, device = load_model(ckpt)

questions = generate_from_text(
    "Photosynthesis is the process by which green plants prepare food using sunlight, carbon dioxide, and water.",
    question_type="short_answer",
    model=model,
    enc=enc,
    device=device,
    temperature=0.0,
    top_p=1.0,
    max_new_tokens=48,
)

print(questions[0])
```

## Inference Notes

- `ask.py` loads the model once and supports multiple prompts interactively.
- Inference now uses a KV-cache path for faster decoding.
- `torch.compile()` is disabled by default for more predictable GPU behavior.
- By default, inference loads the latest completed SFT model from `checkpoints/sft/best_model.pt`.
- You can re-enable it manually with:

```bash
USE_TORCH_COMPILE=1 python ask.py
```

## Repository Layout

```text
SLM/
├── src/
│   ├── model/        # Architecture and config
│   ├── train/        # Pretraining and SFT
│   ├── data/         # Data download, preprocessing, tokenization, SFT building
│   ├── inference/    # Generation and decoding
│   └── evaluate.py   # Evaluation utilities
├── ask.py            # Interactive CLI
├── report/           # Report and presentation material
├── requirements.txt
└── push_to_hub.py
```

## Demo-Friendly Talking Points

- Lightweight LLM trained from scratch rather than adapting an external API model
- End-to-end engineering pipeline is complete
- Multiple question types supported
- Latest completed SFT checkpoint available as the default demo model
- Archived checkpoints remain available as fallback options for comparison