---
language: en
tags:
  - question-generation
  - education
  - SLM
  - causal-lm
license: mit
---

# SLM Question Generator - SFT (114M)

A 114-million-parameter Small Language Model fine-tuned to generate
educational questions from a given passage of text.

## Model Details
- **Architecture**: Decoder-only Transformer
- **Parameters**: 114.1 M
- **Layers**: 12
- **d_model**: 768
- **Attention**: GQA (12 query heads / 4 KV heads)
- **Vocabulary**: tiktoken `r50k_base` + 3 special tokens (50,260 total)
- **Context window**: 4096 tokens

## Evaluation Metrics

Evaluated on SQuAD v2 validation set (50 samples):

| Metric | Score |
|--------|-------|
| BERTScore Precision | 0.8972 |
| BERTScore Recall | 0.8920 |
| **BERTScore F1** | **0.8945** |

## Dataset
- **Training samples**: 83,861
- **Validation samples**: 4,414
- **Sources**: SQuAD v2, HotpotQA

## Usage
```python
from src.inference.generate import load_model, generate_from_text

model, enc, device = load_model()
questions = generate_from_text(
    "Photosynthesis is the process by which green plants prepare food using sunlight.",
    question_type="short_answer",
    model=model,
    enc=enc,
    device=device,
    temperature=0.0,
)
print(questions[0])
```

## Links
- [HuggingFace Model](https://huggingface.co/nayan90k/slm-question-gen-sft)
- [HuggingFace Dataset](https://huggingface.co/datasets/nayan90k/slm-question-gen-sft-data)
- [GitHub Repository](https://github.com/Kamal-Nayan-Kumar/SLM)

## Tokenizer Note
This model uses the `tiktoken` library with the `r50k_base` encoding, plus `<|im_start|>`, `<|im_end|>`, and `<|pad|>` tokens.