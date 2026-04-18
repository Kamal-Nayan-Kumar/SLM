---
language: en
tags:
  - causal-lm
  - SLM
  - pretraining
license: mit
---

# SLM Question Generator - Pretrained (114M)

A 114-million-parameter Small Language Model pretrained from scratch on ~3B tokens
of Wikipedia, OpenWebText2, Gutenberg, and Medium articles.

## Model Details
- **Architecture**: Decoder-only Transformer
- **Parameters**: 114.1 M
- **Layers**: 12
- **d_model**: 768
- **Attention**: GQA (12 query / 4 KV heads)
- **Vocabulary**: tiktoken `r50k_base` + 3 special tokens (50,260 total)
- **Context window**: 4096 tokens

## Tokenizer Note
This model uses the `tiktoken` library with the `r50k_base` encoding, plus `<|im_start|>`, `<|im_end|>`, and `<|pad|>` tokens.
