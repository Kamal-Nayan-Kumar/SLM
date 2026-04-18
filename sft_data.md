---
language: en
tags:
  - question-generation
  - education
license: mit
---

# SLM Question Gen — SFT Dataset

Supervised fine-tuning dataset for the `slm-question-gen` model.

## Dataset Statistics

| Split | Samples |
|-------|----------|
| Train | 83,861 |
| Validation | 4,414 |
| **Total** | **88,275** |

## Data Sources

1. **SQuAD v2** (Stanford Question Answering Dataset)
   - https://huggingface.co/datasets/rajpurkar/squad_v2
   - Used for generating short questions

2. **HotpotQA** (Multi-hop reasoning)
   - https://huggingface.co/datasets/hotpot_qa
   - Used for generating complex reasoning questions

## Supported Prompt Format

The model supports grouped question generation using the following format:
`Generate {N} {difficulty} questions from the following text:\n\n{text}`

- `{N}`: Number of questions (e.g., 1 to 5)
- `{difficulty}`: `short` or `complex`

## Format

Dataset is in ChatML format:

```
<|im_start|>system
You are an expert educational assessment generator. Given a passage of text, generate high-quality questions in the requested format.<|im_end|>
<|im_start|>user
Generate {N} {difficulty} questions from the following text:

{context}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

## HuggingFace Dataset

- **Link**: https://huggingface.co/datasets/nayan90k/slm-question-gen-sft-data
- **Access**: Public

## Citation

If using this dataset, please cite:
```
@software{slm_question_gen,
  author = {Kamal Nayan Kumar},
  title = {SLM Question Generation Dataset},
  year = {2024},
  url = {https://huggingface.co/datasets/nayan90k/slm-question-gen-sft-data}
}
```