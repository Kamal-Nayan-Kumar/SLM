# Evaluation Methodology

## Dataset
- **Source**: SQuAD v2 (validation set)
- **Samples**: 50 answerable questions
- **Link**: https://huggingface.co/datasets/rajpurkar/squad_v2

## Metrics

| Metric | Score |
|--------|-------|
| ROUGE-L F1 | 0.2542 |
| BLEU-1 | 0.1902 |
| BLEU-2 | 0.1003 |
| BLEU-4 | 0.0342 |
| BERTScore Precision | 0.7595 |
| BERTScore Recall | 0.7966 |
| BERTScore F1 | **0.7771** |

## Process
1. Load SFT model from HuggingFace
2. For each of 50 passages, generate question
3. Compare with reference questions
4. Compute ROUGE-L, BLEU, BERTScore

## Code
```python
from src.inference.generate import load_model, build_prompt, generate
model, enc, device = load_model()
prompt = build_prompt(passage, "short_answer")
output = generate(model, enc, prompt, device, max_new_tokens=96, temperature=0.0)
```

## Interpretation
- BERTScore F1 (0.77): Strong semantic alignment
- ROUGE-L (0.25): Moderate n-gram overlap
- Model generates different but semantically correct questions