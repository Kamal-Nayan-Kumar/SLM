"""
Phase 7 — Final Evaluation (Compulsory).

Steps:
  1. Load best SFT checkpoint
  2. ROUGE-L evaluation on SQuAD v2 validation passages (first 20 examples)
  3. Qualitative test on 5 hardcoded NCERT-style passages
  4. Save final_model.pt (weights only, no optimizer)
  5. Write logs/eval_summary.txt

Usage:
  python src/evaluate.py
"""

import sys
import json
import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bert_score_fn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.generate import load_model, build_prompt, generate

LOGS_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints" / "sft"

# ── NCERT qualitative passages ────────────────────────────────────────────────
NCERT_PASSAGES = [
    (
        "Photosynthesis is the process by which green plants prepare food using "
        "sunlight, carbon dioxide, and water. Chlorophyll in the leaves absorbs "
        "sunlight. The light energy is used to convert CO2 and H2O into glucose "
        "and oxygen.",
        "short_answer",
    ),
    (
        "The Indian National Congress was founded in 1885 by A.O. Hume. It played "
        "a pivotal role in the Indian independence movement. Leaders like Mahatma "
        "Gandhi led non-violent civil disobedience campaigns against British rule.",
        "short_answer",
    ),
    (
        "Newton's second law of motion states that the force acting on an object "
        "equals the product of its mass and acceleration (F = ma). A heavier object "
        "requires more force to achieve the same acceleration as a lighter one.",
        "reasoning",
    ),
    (
        "The water cycle involves evaporation, condensation, precipitation, and "
        "collection. Water from oceans and lakes evaporates, rises as vapour, "
        "condenses into clouds, and falls as rain or snow before collecting again.",
        "fill_blank",
    ),
    (
        "Democracy is a form of government in which citizens have the right to elect "
        "their representatives. Key features include universal adult franchise, "
        "periodic elections, rule of law, and protection of fundamental rights.",
        "mcq",
    ),
]


# ── ROUGE-L evaluation on SQuAD v2 ───────────────────────────────────────────

def eval_rouge(model, enc, device, n_examples: int = 20) -> dict:
    """
    Run model on n_examples from SQuAD v2 validation set.
    Returns dict with macro-average ROUGE-L precision/recall/F1.
    """
    print(f"\n[ROUGE] Loading SQuAD v2 validation set ({n_examples} examples)...")
    ds     = load_dataset("rajpurkar/squad_v2", split="validation")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    scores_p, scores_r, scores_f = [], [], []
    for i, sample in enumerate(tqdm(ds, total=n_examples, desc="ROUGE eval")):
        if i >= n_examples:
            break
        if not sample["answers"]["text"]:
            continue  # unanswerable question — skip

        passage   = sample["context"]
        reference = sample["answers"]["text"][0]

        prompt    = build_prompt(passage, "short_answer")
        output    = generate(model, enc, prompt, device,
                             max_new_tokens=150, temperature=0.0, top_p=1.0)

        score = scorer.score(reference, output)
        scores_p.append(score["rougeL"].precision)
        scores_r.append(score["rougeL"].recall)
        scores_f.append(score["rougeL"].fmeasure)

    n = max(len(scores_f), 1)
    return {
        "n_evaluated": n,
        "rougeL_precision": round(sum(scores_p) / n, 4),
        "rougeL_recall":    round(sum(scores_r) / n, 4),
        "rougeL_f1":        round(sum(scores_f) / n, 4),
    }


# ── BLEU-4 evaluation on SQuAD v2 ───────────────────────────────────────────

def eval_bleu(model, enc, device, n_examples: int = 200) -> dict:
    """
    Run model on n_examples from SQuAD v2 validation set.
    Reference = ground-truth QUESTION (not answer) — correct metric for QG.
    Returns BLEU-1/2/3/4 corpus scores.
    """
    print(f"\n[BLEU-4] Loading SQuAD v2 validation set ({n_examples} examples)...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    smooth = SmoothingFunction().method1

    refs_corpus   = []   # list of list-of-tokens (one ref per sample)
    hyps_corpus   = []   # list of tokens

    for i, sample in enumerate(tqdm(ds, total=n_examples, desc="BLEU eval")):
        if i >= n_examples:
            break
        if not sample["answers"]["text"]:
            continue  # unanswerable — skip

        passage   = sample["context"]
        reference = sample["question"]  # ← compare to reference question

        prompt = build_prompt(passage, "short_answer")
        output = generate(model, enc, prompt, device,
                          max_new_tokens=80, temperature=0.0, top_p=1.0)

        refs_corpus.append([reference.lower().split()])
        hyps_corpus.append(output.lower().split())

    n = len(hyps_corpus)
    # corpus-level BLEU scores
    bleu1 = corpus_bleu(refs_corpus, hyps_corpus, weights=(1,0,0,0), smoothing_function=smooth)
    bleu2 = corpus_bleu(refs_corpus, hyps_corpus, weights=(.5,.5,0,0), smoothing_function=smooth)
    bleu3 = corpus_bleu(refs_corpus, hyps_corpus, weights=(.34,.33,.33,0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_corpus, hyps_corpus, weights=(.25,.25,.25,.25), smoothing_function=smooth)

    return {
        "n_evaluated": n,
        "bleu1":       round(bleu1, 4),
        "bleu2":       round(bleu2, 4),
        "bleu3":       round(bleu3, 4),
        "bleu4":       round(bleu4, 4),
    }


# ── BERTScore evaluation on SQuAD v2 ────────────────────────────────────────

def eval_bertscore(model, enc, device, n_examples: int = 100) -> dict:
    """
    Run model on n_examples from SQuAD v2 val.
    Reference = ground-truth QUESTION.  Uses distilbert-base-uncased (fast).
    Returns macro-average Precision / Recall / F1.
    """
    print(f"\n[BERTScore] Loading SQuAD v2 validation set ({n_examples} examples)...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")

    references, hypotheses = [], []

    for i, sample in enumerate(tqdm(ds, total=n_examples, desc="BERTScore gen")):
        if i >= n_examples:
            break
        if not sample["answers"]["text"]:
            continue

        prompt = build_prompt(sample["context"], "short_answer")
        output = generate(model, enc, prompt, device,
                          max_new_tokens=80, temperature=0.0, top_p=1.0)
        references.append(sample["question"])
        hypotheses.append(output.strip() or "none")

    print(f"  Computing BERTScore on {len(hypotheses)} samples...")
    P, R, F1 = bert_score_fn(
        hypotheses, references,
        model_type="distilbert-base-uncased",
        lang="en", verbose=False,
        device=device,
    )
    return {
        "n_evaluated": len(hypotheses),
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall":    round(R.mean().item(), 4),
        "bertscore_f1":        round(F1.mean().item(), 4),
    }


# ── Qualitative tests ─────────────────────────────────────────────────────────

def eval_qualitative(model, enc, device) -> list[dict]:
    """Run 5 NCERT passages and store model outputs for human review."""
    results = []
    print("\n[Qualitative] Running NCERT passages...")
    for idx, (passage, qtype) in enumerate(tqdm(NCERT_PASSAGES, desc="Qualitative")):
        prompt = build_prompt(passage, qtype)
        output = generate(model, enc, prompt, device,
                          max_new_tokens=512, temperature=0.7, top_p=0.9)
        results.append({
            "passage_id":    idx + 1,
            "question_type": qtype,
            "passage":       passage[:120] + "...",
            "model_output":  output.strip(),
        })
    return results


# ── Save final model ──────────────────────────────────────────────────────────

def save_final_model(model) -> Path:
    """Strip optimizer / AMP scaler and save weights-only checkpoint."""
    final_path = CKPT_DIR / "final_model.pt"

    # torch.compile wraps the model; unwrap if needed
    raw = getattr(model, "_orig_mod", model)
    torch.save(raw.state_dict(), final_path)
    print(f"\n[Save] final_model.pt → {final_path}")
    return final_path


# ── Write summary ─────────────────────────────────────────────────────────────

def write_summary(rouge_results: dict, bleu_results: dict, bert_results: dict, qual_results: list, best_val_loss: float):
    """Write human-readable eval_summary.txt to logs/."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = LOGS_DIR / "eval_summary.txt"

    lines = [
        f"=== Final Evaluation Summary ===",
        f"Timestamp : {ts}",
        f"Best Val Loss (SFT) : {best_val_loss:.4f}",
        "",
        "--- ROUGE-L (SQuAD v2 val, first 20 examples) ---",
        f"  Precision : {rouge_results['rougeL_precision']}",
        f"  Recall    : {rouge_results['rougeL_recall']}",
        f"  F1        : {rouge_results['rougeL_f1']}",
        f"  Evaluated : {rouge_results['n_evaluated']} examples",
        "",
        "--- BLEU (SQuAD v2 val, ref=question, first 200 examples) ---",
        f"  BLEU-1    : {bleu_results['bleu1']}",
        f"  BLEU-2    : {bleu_results['bleu2']}",
        f"  BLEU-3    : {bleu_results['bleu3']}",
        f"  BLEU-4    : {bleu_results['bleu4']}",
        f"  Evaluated : {bleu_results['n_evaluated']} examples",
        "",
        "--- BERTScore (SQuAD v2 val, ref=question, first 100 examples, distilbert-base-uncased) ---",
        f"  Precision : {bert_results['bertscore_precision']}",
        f"  Recall    : {bert_results['bertscore_recall']}",
        f"  F1        : {bert_results['bertscore_f1']}",
        f"  Evaluated : {bert_results['n_evaluated']} examples",
        "",
        "--- Qualitative Outputs (NCERT passages) ---",
    ]

    for r in qual_results:
        lines += [
            f"\n  [#{r['passage_id']} | {r['question_type']}]",
            f"  Passage : {r['passage']}",
            f"  Output  : {r['model_output']}",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Summary] Written → {path}")
    return path


# ── Read best val loss from SFT status ───────────────────────────────────────

def read_best_val_loss() -> float:
    status_path = ROOT / "status" / "phase5_sft_train.json"
    if status_path.exists():
        data = json.loads(status_path.read_text())
        return float(data.get("best_val_loss", float("nan")))
    return float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 7 — Final Evaluation")
    print("=" * 60)

    # 1. Load model
    model, enc, device = load_model()

    # 2. ROUGE-L
    rouge_results = eval_rouge(model, enc, device, n_examples=20)
    print(f"\nROUGE-L F1: {rouge_results['rougeL_f1']}")

    # 3. BLEU-4 (ref = reference question)
    bleu_results = eval_bleu(model, enc, device, n_examples=200)
    print(f"BLEU-4    : {bleu_results['bleu4']}")
    print(f"BLEU-1    : {bleu_results['bleu1']}")

    # 4. BERTScore (ref = reference question)
    bert_results = eval_bertscore(model, enc, device, n_examples=100)
    print(f"BERTScore F1        : {bert_results['bertscore_f1']}")
    print(f"BERTScore Precision : {bert_results['bertscore_precision']}")
    print(f"BERTScore Recall    : {bert_results['bertscore_recall']}")

    # 5. Qualitative
    qual_results = eval_qualitative(model, enc, device)

    # 6. Save final model
    save_final_model(model)

    # 7. Write summary
    best_val_loss = read_best_val_loss()
    write_summary(rouge_results, bleu_results, bert_results, qual_results, best_val_loss)

    # 7. Update status
    status = {
        "done":              True,
        "timestamp":         datetime.datetime.now().isoformat(),
        "rougeL_f1":         rouge_results["rougeL_f1"],
        "bleu4":             bleu_results["bleu4"],
        "bertscore_f1":      bert_results["bertscore_f1"],
        "best_val_loss":     best_val_loss,
    }
    (ROOT / "status" / "phase7_eval.json").write_text(
        json.dumps(status, indent=2)
    )
    print("\n✓ Phase 7 complete. All done!")


if __name__ == "__main__":
    main()
