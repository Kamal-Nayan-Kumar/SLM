"""
Phase 7 — Benchmark Evaluation Pipeline

Evaluates the trained SLM on short (SQuAD) and complex (HotpotQA) questions.
Metrics: ROUGE-L, BLEU-4, METEOR.
Generates evaluation graphs for the final report.
"""

import json
import torch
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model.config import ModelConfig
from src.model.model import SLM
from src.inference.generate import generate
import tiktoken

REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR = ROOT / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Metrics
rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
bertscore = evaluate.load('bertscore')

def load_eval_data(dataset_name, split="validation", num_samples=100):
    from datasets import load_dataset
    if dataset_name == "squad":
        ds = load_dataset("rajpurkar/squad_v2", split=split)
        samples = []
        for row in ds:
            if row.get("is_impossible", False): continue
            if not row.get("answers", {}).get("text", []): continue
            samples.append({
                "context": row["context"],
                "target": row["question"]
            })
            if len(samples) >= num_samples: break
        return samples
    elif dataset_name == "hotpotqa":
        ds = load_dataset("hotpot_qa", "distractor", split=split)
        samples = []
        for row in ds:
            ctx = row.get("context", {})
            titles = ctx.get("title", [])
            sentences = ctx.get("sentences", [])
            passage = "\n".join(t + ": " + " ".join(s) for t, s in zip(titles, sentences)).strip()
            q = row.get("question", "")
            if not q.lower().startswith(("why ", "how ")): continue
            samples.append({
                "context": passage,
                "target": q
            })
            if len(samples) >= num_samples: break
        return samples
    return []

def evaluate_model(model, enc, device, dataset_name, difficulty, num_samples=50):
    print(f"Evaluating {dataset_name} ({difficulty})...")
    data = load_eval_data(dataset_name, num_samples=num_samples)
    predictions = []
    references = []
    
    for item in tqdm(data, desc=f"Generating for {dataset_name}"):
        context = item["context"]
        ref_q = item["target"]
        
        prompt = f"<|im_start|>system\nYou are an expert educational assessment generator. Given a passage of text, generate high-quality questions in the requested format.<|im_end|>\n<|im_start|>user\nGenerate 1 {difficulty} questions from the following text:\n\n{context}<|im_end|>\n<|im_start|>assistant\n"
        
        pred = generate(model, enc, prompt, device, question_type=difficulty, max_new_tokens=100, temperature=0.3, top_p=0.9)
        
        # Clean predictions
        pred = pred.strip()
        predictions.append(pred)
        references.append(ref_q)

    # Compute metrics
    r_results = rouge.compute(predictions=predictions, references=references)
    
    # Bleu requires references to be lists of lists of strings for evaluate library
    # e.g., [[ref1_v1, ref1_v2], [ref2_v1]]
    b_refs = [[ref] for ref in references]
    b_results = bleu.compute(predictions=predictions, references=b_refs)
    
    m_results = meteor.compute(predictions=predictions, references=references)
    
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")
    
    return {
        "ROUGE-L": r_results['rougeL'],
        "BLEU-4": b_results['bleu'],
        "METEOR": m_results['meteor'],
        "BERTScore": np.mean(bert_results['f1'])
    }

def plot_metrics(results):
    print("Generating evaluation graphs...")
    datasets = list(results.keys())
    metrics = list(results[datasets[0]].keys())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e']
    for i, dataset in enumerate(datasets):
        scores = [results[dataset][m] for m in metrics]
        ax.bar(x + i*width - width/2, scores, width, label=dataset, color=colors[i])
        
    ax.set_ylabel('Scores')
    ax.set_title('Model Question Generation Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.tight_layout()
    plot_path = REPORT_DIR / "evaluation_metrics.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Graph saved to {plot_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    enc = tiktoken.get_encoding("r50k_base")
    # Load model
    ckpt_path = ROOT / "checkpoints" / "sft" / "best_model.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}. Please complete Phase 5 first.")
        return
        
    cfg = ModelConfig()
    model = SLM(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    
    results = {}
    
    # 1. SQuAD (Short questions)
    res_squad = evaluate_model(model, enc, device, "squad", "short", num_samples=50)
    results["SQuAD (Short)"] = res_squad
    
    # 2. HotpotQA (Complex questions)
    res_hotpot = evaluate_model(model, enc, device, "hotpotqa", "complex", num_samples=50)
    results["HotpotQA (Complex)"] = res_hotpot
    
    print("\n--- Final Results ---")
    print(json.dumps(results, indent=2))
    
    with open(EVAL_DIR / "benchmark_scores.json", "w") as f:
        json.dump(results, f, indent=2)
        
    plot_metrics(results)

if __name__ == "__main__":
    main()
