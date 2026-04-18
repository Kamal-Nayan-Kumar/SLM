"""
Push Pretraining and SFT models & datasets to Hugging Face Hub separately.

Usage:
  python push_to_hub.py --user nayan90k --pretrain-model
  python push_to_hub.py --user nayan90k --pretrain-data
  python push_to_hub.py --user nayan90k --sft-model
  python push_to_hub.py --user nayan90k --sft-data
  python push_to_hub.py --user nayan90k --all
"""

import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

ROOT = Path(__file__).resolve().parent

# ── Model Cards ───────────────────────────────────────────────────────────────

PRETRAIN_MODEL_CARD = """---
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
"""

SFT_MODEL_CARD = """---
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
- **Vocabulary**: tiktoken `r50k_base` + 3 special tokens (50,260 total)

## Tokenizer Note
This model uses the `tiktoken` library with the `r50k_base` encoding, plus `<|im_start|>`, `<|im_end|>`, and `<|pad|>` tokens.
"""

PRETRAIN_DATA_CARD = """---
language: en
tags:
  - pretraining
license: mit
---

# SLM Question Gen — Pretraining Data

The raw processed `.jsonl` and tokenized `.bin` datasets used for the pretraining phase
of the `slm-question-gen` model. (~3 Billion tokens total).

## Sources
- English Wikipedia
- OpenWebText2
- Project Gutenberg
- Medium Articles
"""

SFT_DATA_CARD = """---
language: en
tags:
  - question-generation
  - education
license: mit
---

# SLM Question Gen — SFT Dataset

Supervised fine-tuning dataset for the `slm-question-gen` model.
171,360 train / 9,018 validation samples formatted as ChatML conversations.
"""

# ── Upload Functions ──────────────────────────────────────────────────────────

def push_pretrain_model(api: HfApi, user: str, dry_run: bool):
    repo_id = f"{user}/slm-question-gen-pretrained"
    print(f"\nCreating Pretrain Model repo: {repo_id}")
    if not dry_run:
        create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

    readme_path = ROOT / "pretrain.md"
    
    ckpt_path = ROOT / "checkpoints" / "pretrain" / "step_0009156" / "checkpoint.pt"
    if not ckpt_path.exists():
        print(f"  [ERROR] {ckpt_path} not found!") # Will fail gracefully if it doesn't exist
        
    if not dry_run:
        upload_file(path_or_fileobj=str(readme_path), path_in_repo="README.md", repo_id=repo_id, repo_type="model")
        if ckpt_path.exists():
            upload_file(path_or_fileobj=str(ckpt_path), path_in_repo="checkpoint.pt", repo_id=repo_id, repo_type="model")
    print(f"  ✓ Pretrain Model repo done -> https://huggingface.co/{repo_id}")


def push_pretrain_data(api: HfApi, user: str, dry_run: bool):
    repo_id = f"{user}/slm-question-gen-pretrain-data"
    print(f"\nCreating Pretrain Data repo: {repo_id}")
    if not dry_run:
        create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)

    readme_path = ROOT / "pretrain_data.md"
    
    if not dry_run:
        upload_file(path_or_fileobj=str(readme_path), path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
        
        # Upload processed jsonl files
        if (ROOT / "data" / "processed").exists():
            print("  Uploading data/processed/ ...")
            upload_folder(folder_path=str(ROOT / "data" / "processed"), path_in_repo="processed", repo_id=repo_id, repo_type="dataset")
        
        # Upload tokenized bin files
        if (ROOT / "data" / "tokenized").exists():
            print("  Uploading data/tokenized/ ...")
            upload_folder(folder_path=str(ROOT / "data" / "tokenized"), path_in_repo="tokenized", repo_id=repo_id, repo_type="dataset")
            
    print(f"  ✓ Pretrain Data repo done -> https://huggingface.co/datasets/{repo_id}")


def push_sft_model(api: HfApi, user: str, dry_run: bool):
    repo_id = f"{user}/slm-question-gen-sft"
    print(f"\nCreating SFT Model repo: {repo_id}")
    if not dry_run:
        create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

    readme_path = ROOT / "sft.md"
    
    ckpt_path = ROOT / "checkpoints" / "sft" / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  [ERROR] {ckpt_path} not found!")
        
    if not dry_run:
        upload_file(path_or_fileobj=str(readme_path), path_in_repo="README.md", repo_id=repo_id, repo_type="model")
        if ckpt_path.exists():
            upload_file(path_or_fileobj=str(ckpt_path), path_in_repo="best_model.pt", repo_id=repo_id, repo_type="model")
    print(f"  ✓ SFT Model repo done -> https://huggingface.co/{repo_id}")


def push_sft_data(api: HfApi, user: str, dry_run: bool):
    repo_id = f"{user}/slm-question-gen-sft-data"
    print(f"\nCreating SFT Dataset repo: {repo_id}")
    if not dry_run:
        create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)

    readme_path = ROOT / "sft_data.md"
    
    if not dry_run:
        upload_file(path_or_fileobj=str(readme_path), path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
        print("  Uploading data/sft/ ...")
        upload_folder(folder_path=str(ROOT / "data" / "sft"), path_in_repo=".", repo_id=repo_id, repo_type="dataset", ignore_patterns=["README.md", "README_sft_data.md"])
            
    print(f"  ✓ SFT Data repo done -> https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="Hugging Face username")
    parser.add_argument("--pretrain-model", action="store_true")
    parser.add_argument("--pretrain-data", action="store_true")
    parser.add_argument("--sft-model", action="store_true")
    parser.add_argument("--sft-data", action="store_true")
    parser.add_argument("--all", action="store_true", help="Push all four repos")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()

    if args.all or args.pretrain_model:
        push_pretrain_model(api, args.user, args.dry_run)
    if args.all or args.pretrain_data:
        push_pretrain_data(api, args.user, args.dry_run)
    if args.all or args.sft_model:
        push_sft_model(api, args.user, args.dry_run)
    if args.all or args.sft_data:
        push_sft_data(api, args.user, args.dry_run)

    print("\n✓ Finished specified uploads!")

if __name__ == "__main__":
    main()
