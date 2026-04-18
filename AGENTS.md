# Agent TODO — SLM Question Generation Project
> This file is for the AI agent's use. It tracks every task, sub-task, checkpoint, and context needed to build this project end-to-end without losing progress.  
> Blueprint reference: `SLM_QuestionGen_Blueprint.md`  
> Stack: Python 3.11 · PyTorch 2.x · tiktoken `r50k_base` · Lightning AI (L4 → A100 → L4)

---

## How Checkpointing Works Across Disconnections

Every phase that runs a script must:
1. Write a **phase status file** (`status/phase_X.json`) at the start and end of each phase.
2. On re-run, the script reads this file first — if `"done": true`, it skips that phase entirely.
3. For long-running scripts (tokenization, pretraining, SFT), the resume logic is built into the script itself (see per-phase notes).
4. The agent must check `status/` at the start of every session to know where to pick up.

Phase status file format:
```json
{
  "phase": "pretrain_tokenize",
  "done": false,
  "last_completed_source": "wikipedia",
  "sources_remaining": ["gutenberg", "medium"],
  "timestamp": "2026-03-07T14:00:00"
}
```

---

## GPU Switch Protocol (CRITICAL — Inform User Every Time)

**Before Phase 3 (Pretraining launch): STOP and tell the user:**
> "Before running the pretraining script, please switch your Lightning AI studio to an **A100 (40GB)** instance. Go to Lightning AI → Studio Settings → Machine → select A100 40GB → Restart. Come back and run the script after the switch."

**Before Phase 5 (SFT launch): STOP and tell the user:**
> "Pretraining is done. Please switch your Lightning AI studio back to an **L4 (24GB)** instance to save credits. Go to Lightning AI → Studio Settings → Machine → select L4 24GB → Restart. Then rerun the SFT script."

**Before Phase 1 (Environment setup) and Phase 2 (Data download):**
> User can stay on **CPU-only or L4** — no GPU needed. Inform them to avoid wasting A100 credits on data prep.

---

## Project Directory Structure (Create This First)

```
/teamspace/studios/this_studio/
├── AGENT_TODO.md                  ← this file
├── SLM_QuestionGen_Blueprint.md   ← reference doc
├── .gitignore
├── requirements.txt
├── status/                        ← phase completion flags (JSON)
│   ├── phase1_env.json
│   ├── phase2_download.json
│   ├── phase2_preprocess.json
│   ├── phase2_tokenize.json
│   ├── phase3_pretrain.json
│   ├── phase4_sft_data.json
│   └── phase5_sft_train.json
├── data/
│   ├── raw/                       ← downloaded raw text (jsonl per source)
│   │   ├── owt2/                  ← sharded .jsonl files
│   │   ├── wikipedia/             ← sharded .jsonl files
│   │   ├── gutenberg/             ← .txt files
│   │   └── medium/                ← .jsonl files
│   ├── processed/                 ← cleaned/deduped text
│   │   ├── owt2_clean.jsonl
│   │   ├── wiki_clean.jsonl
│   │   ├── gutenberg_clean.jsonl
│   │   └── medium_clean.jsonl
│   ├── tokenized/                 ← memmap .bin files
│   │   ├── owt2_train.bin
│   │   ├── wiki_train.bin
│   │   ├── gutenberg_train.bin
│   │   ├── medium_train.bin
│   │   ├── train.bin              ← final interleaved mix
│   │   └── val.bin
│   └── sft/                       ← SFT instruction pairs
│       ├── raw/                   ← raw downloaded SFT datasets
│       ├── squad_sft.jsonl
│       ├── race_sft.jsonl
│       ├── ncert_sft.jsonl
│       ├── hotpot_sft.jsonl
│       ├── sft_train.jsonl        ← final merged+shuffled train
│       └── sft_val.jsonl
├── src/
│   ├── data/
│   │   ├── download_pretrain.py
│   │   ├── preprocess.py
│   │   ├── tokenize_pretrain.py
│   │   ├── download_sft.py
│   │   └── build_sft_dataset.py
│   ├── model/
│   │   ├── config.py              ← ModelConfig dataclass
│   │   ├── model.py               ← full model definition
│   │   ├── attention.py           ← GQA + RoPE
│   │   └── ffn.py                 ← SwiGLU FFN
│   ├── train/
│   │   ├── pretrain.py            ← pretraining loop
│   │   ├── sft.py                 ← SFT loop
│   │   └── lr_schedule.py         ← cosine decay + warmup
│   └── inference/
│       └── generate.py            ← generation + PDF chunking
├── checkpoints/
│   └── pretrain/                  ← step_XXXXXX/ directories
│   └── sft/                       ← epoch_X/ directories
└── logs/
    ├── pretrain_loss.csv
    └── sft_loss.csv
```

---

## PHASE 0 — Project Scaffold

**Context:** Create the full directory structure and all config/requirements files. No GPU needed. Can run on CPU studio.

### Tasks

- [ ] **0.1** Verify current working directory is `/teamspace/studios/this_studio`
- [ ] **0.2** Create all directories listed in the structure above using `mkdir -p`
- [ ] **0.3** Create `requirements.txt` with pinned versions:
  ```
  torch>=2.3.0
  tiktoken>=0.7.0
  datasets>=2.19.0
  huggingface_hub>=0.23.0
  transformers>=4.40.0
  numpy>=1.26.0
  tqdm>=4.66.0
  zstandard>=0.22.0
  pandas>=2.2.0
  pymupdf>=1.24.0
  datasketch>=1.6.4
  beautifulsoup4>=4.12.3
  langdetect>=1.0.9
  lm_dataformat>=0.0.20
  ```
- [ ] **0.4** Install requirements: `pip install -r requirements.txt`
- [ ] **0.5** Create `status/` directory and initialize all phase JSON files with `"done": false`
- [ ] **0.6** Verify `.gitignore` exists and covers `data/`, `checkpoints/`, `logs/`, `*.bin`, `*.pt`

**Checkpoint:** After 0.6, write `status/phase0_scaffold.json` → `"done": true`

---

## PHASE 1 — Environment Verification

**Context:** Confirm all libraries import correctly, tiktoken tokenizer loads, and CUDA is available (or CPU is noted). This phase runs on CPU/L4 — no A100 needed.

### Tasks

- [ ] **1.1** Run a sanity check script that:
  - Imports `torch`, `tiktoken`, `datasets`, `numpy`, `fitz` (pymupdf)
  - Loads `tiktoken.get_encoding("r50k_base")` and verifies `enc.n_vocab == 50257`
  - Prints `torch.__version__` and `torch.cuda.is_available()`
  - Encodes a test string and decodes it back; verifies round-trip
- [ ] **1.2** Confirm `datasets` library can reach HuggingFace (network test): load the first 5 rows of `wikimedia/wikipedia` in streaming mode
- [ ] **1.3** Verify available disk space on `/teamspace/studios/this_studio` — need at least **80GB free** for raw + tokenized data combined
  - If less than 80GB: inform user and ask them to expand storage in Lightning AI before proceeding

**Checkpoint:** Write `status/phase1_env.json` → `"done": true`

---

## PHASE 2A — Pretraining Data Download

**Context:** Download all 4 raw corpus sources. Each source is downloaded independently so partial progress is resumable. Expected total download: ~50–70GB. Runs on L4 or CPU studio — no A100 needed.

**Resume logic:** `download_pretrain.py` maintains a per-source completion flag in `status/phase2_download.json` under key `completed_sources`. On re-run, skip any source already in that list.

### Tasks

#### Source 1 — OpenWebText2
- [ ] **2A.1** Check if `data/raw/owt2/` already contains `.jsonl` shards (resume check)
- [ ] **2A.2** Attempt HuggingFace Pile route first (more reliable):
  - `datasets.load_dataset("EleutherAI/pile", "all", split="train", streaming=True)`
  - Filter: `example['meta']['pile_set_name'] == 'OpenWebText2'`
  - Stream and write in shards of 100,000 documents each to `data/raw/owt2/owt2_shard_XXXX.jsonl`
  - Track last written shard index in status file for resume
  - Target: ~5B tokens worth (~190GB raw text before tokenization, ~2M documents)
  - **Note:** If EleutherAI/pile is gated or unavailable, fall back to `academictorrents` download via `lm_dataformat`. Inform user if authentication is needed.
- [ ] **2A.3** Mark `owt2` as complete in `status/phase2_download.json`

#### Source 2 — Wikipedia (English)
- [ ] **2A.4** Check if `data/raw/wikipedia/` already has data
- [ ] **2A.5** `datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")`
  - Write to `data/raw/wikipedia/wiki_shard_XXXX.jsonl` in shards of 50,000 articles
  - Only write the `text` field (prepend `title + "\n\n"` for context)
  - This is ~6.7M articles; track last shard index for resume
- [ ] **2A.6** Mark `wikipedia` as complete in status file

#### Source 3 — Project Gutenberg
- [ ] **2A.7** Check if `data/raw/gutenberg/` already has data
- [ ] **2A.8** Try HuggingFace route: `datasets.load_dataset("sedthh/gutenberg_english", split="train")`
  - If unavailable (dataset often requires manual download), inform user:
    > "The Gutenberg dataset requires manual access. Visit https://huggingface.co/datasets/sedthh/gutenberg_english and accept terms, then rerun."
  - Write full text of each book to `data/raw/gutenberg/gutenberg_shard_XXXX.jsonl`
  - Track shard index for resume
- [ ] **2A.9** Mark `gutenberg` as complete in status file

#### Source 4 — Medium Articles
- [ ] **2A.10** Check if `data/raw/medium/` already has data
- [ ] **2A.11** `datasets.load_dataset("fabiochiu/medium-articles", split="train")`
  - Write `title + "\n\n" + text` to `data/raw/medium/medium.jsonl` (single file, ~190k articles)
- [ ] **2A.12** Mark `medium` as complete in status file

**Checkpoint:** Write `status/phase2_download.json` → `"done": true` only after all 4 sources complete

---

## PHASE 2B — Pretraining Data Preprocessing

**Context:** Clean each source independently. Heavy-duty CPU work. Pipeline: dedup → quality filter → HTML strip → unicode normalize → shuffle. Each source is processed separately so any source can be rerun independently. Runs on L4 or CPU studio.

**Resume logic:** `preprocess.py` checks per-source output file existence in `data/processed/`. If `{source}_clean.jsonl` exists and is non-empty, skip that source. Status tracked in `status/phase2_preprocess.json`.

### Tasks

#### For each source (owt2, wikipedia, gutenberg, medium) — apply in order:

- [ ] **2B.1** **MinHash Deduplication** (apply to owt2 and medium only; wiki and gutenberg are pre-curated)
  - Use `datasketch.MinHash` with 128 permutations, 5-gram shingles
  - Use `datasketch.MinHashLSH` with threshold=0.85
  - Process documents in batches of 10,000 to avoid OOM
  - Track progress: write dedupe progress (last batch index) to status file every 10 batches
  - Expected: remove ~15–20% of owt2 documents

- [ ] **2B.2** **Quality Heuristic Filtering** (all sources)
  Apply in a single pass over each source's raw jsonl:
  - Minimum length: 200 characters (discard shorter)
  - Max line repetition ratio: >20% duplicate lines → discard
  - Alphabetic character ratio: <80% → discard
  - Average word length: outside [3, 10] → discard
  - More than 5 consecutive newlines → discard

- [ ] **2B.3** **HTML and Markup Removal** (owt2 and medium only)
  - Strip HTML tags with `BeautifulSoup(text, "html.parser").get_text()`
  - Strip `**`, `__`, `##` markdown symbols (keep text)
  - Remove URLs: `re.sub(r'https?://\S+', '', text)`

- [ ] **2B.4** **Unicode Normalization** (all sources)
  - `unicodedata.normalize('NFKC', text)`
  - Remove Unicode control chars except `\n` and `\t`
  - Strip null bytes `\x00`

- [ ] **2B.5** **Paragraph Normalization** (all sources)
  - Collapse 3+ consecutive `\n` to exactly `\n\n`
  - Strip leading/trailing whitespace per document

- [ ] **2B.6** **Language Filter** (Gutenberg only)
  - Use `langdetect.detect(text[:500])` on first 500 chars
  - Keep only `en` documents
  - Cache detection results to avoid re-running on resume

- [ ] **2B.7** Write cleaned documents to `data/processed/{source}_clean.jsonl`, one JSON per line with key `"text"`

- [ ] **2B.8** Log per-source stats after processing:
  - Raw document count → Filtered document count → Reduction percentage
  - Estimated token count (rough: `len(text.split()) * 1.3`)

**Checkpoint:** Write `status/phase2_preprocess.json` → `"done": true` after all 4 sources complete

---

## PHASE 2C — Pretraining Data Tokenization

**Context:** Convert cleaned text to memory-mapped binary token arrays using tiktoken `r50k_base`. This is the most I/O-intensive step. Expected output: ~20GB of `.bin` files. Runs on L4 or CPU studio.

**Resume logic:** For each source, tokenization writes chunks of 10M tokens at a time, appending to the `.bin` file. A progress file `status/tokenize_{source}_progress.json` tracks the last completed document index. On re-run, open the `.bin` in append mode and skip already-tokenized documents.

### Tasks

- [ ] **2C.1** Load `enc = tiktoken.get_encoding("r50k_base")`, verify `enc.n_vocab == 50257`

- [ ] **2C.2** For each source (owt2, wikipedia, gutenberg, medium):
  - Read `data/processed/{source}_clean.jsonl`
  - For each document:
    - Encode: `ids = enc.encode_ordinary(text)`
    - Append EOT token (ID 50256): `ids.append(50256)`
  - Accumulate IDs in a buffer; flush every 10M tokens to `data/tokenized/{source}_train.bin` as `numpy.uint16`
  - Track `last_doc_index` in `status/tokenize_{source}_progress.json` every flush
  - Print progress bar with estimated time remaining

- [ ] **2C.3** Compute token counts per source; verify ratios approximately match target mix (50/30/15/5)

- [ ] **2C.4** **Build Interleaved Training File** (`data/tokenized/train.bin`)
  - Sample blocks of 4096 tokens from each source's `.bin` according to mix ratios
  - Shuffle sampled blocks with `random.seed(42)` before writing
  - Write to `data/tokenized/train.bin`
  - Target: exactly 10B tokens (adjust sampling if any source runs short)
  - Track interleave progress in `status/phase2_tokenize.json` under `blocks_written`

- [ ] **2C.5** **Build Validation File** (`data/tokenized/val.bin`)
  - Hold out last 5,000 documents from each source OR last 0.5% of tokens, whichever is smaller
  - Tokenize and write to `data/tokenized/val.bin`
  - Target: ~50–100M tokens in validation

- [ ] **2C.6** **Sanity Check**
  - Load first 100 blocks of `train.bin` using `numpy.memmap`
  - Decode each block with `enc.decode(block.tolist())`
  - Print first 3 decoded blocks to verify they look like clean English text
  - Verify token ID 50256 appears (EOT present)
  - Check token frequency distribution (most common token should be ~ID 284 "the" or similar)

**Checkpoint:** Write `status/phase2_tokenize.json` → `"done": true`

---

## PHASE 3 — Pretraining

### ⚠️ GPU SWITCH REQUIRED BEFORE THIS PHASE
> **STOP. Inform user:**  
> "Phase 3 is the pretraining run. This requires an **A100 40GB GPU** on Lightning AI.  
> Please do the following before running any pretraining script:  
> 1. Go to Lightning AI Studio → top-right machine selector  
> 2. Select **A100 (40GB)**  
> 3. Wait for the studio to restart  
> 4. Come back and run `python src/train/pretrain.py`  
> Do NOT run this on CPU or L4 — it will be ~1000× slower."

**Context:** Train the 120M decoder-only transformer on 10B tokens. Training runs in sessions; each session checkpoints frequently. On reconnect, just rerun `python src/train/pretrain.py` — it automatically finds and loads the latest checkpoint.

### Tasks

#### Model Definition (src/model/)
- [ ] **3.1** Create `src/model/config.py` — `ModelConfig` dataclass with all hyperparameters:
  - `n_layers=12`, `d_model=768`, `n_heads=12`, `n_kv_heads=4`, `ffn_dim=2048`
  - `max_seq_len=4096`, `vocab_size=50257`, `rope_theta=10000.0`
  - `dropout=0.0` (no dropout during pretraining — standard for modern LLMs)

- [ ] **3.2** Create `src/model/attention.py` — GQA with RoPE:
  - Implement `RotaryEmbedding` class: precompute `cos` and `sin` tables for positions 0..4095
  - Implement `apply_rotary_emb(q, k, cos, sin)` function — applies RoPE to Q and K
  - Implement `GroupedQueryAttention` module:
    - Q: Linear(d_model, n_heads × head_dim, bias=False)
    - K: Linear(d_model, n_kv_heads × head_dim, bias=False)
    - V: Linear(d_model, n_kv_heads × head_dim, bias=False)
    - O: Linear(d_model, d_model, bias=False)
    - Expand K/V heads to match Q heads by repeating: `k.repeat_interleave(n_heads // n_kv_heads, dim=2)`
    - Use `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True` (uses FlashAttention if available)

- [ ] **3.3** Create `src/model/ffn.py` — SwiGLU FFN:
  - `gate_proj`: Linear(d_model, ffn_dim, bias=False)
  - `up_proj`: Linear(d_model, ffn_dim, bias=False)
  - `down_proj`: Linear(ffn_dim, d_model, bias=False)
  - Forward: `down_proj(F.silu(gate_proj(x)) * up_proj(x))`

- [ ] **3.4** Create `src/model/model.py` — full model:
  - `TransformerBlock`: RMSNorm → GQA → residual → RMSNorm → SwiGLU → residual
  - `SLM` model class:
    - `tok_emb`: Embedding(vocab_size, d_model)
    - Stack of 12 `TransformerBlock`s
    - Final `RMSNorm`
    - `lm_head`: Linear(d_model, vocab_size, bias=False) — initialized with `lm_head.weight = tok_emb.weight` (weight tying)
  - Add `get_num_params()` method — verify output is ~118–122M
  - Add `configure_optimizers(weight_decay, lr, betas, device)` method:
    - Apply weight decay to all params except norms, biases, and embeddings
    - Use `torch.optim.AdamW` with `fused=True` if CUDA available

- [ ] **3.5** Create `src/train/lr_schedule.py`:
  - `get_lr(step, warmup_steps, max_steps, max_lr, min_lr)` function
  - Linear warmup for steps 0..warmup_steps
  - Cosine decay from max_lr to min_lr for steps warmup_steps..max_steps
  - Returns `min_lr` after max_steps

#### Training Script (src/train/pretrain.py)
- [ ] **3.6** Create `src/train/pretrain.py` with these sections, in order:

  **A. Config block** (all hyperparameters at top of file, easy to change):
  - `train_bin = "data/tokenized/train.bin"`
  - `val_bin = "data/tokenized/val.bin"`
  - `checkpoint_dir = "checkpoints/pretrain"`
  - `log_file = "logs/pretrain_loss.csv"`
  - `max_tokens = 10_000_000_000` (10B)
  - `micro_batch = 5`
  - `grad_accum_steps = 16`
  - `seq_len = 4096`
  - `grad_clip = 1.0`
  - `checkpoint_interval = 1000` (steps)
  - `val_interval = 500` (steps)
  - `log_interval = 10` (steps)

  **B. Resume Logic** (runs before anything else):
  - Scan `checkpoints/pretrain/` for directories named `step_XXXXXX`
  - Sort numerically, pick highest
  - If found: load `checkpoint.pt` containing `{model_state, optimizer_state, global_step, tokens_processed, lr_scheduler_step, rng_states}`
  - If not found: initialize model fresh, set `global_step = 0`, `tokens_processed = 0`
  - Print clearly: `"Resuming from step {global_step} ({tokens_processed/1e9:.2f}B tokens)"` or `"Starting fresh pretraining"`

  **C. DataLoader**:
  - Use `numpy.memmap` to open `train.bin` in read mode
  - `get_batch(split)`: sample a random starting position, return `(x, y)` where `y = x shifted by 1`
  - Use a seeded offset that advances based on `global_step` so resume gives the next unseen batch

  **D. Training Loop**:
  - `torch.compile(model)` before training starts
  - Enable `torch.backends.cuda.matmul.allow_tf32 = True`
  - Enable `torch.backends.cudnn.allow_tf32 = True`
  - For each micro-batch: forward + loss under `torch.autocast("cuda", dtype=torch.bfloat16)`
  - Accumulate gradients over `grad_accum_steps` micro-batches before calling `optimizer.step()`
  - Scale loss by `1.0 / grad_accum_steps` before each backward
  - After accumulation: `torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)`
  - Update LR via `get_lr()` before each optimizer step
  - Increment `global_step` and `tokens_processed` after each full gradient step

  **E. Logging** (every `log_interval` steps):
  - Print: `f"step {global_step} | loss {loss:.4f} | lr {lr:.2e} | tokens {tokens_processed/1e9:.3f}B | time/step {dt:.2f}ms"`
  - Append row to `logs/pretrain_loss.csv`: `step, loss, lr, tokens`

  **F. Validation** (every `val_interval` steps):
  - Run model on 20 batches from `val.bin` with `torch.no_grad()`
  - Compute mean cross-entropy loss, compute perplexity: `exp(val_loss)`
  - Print: `f"VAL step {global_step} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}"`
  - Append to `logs/pretrain_loss.csv` with a `"val"` flag column

  **G. Checkpointing** (every `checkpoint_interval` steps):
  - Save to `checkpoints/pretrain/step_{global_step:07d}/checkpoint.pt`:
    ```python
    {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'global_step': global_step,
        'tokens_processed': tokens_processed,
        'lr_step': lr_step,
        'torch_rng': torch.random.get_rng_state(),
        'cuda_rng': torch.cuda.random.get_rng_state(),
        'numpy_rng': numpy.random.get_state(),
        'python_rng': random.getstate(),
    }
    ```
  - After saving, delete all but the 3 most recent step directories to save disk space
  - Print: `f"Checkpoint saved: step_{global_step:07d}"`

  **H. Termination Check**:
  - After each step: if `tokens_processed >= max_tokens`, break loop
  - Write `status/phase3_pretrain.json` → `"done": true, "final_step": global_step`
  - Print: `"Pretraining complete. Total tokens: {tokens_processed/1e9:.2f}B"`

- [ ] **3.7** After model definition is complete, run a **1-batch smoke test** on A100:
  - Initialize model, run single forward pass, verify output shape is `(batch, seq_len, vocab_size)`
  - Verify `get_num_params()` returns 118–122M
  - Verify no OOM at `micro_batch=5, seq_len=4096` with gradient checkpointing ON

- [ ] **3.8** Enable gradient checkpointing in `TransformerBlock.forward()`:
  - Use `torch.utils.checkpoint.checkpoint(block_fn, x)` for each transformer block
  - This trades compute for memory: reduces activation memory from O(n_layers × batch × seq_len × d_model) to O(seq_len × d_model)

- [ ] **3.9** Launch pretraining: `python src/train/pretrain.py`
  - Monitor first 100 steps manually — loss should start around 10–11 and drop toward 5–6 within 50 steps
  - If loss does not move or NaN appears: stop and debug (likely LR or initialization issue)

- [ ] **3.10** **Mid-training check at 500M tokens**:
  - Val loss should be ~2.9–3.2 at this point
  - If val loss is >3.5: check LR schedule is applied correctly, check data mix
  - Run qualitative generation test: prompt the model with `"The French Revolution began"` and check if output is coherent English (not random tokens)

- [ ] **3.11** **End-of-pretraining qualitative test** (after target token count reached):
  - Test prompts: `"The mitochondria is"`, `"Photosynthesis is the process"`, `"World War II ended when"`
  - Model should produce factually plausible, grammatically correct continuations
  - If output is gibberish: do not proceed to SFT; investigate loss curve

**Checkpoint:** Write `status/phase3_pretrain.json` → `"done": true`

---

## PHASE 4 — SFT Data Preparation

### ⚠️ GPU SWITCH
> "Phase 4 is SFT data preparation — **no GPU needed**. You can switch back to **L4 (24GB)** or even CPU to save A100 credits. Switch machine in Lightning AI Studio Settings before running these scripts."

**Context:** Download and format the 4 SFT datasets into ChatML-format JSONL files. Produces `data/sft/sft_train.jsonl` and `data/sft/sft_val.jsonl`.

**Critical format rule:** The assistant target in SFT must exactly match the desired inference output:
- one prompt in → one output item out
- no answer key for `short_answer`, `reasoning`, or `fill_blank`
- `mcq` outputs exactly one question plus four options
- do not train on prompts that ask for 3 or 5 questions when inference asks for 1

**Resume logic:** Each source dataset saves to its own file in `data/sft/`. `build_sft_dataset.py` checks if each file exists before downloading/building. Status tracked in `status/phase4_sft_data.json` under `completed_sources`. To intentionally rebuild corrected SFT data, run with `SFT_FORCE_REBUILD=1`.

### Tasks

#### Download SFT Sources
- [ ] **4.1** Download SQuAD v2:
  - `datasets.load_dataset("rajpurkar/squad_v2", split="train")` → `data/sft/raw/squad_v2/`
  - Verify 130,319 training examples present

- [ ] **4.2** Download RACE:
  - `datasets.load_dataset("ehovy/race", "all", split="train")` → `data/sft/raw/race/`
  - Verify ~87,866 training examples present

- [ ] **4.3** Download NCERT:
  - `datasets.load_dataset("KadamParth/ncert-dataset")` — check actual schema first (print first 3 rows)
  - Adapt field mapping based on actual schema (fields may be `question`, `answer`, `context`, `subject`)
  - → `data/sft/raw/ncert/`

- [ ] **4.4** Download HotpotQA:
  - `datasets.load_dataset("hotpot_qa", "distractor", split="train")` → `data/sft/raw/hotpotqa/`
  - Note: HotpotQA has `{context, question, answer, type}`. Use `context.sentences` (list of lists) — flatten and join as passage
  - Verify ~90,447 training examples present

#### Build Instruction Pairs

- [ ] **4.5** Define instruction format using natural language: `"Generate {N} {difficulty} questions from the following text:\n\n{text}"`.
- [ ] **4.6** Convert SQuAD → ChatML JSONL (`data/sft/squad_sft.jsonl`):
  - Target: Group multiple questions per context (1-5 questions per sample)
  - Difficulty: `short`
- [ ] **4.7** Convert NCERT → ChatML JSONL (`data/sft/ncert_sft.jsonl`):
  - Target: Group multiple questions per context
  - Difficulty: `short`
- [ ] **4.8** Convert HotpotQA → ChatML JSONL (`data/sft/hotpot_sft.jsonl`):
  - Target: Group multiple questions per context
  - Difficulty: `complex`
- [ ] **4.9** Note: RACE dataset is explicitly excluded as it is purely MCQ and we are focusing strictly on short and complex question generation.

#### Merge, Deduplicate, Split

- [ ] **4.10** Merge all 4 JSONL files into one
- [ ] **4.11** Deduplicate by exact `input_text` match (same passage used in multiple datasets)
- [ ] **4.12** Shuffle with `random.seed(42)`
- [ ] **4.13** Split: 95% → `data/sft/sft_train.jsonl`, 5% → `data/sft/sft_val.jsonl`
- [ ] **4.14** Print final counts: train samples, val samples, average instruction length, average output length
- [ ] **4.14a** Before training, manually inspect 20 random SFT records and verify:
  - no output contains `Answer:`
  - no template asks for more than one question
  - MCQ records contain exactly one question and four options

#### Tokenize SFT Data

- [ ] **4.15** Load `enc = tiktoken.get_encoding("r50k_base")`
- [ ] **4.16** Add 3 special tokens: `<|im_start|>` (ID 50257), `<|im_end|>` (ID 50258), `<|pad|>` (ID 50259)
  - Since tiktoken `r50k_base` does not natively support custom special tokens, create a wrapper:
    - Replace `<|im_start|>` string with the existing token `<|endoftext|>` (50256) during tokenization temporarily, OR
    - Use tiktoken's `Encoding` constructor to create a custom encoding based on `r50k_base` with added special tokens
  - **Important:** Document this mapping so inference uses the same encoding
- [ ] **4.17** For each sample: tokenize the full ChatML sequence, build `input_ids` and `loss_mask` arrays
  - `loss_mask[i] = 1` only for tokens in the `<|im_start|>assistant ... <|im_end|>` section
  - `loss_mask[i] = 0` for system and user sections
- [ ] **4.18** Save tokenized SFT data as a `torch.save` file: `data/sft/sft_train_tokenized.pt` and `data/sft/sft_val_tokenized.pt`
  - Each is a list of `{'input_ids': tensor, 'loss_mask': tensor}`
  - Discard samples where total length > 4096 tokens

**Checkpoint:** Write `status/phase4_sft_data.json` → `"done": true`

---

## PHASE 5 — Supervised Fine-Tuning

### ⚠️ GPU SWITCH
> "Phase 5 is the SFT training run. This needs a GPU — **L4 (24GB)** is sufficient and more cost-effective than A100 for this step. Switch to L4 in Lightning AI Studio Settings before running `python src/train/sft.py`."

**Context:** Fine-tune the pretrained 120M model on corrected single-output instruction pairs for 3–5 epochs with response-only loss. Checkpoints every epoch. Auto-resumes from last epoch checkpoint unless a fresh restart is explicitly requested.

**Resume logic:** `sft.py` checks `checkpoints/sft/` for `epoch_X/` directories. Loads the highest complete epoch checkpoint. Starts from the next epoch. To intentionally retrain from scratch on rebuilt SFT data, run with `SFT_FRESH_START=1`; archive the previous SFT checkpoints instead of deleting them.

### Tasks

- [ ] **5.1** Create `src/train/sft.py`:

  **A. Config block**:
  - `pretrain_ckpt_dir = "checkpoints/pretrain"` — load best (latest) pretrain checkpoint
  - `sft_ckpt_dir = "checkpoints/sft"`
  - `train_data = "data/sft/sft_train_tokenized.pt"`
  - `val_data = "data/sft/sft_val_tokenized.pt"`
  - `log_file = "logs/sft_loss.csv"`
  - `max_epochs = 5`
  - `lr = 2e-5`
  - `min_lr = 0.0`
  - `warmup_steps = 100`
  - `weight_decay = 0.01`
  - `grad_clip = 1.0`
  - `micro_batch = 8`
  - `grad_accum = 4`

  **B. Resume Logic**:
  - Scan `checkpoints/sft/` for `epoch_X/` directories
  - Load highest complete epoch; start from `epoch + 1`
  - If no SFT checkpoint found: load latest pretrain checkpoint from `checkpoints/pretrain/`
  - **Critical:** When loading pretrain checkpoint for SFT, do NOT load the optimizer state — initialize a fresh optimizer with LR = 2e-5
  - Print: `"Starting SFT from epoch {start_epoch}"` or `"Resuming SFT from epoch {resume_epoch+1}"`

  **C. DataLoader**:
  - Implement `SFTDataset(torch.utils.data.Dataset)` that reads tokenized `.pt` file
  - `__getitem__` returns `(input_ids, loss_mask)` padded/truncated to 4096
  - Use `DataLoader` with `shuffle=True` (for training), `batch_size=micro_batch`
  - `collate_fn`: pad sequences to max length in batch using `<|pad|>` (ID 50259), extend loss_mask with 0s

  **D. Training Loop** (per epoch):
  - Same gradient accumulation pattern as pretraining
  - Loss computation: `F.cross_entropy(logits, labels, reduction='none')` — then multiply by `loss_mask`, sum, divide by `loss_mask.sum()` (only count response tokens)
  - Log both `total_loss` and `response_loss` (response_loss is the one that matters)
  - Show `tqdm` epoch progress bar with live loss display

  **E. Validation** (end of every epoch):
  - Compute `val_response_loss` and `val_perplexity` on validation set
  - Print epoch summary: `"Epoch {e} | Train Response Loss: {:.4f} | Val Response Loss: {:.4f}"`
  - Early stopping: if `val_response_loss` increases for 2 consecutive epochs, stop and use best epoch

  **F. Checkpointing** (end of every epoch):
  - Save `checkpoints/sft/epoch_{epoch}/checkpoint.pt`:
    ```python
    {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'val_response_loss': val_response_loss,
        'best_epoch': best_epoch,
    }
    ```
  - Also maintain `checkpoints/sft/best_model.pt` — copy of the best-val-loss epoch's weights

  **G. Mid-epoch checkpointing** (every 500 steps within an epoch):
  - Save `checkpoints/sft/epoch_{epoch}_step_{step}/checkpoint.pt` with same structure + `step` key
  - This handles disconnection mid-epoch: on resume, reload this mid-epoch checkpoint and continue from that step within the epoch

- [ ] **5.2** **Epoch-by-epoch qualitative evaluation**:
  After each epoch, test on 5 fixed held-out passages and observe:
  - Does output follow the requested instruction format (MCQ vs. short answer)?
  - Are questions actually answerable from the given passage?
  - Are MCQ distractors plausible?
  - Does output terminate cleanly with `<|im_end|>`?
  - Does the model avoid emitting `Answer:` when the requested format is question-only?
  - Record observations in `logs/sft_qualitative.txt`

- [ ] **5.3** Track response loss targets:
  - Epoch 1: response loss should drop from ~3.5–4.0 → ~1.5–2.0
  - Epoch 3: response loss should be ~0.8–1.2
  - If loss < 0.5 before epoch 3: likely overfitting, stop early
  - If loss > 2.5 after epoch 2: check loss masking implementation

**Checkpoint:** Write `status/phase5_sft_train.json` → `"done": true, "best_epoch": N, "best_val_loss": X`

---

## PHASE 6 — Inference Module *(Compulsory)*

**Context:** Build the inference script that takes a file (`.pdf`, `.txt`, or raw text string) and generates questions. Runs on L4 or CPU. This phase is mandatory — the model is not deployable without it.

### Tasks

- [ ] **6.1** Create `src/inference/generate.py`:
  - `load_model(ckpt_path)`: loads best SFT checkpoint, puts model in `eval()` mode, wraps with `torch.compile()`
  - `generate(model, enc, prompt, max_new_tokens, temperature, top_p)`:
    - Tokenize prompt
    - Autoregressive generation loop with `top_p` sampling (nucleus sampling)
    - Stop when `<|im_end|>` token (ID 50258) is generated
    - Return decoded string
  - `chunk_text(text, enc, max_tokens=3000, overlap=500)`: splits long text into overlapping windows
  - `generate_from_pdf(pdf_path, question_type, model, enc)`:
    - Extract text with `pymupdf`
    - Chunk with `chunk_text`
    - Run `generate()` on each chunk
    - Return list of questions

- [ ] **6.2** Test inference with 3 manual examples:
  - Generate 3 short questions from a biology definition.
  - Generate 2 complex questions from a history chapter.
  - Generate 5 short questions from a physics law description.
  - Verify output is formatted correctly and respects the count.

**Checkpoint:** Write `status/phase6_inference.json` → `"done": true`

---

## PHASE 7 — Final Validation *(Compulsory)*

**Context:** Mandatory evaluation before the project is considered complete. Tests both automatic metrics and qualitative output.

- [ ] **7.1** Run `python src/evaluation/run_eval.py` to evaluate the model on SQuAD and HotpotQA datasets. This will compute ROUGE-L, BLEU-4, METEOR, and BERTScore metrics.
- [ ] **7.2** Run on 5 NCERT passages not seen during SFT — qualitative check
- [ ] **7.3** Check `report/evaluation_metrics.png` to verify the generated performance graphs.
- [ ] **7.3a** Keep a small fixed regression set of 20-50 passages covering `short_answer` and `complex` reasoning; rerun it after every SFT rebuild
- [ ] **7.4** Save final model weights to `checkpoints/sft/final_model.pt` (clean, only model weights, no optimizer state)
- [ ] **7.5** Compile the LaTeX report located at `report/report.tex` into a PDF to generate the final project report.

---

## Session Restart Checklist

When resuming after any disconnection or session restart, do these in order:

1. `cd /teamspace/studios/this_studio`
2. `cat status/*.json` — read all status files to see which phases are done
3. Identify the first phase with `"done": false`
4. Re-run only that phase's script
5. For pretraining: just run `python src/train/pretrain.py` — resume logic is automatic
6. For SFT: just run `python src/train/sft.py` — resume logic is automatic
7. Never manually delete checkpoint directories unless intentionally starting over

---

## Quick Reference: Which GPU for Which Phase

| Phase | Task | GPU Needed | Lightning AI Instance |
|---|---|---|---|
| 0 | Scaffold | None | CPU or L4 |
| 1 | Env verify | None | CPU or L4 |
| 2A | Download | None | CPU or L4 |
| 2B | Preprocess | None | CPU or L4 |
| 2C | Tokenize | None | CPU or L4 |
| 3 | **PRETRAIN** | **Yes — A100 40GB** | **A100 40GB** |
| 4 | SFT data prep | None | L4 or CPU |
| 5 | **SFT TRAIN** | **Yes — L4/A100** | **L4 24GB** |
| 6 | Inference | **Compulsory** | L4 or CPU |
| 7 | Evaluation | **Compulsory** | L4 or CPU |
