# Small Language Model for Educational Question Generation
**Presentation Notes & Outline (20 Slides)**

---

## Slide 1: Title Slide
*   **Title:** Small Language Model for Educational Question Generation
*   **Subtitle:** End-to-End Implementation of a 114M-Parameter Decoder-Only Transformer
*   **Speaker:**  \textbf{Kamal Nayan Kumar (23BDS026)}\\
      \textbf{Altaf Raja (23BCS011)}\\
      \textbf{Rahul Patel (23BDS047)}\\
      \textbf{Vijaypal Singh Rathore (23BDS067)}\\[0.8cm]
      Under the guidance of\\[0.2cm]
      \textbf{Dr. Krishnendu Ghosh}\\
      \textbf{Project Supervisor}
*   **Visual:** Title text centered. Provide a link to the GitHub repository at the bottom.

---

## Slide 2: Executive Summary
*   **Heading:** Project Overview
*   **Bullet Points:**
    *   **Goal:** Build a specialized, efficient language model for educational use without relying on expensive cloud APIs.
    *   **Architecture:** 114.1 million parameter decoder-only transformer.
    *   **Training:** Pretrained from scratch on 3B tokens; fine-tuned for specialized question generation.
    *   **Result:** A highly capable, locally deployable SLM that generates semantically accurate short-answer and complex reasoning questions.
*   **Visual Suggestion:** A clean summary box showing key numbers: 114M parameters, 3B tokens, 88k SFT samples.

---

## Slide 3: Motivation & Problem Statement
*   **Heading:** Why Build an SLM for Education?
*   **Bullet Points:**
    *   **The Big Tech Bottleneck:** Current state-of-the-art models (e.g., GPT-4) are massive, requiring API keys, cloud dependency, and ongoing costs.
    *   **Privacy & Control:** Educational datasets may be sensitive; calling external APIs poses privacy risks.
    *   **Over-parameterization:** Generating educational questions does not strictly require 70B+ parameters. A highly specialized small model is often sufficient.
    *   **Solution:** An open, transparent SLM optimized purely for generating questions from text passages.

---

## Slide 4: Project Objectives
*   **Heading:** Core Objectives
*   **Bullet Points:**
    *   **Educational Completeness:** Experience the full lifecycle (data prep $\rightarrow$ pretraining $\rightarrow$ fine-tuning $\rightarrow$ evaluation).
    *   **Practical Compactness:** Train within academic hardware limits (H100/A100) and deploy locally on modest hardware (Mac CPU).
    *   **Task Specialization:** Use Supervised Fine-Tuning (SFT) to shift the model from a basic text-predictor to a targeted question generator.
    *   **Modern Architecture:** Integrate state-of-the-art transformer efficiencies (GQA, RoPE, SwiGLU).

---

## Slide 5: High-Level Pipeline
*   **Heading:** End-to-End Training Pipeline
*   **Bullet Points:**
    *   The project was segmented into five distinct phases, ensuring checkpointing and clear progress tracking.
    *   Phase 2: Data Preprocessing \& Tokenization
    *   Phase 3: Base Model Pretraining
    *   Phase 4 \& 5: SFT Data Prep and Supervised Fine-Tuning
    *   Phase 6: Inference \& Evaluation 
*   **Visual:** Insert `evaluation/pipeline_summary.png`.

---

## Slide 6: Pretraining Data Engineering
*   **Heading:** Building the Pretraining Corpus
*   **Bullet Points:**
    *   **Objective:** Give the model a strong foundation in human language, facts, and reasoning.
    *   **Preprocessing Pipeline:**
        *   MinHash Deduplication (removing $\sim$15-20\% noise).
        *   Quality filtering (length, rep-ratio) & HTML stripping.
        *   Unicode normalization.
    *   **Tokenization:** Used OpenAI's `tiktoken` (\texttt{r50k\_base}) to condense cleaned text into a 3B token binary training file.

---

## Slide 7: Pretraining Corpus Mix
*   **Heading:** The 3 Billion Token Recipe
*   **Bullet Points:**
    *   To prevent domain-overfitting, the pretraining corpus was carefully blended.
    *   **OpenWebText ($\sim$50\%):** Diverse internet text for general vocabulary.
    *   **Wikipedia ($\sim$30\%):** High-quality, encyclopedic factual knowledge.
    *   **Project Gutenberg ($\sim$15\%):** Long-form literary structures and complex grammar.
    *   **Medium Articles ($\sim$5\%):** Contemporary, conversational explanatory text.
*   **Visual:** Insert the *left panel* of `evaluation/dataset_composition.png` (Pretraining Corpus Mix pie chart).

---

## Slide 8: Model Architecture
*   **Heading:** The 114M Parameter SLM
*   **Bullet Points:**
    *   **Type:** Autoregressive decoder-only transformer.
    *   **Scale:** 12 Layers, 768 hidden dimension.
    *   **Context:** 4096-token maximum sequence length.
    *   **Vocabulary:** 50,257 tokens + 3 custom special instruction tokens (50,260 total).
    *   **Weight Tying:** Shared weights between the token embedding map and the final LM projection head to save parameters.
*   **Visual:** Insert `evaluation/param_comparison.png` (specifically the right side strategy card comparison).

---

## Slide 9: Architectural Innovations
*   **Heading:** State-of-the-Art Efficiencies
*   **Bullet Points:**
    *   **Grouped Query Attention (GQA):** 12 Query heads paired with 4 Key/Value heads. Dramatically reduces KV-cache memory during inference without hurting perplexity.
    *   **Rotary Positional Embeddings (RoPE):** Encodes relative token distances seamlessly, eliminating fixed absolute position limits.
    *   **SwiGLU FFN:** Replaces standard ReLU with a gated linear unit—improves learning dynamics and representation capacity.
    *   **RMSNorm:** Pre-normalization strategy that is faster and more stable than LayerNorm.

---

## Slide 10: Pretraining Phase (Compute \& Results)
*   **Heading:** Pretraining from Scratch
*   **Bullet Points:**
    *   **Hardware:** 1$\times$ NVIDIA H100 GPU on Lightning AI.
    *   **Compute Time:** Approximately 8 hours to process 3 Billion tokens.
    *   **Training Hyperparameters:** `bfloat16` precision, gradient checkpointing, cosine learning rate decay with linear warmup.
    *   **Result:** Stable convergence from random noise to a fluent language model (Final Validation Loss: $\sim$2.71).
*   **Visual:** Insert `evaluation/pretrain_loss_curve.png`.

---

## Slide 11: Supervised Fine-Tuning (SFT) Rationale
*   **Heading:** From Continuer to Question-Generator
*   **Bullet Points:**
    *   Pretrained models only "continue" text. They don't naturally answer instructions.
    *   SFT bridges this gap by training the model on strictly formatted dialogue pairs (context $+$ desired output).
    *   Implemented via the **ChatML format**: structurally separating the `<|system|>`, `<|user|>`, and `<|assistant|>` turns.
    *   **Loss Masking:** Critical technique—gradients are calculated \emph{only} on the assistant's generated question. The context prompt is ignored in the loss calculation.

---

## Slide 12: SFT Dataset Composition
*   **Heading:** Two Distinct Difficulty Levels
*   **Bullet Points:**
    *   Compiled exactly **88,275** training samples focused entirely on educational querying.
    *   **SQuAD v2 (98.4\%):** High-quality short-answer queries teaching the model to extract explicit facts from text.
    *   **HotpotQA (1.6\%):** Complex multi-hop reasoning teaching the model to synthesize information across different sentences.
*   **Visual:** Insert the *right panel* of `evaluation/dataset_composition.png` (SFT Dataset Composition pie chart).

---

## Slide 13: SFT Training Execution
*   **Heading:** Specialized Fine-Tuning Execution
*   **Bullet Points:**
    *   **Hardware:** 1$\times$ NVIDIA A100 (40GB) GPU.
    *   **Compute Time:** Approximately 5 hours.
    *   **Approach:** Low learning rate (1e-5) with heavy weight decay to preserve pretraining knowledge while adapting to the ChatML format.
    *   **Early Stopping:** Best validation results achieved early (Epoch 1, validation response loss: 1.6262), preventing overfitting to the strict dataset syntaxes.
*   **Visual:** Insert `evaluation/sft_loss_curve.png` (Loss & perplexity per epoch).

---

## Slide 14: Inference Engine Implementation
*   **Heading:** Fast Local Inference
*   **Bullet Points:**
    *   **Hardware:** Runs flawlessly on local edge devices, including Apple Silicon (Mac M-series CPUs).
    *   **KV-Caching Mechanism:** Caches previously computed keys and values, drastically saving compute on long educational PDFs.
    *   **Generation Strategy:** Employs \textit{Nucleus Sampling} (\texttt{top\_p = 0.90}) paired with a low temperature (\texttt{T = 0.3}) for factual, deterministic questioning.
    *   **Repetition Penalties:** Integrated repetition scaling to prevent infinite loops.

---

## Slide 15: Quantitative Evaluation Methods
*   **Heading:** How Do We Measure "Good" Questions?
*   **Bullet Points:**
    *   Evaluated strictly on held-out test sets from SQuAD (Short) and HotpotQA (Complex).
    *   **Lexical Metrics (BLEU-4, ROUGE-L):** Measure exact word overlap. Often score poorly on generative tasks since there are many ways to ask the same concept.
    *   **Semantic Metrics (BERTScore):** Uses deep embeddings to measure if the \textit{meaning} of the generated question matches the ground-truth. The ultimate indicator of success.

---

## Slide 16: Quantitative Results
*   **Heading:** Semantic Excellence
*   **Bullet Points:**
    *   Achieved an exceptional **BERTScore F1 of 0.89+** on Short questions and **0.86+** on Complex reasoning.
    *   BLEU-4 remains near-zero, proving the model doesn't just copy/paste from the text, but rather paraphrases dynamically.
    *   The model performs slightly better on Short-answer extractions, reflecting the SFT dataset balance.
*   **Visual:** Insert `evaluation/evaluation_metrics.png` (The grouped bar chart showing all metrics).

---

## Slide 17: Radar Profile \& Efficiency
*   **Heading:** Multi-Metric Profile & Training Speed
*   **Bullet Points:**
    *   The radar profile highlights the strong spike in semantic similarity (BERTScore) versus traditional overlapping metrics.
    *   The entire project from empty folder to 114M parameters required just 13 hours of combined GPU compute (8h on H100; 5h on A100).
    *   Proves that highly capable specific-purpose AI models can be entirely engineered within university resource budgets.
*   **Visual:** Insert `evaluation/radar_chart.png` or `evaluation/training_efficiency.png`.

---

## Slide 18: Qualitative Demo (Short-Answer)
*   **Heading:** Demonstration: Photosynthesis Extraction
*   **Bullet Points:**
    *   **Passage Input:** "Photosynthesis is the process... converting sunlight into glucose and oxygen... The overall equation is 6CO2 + 6H2O + light $\rightarrow$ C6H12O6 + 6O2."
    *   **Task:** Generate a short-answer difficulty question. 
    *   **Model Output:** \textit{"Provide an example of a plant that uses the energy produced by photosynthesis to form water."} (Demonstrating accurate structural formatting and topic extraction).
*   **Visual:** Insert `evaluation/demo_short_1.png` to show the terminal UX.

---

## Slide 19: Qualitative Demo (Complex Reasoning)
*   **Heading:** Demonstration: Multi-Hop History
*   **Bullet Points:**
    *   **Passage Input:** "World War II began September 1, 1939... Expanded after Japan attacked Pearl Harbor in Dec 1941... War ended in the Pacific on September 2, 1945."
    *   **Task:** Generate a complex difficulty question.
    *   **Model Output:** \textit{"Provide an example of a significant event that occurred on September 2, 1945."}
*   **Visual:** Insert `evaluation/demo_complex.png` to show the terminal UX.

---

## Slide 20: Conclusion & Future Scope
*   **Heading:** Conclusion & Future Scope
*   **Bullet Points:**
    *   We successfully built, documented, and evaluated a domain-specific SLM completely from scratch.
    *   Modern architecture (GQA, RoPE) enables 114M parameters to punch far above their weight.
    *   **Future Scope:**
        *   Implement Direct Preference Optimization (DPO) to correct hallucinated formatting.
        *   Expand the SFT corpus with high-quality, human-curated MCQ and fill-in-the-blank structures.
        *   Export weights to ONNX/CoreML for real-time mobile application integration.
*   **Visual:** Final "Thank You / Q\&A" text. Link to the Github Repository.
