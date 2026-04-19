"""
generate_charts.py — Generates all evaluation figures for the SLM report.
Run from project root: python evaluation/generate_charts.py
Outputs saved to evaluation/ directory.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. Pretraining loss curve (detailed)
# ─────────────────────────────────────────────────────────────────────────────
def plot_pretrain_loss():
    steps = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000,
             5500,6000,6500,7000,7500,8000,8500,9000]
    val_losses = [5.1513,4.0066,3.5391,3.3122,3.2317,3.0933,
                  3.0416,2.9629,2.9297,2.8063,2.8828,2.9057,
                  2.8161,2.8337,2.7661,2.7085,2.7235,2.7680]
    tokens_B = [s * 10 * 8 * 4096 / 1e9 for s in steps]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: loss vs step
    ax = axes[0]
    ax.plot(steps, val_losses, "o-", color="#2563eb", linewidth=2,
            markersize=5, markerfacecolor="white", markeredgewidth=1.5)
    ax.fill_between(steps, val_losses, 6.0, alpha=0.08, color="#2563eb")
    ax.axhline(y=min(val_losses), color="#dc2626", linestyle="--", linewidth=1.0,
               label=f"Min val loss = {min(val_losses):.4f}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Pretraining Validation Loss vs. Step")
    ax.legend()
    ax.set_ylim(2.4, 5.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Right: loss vs tokens processed
    ax = axes[1]
    ax.plot(tokens_B, val_losses, "s-", color="#7c3aed", linewidth=2,
            markersize=5, markerfacecolor="white", markeredgewidth=1.5)
    ax.axhline(y=min(val_losses), color="#dc2626", linestyle="--", linewidth=1.0,
               label=f"Min val loss = {min(val_losses):.4f}")
    ax.set_xlabel("Tokens Processed (Billions)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Pretraining Validation Loss vs. Tokens Processed")
    ax.legend()
    ax.set_ylim(2.4, 5.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(pad=2.0)
    path = OUT / "pretrain_loss_curve.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SFT validation loss per epoch
# ─────────────────────────────────────────────────────────────────────────────
def plot_sft_loss():
    epochs = [0, 1, 2, 3]
    val_losses = [1.6517, 1.6262, 1.6406, 1.6725]
    perplexities = [np.exp(l) for l in val_losses]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    bars = ax.bar(epochs, val_losses, color=["#f59e0b","#16a34a","#f59e0b","#dc2626"],
                  edgecolor="white", linewidth=1.2, width=0.55)
    ax.axhline(y=min(val_losses), color="#2563eb", linestyle="--", linewidth=1.2,
               label=f"Best = {min(val_losses):.4f} (Epoch 1)")
    for bar, v in zip(bars, val_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(epochs)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax.set_ylabel("Validation Response Loss")
    ax.set_title("SFT Validation Loss per Epoch")
    ax.legend()
    ax.set_ylim(1.58, 1.72)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    ax.plot(epochs, perplexities, "D-", color="#0891b2", linewidth=2,
            markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax.axhline(y=min(perplexities), color="#dc2626", linestyle="--", linewidth=1.0,
               label=f"Best PPL = {min(perplexities):.2f}")
    for i, (e, p) in enumerate(zip(epochs, perplexities)):
        ax.annotate(f"{p:.3f}", (e, p), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(epochs)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax.set_ylabel("Perplexity")
    ax.set_title("SFT Validation Perplexity per Epoch")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(pad=2.0)
    path = OUT / "sft_loss_curve.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset composition charts (pretrain + SFT)
# ─────────────────────────────────────────────────────────────────────────────
def plot_dataset_composition():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pretraining corpus
    pretrain_labels = ["OpenWebText\n(~50%)", "Wikipedia\n(~30%)",
                        "Gutenberg\n(~15%)", "Medium\n(~5%)"]
    pretrain_sizes  = [50, 30, 15, 5]
    pretrain_colors = ["#3b82f6","#10b981","#f59e0b","#ef4444"]
    explode = (0.04, 0.04, 0.04, 0.04)

    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        pretrain_sizes, labels=pretrain_labels, colors=pretrain_colors,
        autopct="%1.0f%%", startangle=90, explode=explode,
        wedgeprops=dict(edgecolor="white", linewidth=2))
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax.set_title("Pretraining Corpus Mix\n(~3B tokens total)", fontweight="bold", pad=12)

    # SFT dataset
    sft_labels  = ["SQuAD v2\n(86,821)", "HotpotQA\n(1,454)"]
    sft_sizes   = [86821, 1454]
    sft_colors  = ["#0ea5e9","#84cc16"]
    explode_sft = (0.04, 0.04)

    ax2 = axes[1]
    wedges2, texts2, autotexts2 = ax2.pie(
        sft_sizes, labels=sft_labels, colors=sft_colors,
        autopct="%1.1f%%", startangle=90, explode=explode_sft,
        wedgeprops=dict(edgecolor="white", linewidth=2))
    for t in autotexts2:
        t.set_fontsize(9)
        t.set_fontweight("bold")
    ax2.set_title("SFT Dataset Composition\n(88,275 total samples)", fontweight="bold", pad=12)

    plt.tight_layout()
    path = OUT / "dataset_composition.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation metrics — grouped bar chart (enhanced)
# ─────────────────────────────────────────────────────────────────────────────
def plot_evaluation_metrics_bar():
    metrics  = ["ROUGE-L", "BLEU-4", "METEOR", "BERTScore"]
    squad    = [0.2567, 0.0000, 0.2658, 0.8945]
    hotpot   = [0.1943, 0.0248, 0.2256, 0.8643]

    x     = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - width/2, squad,  width, label="SQuAD (Short)",
                   color="#3b82f6", edgecolor="white", linewidth=1.2)
    bars2 = ax.bar(x + width/2, hotpot, width, label="HotpotQA (Complex)",
                   color="#f97316", edgecolor="white", linewidth=1.2)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.012,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, color="#1e40af")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.012,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, color="#c2410c")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics: SQuAD vs. HotpotQA\n(114M SLM — 50 samples each)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Annotate BERTScore region
    ax.axhspan(0.85, 1.05, alpha=0.05, color="green")
    ax.text(3.55, 0.87, "High\nSemantics", fontsize=8, color="green", style="italic")

    plt.tight_layout()
    path = OUT / "evaluation_metrics.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved (updated): {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Parameter efficiency comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_param_comparison():
    models  = ["Our SLM\n(Decoder)", "T5-Small\n(Enc-Dec)", "T5-Base\n(Enc-Dec)",
               "T5-Large\n(Enc-Dec)", "BART-Base\n(Enc-Dec)", "GPT-2\nMedium"]
    params  = [114, 60, 220, 770, 139, 345]  # millions
    bertscore = [0.89, 0.84, 0.87, None, None, None]  # illustrative where known
    colors  = ["#2563eb","#64748b","#64748b","#64748b","#64748b","#64748b"]
    highlights = [True] + [False]*5

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: parameter count
    ax = axes[0]
    bars = ax.barh(models, params, color=colors, edgecolor="white", linewidth=1.2, height=0.55)
    for bar, p, hl in zip(bars, params, highlights):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
                f"{p}M", va="center", fontsize=10,
                fontweight="bold" if hl else "normal",
                color="#2563eb" if hl else "#374151")
    ax.set_xlabel("Number of Parameters (Millions)")
    ax.set_title("Model Size Comparison\n(Question Generation Models)")
    ax.set_xlim(0, 900)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.invert_yaxis()

    # Right: architecture strategy cards − individual line placement to avoid overlap
    ax2 = axes[1]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("Architecture Strategy Comparison", pad=10, fontsize=12)

    cards = [
        {
            "title": "Decoder-only (Ours)",
            "color": "#2563eb",
            "bg":    "#dbeafe",
            "y_box": 0.54,          # bottom of box in axes coords
            "box_h": 0.40,
            "bullets": [
                "• No cross-attention overhead",
                "• KV-cache for fast inference",
                "• Native generation format",
                "• Modern: GQA, RoPE, SwiGLU",
            ],
        },
        {
            "title": "Encoder-Decoder (T5, BART)",
            "color": "#64748b",
            "bg":    "#f1f5f9",
            "y_box": 0.06,
            "box_h": 0.40,
            "bullets": [
                "• Established QG baseline",
                "• Separate encoder / decoder",
                "• Larger parameter budgets",
                "• Heavier inference overhead",
            ],
        },
    ]

    for card in cards:
        # Background box
        ax2.add_patch(mpatches.FancyBboxPatch(
            (0.04, card["y_box"]), 0.92, card["box_h"],
            boxstyle="round,pad=0.02",
            facecolor=card["bg"], edgecolor=card["color"],
            linewidth=1.5, transform=ax2.transAxes, clip_on=False))

        # Title  (top of box, padded inward)
        title_y = card["y_box"] + card["box_h"] - 0.055
        ax2.text(0.50, title_y, card["title"],
                 transform=ax2.transAxes, ha="center", va="top",
                 fontsize=10, fontweight="bold", color=card["color"])

        # Bullet lines – spaced 0.07 apart below title
        for j, bullet in enumerate(card["bullets"]):
            by = title_y - 0.075 - j * 0.072
            ax2.text(0.12, by, bullet,
                     transform=ax2.transAxes, ha="left", va="top",
                     fontsize=9, color="#374151")

    plt.tight_layout()
    path = OUT / "param_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")



# ─────────────────────────────────────────────────────────────────────────────
# 6. Radar chart — multi-metric comparison across datasets
# ─────────────────────────────────────────────────────────────────────────────
def plot_radar_chart():
    from matplotlib.patches import FancyArrowPatch

    categories = ["ROUGE-L", "BLEU-4\n(×10)", "METEOR", "BERTScore"]
    squad_vals  = [0.2567, 0.0000, 0.2658, 0.8945]
    hotpot_vals = [0.1943, 0.0248, 0.2256, 0.8643]

    # Scale BLEU-4 ×10 for visibility
    squad_plot  = [squad_vals[0],  squad_vals[1]*10,  squad_vals[2],  squad_vals[3]]
    hotpot_plot = [hotpot_vals[0], hotpot_vals[1]*10, hotpot_vals[2], hotpot_vals[3]]

    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    squad_plot  += squad_plot[:1]
    hotpot_plot += hotpot_plot[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, squad_plot,  "o-", linewidth=2, color="#3b82f6", label="SQuAD (Short)")
    ax.fill(angles, squad_plot, alpha=0.15, color="#3b82f6")

    ax.plot(angles, hotpot_plot, "s-", linewidth=2, color="#f97316", label="HotpotQA (Complex)")
    ax.fill(angles, hotpot_plot, alpha=0.15, color="#f97316")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8)
    ax.set_title("Metric Radar Chart\n(114M SLM on QG Benchmarks)", fontsize=12, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    ax.grid(color="gray", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = OUT / "radar_chart.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Training pipeline summary figure
# ─────────────────────────────────────────────────────────────────────────────
def plot_pipeline_summary():
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    stages = [
        ("Raw Data\nCollection", "#dbeafe", "#1d4ed8", 0.5),
        ("Preprocessing\n& Dedup", "#ede9fe", "#6d28d9", 2.0),
        ("Tokenization\n(tiktoken)", "#fce7f3", "#be185d", 3.5),
        ("Pretraining\n(H100 GPU)\n3B tokens", "#dcfce7", "#166534", 5.0),
        ("SFT Data\nPrep", "#fef3c7", "#92400e", 6.5),
        ("Supervised\nFine-Tuning\n(A100 GPU)", "#ffedd5", "#c2410c", 8.0),
        ("Inference\n& Evaluation", "#f0fdf4", "#15803d", 9.5),
    ]

    for i, (label, bg, fg, x) in enumerate(stages):
        rect = mpatches.FancyBboxPatch(
            (x, 0.6), 1.2, 2.25,
            boxstyle="round,pad=0.1", facecolor=bg, edgecolor=fg,
            linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + 0.6, 1.72, label, ha="center", va="center",
                fontsize=9, color=fg, fontweight="bold", zorder=3,
                linespacing=1.5)
        if i < len(stages) - 1:
            ax.annotate("", xy=(stages[i+1][3], 1.72),
                        xytext=(x + 1.2, 1.72),
                        arrowprops=dict(arrowstyle="->", color="#6b7280",
                                        lw=2.0), zorder=4)

    ax.text(6.5, 3.2, "SLM End-to-End Training Pipeline",
            ha="center", va="center", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = OUT / "pipeline_summary.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Training efficiency: tokens/sec on A100 (illustrative)
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # Left: Compute breakdown
    phases = ["2A: Data\nDownload", "2B: Pre-\nprocessing",
              "2C: Tokeni-\nzation", "Phase 3:\nPretraining\n(H100)",
              "Phase 5:\nSFT\n(A100 40GB)"]
    gpu_type = ["CPU", "CPU", "CPU", "H100", "A100 40GB"]
    hours    = [4, 2, 1, 8, 5]
    colors_p = ["#94a3b8", "#94a3b8", "#94a3b8", "#2563eb", "#7c3aed"]

    ax = axes[0]
    x = range(len(phases))
    bars = ax.bar(x, hours, color=colors_p, edgecolor="white", linewidth=1.2, width=0.55)

    for bar, h, g in zip(bars, hours, gpu_type):
        bx = bar.get_x() + bar.get_width() / 2
        # For tall bars put annotation above; for short bars put it inside/side
        if h >= 6:
            ax.text(bx, h + 0.5, f"{h}h", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
            ax.text(bx, h / 2, f"({g})", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")
        else:
            ax.text(bx, h + 0.5, f"{h}h\n({g})", ha="center", va="bottom",
                    fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(phases, fontsize=8.5)
    ax.set_ylabel("Approximate Time (hours)")
    ax.set_title("Compute Time per Training Phase")
    ax.set_ylim(0, 12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Right: val loss comparison at key token counts
    tokens_B = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]
    approx_loss = [4.0, 3.35, 3.09, 2.96, 2.81, 2.78, 2.72]

    ax2 = axes[1]
    ax2.plot(tokens_B, approx_loss, "o-", color="#2563eb", linewidth=2, markersize=6)
    ax2.axvline(x=3.0, color="#dc2626", linestyle="--", linewidth=1.2,
                label="Training cutoff (3B tokens)")
    ax2.fill_between(tokens_B, approx_loss, max(approx_loss)+0.5,
                     alpha=0.07, color="#2563eb")
    ax2.set_xlabel("Tokens Processed (B)")
    ax2.set_ylabel("Val Loss")
    ax2.set_title("Learning Progress: Validation Loss\nvs. Token Budget")
    ax2.legend()
    ax2.grid(linestyle="--", alpha=0.35)

    plt.tight_layout()
    path = OUT / "training_efficiency.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating all evaluation charts for SLM report...\n")
    plot_pretrain_loss()
    plot_sft_loss()
    plot_dataset_composition()
    plot_evaluation_metrics_bar()
    plot_param_comparison()
    plot_radar_chart()
    plot_pipeline_summary()
    plot_training_efficiency()
    print("\nAll charts generated successfully!")
