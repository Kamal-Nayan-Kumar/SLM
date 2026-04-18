"""
make_demo_screenshots.py — Creates professional terminal-style screenshot images
for the 3 inference demos: 2 short-answer + 1 complex question.
Run from project root: python evaluation/make_demo_screenshots.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("/Users/nayan/Documents/SLM/evaluation")

PASSAGE_PHOTO = (
    "Photosynthesis is the process by which green plants, algae, and some bacteria convert\n"
    "light energy—usually from the Sun—into chemical energy stored in glucose. Chlorophyll,\n"
    "the green pigment found in chloroplasts, absorbs sunlight and uses its energy to combine\n"
    "carbon dioxide from the air with water from the soil, producing glucose and oxygen as\n"
    "byproducts. The overall chemical equation is: 6CO\u2082 + 6H\u2082O + light \u2192 C\u2086H\u2081\u2082O\u2086 + 6O\u2082."
)

PASSAGE_NEWTON = (
    "Isaac Newton's three laws of motion form the foundation of classical mechanics.\n"
    "The first law states that an object at rest stays at rest and an object in motion\n"
    "stays in motion unless acted upon by an external force. The second law establishes\n"
    "that force equals mass times acceleration (F = ma). The third law states that for\n"
    "every action there is an equal and opposite reaction."
)

PASSAGE_WW2 = (
    "World War II began on September 1, 1939, when Nazi Germany, under Adolf Hitler,\n"
    "invaded Poland. Britain and France declared war on Germany two days later. The war\n"
    "expanded rapidly across Europe, North Africa, and eventually the Pacific after Japan\n"
    "attacked Pearl Harbor in December 1941, drawing the United States into the conflict.\n"
    "The war ended in Europe on May 8, 1945 (V-E Day), and in the Pacific on September 2,\n"
    "1945 (V-J Day), after the United States dropped atomic bombs on Hiroshima and Nagasaki."
)

DEMOS = [
    {
        "label": "Demo 1 \u2014 Short Question (Biology)",
        "type_label": "DIFFICULTY: short  |  python src/inference/generate.py --difficulty short",
        "passage": PASSAGE_PHOTO,
        "question": "\u25ba Provide an example of a plant that uses the energy produced by photosynthesis to form water.",
        "filename": "demo_short_1.png",
        "accent": "#34d399",
    },
    {
        "label": "Demo 2 \u2014 Short Question (Physics)",
        "type_label": "DIFFICULTY: short  |  python src/inference/generate.py --difficulty short",
        "passage": PASSAGE_NEWTON,
        "question": "\u25ba Describe the relationship between force, mass, and acceleration\n  according to Isaac Newton's three laws of motion.",
        "filename": "demo_short_2.png",
        "accent": "#34d399",
    },
    {
        "label": "Demo 3 \u2014 Complex Question (History)",
        "type_label": "DIFFICULTY: complex  |  python src/inference/generate.py --difficulty complex",
        "passage": PASSAGE_WW2,
        "question": "\u25ba Provide an example of a significant event that occurred on September 2, 1945.",
        "filename": "demo_complex.png",
        "accent": "#60a5fa",
    },
]


def render_terminal(demo: dict):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.axis("off")

    # Window chrome bar
    chrome = mpatches.FancyBboxPatch(
        (0.0, 0.965), 1.0, 0.035,
        boxstyle="square,pad=0", facecolor="#2d2d2d",
        edgecolor="none", transform=ax.transAxes, clip_on=False)
    ax.add_patch(chrome)

    # Traffic light dots
    for cx, col in [(0.025, "#ff5f57"), (0.055, "#ffbd2e"), (0.085, "#28c840")]:
        dot = plt.Circle((cx, 0.9825), 0.013, color=col,
                         transform=ax.transAxes, clip_on=False)
        ax.add_patch(dot)

    ax.text(0.5, 0.9825, "python src/inference/generate.py  \u2014  SLM Question Generator",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color="#9ca3af", fontfamily="monospace", clip_on=False)

    y = 0.91

    def write(text, color="#d4d4d4", bold=False, size=8.8):
        nonlocal y
        ax.text(0.025, y, text,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=size, color=color, fontfamily="monospace",
                fontweight="bold" if bold else "normal", clip_on=False)
        lines = text.count("\n") + 1
        y -= 0.048 * lines

    write("$ " + demo["type_label"].split("|")[1].strip(), color="#6b7280")
    write("Model loaded from checkpoints/sft/best_model.pt  |  device: cpu", color="#6b7280", size=8)
    y -= 0.014

    write("\u2550" * 64, color="#374151")
    y -= 0.004

    write(f"  {demo['label']}", color=demo["accent"], bold=True, size=9.5)
    write(f"  {demo['type_label']}", color="#6b7280", size=8)
    y -= 0.008

    write("  Input Passage:", color="#9ca3af", bold=True)
    for line in demo["passage"].split("\n"):
        write(f"  {line}", color="#d4d4d4", size=8.4)
    y -= 0.010

    write("  Generated Question:", color="#9ca3af", bold=True)
    write(f"  {demo['question']}", color=demo["accent"], bold=True, size=9.5)

    y -= 0.014
    write("\u2550" * 64, color="#374151")

    path = OUT / demo["filename"]
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="#1e1e1e", edgecolor="none")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    for demo in DEMOS:
        render_terminal(demo)
    print("All demo screenshots created.")
