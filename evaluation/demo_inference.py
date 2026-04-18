"""
demo_inference.py — Run model for 2 short + 1 complex question demos.
Run from project root: python evaluation/demo_inference.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.generate import load_model, generate_from_text

PASSAGE_PHOTO = (
    "Photosynthesis is the process by which green plants, algae, and some bacteria "
    "convert light energy—usually from the Sun—into chemical energy stored in glucose. "
    "Chlorophyll, the green pigment found in chloroplasts, absorbs sunlight and uses "
    "its energy to combine carbon dioxide from the air with water from the soil, "
    "producing glucose and oxygen as byproducts. The overall chemical equation is: "
    "6CO2 + 6H2O + light energy → C6H12O6 + 6O2. This process is fundamental to life "
    "on Earth, as it produces the oxygen we breathe and forms the base of almost all food chains."
)

PASSAGE_NEWTON = (
    "Isaac Newton's three laws of motion form the foundation of classical mechanics. "
    "The first law states that an object at rest stays at rest and an object in motion "
    "stays in motion unless acted upon by an external force. The second law establishes "
    "that force equals mass times acceleration (F = ma). The third law states that for "
    "every action there is an equal and opposite reaction."
)

PASSAGE_WW2 = (
    "World War II began on September 1, 1939, when Nazi Germany, under Adolf Hitler, "
    "invaded Poland. Britain and France declared war on Germany two days later. The war "
    "expanded rapidly across Europe, North Africa, and eventually the Pacific after Japan "
    "attacked Pearl Harbor in December 1941, drawing the United States into the conflict. "
    "The war ended in Europe on May 8, 1945 (V-E Day), following Germany's unconditional "
    "surrender, and in the Pacific on September 2, 1945 (V-J Day), after the United States "
    "dropped atomic bombs on Hiroshima and Nagasaki."
)


def demo():
    print("Loading model...")
    model, enc, device = load_model()
    print(f"Model ready on device: {device}\n")
    print("=" * 72)

    # ── Demo 1: Short — Biology ───────────────────────────────────────────
    print("\n[Demo 1] SHORT QUESTION — Biology (Photosynthesis)")
    print("─" * 60)
    print(f"Passage:\n{PASSAGE_PHOTO}\n")
    q1 = generate_from_text(
        PASSAGE_PHOTO, difficulty="short",
        model=model, enc=enc, device=device,
        temperature=0.3, top_p=0.9, max_new_tokens=64
    )
    print(f"Generated Question:\n  ► {q1[0]}")

    print("\n" + "=" * 72)

    # ── Demo 2: Short — Physics ───────────────────────────────────────────
    print("\n[Demo 2] SHORT QUESTION — Physics (Newton's Laws)")
    print("─" * 60)
    print(f"Passage:\n{PASSAGE_NEWTON}\n")
    q2 = generate_from_text(
        PASSAGE_NEWTON, difficulty="short",
        model=model, enc=enc, device=device,
        temperature=0.3, top_p=0.9, max_new_tokens=64
    )
    print(f"Generated Question:\n  ► {q2[0]}")

    print("\n" + "=" * 72)

    # ── Demo 3: Complex — History ─────────────────────────────────────────
    print("\n[Demo 3] COMPLEX QUESTION — History (World War II)")
    print("─" * 60)
    print(f"Passage:\n{PASSAGE_WW2}\n")
    q3 = generate_from_text(
        PASSAGE_WW2, difficulty="complex",
        model=model, enc=enc, device=device,
        temperature=0.5, top_p=0.9, max_new_tokens=128
    )
    print(f"Generated Question:\n  ► {q3[0]}")

    print("\n" + "=" * 72)
    print("Demo complete.")


if __name__ == "__main__":
    demo()
