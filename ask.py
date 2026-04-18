"""
Usage:
  python ask.py
  python ask.py --type complex
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.inference.generate import load_model, generate_from_text

VALID_TYPES = ["short_answer", "complex"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="short_answer", choices=VALID_TYPES,
                        help="Question type (default: short_answer)")
    parser.add_argument("--ckpt", default=None,
                        help="Path to a specific checkpoint to test.")
    parser.add_argument("--temp", type=float, default=0.2,
                        help="Sampling temperature (use 0 for greedy decoding).")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling top-p.")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override maximum generation length.")
    args = parser.parse_args()

    print("Loading model...")
    model, enc, device = load_model(args.ckpt)
    print("Model ready.\n")

    while True:
        para = input("Input: ").strip()
        if not para:
            continue

        questions = generate_from_text(
            para,
            args.type,
            model,
            enc,
            device,
            temperature=args.temp,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"\nQuestion: {questions[0] if questions else 'No question generated.'}\n")

        again = input("Next? (y/n): ").strip().lower()
        if again != "y":
            print("Exiting.")
            break

if __name__ == "__main__":
    main()
