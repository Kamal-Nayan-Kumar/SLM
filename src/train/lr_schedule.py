"""Cosine decay learning rate schedule with linear warmup."""

import math


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Returns the learning rate at a given step.
    - Linear warmup from 0 → max_lr over [0, warmup_steps)
    - Cosine decay from max_lr → min_lr over [warmup_steps, max_steps]
    - Constant min_lr after max_steps
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    # Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
