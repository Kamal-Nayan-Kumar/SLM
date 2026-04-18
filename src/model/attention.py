"""Grouped Query Attention (GQA) with Rotary Positional Embeddings (RoPE)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from src.model.config import ModelConfig


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for RoPE. Not a learned module."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        # Frequency inverse for each pair of dimensions
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute tables up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, start_pos: int = 0):
        end_pos = start_pos + seq_len
        return (
            self.cos_cached[:, :, start_pos:end_pos, :],
            self.sin_cached[:, :, start_pos:end_pos, :],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate each pair: [x1, x2] -> [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors. q/k shape: (B, n_heads, T, head_dim)."""
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class GroupedQueryAttention(nn.Module):
    """GQA: n_heads query heads, n_kv_heads key/value heads. No bias anywhere."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.head_dim
        self.n_rep      = cfg.n_heads // cfg.n_kv_heads  # heads per KV group

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads    * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model,                   bias=False)

        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        past_len = 0 if past_kv is None else past_kv[0].size(2)

        # Project and split into heads
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(T, start_pos=past_len)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Expand KV heads to match Q heads (GQA repeat)
        k_attn = k
        v_attn = v
        if self.n_rep > 1:
            k_attn = k.repeat_interleave(self.n_rep, dim=1)
            v_attn = v.repeat_interleave(self.n_rep, dim=1)

        # During cached decoding we feed one new token at a time, so there are
        # no future positions inside the current query block.
        is_causal = past_kv is None
        y = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=is_causal)

        # Re-assemble and project
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        y = self.o_proj(y)
        return (y, present_kv) if use_cache else y
