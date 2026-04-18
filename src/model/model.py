"""
Full decoder-only transformer model.
Architecture: Embedding → 12x (RMSNorm→GQA→Residual + RMSNorm→SwiGLU→Residual) → RMSNorm → LM Head
Weight tying: lm_head.weight == tok_emb.weight
"""

import torch
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.model.config import ModelConfig
from src.model.attention import GroupedQueryAttention
from src.model.ffn import SwiGLUFFN


class RMSNorm(nn.Module):
    """Pre-normalization. No bias. Numerically stable in bfloat16."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class TransformerBlock(nn.Module):
    """Single transformer block: Pre-Norm GQA + Pre-Norm SwiGLU."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn      = GroupedQueryAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg.d_model)
        self.ffn       = SwiGLUFFN(cfg)

    def _attn_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.attn_norm(x))

    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.ffn_norm(x))

    def forward(
        self,
        x: torch.Tensor,
        use_grad_ckpt: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        if use_cache:
            attn_out, present_kv = self.attn(self.attn_norm(x), past_kv=past_kv, use_cache=True)
            x = x + attn_out
            x = self._ffn_forward(x)
            return x, present_kv

        if use_grad_ckpt and self.training:
            # Gradient checkpointing: trade recompute for memory
            x = checkpoint(self._attn_forward, x, use_reentrant=False)
            x = checkpoint(self._ffn_forward,  x, use_reentrant=False)
        else:
            x = self._attn_forward(x)
            x = self._ffn_forward(x)
        return x


class SLM(nn.Module):
    """120M-class Small Language Model for question generation."""

    def __init__(self, cfg: ModelConfig, use_grad_ckpt: bool = False):
        super().__init__()
        self.cfg            = cfg
        self.use_grad_ckpt  = use_grad_ckpt

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: input embedding == output projection
        self.lm_head.weight = self.tok_emb.weight

        # Init weights (GPT-2 style scaled init)
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layers) for stability
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        past_kvs: Optional[list[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list[Tuple[torch.Tensor, torch.Tensor]]]]:
        """idx: (B, T) int64 → logits: (B, T, vocab_size)"""
        x = self.tok_emb(idx)
        present_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = None if past_kvs is None else past_kvs[i]
            if use_cache:
                x, present_kv = block(
                    x,
                    use_grad_ckpt=False,
                    past_kv=past_kv,
                    use_cache=True,
                )
                present_kvs.append(present_kv)
            else:
                x = block(x, use_grad_ckpt=self.use_grad_ckpt)
        x = self.norm(x)
        logits = self.lm_head(x)
        return (logits, present_kvs) if use_cache else logits

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Count trainable params. With weight tying, embedding is counted once."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def configure_optimizers(
        self, weight_decay: float, lr: float,
        betas: tuple, device: str
    ) -> torch.optim.AdamW:
        """AdamW with weight decay applied only to 2D+ params (not norms/biases/embeddings)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # Embeddings and norm gains are 1D → no decay
            if p.dim() < 2 or "norm" in name or "bias" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # Use fused AdamW on CUDA (faster)
        use_fused = device == "cuda" and hasattr(torch.optim, "AdamW")
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=use_fused)
