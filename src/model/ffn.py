"""SwiGLU Feed-Forward Network as used in Llama/Mistral architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import ModelConfig


class SwiGLUFFN(nn.Module):
    """
    SwiGLU: output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
    Three linear projections, no bias, no dropout.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up_proj   = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
