"""Model configuration dataclass — single source of truth for all hyperparameters."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Architecture
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12          # query heads
    n_kv_heads: int = 4        # key/value heads (GQA 3:1 ratio)
    ffn_dim: int = 2048        # 8/3 * 768 rounded to nearest 256
    max_seq_len: int = 4096
    vocab_size: int = 50260    # r50k_base 50257 + 3 special tokens (im_start/im_end/pad)
    rope_theta: float = 10000.0
    dropout: float = 0.0       # no dropout during pretraining

    # Special token IDs (added on top of r50k_base)
    eot_token_id: int = 50256  # <|endoftext|> — native to r50k_base
    im_start_id: int = 50257   # <|im_start|>
    im_end_id: int = 50258     # <|im_end|>
    pad_id: int = 50259        # <|pad|>

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
