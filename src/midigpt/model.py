"""Decoder-only transformer over MIDI event tokens.

Port of the cicero Latin-LLM v2 substrate (train_token_gpt_v2.py): RoPE,
RMSNorm (f32 compute), SwiGLU MLP, optional GQA, flash attention via SDPA,
weight tying. Additions here: GPT-2-style init and an optional per-token
loss mask in the forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_kv_head: int          # GQA: K/V heads (must divide n_head); == n_head disables GQA
    n_embd: int
    mlp_ratio: float = 3.0  # SwiGLU; mlp dim = round(n_embd * mlp_ratio / 64) * 64
    dropout: float = 0.0
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rms).to(orig_dtype) * self.weight


def precompute_rope_cache(head_dim: int, max_seq_len: int, base: float,
                          device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) tensors of shape (max_seq_len, head_dim/2)."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                            dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, H, T, D). cos/sin are (T, D/2)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rx1 = x1 * cos - x2 * sin
    rx2 = x2 * cos + x1 * sin
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.group_size = cfg.n_head // cfg.n_kv_head

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor,
                rope_sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos[:T], rope_sin[:T])
        k = apply_rope(k, rope_cos[:T], rope_sin[:T])

        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        if torch.onnx.is_in_onnx_export():
            # explicit attention with a dynamic causal mask — exports to plain ONNX
            # ops (MatMul/Softmax/Where) and keeps T dynamic (SDPA's is_causal can
            # bake in a fixed mask size). Numerically equal to the SDPA path.
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            ar = torch.arange(T, device=x.device)
            att = att + (ar[None, :] > ar[:, None]).to(att.dtype) * (-1e9)
            y = F.softmax(att, dim=-1) @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


def _round_to_multiple(n: int, multiple: int = 64) -> int:
    return ((n + multiple - 1) // multiple) * multiple


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        mlp_dim = _round_to_multiple(int(n_embd * mlp_ratio), 64)
        self.gate_proj = nn.Linear(n_embd, mlp_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        return self.dropout(self.down_proj(gate * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg.n_embd, cfg.mlp_ratio, cfg.dropout)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor,
                rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x


class MusicGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self._rope_cos: torch.Tensor | None = None
        self._rope_sin: torch.Tensor | None = None

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weight counted once

    def _ensure_rope(self, device, dtype) -> None:
        if (self._rope_cos is None or self._rope_cos.device != device
                or self._rope_cos.dtype != dtype):
            head_dim = self.cfg.n_embd // self.cfg.n_head
            self._rope_cos, self._rope_sin = precompute_rope_cache(
                head_dim, self.cfg.block_size, self.cfg.rope_base, device, dtype)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        x = self.drop(self.tok_emb(idx))
        self._ensure_rope(x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, self._rope_cos, self._rope_sin)
        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size),
                                 targets.reshape(-1), reduction="none").view(B, T)
            if loss_mask is not None:
                loss = (ce * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            else:
                loss = ce.mean()
        return logits, loss
