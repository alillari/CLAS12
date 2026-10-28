"""Causal pretraining and bidirectional adapter transformers.

Sequence encoding is independent of detector-coordinate embedding. NoPE (none)
adds no sequence features; RoPE rotates queries/keys using valid-token indices.
Padding masks follow the dataset convention: True denotes a real token.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from fm4npp.models.embed import (
    EmbedderAdd, EmbedderConcat, EmbedderPosOnly, EmbedderPosPlusAux,
)


def valid_mask(x, padding_mask=None):
    if padding_mask is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if padding_mask.dtype != torch.bool or padding_mask.shape != x.shape[:2]:
        raise ValueError("padding_mask must be boolean (batch, sequence), True for valid tokens")
    return padding_mask.to(x.device)


class SequenceAttention(nn.Module):
    def __init__(self, d_model, num_heads, *, causal, sequence_encoding="rope",
                 rope_theta=10000., dropout=0.):
        super().__init__()
        if num_heads <= 0 or d_model <= 0 or d_model % num_heads:
            raise ValueError("Transformer width must be positive and divisible by num_heads")
        if sequence_encoding not in {"none", "rope"}:
            raise ValueError("sequence_encoding must be 'none' (NoPE) or 'rope'")
        self.head_dim = d_model // num_heads
        if sequence_encoding == "rope" and self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.num_heads = num_heads
        self.causal = causal
        self.sequence_encoding = sequence_encoding
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _rotate(self, x, positions):
        # Compute phases in FP32 even under autocast; no fixed-length cache.
        inv_freq = self.rope_theta ** (-torch.arange(
            0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim)
        phase = positions.float()[:, None, :, None] * inv_freq
        cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x, padding_mask=None):
        mask = valid_mask(x, padding_mask)
        x = x.masked_fill(~mask[..., None], 0.)
        b, n, d = x.shape
        if n == 0:
            raise ValueError("Transformer sequences must contain at least one slot")
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.sequence_encoding == "rope":
            positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
            q, k = self._rotate(q, positions), self._rotate(k, positions)
        # SDPA boolean masks use True=allowed. Include the diagonal for invalid
        # queries to avoid all-masked softmax rows; their outputs are discarded.
        allowed = mask[:, None, None, :].expand(b, 1, n, n)
        diagonal = torch.eye(n, dtype=torch.bool, device=x.device)[None, None]
        if self.causal:
            allowed = allowed & torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
        allowed = allowed | (~mask[:, None, :, None] & diagonal)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out).masked_fill(~mask[..., None], 0.)


class TransformerBlock(nn.Module):
    """Pre-norm attention + FFN, including both residual additions."""

    def __init__(self, d_model, num_heads=4, *, causal=False,
                 sequence_encoding="rope", rope_theta=10000., mlp_ratio=4., dropout=0.):
        super().__init__()
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must produce a positive FFN width")
        hidden = int(d_model * mlp_ratio)
        if hidden < 1:
            raise ValueError("mlp_ratio must produce a positive FFN width")
        self.norm1 = nn.LayerNorm(d_model)
        self.sequence_attn = SequenceAttention(d_model, num_heads, causal=causal,
                                     sequence_encoding=sequence_encoding,
                                     rope_theta=rope_theta, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, d_model))
        self.dropout = nn.Dropout(dropout)
        self._transformer_config = dict(version="transformer", d_model=d_model,
                                        num_heads=num_heads, causal=causal,
                                        sequence_encoding=sequence_encoding,
                                        rope_theta=rope_theta, mlp_ratio=mlp_ratio,
                                        dropout=dropout)
        for parameter in self.parameters():
            parameter._preserve_transformer_init = True

    def forward(self, x, padding_mask=None):
        mask = valid_mask(x, padding_mask)
        x = x.masked_fill(~mask[..., None], 0.)
        x = x + self.dropout(self.sequence_attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x.masked_fill(~mask[..., None], 0.)


class TransformerGPT(nn.Module):
    """Full-sequence causal predictor with the existing backbone feature API."""

    def __init__(self, embed_dim=256, num_layers=12, num_heads=4, klen=1,
                 dropout=0.1, embed_method="pos_only", pe_method="nerf",
                 pos_dim=3, aux_dim=4, sequence_encoding="rope", rope_theta=10000.,
                 mlp_ratio=4.):
        super().__init__()
        if num_layers < 1:
            raise ValueError("Transformer backbones need at least one layer")
        self.backbone_version = "transformer"
        self.embed_dim = embed_dim
        self._transformer_input_config = dict(embed_method=embed_method, pe_method=pe_method,
                                              pos_dim=pos_dim, aux_dim=aux_dim, klen=klen)
        embedders = dict(add=EmbedderAdd, concat=EmbedderConcat,
                         pos_only=EmbedderPosOnly, pos_plus_aux=EmbedderPosPlusAux)
        if embed_method not in embedders:
            raise ValueError(f"Unknown embed_method: {embed_method}")
        options = dict(pos_dim=pos_dim, aux_dim=aux_dim) if embed_method == "pos_plus_aux" else {}
        self.embedder = embedders[embed_method](
            pe_method=pe_method, embed_dim=embed_dim, learnable_projection=False, **options)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, causal=True,
                             sequence_encoding=sequence_encoding, rope_theta=rope_theta,
                             mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, klen * 3)

    def forward(self, x, return_z=False, padding_mask=None):
        inferred = x[..., 0] != -100
        mask = inferred if padding_mask is None else valid_mask(x, padding_mask) & inferred
        x = x.masked_fill(~mask[..., None], 0.)
        x, pos = self.embedder(x)
        features = []
        for block in self.blocks:
            x = block(x, mask)
            features.append(x)
        if return_z:
            return None, features, pos.masked_fill(~mask[..., None], 0.)
        return self.output_layer(self.norm(x)).masked_fill(~mask[..., None], 0.)
