"""How an adapter combines the frozen backbone's layer outputs (`feature_mix`).

The backbone returns its residual stream x_l after each layer, shape (L, B, N, D).
Every adapter mixes them as  sum_l softmax(w)_l * f_l(x_l)  before its own input
LayerNorm and projection. The config key `feature_mix` (with `feature_stat_batches`)
selects f_l; fm4npp.models.mamba_config.adapter_options passes it to the heads.

  raw          f_l(x) = x (historical default). A layer whose residual stream is
               large dominates the mix whatever its weight.
  standardize  f_l(x) = (x - mean_l) / std_l per channel, with mean and std of each
               layer estimated over the real (unpadded) clusters of the first
               `feature_stat_batches` training batches and then frozen. Removes
               constant offsets and puts every layer on the same scale, so the
               learned weights alone decide each layer's share. The statistics are
               buffers of the head and are saved with its checkpoint.
"""
import torch
from torch import nn

from fm4npp.models.mamba_config import FEATURE_MIXES

DEFAULT_STAT_BATCHES = 200


class LayerStandardizer(nn.Module):
    """Per-layer, per-channel standardization of (L, B, N, D) features.

    In training mode the first `stat_batches` calls accumulate the mean and variance
    over valid tokens (float64), and each call uses the estimate so far; afterwards,
    and always in eval mode, the stored statistics are used unchanged. A channel's
    variance is floored at `rel_floor` x the layer's mean channel variance so that a
    near-constant channel is centred but not blown up to unit size.
    """

    def __init__(self, num_layers, dim, stat_batches=DEFAULT_STAT_BATCHES, rel_floor=1e-3, eps=1e-12):
        super().__init__()
        self.stat_batches, self.rel_floor, self.eps = int(stat_batches), float(rel_floor), float(eps)
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("batches", torch.zeros((), dtype=torch.long))
        self.register_buffer("sum", torch.zeros(num_layers, dim, dtype=torch.float64))
        self.register_buffer("sumsq", torch.zeros(num_layers, dim, dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(num_layers, dim))
        self.register_buffer("std", torch.ones(num_layers, dim))

    @torch.no_grad()
    def _update(self, feature, valid):
        x = feature.detach().double()                                # (L, B, N, D)
        if valid is not None:
            x = x[:, valid]                                          # (L, T, D)
        else:
            x = x.flatten(1, 2)
        if x.shape[1] == 0:
            return
        self.count += x.shape[1]
        self.sum += x.sum(1)
        self.sumsq += x.square().sum(1)
        self.batches += 1
        mean = self.sum / self.count
        var = (self.sumsq / self.count - mean.square()).clamp_min(0)
        var = torch.maximum(var, self.rel_floor * var.mean(1, keepdim=True)).clamp_min(self.eps)
        self.mean.copy_(mean.float())
        self.std.copy_(var.sqrt().float())

    def forward(self, feature, valid=None):
        if self.training and int(self.batches) < self.stat_batches:
            self._update(feature, valid)
        if int(self.batches) == 0:
            raise RuntimeError("feature_mix='standardize' has no statistics: train first or load a checkpoint")
        # float32: a residual stream of ~1e4 has steps of ~64 in bfloat16, so centring
        # in half precision would erase the variation being kept
        out = (feature.float() - self.mean[:, None, None]) / self.std[:, None, None]
        return out.to(feature.dtype)


def make_layer_mixer(num_layers, dim, feature_mix="raw", feature_stat_batches=DEFAULT_STAT_BATCHES):
    if feature_mix not in FEATURE_MIXES:
        raise ValueError(f"Unknown feature_mix {feature_mix!r}; choose one of {FEATURE_MIXES}")
    return None if feature_mix == "raw" else LayerStandardizer(num_layers, dim, feature_stat_batches)


def mix_layers(feature, weights, standardizer=None, valid=None):
    """feature (L, B, N, D) -> (B, N, D): sum_l softmax(weights)_l f_l(x_l)."""
    if standardizer is not None:
        feature = standardizer(feature, valid)
    x = feature.permute(1, 2, 0, 3)
    w = torch.softmax(weights, dim=0).to(x.dtype)
    return torch.einsum("bsnd,n->bsd", x, w)


def valid_tokens(x, padding_mask):
    """(B, N) bool of real clusters: the head's padding_mask (True = real) or the -100 fill."""
    if padding_mask is not None:
        return padding_mask.bool()
    return x[..., 0] != -100
