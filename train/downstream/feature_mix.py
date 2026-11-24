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
  last         only the final layer, through the backbone's own final norm: exactly
               the representation the pretraining head read (the usual "last hidden
               state"). The layer weights are kept but unused. The norm's parameters
               are copied from the frozen backbone when the head is built
               (attach_backbone_norm) and saved with the head.
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


class LastLayerNorm(nn.Module):
    """The backbone's final norm applied to its last layer. The norm (RMSNorm for Mamba
    backbones, LayerNorm for transformers) is held as frozen buffers copied by
    attach_backbone_norm; `kind` 0 means not attached yet."""
    RMS, LAYER = 1, 2

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("kind", torch.zeros((), dtype=torch.long))
        self.register_buffer("eps", torch.zeros((), dtype=torch.float64))
        self.register_buffer("weight", torch.ones(dim))
        self.register_buffer("bias", torch.zeros(dim))

    @torch.no_grad()
    def attach(self, norm):
        self.weight.copy_(norm.weight.detach().float())
        if isinstance(norm, nn.LayerNorm):
            self.kind.fill_(self.LAYER)
            self.eps.fill_(norm.eps)
            if norm.bias is not None:
                self.bias.copy_(norm.bias.detach().float())
        elif hasattr(norm, "eps") and not hasattr(norm, "bias"):
            self.kind.fill_(self.RMS)
            self.eps.fill_(norm.eps)
        else:
            raise TypeError(f"Unsupported backbone final norm {type(norm).__name__}")

    def forward(self, feature, valid=None):
        kind = int(self.kind)
        if kind == 0:
            raise RuntimeError("feature_mix='last' needs the backbone's final norm: call attach_backbone_norm "
                               "after building the head, or load a checkpoint")
        x, eps = feature[-1].float(), float(self.eps)
        if kind == self.RMS:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * self.weight
        else:
            x = nn.functional.layer_norm(x, x.shape[-1:], self.weight, self.bias, eps)
        return x.to(feature.dtype)


def make_layer_mixer(num_layers, dim, feature_mix="raw", feature_stat_batches=DEFAULT_STAT_BATCHES):
    if feature_mix not in FEATURE_MIXES:
        raise ValueError(f"Unknown feature_mix {feature_mix!r}; choose one of {FEATURE_MIXES}")
    if feature_mix == "raw":
        return None
    if feature_mix == "last":
        return LastLayerNorm(dim)
    return LayerStandardizer(num_layers, dim, feature_stat_batches)


def feature_mix_name(mixer):
    return "raw" if mixer is None else "last" if isinstance(mixer, LastLayerNorm) else "standardize"


def attach_backbone_norm(head, backbone):
    """For feature_mix='last': copy the frozen backbone's final norm into the head.
    No effect for the other mixes. Call once both are built and the backbone is loaded."""
    head = getattr(head, "module", head)
    mixer = getattr(head, "feature_mixer", None)
    if isinstance(mixer, LastLayerNorm) and backbone is not None:
        mixer.attach(getattr(backbone, "module", backbone).norm)
    return head


def mix_layers(feature, weights, standardizer=None, valid=None):
    """feature (L, B, N, D) -> (B, N, D): sum_l softmax(weights)_l f_l(x_l), or the
    normed last layer for feature_mix='last'."""
    if isinstance(standardizer, LastLayerNorm):
        return standardizer(feature, valid)
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
