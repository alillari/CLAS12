"""Frozen full-event backbone features gathered into ordinary track batches."""

import torch


EVENT_CONTEXT_MODE = "event_segment_context"


def validate_event_context_config(params):
    if getattr(params, "adapter_sample_mode", "event_segment") != EVENT_CONTEXT_MODE:
        return
    if not getattr(params, "pretrained_ckpt", None):
        raise ValueError("event_segment_context requires --usepretrain and --pretrained_ckpt")
    if getattr(params, "chunk_training", False):
        raise ValueError("event_segment_context requires chunk_training=False")
    if getattr(params, "input_representation", "center_only") != "center_only":
        raise ValueError("event_segment_context currently supports center_only inputs")
    if getattr(params, "embed_method", "pos_only") != "pos_only":
        raise ValueError("event_segment_context currently requires embed_method=pos_only")
    if getattr(params, "mambaversion", "mamba1") != "mamba1":
        raise ValueError("event_segment_context is initially supported for Mamba1 backbones")


def validate_checkpoint_sample_mode(checkpoint, sample_mode):
    # Existing adapter checkpoints predate the event-context mode.
    saved_mode = checkpoint.get("adapter_sample_mode", "event_segment")
    if (saved_mode == EVENT_CONTEXT_MODE) != (sample_mode == EVENT_CONTEXT_MODE):
        raise ValueError(
            "Adapter checkpoint backbone context differs from configuration: "
            f"checkpoint={saved_mode!r}, config={sample_mode!r}. "
            "Train a fresh adapter for the new context."
        )


def backbone_features(model, track_points, batch, sample_mode="event_segment"):
    """Return (layers, tracks, track_tokens, width) for the existing adapter.

    The caller controls no_grad. All real event tokens, including noise and
    unlabelled tracks, remain in backbone context. Track selection follows
    inference and precedes the adapter's own sequence-mixing layer.
    """
    has_context = "backbone_points" in batch
    if has_context != (sample_mode == EVENT_CONTEXT_MODE):
        raise ValueError("Backbone context tensors do not match adapter_sample_mode")
    if not has_context:
        _, layers, _ = model(track_points, return_z=True)
        return torch.stack(layers)
    if model.training:
        raise ValueError("The frozen event-context backbone must be in eval mode")
    event_points = batch["backbone_points"].to(track_points.device)
    event_index = batch["backbone_event_index"].to(track_points.device)
    token_index = batch["backbone_token_index"].to(track_points.device)
    if token_index.shape != track_points.shape[:2] or event_index.shape != track_points.shape[:1]:
        raise ValueError("Event-to-track gather indices have incompatible shapes")
    valid = track_points[..., 0] != -100
    # A stale or corrupted gather must fail before a misleading training run.
    gathered_points = event_points[event_index[:, None], token_index]
    # Pointwise coordinate conversion can differ by FP32 rounding when the
    # event and track have different lengths. Identity comes from row indices.
    if not torch.allclose(gathered_points[valid], track_points[valid], rtol=1e-5, atol=1e-6):
        raise ValueError("Event-to-track gather does not reproduce the input track tokens")
    _, event_layers, _ = model(event_points, return_z=True)
    # Gather each layer before stacking: avoid a second full-event feature copy.
    return torch.stack([
        layer[event_index[:, None], token_index].masked_fill(~valid[..., None], 0)
        for layer in event_layers
    ])
