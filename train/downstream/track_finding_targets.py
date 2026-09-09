"""Target construction for event-level track-finding instance segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


SIGNAL_ONLY = "signal_only"
UNIFIED_NOISE_INSTANCE = "unified_noise_instance"
VALID_TARGET_MODES = frozenset({SIGNAL_ONLY, UNIFIED_NOISE_INSTANCE})


def validation_ari_metric(mode: str) -> str:
    """Return the event-metric key used for checkpoint selection."""
    metrics = {
        "signal": "ari_signal",
        "inclusive": "ari_with_background",
    }
    try:
        return metrics[mode]
    except KeyError as exc:
        raise ValueError("validation_ari_mode must be 'signal' or 'inclusive'") from exc


def build_track_instance_targets(
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    mode: str,
    background_label: int = -1,
    padding_label: int = -100,
) -> tuple[list[dict[str, torch.Tensor]], list[torch.Tensor]]:
    """Build per-event Hungarian targets without ever promoting padding.

    ``unified_noise_instance`` follows the published FM4NPP target intent: the
    raw background label is one ordinary instance mask with object class 1.
    ``signal_only`` preserves the earlier CLAS12 compatibility behavior.
    """
    if mode not in VALID_TARGET_MODES:
        raise ValueError(
            f"Unknown track_target_mode {mode!r}; expected one of {sorted(VALID_TARGET_MODES)}"
        )
    if labels.ndim != 2 or valid_mask.shape != labels.shape:
        raise ValueError("labels and valid_mask must be matching (batch, points) tensors")

    targets: list[dict[str, torch.Tensor]] = []
    inverse_valid_list: list[torch.Tensor] = []
    for batch_idx in range(labels.size(0)):
        sample_labels = labels[batch_idx]
        # Point geometry defines validity; the label guard makes the padding
        # boundary explicit even if a malformed batch marks a padded row valid.
        selected = valid_mask[batch_idx].bool() & (sample_labels != int(padding_label))
        if mode == SIGNAL_ONLY:
            selected &= sample_labels != int(background_label)

        selected_labels = sample_labels[selected]
        if selected_labels.numel() == 0:
            targets.append({
                "masks": torch.zeros(
                    0,
                    sample_labels.numel(),
                    device=labels.device,
                    dtype=torch.float32,
                ),
                "labels": torch.zeros(0, dtype=torch.long, device=labels.device),
            })
            inverse_valid_list.append(torch.zeros(0, dtype=torch.long, device=labels.device))
            continue

        _unique_labels, inverse_valid = torch.unique(
            selected_labels,
            sorted=True,
            return_inverse=True,
        )
        n_instances = int(_unique_labels.numel())
        one_hot = F.one_hot(inverse_valid, num_classes=n_instances).float()
        masks = torch.zeros(
            n_instances,
            sample_labels.numel(),
            device=labels.device,
            dtype=one_hot.dtype,
        )
        masks[:, selected] = one_hot.transpose(0, 1)
        targets.append({
            "masks": masks,
            "labels": torch.ones(n_instances, dtype=torch.long, device=labels.device),
        })
        inverse_valid_list.append(inverse_valid)

    return targets, inverse_valid_list
