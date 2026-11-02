"""Historical calibrated outward-band target criterion, after serialization."""

import torch

from fm4npp.hilbert import assign_clas12_layer


def outward_band_targets(points, k, band_config="calibrated12"):
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Expected a nonempty (N,3) normalized eta/phi/r sequence")
    bands = assign_clas12_layer(points[:, 2], band_config=band_config)
    if not torch.all(bands[1:] >= bands[:-1]):
        raise ValueError("Input must already be serialized in nondecreasing band order")
    distances = torch.norm(points[:, None, :] - points[None, :, :], dim=-1)
    indices = torch.full((len(points), k), -1, dtype=torch.long, device=points.device)
    targets = torch.full((len(points), k, 3), -100., dtype=points.dtype, device=points.device)
    for slot in range(k):
        eligible = bands[None, :] >= bands[:, None] + slot + 1
        values, nearest = distances.masked_fill(~eligible, float("inf")).min(-1)
        valid = torch.isfinite(values)
        indices[valid, slot] = nearest[valid]
        targets[valid, slot] = points[nearest[valid]]
    visible_positions = torch.arange(len(points), device=points.device)[:, None]
    if ((indices >= 0) & (indices <= visible_positions)).any():
        raise AssertionError("Target is already visible in the causal input")
    return targets.reshape(len(points), 3 * k), indices
