"""Check whether target coordinates are already present in the causal input."""

import json
from pathlib import Path

import torch

from fm4npp.utils import YParams
from dev_scripts.mamba3_cluster_pretraining.run import CohortDataset, RECIPE, CONFIG, OUT

torch.set_num_threads(2)
params = YParams(str(RECIPE), CONFIG)
params.stat_dir = str(OUT)
dataset = CohortDataset(params, "pretrain", 256, params.seed)
counts = {"events": 0, "valid_target_slots": 0, "target_slots_already_in_input_prefix": 0,
          "events_with_visible_targets": 0, "coordinate_matches_not_unique": 0}
for i in range(len(dataset)):
    points, _, targets = dataset[i]
    values = targets.reshape(len(points), params.klen, 3)
    valid = (values != -100).all(-1)
    matches = (values[:, :, None, :] - points[None, None, :, :]).abs().amax(-1) < 1e-7
    prefix = torch.arange(len(points))[None, :] <= torch.arange(len(points))[:, None]
    visible = (matches & prefix[:, None, :]).any(-1) & valid
    counts["events"] += 1
    counts["valid_target_slots"] += int(valid.sum())
    counts["target_slots_already_in_input_prefix"] += int(visible.sum())
    counts["events_with_visible_targets"] += int(visible.any())
    counts["coordinate_matches_not_unique"] += int(((matches.sum(-1) > 1) & valid).sum())
counts["visible_fraction"] = counts["target_slots_already_in_input_prefix"] / counts["valid_target_slots"]
old = Path("/historical/source/fm4npp/datasets/dataset_pretrain.py")
if old.exists():
    lines = old.read_text().splitlines()
    counts["historical_target_lines"] = [f"{i+1}: {line}" for i, line in enumerate(lines)
                                         if any(key in line.lower() for key in ["def knn", "min_band", "band_offset", "band_delta", "future", "layer_idx", "knearest_points ="])][:60]
    counts["historical_source_excerpts"] = {"target_function": "\n".join(lines[25:150]),
                                           "serialized_target_override": "\n".join(lines[420:485])}
(OUT / "target_causality.json").write_text(json.dumps(counts, indent=2) + "\n")
print(json.dumps(counts), flush=True)
