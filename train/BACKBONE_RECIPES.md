# Backbone pretraining recipes

This reference specifies the Chinchilla sweep and the width-512 Mamba1/2/3
recipe. For downstream use, follow [Training workflow](TRAINING_RECIPES.md).
For family-specific block controls, see [Mamba support](MAMBA_SUPPORT.md).

## Chinchilla width sweep

Mike's sweep uses Mamba1 blocks, depth 12, state size 16, convolution width 4,
expansion 2, `klen: 1` and position-only NeRF embedding. M1–M6 below denote
width tiers; `backbone_version: mamba1|mamba2|mamba3` denotes a block family.

| Tier | `embed_dim` | `num_layers_backbone` | `total_steps` | `warmup_steps` |
| --- | ---: | ---: | ---: | ---: |
| M1 | 64 | 12 | 100,000 | 10,000 |
| M2 | 128 | 12 | 100,000 | 10,000 |
| M3 | 256 | 12 | 100,000 | 10,000 |
| M4 | 512 | 12 | 200,000 | 20,000 |
| M5 | 1,024 | 12 | 800,000 | 80,000 |
| M6 | 1,536 | 12 | 1,600,000 | 160,000 |

| Shared control | Value and meaning |
| --- | --- |
| Effective batch | 64 events on one GPU (`local_batch_size`); nominal `batch_size` is 128 in the sweep YAML. Count exposure using the effective batch. |
| Optimizer | AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`. |
| `min_lr`, `max_lr` | Approximately `9.888e-4`, `9.888e-3`; linear warmup then cosine decay over the update budget. |
| `use_mup`, `mup_base_width` | Enabled, base width 256. Rank > 1 parameter tensors use scheduled LR / `(embed_dim / 256)`; vector parameters use scheduled LR. |
| `dropout` | Approximately 0.05019. |
| `weight_decay` | Approximately 0.04698. |
| `grad_clip_value` | Approximately 1.08699, **per-element value clipping**. |
| `n_eval_steps` | Validation/checkpoint interval: 1,000 updates. |
| `local_valid_batch_size`, `max_val_batches` | 2 events/batch, at most 1,000 batches: 2,000 validation events. |
| Objective / selection | MSE on valid next-point coordinates; select lowest validation MSE. Padding/missing targets are masked. |

The rank-based LR adjustment runs after the scheduler step; the first update
uses the initial LR before that adjustment. Preserving these parameter groups
and timing is part of this recipe. A peak LR alone does not specify it.

Full-precision values for all tiers are in
[the sweep YAML](../dev_scripts/chinchilla_restart/reference/chinchilla_original.yaml).
Set data/stat/output paths for the execution environment. The generic
`scripts/configs/mamba_pretrain.yaml` describes a different recipe; changing
only its width does not select the sweep's optimization policy.

## Width-512 Mamba1/2/3 recipe

The [three-family YAML](../dev_scripts/backbone_pretraining_comparison/recipe.yaml)
sets width 512/depth 12/state 16/k=1, 200k updates and actual batch 64. Input is
the CLAS12 v6_02 cluster-center event product. It selects the first 10 million
eligible training events and first 2,000 eligible validation events, accepting
1–50 clusters/event, with shuffle seed 42.

The loader converts xyz to normalized eta/phi/r and uses calibrated12
band-Hilbert ordering. The target is the nearest eligible outward-band point
after serialization; target coordinates must be outside the visible causal
prefix. `klen: 1` means one coordinate triple per token, not sequence length.
Use the supplied target constructor and its visibility checks with this recipe.

| Block family | Additional settings | Execution |
| --- | --- | --- |
| Mamba1 | State 16, conv 4, expand 2 | FP32 blocks |
| Mamba2 | State 16, conv 4, expand 2, head dimension 64, one group, chunk size 64; fused path | FP32 blocks |
| Mamba3 | State 16, expand 2, head dimension 64, one group, chunk size 64, SISO, rotary fraction 0.5, no output-projection norm | BF16 inside blocks; FP32 parameters, outputs and loss |

Native SSM initialization and `_no_weight_decay` flags are retained in the
three-family runner. Mamba3 has no convolution. Equal width/depth does not imply
equal parameter count or precision.

## Launcher contract

The supplied launchers are single-GPU, container-oriented modules:

```bash
# Mamba2 or Mamba3
python -m dev_scripts.chinchilla_restart.run --backbone mamba3 --phase preflight
python -m dev_scripts.chinchilla_restart.run --backbone mamba3 --phase train

# Mamba1
python -m dev_scripts.backbone_pretraining_comparison.train --backbone mamba1 --phase preflight
python -m dev_scripts.backbone_pretraining_comparison.train --backbone mamba1 --phase train
```

Run preflight and training with separate output directories. Mount the writable
output at `/results`, the successful preflight output read-only at `/pilot` for
training, and the data at the absolute path set in the recipe. The launchers do
not accept general path overrides; edit the recipe and mount configuration
together. Production verifies preflight recipe/runner/target hashes, starts
fresh weights and requires CUDA. A changed recipe requires a new preflight.

The directory's `*-run-spec.json` files show commands, image identity, mounts
and expected outputs for [Edge MCP](../integrations/edge_mcp/README.md). Use new
job/output identities for each run.

Preflight checks finite updates, causal target visibility and checkpoint/model,
optimizer and scheduler round trips. It is an execution check. Training writes
resolved recipe, cohort indices/checksums, progress, validation MSE, latest and
best checkpoints and a completion receipt. Plot raw validation MSE against
updates to inspect learning; the comparison tooling also supports token/compute
axes. Keep implementation precision and the estimation method explicit when
using those axes.

Implementation references: [LR groups](../dev_scripts/chinchilla_restart/policy.py),
[training loop](../dev_scripts/chinchilla_restart/trainer.py),
[Mamba2/3 launcher](../dev_scripts/chinchilla_restart/run.py), and
[comparison/plot tooling](../dev_scripts/backbone_pretraining_comparison/README.md).
