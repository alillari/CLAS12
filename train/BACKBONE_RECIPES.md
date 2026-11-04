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
[mamba_chinchilla_sweep.yaml](../scripts/configs/mamba_chinchilla_sweep.yaml).
The maintained [v6_02 sweep](../scripts/configs/mamba_v602_sweep.yaml) uses the
same principal optimizer settings but a flat 100k-update budget for all widths;
the Chinchilla table above uses width-dependent budgets. Select the intended
YAML explicitly. Auxiliary length/pitch inputs have their own maintained
[v7 length](../scripts/configs/mamba_v701_aux_sweep.yaml) and
[v7 pitch](../scripts/configs/mamba_v701_pitch_sweep.yaml) sweeps.

## Inputs, targets and block-family choices

The standard center-only recipe converts xyz to normalized eta/phi/r, accepts
1–50 clusters/event and uses calibrated12 band-Hilbert ordering. Each token
predicts the nearest eligible point in an outward band. `klen: 1` means one
coordinate triple per token, not sequence length. Use the maintained pretraining
loader so serialization, target construction and calibration agree.

| Choice | Required settings / consequence |
| --- | --- |
| Mamba1 | `backbone_version: mamba1`, state 16, conv 4, expand 2; standard sweep family. |
| Mamba2 | `backbone_version: mamba2`; match width/state/conv/expand and choose valid `headdim`, `ngroups`, `chunk_size`. Use the fused CUDA path for training. |
| Mamba3 | `backbone_version: mamba3`, SISO; no convolution. Specify state/expand/head/group/chunk/rotary options. Blocks execute BF16 internally with FP32 parameters/output. |
| Transformer | `backbone_version: transformer`; specify head count and independent NoPE/RoPE settings. See [Transformer support](TRANSFORMER_SUPPORT.md). |
| Center + geometry | `embed_method: pos_plus_aux`, `pos_dim: 3`, `aux_dim: 4`, with aligned geometry sidecars and the length or pitch recipe. Match downstream input representation and auxiliary feature choice. |

For a width-512/depth-12/state-16/k1 run, start with
`scale_w512_d12_chinchilla` and retain actual batch 64, 200k updates, 20k warmup
and the shared optimizer settings above. Other block families can use this as a
starting recipe; equal width/depth does not imply equal parameter count or
precision. The independent downstream head family is not set by pretraining.

## Run pretraining

From the repository root in a compatible CUDA environment:

```bash
export CLAS12_PRETRAIN_DATA=/absolute/path/to/cluster_event_product
export CLAS12_ARTIFACT_ROOT=/absolute/path/to/result_deep_storage

python -m train.pretrain \
  --yaml_config scripts/configs/mamba_chinchilla_sweep.yaml \
  --config scale_w512_d12_chinchilla --run_num seed42 \
  --root_dir "$CLAS12_ARTIFACT_ROOT/pretrain"
```

`run_num` names the run; it is not a seed option. The imported dataset module
initializes Torch's seed to 42. For a controlled seed change, inspect that setup
rather than renaming the run. Render a copy of the YAML when changing model or
optimizer options, as shown in the [pretraining workflow](pretrain/README.md).

Set `CLAS12_PRETRAIN_STATS` (or the recipe's `stat_dir`) only for
multiplicity-weighted pretraining. With `loss_reweight: true` the trainer
loads the bin edges from `loss_bin_pp.pkl` and the corresponding event
weights from `loss_weight_pp.pkl`. The unweighted profiles need neither file.
See [pretraining setup](pretrain/README.md) for the HDF5 and mmap profiles.

The trainer writes under `<root_dir>/<config>/<run_num>/training_checkpoints/`:
`ckpt.tar` is the latest state and `ckpt_best.tar` is validation-selected.
Retain the model configuration and environment/data provenance. Downstream
frozen-feature fits require matching family, width, depth, state, `klen`,
embedding and auxiliary contract. Resume restores model/optimizer/scheduler;
the epoch-based loader does not restore an exact mid-epoch data cursor.

Plot the saved CSV without inference or retraining:

```bash
python train/pretrain/plot_pretraining_history.py \
  --log /path/to/pretrain/run/config_PROFILE_run_RUN.csv \
  --output-dir /path/to/plots
```

This writes `pretraining_history.png/.pdf` with raw loss and logged LR versus
updates and marks the best validation observation. The log's training losses
are sampled batches, not full-epoch averages; the LR is the logged schedule
value, not every rank-scaled parameter group's rate. See
[the complete worked example](WORKED_EXAMPLE.md) for exact run/log paths.
Track the
actual effective batch and validation sample count when comparing runs. A
short pilot checks execution; it does not establish convergence.
