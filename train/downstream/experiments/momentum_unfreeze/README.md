# Momentum adapter: unfreeze the last two M6 blocks

This branch tests whether supervised momentum training benefits from updating
blocks 11 and 12 (zero-based indices 10 and 11) of campaign-4 M6. The embedding,
first ten blocks, final pretraining norm and output head remain frozen. All 12
accumulated states still enter the usual learned layer mixture; the head is
unchanged. Backbone dropout stays disabled in both pretrained arms.

Three matched arms: AdapterOnly, frozen M6 + adapter, and last-two M6 + adapter.
All use Cartesian `(px,py,pz)` MAE, attention pooling, the conventional 256-wide
one-Mamba2-block momentum head, seed 42, 50k training tracks, batch 128, 30k
optimizer steps, 50k validation/evaluation tracks and best-validation-loss
selection. Settings follow the vanilla v6_02 momentum recipe, with M6 width
1536, 12 layers, state size 16 and checkpoint-compatible klen 1. Adapter LR is
1e-5 to 3e-4 with the existing 3000-step cosine restarts / 300-step warmup;
the tuned backbone follows exactly the same schedule at 0.1 times the LR.
AdamW decay is 1e-4. Gradients are clipped separately per optimizer group at
0.1 to preserve the adapter clipping rule. There is no unfreeze delay.

This is a minimal single-seed screen. Validation and evaluation share the source
test split; evaluation is not an independent holdout. A promising result needs
paired additional seeds and an independent holdout before a general claim.
Compare validation loss, component MAE, momentum/angle resolution and tail
fractions together. Learning-rate tuning is outside this first screen.

## Implementation

YAML `backbone_unfreeze_last_n: 0` preserves the frozen default. Setting it to
`2` requires pretrained mode and a Mamba backbone. `backbone_lr_scale: 0.1`
sets the extra optimizer group's ratio. Single-process training only is
supported. Autograd starts at the first trainable block because the prefix has
no trainable parameters or grad-requiring inputs. No backbone API changes are
needed. The scheduler wrapper preserves LR ratios through restarts and resume.

Tuned adapter checkpoints include the entire updated backbone and the unfreeze
count. Training resume and the standard evaluator strictly restore that state;
configuration/weight-contract mismatches fail. Legacy frozen adapter checkpoints
remain loadable. Current launch still requires the original pretrained path for
architecture initialization before the tuned state is restored.

## Run

Use a clean committed checkout and the existing CUDA/Mamba Docker image.

```bash
bash train/downstream/experiments/momentum_unfreeze/launch.sh   /home/alessio/ML-work/result_deep_storage/campaigns/m6_last_two_50k_20261026   /home/alessio/ML-work/HIPO_processing/multi_particle_storage/big-data/mmap_canonical_loose_truthseg_event_v6_02   /home/alessio/ML-work/pretrained-FMs/campaign_4/scale_w1536_d12_n39553933/ckpt_best.tar
```

The driver records source/checkpoint hashes, strict checkpoint loading,
regenerated training target statistics, ordered train/validation cohort hashes,
and real-data CUDA gradients/updates restricted to blocks 11-12. It then runs
10-step train/evaluate pilots for all arms before launching production. Pilot
checks include LR ratios, saved tuned weights, unchanged prefix and matching
evaluation identities/truth. Full completion requires all final artifacts and
30k steps per arm; `completed.json` is written only after these checks.

Inspect `launch.log`, `pilot_pass.json`, `preflight_pass.json`, `status.yaml`,
`runs/*/train/artifacts.json`, `runs/*/evaluation/summary.json` and
`launcher_result.txt`. The existing campaign runner handles per-arm logs,
status and summary collation. Rerunning the driver skips completed arms.
