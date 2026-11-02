# First Mamba3 cluster pretraining run

Historical k=5 experiment, stopped and superseded for the matched backbone
comparison by the [k=1 Chinchilla restart](../chinchilla_restart/README.md).
Use [Training recipes](../../train/TRAINING_RECIPES.md) for onboarding defaults.

This experiment uses the shared Mamba pretraining Trainer and current
`TPCBatchDataset` coordinate normalization and serialization. It selects Mamba3 **SISO**;
it does not use the historical M3 width-tier naming.

The full configuration is in `recipe.yaml`: width 512, depth 12, state size 128,
expansion 2, head dimension 64, chunk size 64, position-only NeRF embedding,
five next-point targets, dropout 0.1, and seed 42. The shared Trainer supplies
AdamW with betas (0.9, 0.95), weight decay 0.1, upstream Mamba3 no-decay flags,
value clipping at 0.1, and local BF16 execution inside Mamba3 blocks. The learning
rate warms from 2e-5 to 2e-4 over 2,000 steps, then follows cosine decay over a
100,000-step budget. This uses standard AdamW rather than the earlier M4 run's
mu-transfer/high-learning-rate configuration.

The input is the Cartesian cluster-center dataset
`mmap_canonical_loose_truthseg_event_v6_02`. The existing dataset converts points
to normalized eta/phi/r and orders them using calibrated12 band-Hilbert
serialization. This experiment then restores the historical M4 outward-band
target criterion: slot m predicts the nearest point at least m calibrated bands
farther out. All targets must occur strictly later in the serialized sequence.
Successive slots may select the same point; their minimum allowed band offsets
increase, rather than enforcing distinct target identities. Repeated-slot counts
are reported. This is an initial Mamba3 run, not a controlled architecture
comparison against historical loss numbers.

A target audit found that the current checkout's later-radius-neighbor targets,
constructed before Hilbert ordering, could already be visible in the input
prefix. On 256 sampled events, 1,719/40,965 target slots (4.20%) were visible, with
no ambiguous coordinate matches. The initial launch was stopped and retained
for diagnosis. The corrected pilot/full run requires zero visible target
coordinates in its audited events. `audit_targets.py` reproduces the coordinate
visibility check, and `test_targets.py` checks band offsets, causality, padding,
and permissible repeated-slot selections. The core dataset file is unchanged.

Training uses 10,000,000 events drawn without replacement from the eligible
`pretrain` split with seed 42. Validation uses a fixed 50,000-event subset from
the separate historical `test` split with seed 43. Two to 100 clusters per event
are accepted; the lower limit prevents target-free singleton events, and the
upper limit preserves the existing cluster pretraining filter. The inspected
data contains no events below the minimum. Above-100 events comprise about
0.18% of each split. Indices, checksums and population counts are saved in each
run's output. This validation set selects checkpoints and is not an independent
final evaluation holdout.

`run.py` keeps the existing coordinate transformations, model, optimizer and
training loop. Its dataset subclass supplies outward-band targets and replaces
the slow filtering scan with vectorized metadata and reproducible cohort selection. Its Trainer subclass
checks gradients before updates, saves progress and validation metadata, and
writes checkpoints through temporary files. The shared Trainer still requires
two legacy weighting files: unit arrays are written into the run's private
output directory, documented as inactive because `loss_reweight=false`.

The `preflight` phase uses the full model and batch size for 20 optimizer steps,
with 4,096 training and 512 validation events and a short warmup. It checks
target coverage, finite updates, validation, strict checkpoint reload,
prediction equivalence, and optimizer/scheduler restoration. The `train` phase
requires the pilot's passing artifact with matching recipe/runner checksums;
it starts fresh from seed 42 and uses the full training schedule. Pilot weights
are not used as initialization. Checkpoint continuation preserves the shared
Trainer's model/optimizer/scheduler/step state; sample order after interruption
is not guaranteed bit-exact.

MCP owns source staging and detached Docker execution. The full run has no
wall-clock deadline or default CPU/RAM/process cap. The dataset and pilot output
are mounted read-only; outputs and checkpoints are written under the full job's
`output` directory. The full run copies the pilot's Triton cache into its own
writable cache, leaving the pilot intact. Checkpoints and validation are saved
every 1,000 updates.

Useful output artifacts are `resolved_recipe.yaml`, `cohorts.json`, the two
event-index arrays, `started.json`, `progress.json`, `latest_validation.json`,
and the final `completed.json`. Checkpoints reside under
`training/mamba3_clusters_w512_d12/seed42/training_checkpoints/`.
