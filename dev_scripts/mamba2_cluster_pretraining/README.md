# Matched Mamba2 cluster pretraining on a518

Historical k=5 experiment, stopped and superseded for the matched backbone
comparison by the [k=1 Chinchilla restart](../chinchilla_restart/README.md).
Use [Training recipes](../../train/TRAINING_RECIPES.md) for onboarding defaults.

This run matches the a02c Mamba3 job
`mamba3-clusters-w512-d12-k5-causal-20261027` in dataset, cohort selection,
normalization, calibrated12 band-Hilbert serialization, causal outward-band
targets, width/depth, state size, batch sizes, seeds, optimizer hyperparameters,
validation frequency and 100,000-step learning-rate schedule. The saved Mamba3
source manifest was checked against the local shared pretraining/model/dataset
files before staging this experiment. The Mamba3 experiment is unchanged.

The backbone is Mamba2: width 512, 12 layers, state size 128, expansion 2,
head dimension 64, one group, chunk size 64, convolution width 4, and fused
CUDA execution required. Mamba3-specific rotation/SISO options are removed.
The existing repository's Mamba2 implementation is used; the approximate
manual fallback is refused. Both jobs use the same existing Docker image ID.

`run.py` is a separate copy of the audited Mamba3 runner with only the backbone
selection, experiment name, actual optimizer-policy metadata, a fused-path
assertion and full-cohort equality checks changed. `targets.py` is byte-identical
to the Mamba3 target implementation. During full training, selected event-index
checksums and population counts must equal `reference_cohorts.json` before
optimization starts. `reference_started.json` records the Mamba3 model/runtime.

Training draws the identical 10,000,000 events from `pretrain` with seed 42 and
50,000 events from `test` with seed 43, accepting 2–100 clusters per event.
The historical test subset selects checkpoints and is not a final holdout.
Batch size is 128, warmup is 2,000 steps, LR is 2e-5 → 2e-4 → 2e-5, and
validation/checkpoints occur every 1,000 steps. Training starts from scratch;
pilot weights are never used as initialization.

The shared Trainer retains each family's implementation policies: Mamba2 uses
its existing FP32/fused execution and a single AdamW parameter group; Mamba3
uses local BF16 blocks and honors its upstream no-decay flags. Parameter counts
therefore differ. This is a matched practical recipe, not an equal-parameter
or equal-precision ablation. Optimizer betas (0.9, 0.95), weight decay 0.1,
value clipping 0.1 and all schedule settings are unchanged.

Jobs managed through `clas12-edge`:

- `mamba2-clusters-data-a02c-20261027` and
  `mamba2-clusters-data-a518-20261027`: CPU-only full SHA-256 verification of all
  files in the six consumed RaggedMmap trees, including Cartesian features,
  segmentation/regression sidecars and their indexing metadata. The two
  `data_identity.json` manifests must match before the production launch.
- `mamba2-clusters-causal-preflight-20261027` on a518: full model/batch,
  4,096/512 pilot cohorts, 20 updates, finite gradients/validation, causal target
  audit and strict model/optimizer/scheduler checkpoint round trip.
- `mamba2-clusters-w512-d12-k5-causal-20261027` on a518: full training, requiring
  the successful pilot artifact with matching runner/recipe/target hashes.

Each job's records are below
`/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/JOB-ID/`.
Full training has no runtime deadline or integration-imposed resource caps.
Source, dataset and pilot output mounts are read-only. Only private job output
directories receive checkpoints, caches and statistics.

Inspect `progress.json`, `latest_validation.json`, `cohorts.json`, `started.json`
and `completed.json` through MCP. Checkpoints are below
`training/mamba2_clusters_w512_d12/seed42/training_checkpoints/` in the job output.
