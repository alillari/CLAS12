# Matched downstream comparison, 2026-10-29

This is the controlled **zero-layer** momentum comparison. The ordinary momentum
trainer uses one Mamba2 refinement layer; see [Training recipes](../../train/TRAINING_RECIPES.md)
for onboarding defaults and the subsequent one-layer comparisons. Reproducing
this campaign requires its saved cohorts and constructor overrides.

The final user-selected design has **no Mamba blocks in either downstream head**.
There are eight independent fits: track finding and Cartesian momentum regression,
each with frozen Mamba1/Mamba2/Mamba3 features and one AdapterOnly control. The
earlier proposal to vary downstream Mamba families was superseded before launch.

## Architecture

All inputs are center-only cluster tokens, calibrated12 band-Hilbert serialization,
and the same fixed coordinate normalization used by the cluster pipeline.
Frozen backbones are the completed width512/depth12/state16/k1 checkpoints from
the matched 200k-update pretraining campaign. Use their best validation checkpoints.
Frozen feature extraction preserves each checkpoint implementation's native
precision: Mamba1/2 FP32 blocks, Mamba3 local BF16 blocks, with FP32 outputs.
The common downstream heads run in FP32; no Mamba-family adapter selection remains.
Head construction uses an isolated seed42 CPU generator, so different backbone
construction cannot shift initial adapter weights. The actual initial head state
checksum is recorded before the first update and must match within each task.

- Both tasks use input width512 and a projection to width256. Frozen features are
  mixed by learned softmax depth weights. AdapterOnly uses the existing NeRF
  coordinate embedder and normalization at the same input width.
- Track finding uses zero Mamba input/refinement layers, the common MLP/attention
  decoder, 16 prototypes, and two prototype-attention refinement layers.
- Momentum uses zero Mamba input/refinement layers, the common projection,
  normalization, attention pooling, and regression MLP. `run.py` overrides the
  otherwise hard-coded one-layer regression constructor only in its child process.
  Training and evaluation both use the same override. No repository defaults change.
- Every actual head graph is checked for zero `_mamba_config` modules; both input
  and refinement ModuleLists must be empty. Shared shapes are held fixed. Frozen
  and AdapterOnly paths have different active inputs, as intended by the control.

## Recipe

50,000 labeled examples per task, seed42, 30,000 optimizer updates, no early
termination, AdamW LR1e-5 to3e-4 with1000-step warmup then one30k-step cosine
cycle, weight decay1e-4, gradient norm clipping0.1, dropout0.1. Batch32 for event
track finding; batch128 for per-track momentum. Validate every1000 updates.

Track finding uses signal-only targets, the existing Hungarian class/Dice/focal
loss with weights0.5/0.1/1/30, and checkpoint selection by signal option2 ARI with
validation-loss tie breaking. Assignment option2 and threshold0.2 are fixed for
all runs; no per-model evaluation tuning. Retain inclusive/background metrics,
efficiency/purity/fake/split/merge rates and the COATJAVA reference.

Momentum predicts entrance `(px,py,pz)` using standardized MAE and attention
pooling. Target means/stds are calculated only from the selected50k training
tracks. Checkpoint selection uses validation loss. Evaluation includes native
and physical momentum/angular errors, with the existing COATJAVA comparison.

## Cohorts and provenance

`prepare_data.py` writes immutable NPZ keys/lengths and `data_ready.json` checksums.
The existing v6_02 data are read-only; no data or original checkpoint is rewritten.
Both hosts must match keys, lengths, selected raw feature/label/target content and
dataset metadata checksums before production. Training uses the source pretrain
split. Validation uses source test event IDs[10000,100000); final evaluation uses
IDs>=100000. Whole events stay on one side of validation/evaluation, including
all their momentum segments. The early test events used in pretraining selection
are reserved from this downstream campaign. Independence is conditional on the
existing source pretrain/test split; this does not create a new external dataset.

| Task | Training | Validation | Final evaluation |
|---|---:|---:|---:|
| Track finding |50k events|5k events|10k events|
| Momentum |50k truth tracks|10k truth tracks|50k truth tracks|

Track finding accepts1–512 clusters and1–16 signal instances. Momentum uses
MC-truth segments with12–100 clusters, and rejects invalid accepted truth targets
fatally. No truth joins are repaired or ambiguous rows silently selected.
Evaluation output's ordinal event IDs map to the persisted ordered cohort keys.

## Execution and gates

Use the existing clas12-edge MCP and pinned training image
`sha256:bbc43e230b611637aca8f1824804dc26f3ac0d5ab9703dcc11d7fda9085a4a0f`.
Each machine runs one GPU child at a time in a detached, immutable source snapshot.
The primary checkout, image tag and original checkpoints remain unchanged.

- a02c: finding Mamba1, momentum Mamba3, momentum Mamba1, finding Mamba3.
- a518: finding AdapterOnly, momentum Mamba2, finding Mamba2, momentum AdapterOnly.

Before full training, all eight configurations run5 real optimizer updates on
the first256 training examples,64 validation examples and64 separate evaluation
examples. These are execution checks, not accuracy results. Each training child
requires finite adapter gradients, no backbone gradients, exact budget completion,
an unchanged frozen-backbone digest, strict head reload, identical reloaded
predictions and restored optimizer/scheduler state. Standalone evaluation must
exit0 and produce its summary. Both host pilots must pass before production.
Production imposes no runtime deadline and starts fresh after the pilot.

`run.py --host HOST --pilot` runs the bounded pilot queue.
`run.py --host HOST` runs the production queue with `/pilot/pilot_complete.json`
as its gate. Individual children use `--stage train|eval --task finding|momentum
--backbone adapteronly|mamba1|mamba2|mamba3 --run-dir ...` and require `/cohorts`
plus the appropriate read-only `/pretrained/VERSION` mount.

The worker writes `status.json` while active and `completed.json` only after all
four assigned runs train and evaluate successfully. Each run saves resolved model
and analysis configurations, progress, selected checkpoint, gradient/reload audit,
stdout logs, artifact summary and evaluation outputs. A failed child stops its
queue. Single-seed results need repeated seeds before strong model-family claims.

## Registered jobs

Both production queues launched on2026-10-29 at12:23p.m. America/New_York.
All eight final v4 pilots passed finite-gradient, strict-reload, prediction,
frozen-backbone and exact64-sample evaluation gates. Initial head states match
across all four arms of each task; momentum evaluation truths match as well.

- Shared preparation on each host: `backbone-downstream-cohorts-20261029`.
- a02c final pilot: `backbone-downstream-no-mamba-pilot-v4-a02c-20261029`.
- a518 final pilot: `backbone-downstream-no-mamba-pilot-v4-a518-20261029`.
- a02c production: `backbone-downstream-no-mamba-a02c-20261029`.
- a518 production: `backbone-downstream-no-mamba-a518-20261029`.

Each remote output lives at
`/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/JOB_ID/output`.
The pinned production source snapshot SHA256 is
`31331d76cb655a5dc6bf1bb008d878213a7e00ff04b25ad3b0668677fec3b575`.
Local plans, launch receipts and final pilot reports are saved under
`integrations/edge_mcp/validation/backbone-downstream-no-mamba-*20261029.json`.
