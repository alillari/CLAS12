# Chinchilla recipe restart: Mamba2 and Mamba3, k=1

The original recipe was recovered from the local `origin/mike` Git reference,
commit `38c82ec06265a530a4efccc853bba085d0742875`, file
`scripts/configs/mamba_chinchilla_sweep.yaml`, configuration
`scale_w512_d12_chinchilla`. Exact recovered files are retained in `reference/`.
The YAML's learning-rate schedule, weight decay and 200k-step budget agree with
the actual campaign_4 width-512 CSV and saved checkpoint inspected through MCP.
The recovered pre-Chinchilla dataset code at `bace301` also uses the 50-cluster
filter, so that policy is not inferred from the later YAML alone.

Both replacements use width 512, 12 layers, state size 16, expansion 2, one
predicted coordinate triple (`klen=1`), position-only NeRF embedding, dropout
0.05019431407230154, and the following optimization:

- AdamW, betas (0.9, 0.95), epsilon 1e-8.
- Base learning rate 0.0009888238512457074 → 0.009888238512457074 →
  0.0009888238512457074, 20,000-step linear warmup and cosine decay to 200,000.
- Historical practical muP policy: vectors use base LR; tensors of rank >1 use
  base LR divided by width/base-width = 512/256 = 2. The historical first update
  uses the initial LR for both ranks; scaling is applied after each subsequent
  scheduler step, exactly as in the recovered loop. This timing is tested
  against the saved training CSV, including its peak and final values.
- Weight decay 0.046983716066038746, with native SSM `_no_weight_decay` flags
  honored for both new families. These parameters form zero-decay groups.
- Per-element gradient clipping at 1.0869938178631702.
- Actual single-GPU batch size 64, 200,000 updates: 12.8 million event
  presentations. Historical nominal `batch_size=128` is retained as metadata;
  the recovered single-GPU loader uses `local_batch_size=64`.
- Validation/checkpoint every 1,000 steps, validation batches of 2, capped at
  1,000 batches: 2,000 fixed validation events.

The old `mmap_cluster_full` dataset is unavailable, as confirmed by the user.
Both new runs use the available canonical cluster v6_02 dataset with identical
first-eligible cohort selection: the first 10 million training events with
1–50 clusters and the first 2,000 eligible historical-test events for validation.
The existing full SHA-256 manifests verified the consumed data files match on
both hosts. Each new run also saves event indices and their SHA-256 checksums;
these must match across the new runs. Validation selects checkpoints and is
not an independent final holdout.

The original calibrated12 geometry values are semantically identical to the
current calibration. Normalization, band-Hilbert ordering and nearest
next-band targets are retained. Targets are constructed after serialization;
tests confirm equivalence to the recovered historical function for k=1 and
the runtime audit requires no target to be visible in the causal prefix.
An explicit loader generator gives both families the same shuffle seed,
independently of how many random numbers their initializations consume.

Architecture-specific adaptations are explicit: Mamba2 uses convolution width
4 and requires its fused CUDA path; Mamba3 SISO has no convolution, uses rotary
fraction 0.5 and no output-projection norm. Both use head dimension 64, one
group and computational chunk size 64. Native SSM initialization is preserved;
the shared head/norm initialization follows the recovered loop. Mamba2 uses
FP32 and Mamba3 uses BF16 inside its blocks with FP32 parameters and loss.
Native initialization, zero-decay SSM parameters, and Mamba3 precision are
deliberate differences from the historical Mamba1 implementation.

`trainer.py` is the recovered historical loop with a selectable backbone,
preserved native SSM initialization and rank/decay parameter groups.
`run.py` adds finite-gradient checks, deterministic/vectorized cohort selection,
atomic checkpoints, architecture metadata and optimizer/scheduler restore
checks. The shared primary trainer and the previous k=5 experiments are
unchanged. Offline W&B is disabled; JSON/CSV/log/checkpoint records are retained.

The preflight uses the full architecture and batch size, 4,096/512 pilot
cohorts, and a compressed 20-step schedule with the same peak LR. It exercises
finite updates through that peak, validation, and strict checkpoint/prediction,
optimizer and scheduler restoration. Production starts fresh from seed 42 and
never loads pilot or old k=5 weights. Source manifest checks verify the pilot
and production implementation files agree before launch.

Replacement jobs:

- a02c: `mamba3-chinchilla-w512-d12-k1-20261027`
- a518: `mamba2-chinchilla-w512-d12-k1-20261027`

Records and outputs are below
`/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/JOB-ID/`.
Training has no wall-clock deadline or integration-imposed resource cap.
The old k=5 jobs are stopped with their source, checkpoints and outputs retained.
