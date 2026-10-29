# Matched Transformer pretraining, 2026-10-29

Pretraining only: two FP32 causal full-attention Transformers, NoPE and RoPE,
width512/depth12/8 heads/FFN544 (ratio1.0625), 19,339,651 total parameters.
The original NeRF coordinate embedding is retained and compared bitwise with
the Mamba2 reference checkpoint. Initial parameter tensors and sampler RNG
states must match between Transformer arms.

The copied Mamba comparison harness supplies the unchanged cohorts, normalization,
calibrated12 band/Hilbert serialization, outward-band k1 targets, masked MSE,
optimizer, value clipping and schedule timing. The copied shared trainer has
the exact pretraining loader from the completed Mamba2 source snapshot
(SHA256 f184e312689e4373b66304a01298940b6daeee6703bea07e3d76a5036b157df3), plus
one initialization change: preserve native Transformer block initialization,
just as native SSM block initialization is preserved. Mamba behavior is unchanged.

Production uses the same first10M eligible training events (1–50 clusters),
2,000 validation events, batch64, 200k updates, 20k warmup, seed42, dropout,
AdamW betas, historical matrix LR scaling and checkpoint selection as Mamba.
Both pilot arms use those complete cohorts and 2,000 updates with warmup200,
so each visits the full peak LR. Production starts fresh with the original
schedule and requires both pilots, finite-gradient and strict-reload gates.
The fixed pilot schedule is an integration/stability check, not final evidence.

The queue runs NoPE then RoPE sequentially on a518 in an immutable snapshot.
Original datasets/checkpoints are read-only. Failures stop the queue. No runtime
deadline is imposed. No downstream jobs are launched.

`run.py --phase pilot` writes `pilot_complete.json`; `run.py --phase train`
requires that gate at `/pilot`, trains both arms and automatically creates
`comparison/report.md`, `metrics.json`, CSV and PNG/PDF plots for the pair and
all five backbones against steps, valid tokens and estimated arithmetic.
The common-compute table uses the last checkpoint within the minimum budget.

Compute accounting retains the Mamba estimates and adds full dense attention
QK/AV arithmetic using the replayed padded length of each actual training batch.
It excludes softmax, rotations, normalization, optimizer, nonlinearities and
kernel-specific work. Estimates are not measured GPU operations. Original
Mamba precision and host/session differences remain explicit limitations.
The 2,000-event validation cohort selects checkpoints; results are single-seed
pretraining evidence, without an independent holdout or downstream claims.
