# Three-backbone pretraining comparison

Scope: pretraining only. There are no downstream fits or label-efficiency
campaigns. The primary figures plot the **raw validation MSE recorded at each
checkpoint** against optimizer updates and against estimated training compute
spent reaching that checkpoint. Losses are not summed, integrated or replaced
with a running average/minimum. Stars identify the best recorded checkpoints.

The completed Mamba2/a518 and Mamba3/a02c k=1 Chinchilla-recipe runs are reused.
A new Mamba1/a02c control uses the same available v6_02 data, first 10M eligible
training events, first 2,000 eligible validation events, 1–50 cluster filter,
shuffle seed, width 512, 12 layers, state size 16, k=1, batch 64 and 200,000-step
ported recipe. This is a fresh Mamba1 control, not the historical checkpoint
trained on the unavailable older data. Native SSM initialization and decay
exemptions are preserved for all three. Mamba3 retains its local BF16 boundary;
Mamba1/Mamba2 use FP32. These are native implementations at equal width/depth,
not equal parameter counts or a family-specific hyperparameter search.

Jobs:

- Mamba1 training: `mamba1-chinchilla-w512-d12-k1-20261028` on a02c.
- Same-host core benchmark: `backbone-pretraining-compute-profile-20261028`
  on a518, with 10 warmup and 30 measured updates/model, batch 64 and length 50.
  Fresh fixed-seed models are used to measure graph cost, not validation loss.
  Forward/MSE/backward/value-clipping/AdamW are timed; LR=0 preserves weights.
  Data loading, validation and audit overhead are excluded from this benchmark.
- Interim plots: `backbone-pretraining-interim-curves-20261028` on a02c.
- Final report: `backbone-pretraining-full-comparison-20261028` on a02c.
  This CPU-only job waits for Mamba1's supervisor exit result, requires successful
  200k-step completion and checkpoint reload, then produces the complete report
  and plots automatically. It imposes no runtime deadline and does not restart
  training or alter any checkpoint.

## Compute accounting

`flops.py` documents the estimates. A multiply-add counts as two FLOPs. Dense
projection arithmetic is computed from actual module weight shapes, including
Mamba1 projections called inside fused kernels and the frozen NeRF projection.
Frozen embedding arithmetic is charged only for forward; trainable forward and
backward are estimated as three times forward. The second estimate adds causal
convolution and canonical recurrent update/readout arithmetic:

- Mamba1: `7*inner*state + 3*inner` scalar operations/token/block.
- Mamba2: `5*inner*state + heads*state + heads + 3*inner`.
- Mamba3 SISO: `8*inner*state + 6*heads*state*rope_fraction + 3*inner`.

These are canonical arithmetic estimates, **not measured executed GPU FLOPs**.
They exclude normalization, nonlinear/special functions, optimizer/audit/data/
validation work, parallel-scan overhead, chunk/tile padding and backward
recomputation. Dense-only and dense-plus-recurrence plots are both retained as
an accounting-sensitivity check. They must not be presented as exact total
hardware work or as a precision-independent architecture-efficiency verdict.

Actual padded-token counts are replayed from event lengths and the exact
persistent-worker sampler generator sequence, including its seed draw and
epoch-boundary permutation consumption. This is tested against a real loader
across multiple epochs. All three models' event-index checksums must match.

The profiler's supported-operator counts and named custom/fused operations are
saved separately. PyTorch's `with_flops` support does not establish a full count
for custom SSM kernels; it is not used as the total-model denominator.
See [PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html),
the [Mamba selective-scan reference](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py),
and the [Mamba3 recurrence paper](https://arxiv.org/abs/2603.15569).

## Outputs and interpretation

The final job writes `comparison_complete.json`, `comparison/summary.json`,
`learning_curves.csv`, a short report, and PNG/PDF learning curves versus steps,
estimated compute and dense-only compute. Threshold crossings and the last
recorded loss within the common compute budget are included. Validation is the
same cohort used for checkpoint selection; this is a single-seed pretraining
comparison, not independent holdout or downstream evidence.

Local interim plots/data are under the gitignored
`research_reports/backbone_pretraining_comparison_20261028/interim/comparison/`.
The interim Mamba1 curve is explicitly marked as running. The final remote
report appears automatically when the matched Mamba1 run finishes.
