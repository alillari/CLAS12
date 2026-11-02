# Mamba2/Mamba3 speed audit, 2026-10-28

Both completed runs performed 200,000 optimizer steps. Mamba2 took 14,521.60 s
on a518; Mamba3 took 10,367.46 s on a02c (28.61% less elapsed time).
Precision was confounded: Mamba2 blocks ran FP32 and Mamba3 blocks ran BF16.

## Completed same-host diagnostic

a518 NVIDIA GB10, same pinned training image. Job
`mamba23-precision-host-a518-20261028` exited 0 with both artifacts present.

| Condition | Median GPU ms/update | Round medians, ms |
|---|---:|---|
| Mamba2 FP32 | 64.201 | 63.479, 64.466, 64.201 |
| Mamba2 BF16 blocks | 48.665 | 48.665, 48.925, 48.538 |
| Mamba3 BF16 blocks | 42.754 | 42.604, 42.754, 42.886 |

Three counterbalanced rounds, each 10 warmup and 30 measured updates. Batch64,
sequence length50, width512, depth12, state16, fixed shared inputs and fresh
fixed-seed models. Timed forward/MSE/backward/value-clipping/AdamW, LR=0.
Data loading, validation and checkpoint/audit overhead excluded. BF16 applies
only inside SSM blocks; parameters, gradients, caller outputs and loss remain
FP32. Actual projection dtype and fused Mamba2 path are asserted; loss finite.

Mamba2 BF16 reduced update time by 24.20%; Mamba3 reduced time by another 12.15%
at matched BF16 block boundaries. The precision change removes 72.44% of this
microbenchmark's original millisecond gap, not necessarily that fraction of
historical end-to-end training time. SM clocks were 2515–2541MHz; recorded
post-condition temperatures 49–57C.

First-round profiling showed Mamba2 matrix-multiplication self GPU time dropping
from 12.78ms to 7.02ms under BF16, and its fused scan/convolution backward event
from 13.99ms to 7.47ms. Mamba3 BF16 had 75 aten::mm calls (6.08ms) versus 135
for Mamba2. Our Mamba2 separately projects B/C and packs tensors before its fused
scan/convolution, whereas Mamba3 uses a different projection and SSM kernel path.
These are diagnostic observations: custom events and kernels overlap, so these
categories must not be summed as a disjoint total.

## Interpretation

Host placement cannot be the sole explanation. Precision accounts for a
substantial part of the gap. The remaining same-host, matched-precision result
includes implementation, kernels and this short-sequence workload; it does not
establish a universal architecture-only advantage. Mamba3 has slightly more
parameters (19,437,827 versus Mamba2's 19,285,955).

Cross-host replica `mamba23-precision-host-a02c-20261028` is pending. At the
audit it was explicitly waiting for the running Mamba1 supervisor to exit.
It then requires an idle GPU at <=50C before initializing CUDA and measuring.
Mamba1 continues uninterrupted. Historical thermal/power state, concurrent load,
and data/validation overhead cannot be reconstructed by this microbenchmark.
No training recipe or trained checkpoint was changed.

Full raw a518 report:
`integrations/edge_mcp/validation/mamba23-precision-a518-20261028.json`.
Benchmark source: `precision_benchmark.py`.

