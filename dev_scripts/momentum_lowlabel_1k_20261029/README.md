# Eight momentum fits with 1,000 training tracks

All training runs on a02c. Train fresh AdapterOnly, frozen Mamba1, frozen Mamba2
and frozen Mamba3 heads, each with zero or one Mamba2 refinement block: eight
fits. Original backbones, dataset and 50k-label results are preserved.

The original scientific model/trainer/dataset/configuration sources were
compared byte for byte with the successful 50k campaign snapshot. They match.
The a02c primary checkout differs and is not used or overwritten. MCP stages
the verified local source as an immutable snapshot. The Oct29 storage cleanup
removed v7 data and caches; the retained v6 dataset and the original selected
train/validation/evaluation feature/label/target content were freshly checked
against the earlier manifests, including metadata and exact track lengths.

## Training-label subset and normalization

`prepare.py` chooses1,000 distinct tracks without replacement from the earlier
50k cohort using NumPy default_rng(42), retaining their original order. They
span987 source events. The same ordered1k subset is used by all eight models.
Both the10k validation and50k final evaluation cohorts are unchanged; their
source events are disjoint. This is1k TRAINING labels, not1k total annotations:
the retained10k validation labels still select checkpoints.

Momentum means/stds use ONLY these1,000 training tracks. The complete starting
zero/one-layer heads are generated from the earlier exact initial weights,
replacing all target_normalizer mean/std/constant buffers with the1k statistics.
All remaining initial tensors are unchanged. Every new fit strictly restores
the appropriate initial state after the trainer's legacy initializer. Both
layer groups share identical common tensors. No50k momentum normalization or
trained adapter checkpoint is used as an initialization.

The prepared initial full-state digests are:

- Zero layers:59a5203d4ae20011667bc11eec087b8f174794c0b86e37b91108e500ce2ca87d
- One layer:20e083a244f8b8c9f78f0d838e64ba3e69984aa4f3c9f945fb16e4303ad15958
- Shared tensors:59a5203d4ae20011667bc11eec087b8f174794c0b86e37b91108e500ce2ca87d

## Recipe and execution

Seed42,30k optimizer updates, batch128 (last batch retained), validation every1k
updates, best-validation-loss selection, AdamW LR1e-5→3e-4→1e-5 with1k warmup,
weight decay1e-4, norm clipping0.1, dropout0.1, attention pooling, standardized
Cartesian entrance-momentum MAE. This keeps the earlier optimization budget
and scientific settings. DataLoader workers are zero to avoid repeated worker
startup on the small dataset. There is no early termination or runtime deadline.

Input width512, adapter width256, zero input-embedder Mamba layers. When present,
the refinement block is Mamba2 width256/state64/conv4/expand2 for every backbone.
Counts excluding the backbone are232720 total/200464 declared trainable for
zero layers and664744/632488 for one; refinement adds432024 parameters. Both
include32256 fixed coordinate parameters;12 depth weights are unused in AO.
Heads run FP32. Frozen Mamba1/2 retain FP32 and Mamba3 native local BF16 blocks
with FP32 output. This is a single-seed, single-subset experiment.

The Mamba2 best checkpoint is relayed from a518 to a new isolated a02c input
directory. Its source, local and destination bytes are SHA256-verified. The
worker waits for the atomic completed copy before any GPU work, checks its
expected231340008 bytes and SHA256
242666b0b62fd59b3fcc8eb7ce38c5c55ee9f74a2fd12b464e9e070d001e3556,
then runs all eight5-update/64-track real CUDA pilots. All must pass before
the eight30k-update fits start. Compiler caches are new and experiment-local.
Every GPU child runs sequentially on a02c. Training and standalone evaluation
use the same head/cohort override inside the child process; defaults are unchanged.

Completion requires finite updates, nonzero refinement gradients when present,
FP32 head execution, strict head reload and equivalent predictions, restored
optimizer/scheduler, no backbone gradients and unchanged frozen digests,
exact common cohorts/initial weights, and50k finite final predictions. Native
evaluation truth is independent of subset-specific normalization and must match
the original final truth checksum. A failed child stops the queue.

MCP jobs:

- a02c input directory: `momentum-1k-checkpoint-import-a02c-20261029`
- a518 read-only checkpoint hash probe: `momentum-1k-source-checkpoint-sha-a518-20261029`
- a02c CPU preparation: `momentum-1k-prepare-a02c-20261029`
- a02c complete GPU queue: `momentum-1k-a02c-20261029`

Plans/receipts are in
`integrations/edge_mcp/validation/momentum-1k-launch-20261029.json`.
The GPU output is
`/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/momentum-1k-a02c-20261029/output`.
`status.json` identifies the phase/active child, including input-transfer wait;
each child writes progress and training/evaluation audit markers. Only final
`completed.json` and successful supervisor/container exit establish queue success.

## Finished comparisons

After success, run `collect.py --root research_reports/momentum_lowlabel_1k_20261029`
through the existing SSH alias to download existing reports, raw histories,
configurations and predictions. It preserves remote checkpoints. Then run
`compare.py` with the same root using the fm4npp Python environment. It audits
all eight fits and compares:

- Four inputs within each head-layer group, including raw validation curves
  versus step with linear-x and log-log axes.
- Zero/one refinement for each input, physical component/magnitude/angular
  errors, width68, tails and matched momentum-bin/CVT comparisons.
- Physical errors at1k versus the completed50k-label fits on exactly the same
  final tracks. Standardized validation losses must NOT be directly compared
  across label counts because the target standard deviations are recalculated.

Outputs are a separate report, JSON/CSV tables, five inspected PNG/PDF figures
and `momentum_1k_comparison.zip`. Preserve the earlier reports and mark delivery
in the new report root's `delivery_state.json`. The result supports discussion
of training-label dependence; uncertainty over seeds/subsets is not measured.
