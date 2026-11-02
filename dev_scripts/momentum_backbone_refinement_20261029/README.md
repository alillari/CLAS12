# Momentum backbones with a matched Mamba2 refinement head

Three new fits: frozen Mamba1, Mamba2 and Mamba3, each followed by the same
one-layer Mamba2 momentum head as the already completed refined AdapterOnly run.
The two AdapterOnly controls are reused from the completed reports. No
pretraining or track-finding job is repeated. No repository training default is
changed; the new runner wraps the existing campaign inside child processes.

The refined head uses width256/state64/conv4/expand2 and attention pooling, with
zero input-embedder Mamba layers. It contains664744 total/632488 declared-trainable
parameters, including32256 fixed coordinate parameters. The12 depth-mixing
weights are unused in AdapterOnly mode. Relative to the zero-layer control the
block adds432024 parameters. All refined arms have the same complete starting
head weights, not only the projection/pooling/MLP weights.

The legacy trainer reinitializes Mamba2 parameters after construction. To reuse
the refined control exactly, a zero-update reconstruction on each host saves
the pre-update head state and requires its full digest to equal the control's
recorded initial digest. Each new training child restores that state after the
legacy initializer and before its first forward/update. Strict loading and the
recorded full initial digest guard this operation. The control source manifest
was compared with the local training sources before launch: only its README
had changed. All core model, trainer, dataset and original campaign runner files
were byte-identical to the control snapshot.

The recipe is inherited unchanged: seed42,50k training tracks,10k validation
tracks,50k evaluation tracks, event-disjoint validation/evaluation, Cartesian
entrance momentum targets and train-only standardization, standardized component
MAE, batch128,30k updates, validation every1k updates and validation-loss checkpoint
selection. AdamW LR1e-5→3e-4→1e-5 with1k warmup, weight decay1e-4, norm clipping0.1,
dropout0.1. Heads are FP32; frozen Mamba1/2 blocks retain FP32 and Mamba3 retains
its native local BF16 policy with FP32 output features. No architecture-only
speed conclusions are justified by these host/precision-dependent runtimes.

- a02c: Mamba1 then Mamba3, one GPU child at a time.
- a518: Mamba2.
- Before production, each arm runs a5-update/64-evaluation-track GPU pilot.
- Production has no runtime deadline. Frozen checkpoints and source data are
  read-only, using the same immutable Docker image and exact cohort manifests
  as the preceding campaign.

Each pilot and production run must show finite head gradients including a
nonzero refinement gradient, FP32 refinement forward execution, no backbone
gradients, unchanged frozen weights, exact update budget, strict head reload,
identical reloaded predictions and restored optimizer/scheduler state. The
worker checks head initialization, recipe, cohorts and final truth digests
against the completed AdapterOnly control and original backbone audits.

The immutable plans/launch receipts are saved under
`integrations/edge_mcp/validation/momentum-backbone-refinement-launch-20261029.json`.
Jobs use these stable IDs (reuse the same ID when querying after disconnect):

- `momentum-refined-head-initial-a02c-20261029`
- `momentum-refined-head-initial-a518-20261029`
- `momentum-backbone-refinement-pilot-a02c-20261029`
- `momentum-backbone-refinement-pilot-a518-20261029`
- `momentum-backbone-refinement-a02c-20261029`
- `momentum-backbone-refinement-a518-20261029`

After both queues successfully finish, `collect.py --root ROOT` downloads existing
reports/predictions/histories using the established SSH aliases. It preserves
remote checkpoints and excludes caches and redundant existing evaluator images.
`compare.py --root ROOT` audits all runs, identities, truth vectors and the CVT
reference row by row before producing PNG/PDF learning curves (linear-x and
log-log), final quality, component errors and momentum-binned plots for both
AdapterOnly controls plus all three refined-backbone models. It also exports
per-backbone zero/one-layer comparisons, gains over both controls, CSV tables,
an audited report and a ZIP. ROOT is the existing
`research_reports/backbone_downstream_comparison_20261029` folder; all new results
go into its separate `backbone_refinement` subfolder. Existing reports remain
unchanged. This is single-seed evidence.
