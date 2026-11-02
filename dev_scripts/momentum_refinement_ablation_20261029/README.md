# AdapterOnly momentum refinement ablation

The control is the already scheduled30k-update AdapterOnly momentum fit in
`backbone-downstream-no-mamba-a518-20261029`. This test adds one historical
Mamba2 refinement block and its RMSNorm at width256/state64/conv4/expand2.
Both paths retain zero input-embedder Mamba blocks. The new arm uses FP32,
matching the control and the historical momentum refinement execution policy.
The common projection, pooling, output MLP and coordinate-encoding initial weights
are held identical: the added block is constructed afterward in a separate RNG
scope. The existing trainer's historical Mamba2 initialization still applies.

The new job waits for the existing a518 queue to exit successfully and the GPU
to become idle. It does not interrupt, modify or compete with either live queue.
It runs a5-update/64-evaluation-track CUDA pilot, requires finite gradients,
strict checkpoint reload and matching shared initialization, then trains the
new arm for30k updates and evaluates50k tracks automatically. There is no runtime
deadline. Both arms use the same50k training tracks,10k validation tracks,
event-disjoint50k evaluation tracks, target statistics, batch128, seed42,
LR schedule, dropout, loss, clipping and validation checkpoint selection.

Completion requires identical train/validation/evaluation identity checksums,
evaluation truth and metric population counts, common starting weights and
fixed training budgets. `comparison.json` saves both complete sets of metrics;
`completed.json` is written only after these checks pass. The existing control
is reused rather than retrained. This is single-seed evidence. The new layer
adds432,024 parameters, so the result includes a capacity difference as well as
the sequence-processing change; it does not isolate topology at equal capacity.

For the current512-wide input and256-wide adapter:

| Adapter | Total | Trainable |
|---|---:|---:|
| Track finding, no Mamba |2,217,486|2,185,230|
| Momentum, no Mamba |232,720|200,464|
| Momentum, one Mamba2 |664,744|632,488|

These are actual module counts excluding any backbone. All heads include32,256
frozen coordinate-encoding parameters. Each also contains12 depth-mixing weights
that have `requires_grad=True` but are unused in AdapterOnly mode; subtract12
from the trainable column for parameters active in the AdapterOnly forward path.
The counts refer to this controlled campaign; historical AdapterOnly input width
can differ. Source counts are saved in
`integrations/edge_mcp/validation/adapter-parameter-counts-20261029.json`.

Registered a518 job: `momentum-adapteronly-refinement-ablation-20261029`.
At launch it was verified in `waiting_for_existing_campaign`, with no CUDA work
until the prior queue exits. The immutable source snapshot SHA256 is
`9c3d536d19223cc89124febc12acbff7617dac48ce15fea42ea3a03989460c59`.
The full plan and receipt are saved in
`integrations/edge_mcp/validation/momentum-refinement-ablation-launch-20261029.json`.
