# Branch inventory and historical experiments

Updated 2026-11-02 after the authorized branch cleanup on GitHub, the laptop,
and both EdgeExperts (a02c and a518). The cleanup is based on `main` at
`c629855f1d0a66fa66d2586d4321f2e74eae14c1`; the inventory update is a subsequent
documentation-only commit. No experimental implementation was merged.

## Active branches

| Branch | Code tip before inventory update | Purpose and next action |
| --- | --- | --- |
| `main` | `c629855f1d0a` | Shared research baseline; adapter baseline, pretraining, Mamba2/3, Transformer support and training guides are merged through PRs #1–5 |
| `mike` | `38c82ec06265` | Collaborator-owned development; retain and review selected contributions independently |
| `feature/momentum-coordinate-bypass` | `544f21ba9114` | Separate PR candidate; opt-in coordinate input alongside frozen features; representative CUDA validation remains necessary |
| `tooling/edge-mcp` | `3f209712c096` | Separate PR candidate for isolated Edge inspection/execution tooling |
| `experiment/momentum-last-two-layers` | `5f51ff5cb254` | Retain for a selective port and separate review; conflicts with the current regression trainer and predates current architecture selection |

These five branch tips remain on GitHub. The laptop retains its existing four
local development branches (`main` and the three pending-review branches);
a02c and a518 retain local `main` and fetch all five GitHub branches.
No collaborator branch was rewritten. The cleanup does not approve any pending
implementation for integration.

## Published archive tags

All retired tips are preserved by **annotated tags** published to GitHub and
available on all three machines. The tag prefix is `archive/2026-11-02/`;
the table gives suffixes. Never move these tags. Create a new branch from a tag
to resume an investigation.

The [machine-readable manifest](branch-archive-20261102.json) contains full
commit SHAs, tag object SHAs, mappings for every retired ref and bundle digests.
A tag may serve several retired names when they pointed to the same commit.

| Retired GitHub branch | Exact tip | Archive tag suffix | Disposition |
| --- | --- | --- | --- |
| `archive/upstream-fm4npp-2026-02-28` | `ffe8d6a33a36` | `archive/upstream-fm4npp-2026-02-28` | Preserve upstream provenance as a tag |
| `diagnostic/campaign4-track-loss` | `10b7ddf65e1a` | `diagnostic/campaign4-track-loss` | Preserve historical diagnostic; adapt reusable tools separately |
| `experiment/causal-transformer-adapters` | `ae09d4898619` | `experiment/causal-transformer-adapters` | Fully merged in PR #4 |
| `experiment/mamba-backbone-adapter-support` | `f16093a8cfc2` | `experiment/mamba-backbone-adapter-support` | Fully merged in PR #3 |
| `experiment/transformer-pretraining-comparison` | `d84731c391e0` | `experiment/transformer-pretraining-comparison` | Preserve completed comparison; isolate harness before reuse |
| `experiments/momentum-adapter-experiments` | `1e35394407f6` | `experiments/momentum-adapter-experiments` | Preserve context experiment; selected evaluation fix is already patch-equivalent in main |
| `experiments/momentum-cartesian-factorial` | `30a302de60d5` | `experiments/momentum-cartesian-factorial` | Preserve Cartesian and membership experiment lineage |
| `experiments/momentum-membership-mask` | `96d926be9b3b` | `experiments/momentum-membership-mask` | Preserve endpoint; its history is included in the factorial branch |

## Retired machine-local branches

The following local refs were removed after their exact tips were archived.
Historical worktree directories remain at their original commits with detached
HEADs. The existing detached legacy worktrees also remain unchanged. No data,
checkpoints, campaign outputs, environments or worktree directories were deleted.

| Machine | Retired local branch | Exact tip | Archive tag suffix |
| --- | --- | --- | --- |
| laptop | `adapter-dev` | `3b9170780c16` | `laptop/adapter-dev` |
| laptop | `archive/campaign-runners-20261029` | `825d6b4a43a1` | `laptop/archive/campaign-runners-20261029` |
| laptop | `archive/coordinate-bypass-baseline` | `3b9170780c16` | `laptop/adapter-dev` |
| laptop | `archive/local-preservation-20261102` | `b80d770769cd` | `laptop/archive/local-preservation-20261102` |
| laptop | `archive/mamba-backbone-adapter-support-original` | `ff32c464506e` | `laptop/archive/mamba-backbone-adapter-support-original` |
| laptop | `experiment/causal-transformer-adapters` | `ae09d4898619` | `experiment/causal-transformer-adapters` |
| laptop | `experiment/mamba-backbone-adapter-support` | `f16093a8cfc2` | `experiment/mamba-backbone-adapter-support` |
| laptop | `experiment/transformer-pretraining-comparison` | `d84731c391e0` | `experiment/transformer-pretraining-comparison` |
| laptop | `integration-adapter-pretrain` | `756fc500c949` | `laptop/integration-adapter-pretrain` |
| laptop | `momentum-adapter-experiments` | `690dc33682fb` | `laptop/momentum-adapter-experiments` |
| laptop | `momentum-cleanup` | `dadeb107c76f` | `laptop/momentum-cleanup` |
| a02c | `adapter-dev` | `ca094b7919c9` | `a02c/adapter-dev` |
| a02c | `momentum-adapter-experiments` | `4537f296531d` | `a02c/momentum-adapter-experiments` |
| a02c | `momentum-adapter-primary` | `1e35394407f6` | `experiments/momentum-adapter-experiments` |
| a02c | `momentum-cartesian-factorial` | `30a302de60d5` | `experiments/momentum-cartesian-factorial` |
| a518 | `adapter-dev` | `b566b54a7e2e` | `a518/adapter-dev` |
| a518 | `diagnostic/campaign4-track-loss` | `10b7ddf65e1a` | `diagnostic/campaign4-track-loss` |
| a518 | `momentum-adapter-experiments` | `1e35394407f6` | `experiments/momentum-adapter-experiments` |
| a518 | `momentum-cartesian-factorial` | `30a302de60d5` | `experiments/momentum-cartesian-factorial` |
| a518 | `momentum-membership-mask` | `96d926be9b3b` | `experiments/momentum-membership-mask` |

The laptop's `adapter-dev`, coordinate-bypass baseline and `momentum-cleanup`
are fully ancestral to `main`. The older a02c/a518 `adapter-dev` tips are also
ancestral to `main` and have their own tags. The laptop integration branch has
one unique merge commit, but merging it into the cleanup baseline produces no
file changes; its exact merge endpoint is tagged.

The laptop's additional momentum audit commit is patch-equivalent to an audit
already in `main`. Its full endpoint remains tagged alongside the remote context
experiment. The original Mamba implementation is preserved separately from the
reviewed version integrated by PR #3. The two local preservation snapshots are
not ancestors of one another; both are retained as tags and in recovery bundles.

## Experiment provenance and integration boundaries

### Full-event context, membership and Cartesian experiments

These investigations use the historical CLAS12 event-product RaggedMmap inputs
and momentum targets. Their exact dataset paths, cohort selection, architecture,
training and evaluation settings remain in the source recipes at each tag:

- Context: `train/downstream/MOMENTUM_CONTEXT_EXPERIMENT.md` and
  `train/downstream/experiments/m6_context/experiment.yaml`.
- Membership: `train/downstream/MOMENTUM_MEMBERSHIP_EXPERIMENT.md` and
  `train/downstream/experiments/m6_context/run_membership.sh`.
- Cartesian factorial: `train/downstream/experiments/momentum_cartesian_factorial/`
  contains the AdapterOnly/pretrained templates, renderer and launcher.

The membership endpoint is ancestral to the factorial endpoint. These branches
have conflicts with the current trainer/configuration, and the membership and
factorial branches also conflict in model/evaluation code. They are archived to
preserve investigations without restoring older shared code. Their scientific
outcomes were not re-audited for this cleanup; archival is not a failed-result
classification. Generated reports/checkpoints remain in their original storage.

### Transformer pretraining comparison

The exact recipe and launch/report code are under
`dev_scripts/transformer_pretraining_20261029/` at the archived tip, with the
copied Mamba comparison and Chinchilla harnesses beside it. This completed
pretraining-only comparison used the matched first-10M eligible cluster-event
cohort, 2,000 validation events, 200k updates and seed 42. The local report is
`research_reports/transformer_pretraining_20261029/final/comparison/report.md`.
It records single-seed results, checkpoint-selection validation, differing Mamba
precision and estimated arithmetic rather than measured executed FLOPs.

Do not merge this entire branch even though Git reports no conflicts. Its
`fm4npp/datasets/dataset_pretrain.py` replaces the shared calibrated outward-band
target rule with later-index neighbors and removes auxiliary-feature,
alternate-reader and data-fraction controls. The campaign harness separately
constructs its own outward-band targets. Preserve the reproduction snapshot;
extract or adapt useful harness code into isolated modules before review.

### Campaign 4 diagnostic

The archived tip contains `campaign4_track_loss.py`, `campaign4_analyze.py`,
`campaign4_inventory.py` and `campaign4_recovered.yaml`. These evaluate historical
campaign-4 checkpoints against v6.02 test-event features and truth/noise/reason
sidecars, separating same-track, different-track and ambiguous targets. The
script records sample identities and explicitly leaves overlap with original
campaign training unverified. It branches from an older Mike baseline and does
not include Mike's later `38c82ec` commit. Archive it as an evaluation snapshot;
port reusable diagnostics independently of that baseline. Execution/results
were not re-audited during this cleanup.

### Local campaign and preservation snapshots

The laptop campaign-runner archive preserves the exact reproduction sources in
`dev_scripts/backbone_downstream_20261029/`, `backbone_pretraining_comparison/`,
`chinchilla_restart/`, `mamba2_cluster_pretraining/`,
`mamba3_cluster_pretraining/`, `momentum_backbone_refinement_20261029/`,
`momentum_lowlabel_1k_20261029/` and `momentum_refinement_ablation_20261029/`.
Their README files, recipes and recorded run specifications retain each
experiment's inputs, settings and execution scope. Existing local reports under
`research_reports/` remain outside Git. The broader local-preservation archive
also records the pre-reconciliation tooling, configuration and documentation.
Neither snapshot should replace current shared code wholesale.

### Partial backbone tuning remains pending

The retained last-two-layer branch records its matched M6 experiment in
`train/downstream/experiments/momentum_unfreeze/README.md` and `recipe.yaml`.
All three 30k-update, 50k-track arms completed. The result remains a single-seed
screen with validation/evaluation sharing the source test split. Its optional
fine-tuning capability needs a current-main port, explicit architecture checks,
checkpoint/resume validation and separate review before integration.

## Recovery and verification

Each machine has the recovery directory:

```text
/home/alessio/ML-work/CLAS12-branch-archives/2026-11-02/
```

It contains the laptop/a02c/a518 pre-cleanup Git bundles and ref/worktree
manifests, `all-machines-recovery.bundle`, the cleanup plan and `SHA256SUMS`.
The bundles were verified with `git bundle verify` and their copied files were
verified against SHA-256 digests before branch removal. Complete recovery bundles
remain private because they can include stash and application backup refs;
only the explicitly listed branch archive tags are published to GitHub.

Inspect or resume an archived source snapshot:

```bash
git fetch origin --tags --prune
git show archive/2026-11-02/experiments/momentum-membership-mask
git switch -c experiment/resume-membership \
  archive/2026-11-02/experiments/momentum-membership-mask
```

For complete recovery, verify `SHA256SUMS` from the recovery directory, inspect
`git bundle list-heads all-machines-recovery.bundle`, then clone the bundle into
a **new** directory with `git clone --mirror all-machines-recovery.bundle recovered.git`.
The per-machine bundles preserve the original branch names and exact tips.

For future experiments, record the question, source commit, dataset/schema,
recipe, execution status, results and limitations before retirement. Store
completed investigations as immutable tags; keep active implementation work on
branches. Older `alex` and unprefixed remote experiment names were already
retired before this cleanup; their historical lineage remains in the preserved
Git history. The upstream FM4NPP endpoint is retained by its dated archive tag.
