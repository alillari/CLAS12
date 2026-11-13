# CLAS12 environment and data setup

Start here to prepare a training environment, then use
[Training workflow](train/TRAINING_RECIPES.md) for commands.
[Training options](train/downstream/TRAINING_OPTIONS.md) explains the controls;
[Target statistics](train/downstream/TARGET_STATISTICS.md) explains the stats file.
For a complete pretraining and two-adapter run, follow
[the worked example](train/WORKED_EXAMPLE.md).
Run the commands from the repository root. This is research code; the checked-in
recipes still require access to data and compatible CUDA kernels.

## Environment

Start with the team’s prepared Docker runtime on a02c or a518. Ask the host
operator for access, the approved image (for example
`pp_collision:alessio-latest`), the published CVT cluster HDF5 directory, and
a writable results directory. Follow [Remote edgexpert jobs](train/REMOTE_RUNS.md)
to record the image ID and mount a source snapshot and the data read-only.
Run the checks below from the repository root inside that runtime. Use paths
as seen by the container.

For a separate local environment, use Linux, Python 3.10 or newer, and an NVIDIA
GPU. [requirements.txt](requirements.txt) lists Python dependencies. Choose
PyTorch, CUDA, Triton, and compiled Mamba extensions for the target GPU as
described in [Mamba support](train/MAMBA_SUPPORT.md#mamba3-runtime).

### Check CUDA and trainer imports

```bash
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python -c "import h5py, ruamel.yaml, sklearn, einops, mmap_ninja, plotly, cosine_annealing_warmup"
python -m train.downstream.train_track_finding --help
```

For Mamba2/3 workloads, run the
[existing CUDA kernel tests](train/MAMBA_SUPPORT.md#checkpoints-and-validation)
in the same runtime. Check that the tests for the intended backend ran, since
the suite can skip cases when CUDA or optional dependencies are unavailable.

### Run a short HDF5 smoke test

Use the directory containing `manifest.json` for a published
`cvt-clusters-h5-v1` product with train, validation, and test splits. If you
use a small custom fixture, include eligible truth tracks with at least four
clusters in the validation events. Set
`CLAS12_ARTIFACT_ROOT` to a writable directory outside the source checkout.

```bash
export CLAS12_DATA_ROOT=/absolute/path/to/cvt-clusters-h5
export CLAS12_ARTIFACT_ROOT=/absolute/path/to/writable/results
CLAS12_SMOKE_DIR="$(mktemp -d "$CLAS12_ARTIFACT_ROOT/hdf5-smoke.XXXXXX")"
CLAS12_VALIDATION_DATA_ROOT="$CLAS12_DATA_ROOT" \
  python -m train.downstream.benchmark_hdf5_finding \
    --data-root "$CLAS12_DATA_ROOT" --output-dir "$CLAS12_SMOKE_DIR/finding" \
    --train-size 32 --validation-size 16 --test-size 16 --steps 2 --workers 0
```

In `$CLAS12_SMOKE_DIR/finding/summary.json`, check `optimizer_steps: 2`,
`train_samples: 32`, `validation_samples: 16`, and `test_samples: 16`.
The run trains an AdapterOnly head, reloads its checkpoint, and evaluates
the test split. Use it to check data access and execution before a longer fit.

### GB10 Triton assembler selection

If a fused-kernel probe fails with `ptxas fatal: Value 'sm_121a' is not defined`,
check which assembler Triton invokes. The clean PyTorch 2.9.1/CUDA 13.0,
Triton 3.5.1 environment required the CUDA toolkit assembler instead of Triton's
bundled copy. With a compatible CUDA 13 toolkit installed, set these inside
the job environment and rerun the actual fused forward/backward check:

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST="12.1+PTX"
unset TRITON_OVERRIDE_ARCH
```

Use a fresh, job-specific `TRITON_CACHE_DIR` when changing the compiler/runtime.
This addresses assembler selection; it does not install the toolkit or validate
a different GPU architecture. The same GB10 failure and selection are described
in [the upstream Triton issue](https://github.com/triton-lang/triton/issues/10331).

## Data contract

For HDF5, see
[data representation and sample selection](train/downstream/TRAINING_OPTIONS.md#data-representation-and-sample-selection).
The legacy downstream recipes read a CLAS12 v6 event-product RaggedMmap root:

```text
data_root/
|-- features_pretrain/       # raw Cartesian cluster centers, three columns
|-- features_test/
|-- seg_target_pretrain/     # per-cluster truth segment IDs; background -1
|-- seg_target_test/
|-- reg_target_pretrain/     # real per-cluster truth targets, aligned with features
`-- reg_target_test/
```

Optional sidecars include `pid_target_pretrain/test`, `noise_target_pretrain/test`
and `coatjava_seg_pred_pretrain/test`; their need depends on the task and
comparison being run. Final COATJAVA comparisons need their corresponding
sidecars. See the rendered model/analysis configuration.

Track finding consumes complete events. Momentum uses
`adapter_sample_mode: event_segment`, `segment_target_source: mctrue`, and
`segment_min_clusters: 12`, accepting 12–100 clusters per truth track in the
usual recipe. Legacy track-level regression input is disabled. `--eventnumber`
therefore counts events for track finding and accepted tracks for momentum.

Use `input_layout: xyz`, `embed_method: pos_only`, `voxelize: false`, and
`serialization: clas12_band_hilbert`. The loader normalizes/transforms raw xyz;
do not supply already normalized eta/phi/r files as xyz. Keep the versioned
`dev_scripts/clas12_band_calibration.json` available.

## Paths and outputs

The new onboarding profiles in
[scripts/configs/clas12_adapter_recipes.yaml](scripts/configs/clas12_adapter_recipes.yaml)
use environment variables:

```bash
export CLAS12_DATA_ROOT=/absolute/path/to/mmap_canonical_loose_truthseg_event_v6_02
export CLAS12_ARTIFACT_ROOT=/absolute/path/to/result_deep_storage
export CLAS12_MOMENTUM_STATS="$CLAS12_ARTIFACT_ROOT/stats/momentum_stats_train50k.json"
```

The [training guide](train/TRAINING_RECIPES.md) includes the command that creates
momentum target statistics using only the selected training tracks. Recompute
those statistics if the cohort/filter/label budget changes.

YAML paths accept environment variables and references such as
`checkpoint_dir: "{downstream_dir}/checkpoints"`. **Only the new onboarding
profiles** use `CLAS12_DATA_ROOT` and `CLAS12_MOMENTUM_STATS`; older YAMLs retain
machine-specific paths that must be edited. `data_root` is the training root;
`data_root_train` is compatibility metadata in the normal loader, and
`data_root_test` selects validation data. The onboarding profiles accept
`CLAS12_VALIDATION_DATA_ROOT` for validation and provide separate `*_evaluation`
profiles controlled by `CLAS12_EVAL_DATA_ROOT` for final inference. Momentum
evaluation initializes the trainer and opens the original training trees too;
keep that training root accessible when evaluating an existing checkpoint.

Use external directories for checkpoints, logs, plots and reports. CLI
`--checkpoint_dir` overrides the YAML path; `--root_dir` controls the experiment
root and global logs and does not replace every YAML output path. Give each run
its own checkpoint directory. Save `--resolved_config_path` and the generated
artifact summary with the run.

## Launch and evaluate

Use these public entrypoints with the named profiles in the training guide:

| Task | Entrypoint | Onboarding config |
| --- | --- | --- |
| Event track finding | `python train/downstream/train_track_finding.py` | `clas12_track_finding_50k` |
| Momentum | `python train/downstream/train_track_regression.py` | `clas12_momentum_50k` |

`train_momentum_regression.py` is a backward-compatible wrapper for the second
command. `track_finding_trainer.py` is an implementation module, not the public
training CLI. Omit `--usepretrain` for AdapterOnly. To train on frozen pretrained
features, provide both `--usepretrain` and `--pretrained_ckpt` and match the
backbone architecture/embedding/klen to its saved configuration.

Always pass the batch explicitly: `--train_batch_size 32` for track finding and
`--train_batch_size 128` for momentum. CLI defaults override YAML batch settings.
A step-controlled run requires `max_optimizer_steps`; `total_steps` alone does
not set the downstream update budget.

For multiple fits, see [campaign tools](train/downstream/campaign/README.md).
For scoring, use [evaluation](train/downstream/eval/README.md) with the run's
rendered model/analysis YAML. Normal CLI validation uses the configured test
root; reserve a disjoint final cohort explicitly for scientific comparisons.

## Historical scripts and troubleshooting

`scripts/configs/mamba_pretrain.yaml`, `mamba_tracking.yaml` and the
`scripts/run/` shell/direct launchers are inherited FM4NPP examples, not current
CLAS12 recipes. The direct downstream launcher points at the trainer module;
the shell scripts contain site-specific Python paths and SLURM resource requests.
Their direct-execution branches do not establish multi-GPU correctness. Use the
public single-run CLIs above or the versioned isolated campaign runners.

- Import errors: activate the training environment, run from the repository root,
  and check the imports above. If needed, set
  `PYTHONPATH=/absolute/path/to/CLAS12:$PYTHONPATH`.
- CUDA errors: check `nvidia-smi` and the recorded package/GPU versions; run the
  relevant kernel tests before training.
- Out of memory: reduce the CLI training batch and the YAML validation batch
  (track-finding CLI overrides both), then check accepted sequence lengths.
- Checkpoint mismatch: inspect the saved architecture; names such as M4 describe
  width tiers, not the Mamba implementation family. `klen` is a target horizon,
  not the number of queries.

This code builds on FM4NPP; preserve the attribution and citations in
[README.md](README.md).
