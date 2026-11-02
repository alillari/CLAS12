# CLAS12 environment and data setup

Start here to prepare a training environment, then use
[Training workflow](train/TRAINING_RECIPES.md) for commands.
[Training options](train/downstream/TRAINING_OPTIONS.md) explains the controls;
[Target statistics](train/downstream/TARGET_STATISTICS.md) explains the stats file.
Run the commands from the repository root. This is research code; the checked-in
recipes still require access to data and compatible CUDA kernels.

## Environment

Use Linux, Python 3.10 or newer, and a CUDA-capable NVIDIA GPU for training.
The recent a02c/a518 campaigns used a prepared Docker environment; its recorded
runtime is in [Mamba support](train/MAMBA_SUPPORT.md). A fresh environment must
match PyTorch, CUDA, compiler, `mamba-ssm`, `causal-conv1d`, and Triton versions.
Do not replace a working training environment merely to follow a generic pip
command. Mamba3 requires an upstream revision containing its SISO implementation;
Mamba1 also requires the selective-scan CUDA extension.

For a new local environment, create it first and install a PyTorch build
appropriate to the target GPU/CUDA stack:

```bash
conda create -n clas12-fm python=3.10
conda activate clas12-fm
```

`requirements.txt` is an inherited, unpinned dependency inventory. It is not a
complete tested installation recipe: the trainers also import `ruamel.yaml`,
`scikit-learn`, `einops`, and `cosine_annealing_warmup`, and evaluators/plotting
need packages such as `matplotlib` and `pandas`. The `torch-distributed` entry
is not the source of `torch.distributed`, which is provided by PyTorch itself.
Use the team's working environment or resolve and record a compatible environment
before a long run. The scheduler import's upstream installation reference is
in `train/downstream/track_finding_trainer.py` and
`train/downstream/track_regression_trainer.py`.

Check the actual imports, not just `import fm4npp`:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import ruamel.yaml, sklearn, einops, mmap_ninja, cosine_annealing_warmup"
python train/downstream/train_track_finding.py --help
python train/downstream/train_track_regression.py --help
```

The help commands exercise trainer imports without loading data or starting
training. A successful help command does not validate GPU kernels. Use the
CUDA checks in [Mamba support](train/MAMBA_SUPPORT.md) on the target runtime.
The repository's local Mamba2 CPU fallback is approximate and cannot validate
the fused training path.

## Data contract

The normal downstream CLIs read a CLAS12 v6 event-product RaggedMmap root:

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
`data_root_test` selects validation data.

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
