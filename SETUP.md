# CLAS12 environment and data setup

Start here to prepare a training environment, then use
[Training workflow](train/TRAINING_RECIPES.md) for commands.
[Training options](train/downstream/TRAINING_OPTIONS.md) explains the controls;
[Target statistics](train/downstream/TARGET_STATISTICS.md) explains the stats file.
For a complete pretraining and two-adapter run, follow
[the worked example](train/WORKED_EXAMPLE.md).
Run the commands from the repository root. This is research code; the checked-in
recipes still require access to data and compatible CUDA kernels.

## Install the training environment

The same recipe supports **Linux x86_64** and **Linux aarch64/ARM64**. It installs
Python 3.12, CUDA toolkit 13.0, PyTorch 2.9.1, the Mamba1 selective-scan extension,
causal-conv1d, fused Mamba2, Mamba3 SISO dependencies, plotting and tuning tools.
Choose **one** installation method below. Run from the repository root.

| Target | Independently tested route |
| --- | --- |
| Linux x86_64 with an NVIDIA GPU | A: native Conda (tested on RTX 3050 Ti) |
| Linux ARM64 with a GB10 GPU | B: Docker (tested on a02c) |

Both routes share the installer and pinned Python versions. The full
[validation matrix](environments/README.md#validation-matrix) records the scope.

Requirements: Linux with glibc 2.34+ (for example Ubuntu 22.04/24.04), an NVIDIA
GPU and a working host driver from the 580 series or newer. The full preflight
targets Ampere or newer GPUs (compute capability 8.0+).
The native route needs Conda/Miniforge, Git and a C++ compiler; the container
route needs Docker and NVIDIA Container Toolkit. Both methods supply their own
CUDA compiler, so the host's system CUDA installation can stay as it is.

```bash
uname -m
nvidia-smi
git clone https://github.com/alillari/CLAS12.git
cd CLAS12
```

`uname -m` should print `x86_64` or `aarch64`. This is the **CPU** architecture;
the GPU's compute capability is a separate choice. a02c/a518 use ARM64 CPUs
and GB10 GPUs (SM 12.1). An x86_64 machine can have a different NVIDIA GPU.

### A. Fresh native Conda environment

If Conda is not installed, use the [copy-paste Miniforge bootstrap](environments/README.md#host-prerequisites)
for your CPU architecture first. On Ubuntu, `build-essential` supplies the C++
compiler; Git is needed to fetch the pinned scheduler and kernel sources.
Create a new environment rather than modifying a working training environment:

```bash
conda env create -f environments/cuda13.yml
conda activate clas12-fm
source environments/runtime.sh
bash environments/install.sh
python environments/verify.py --output /tmp/clas12-environment.json
```

The environment file installs the CUDA toolkit **inside this Conda environment**.
`runtime.sh` selects that compiler and assembler. The installer downloads the
correct CUDA-enabled PyTorch wheels for your CPU architecture and builds pinned
kernel sources for the detected GPU. `MAX_JOBS=2` bounds compiler parallelism;
initial installation downloads several GB and compiles CUDA code.

Every subsequent shell/job should activate and configure the same environment:

```bash
conda activate clas12-fm
source environments/runtime.sh
```

If the name `clas12-fm` already exists, choose a new prefix instead of overwriting
it: `conda env create --prefix /absolute/path/to/new-env -f environments/cuda13.yml`,
then activate that prefix. Dataset creation from HIPO uses separate tools;
this environment runs the prepared RaggedMmap training/evaluation workflows.

### B. Build a fresh Docker image

Build on the actual target machine, without CPU emulation. The public CUDA
**devel** base supplies `nvcc`; no private team image is required. This example
builds for the first visible GPU, including PTX for compatible later devices:

```bash
export CLAS12_GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
docker build -f environments/Dockerfile \
  --build-arg "TORCH_CUDA_ARCH_LIST=$CLAS12_GPU_ARCH+PTX" \
  -t clas12:cuda13 .

mkdir -p "$PWD/artifacts"
docker run --rm -it --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace:ro" -v "$PWD/artifacts:/results" \
  -w /workspace -e PYTHONPATH=/workspace \
  -e MPLCONFIGDIR=/results/cache/matplotlib \
  -e XDG_CACHE_HOME=/results/cache -e TRITON_CACHE_DIR=/results/cache/triton \
  clas12:cuda13 bash
```

Inside the container:

```bash
source environments/runtime.sh
python environments/verify.py --output /results/environment.json
```

For training, add read-only dataset mounts and use their **container paths** in
YAMLs, with outputs under `/results`. See [Remote jobs](train/REMOTE_RUNS.md).
Record the immutable image ID with `docker image inspect clas12:cuda13`; tags
are local names. For mixed GPU machines, pass an explicit semicolon-separated
`TORCH_CUDA_ARCH_LIST` containing the capabilities you intend to support.

### What counts as a successful setup?

The verifier imports the actual dependencies and public trainer CLIs, then
runs Mamba1, fused Mamba2, Mamba3 SISO and Transformer CUDA forward/backward,
an optimizer update, and strict checkpoint save/reload. It fails if a selected
backend fails or CUDA is unavailable; it does not silently pass skipped tests.
The JSON receipt records architecture, GPU, package versions and kernel build
provenance. Inspect its `status` and `backbones` fields before training.
The first GPU preflight can take several minutes while Triton compiles and
autotunes kernels; later runs can reuse the same environment's cache.

For a deliberate import-only check on a machine without a GPU:

```bash
python environments/verify.py --imports-only --output /tmp/clas12-imports.json
```

That result is labelled `imports_only` and does not validate training. These
checks do not validate dataset alignment, convergence, multi-GPU execution or
every production model width. Continue with the [worked example](train/WORKED_EXAMPLE.md).
The [environment reference](environments/README.md) explains pins, build options,
the validation matrix and recovery from compiler/kernel errors.

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
