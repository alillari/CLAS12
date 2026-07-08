# Setup Notes

This document describes the current working setup for this early-stage CLAS12
foundation-model research repository. The code is still changing, so treat these
notes as a starting point rather than a stable installation manual.

The repository builds on FM4NPP code and training patterns. Some inherited
scripts and configuration options are still present for compatibility while the
CLAS12 workflow is being developed.

## Prerequisites

### Hardware

- NVIDIA GPU with CUDA support for training.
- Sufficient GPU memory for the selected model and batch size.
- Local or shared storage for CLAS12 RaggedMmap data, checkpoints, and logs.

For quick debugging, start with smaller event counts and batch sizes before
scaling to long training runs.

### Software

- Python 3.10 or newer.
- PyTorch with CUDA support, if training on GPU.
- Linux or a comparable HPC environment.
- SLURM, only if using the provided batch scripts.

## Installation

Clone the repository and enter the project root:

```bash
git clone <repository-url>
cd CLAS12
```

Create and activate an environment:

```bash
conda create -n clas12-fm python=3.10
conda activate clas12-fm
```

Install PyTorch for your CUDA version. For CUDA 12.1, one option is:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

Depending on your platform, `mamba-ssm`, `causal-conv1d`, or `triton` may need
versions matched to your CUDA, PyTorch, and compiler stack.

## Verify Imports

From the repository root:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import fm4npp; print('fm4npp package import ok')"
```

If imports fail from a script launched outside the repository root, set
`PYTHONPATH` explicitly:

```bash
export PYTHONPATH=/absolute/path/to/CLAS12:$PYTHONPATH
```

## Data Layout

Current CLAS12 downstream configs expect RaggedMmap-style directories. The exact
data paths are local to the machine or cluster where experiments are run, so
review and edit the YAML files before launching a job.

For CLAS12 regression workflows, the data root commonly needs directories like:

```text
data_root/
|-- features_pretrain/
|-- features_test/
|-- seg_target_pretrain/
|-- seg_target_test/
|-- reg_target_pretrain/
`-- reg_target_test/
```

Some code paths may also expect optional PID targets:

```text
data_root/
|-- pid_target_pretrain/
`-- pid_target_test/
```

Regression target statistics are configured separately, usually through a file
such as:

```text
stats/regression_target_stats.json
```

## Configuration Files

The main CLAS12-oriented configs currently live in `scripts/configs/`:

```text
scripts/configs/
|-- mamba_clas12_momentum_adapteronly.yaml
|-- mamba_clas12_track_regression_adapteronly.yaml
`-- mamba_clas12_track_regression_pretrained.yaml
```

Before running, check at least these fields:

```yaml
data_root: /path/to/clas12/raggedmmap
data_root_train: /path/to/clas12/raggedmmap
data_root_test: /path/to/clas12/raggedmmap
stat_dir: /path/to/stats
regression_target_stats: /path/to/stats/regression_target_stats.json
downstream_dir: ./downstream_log/<run-name>
checkpoint_dir: ./downstream_log/<run-name>/checkpoints
```

The inherited FM4NPP configs are still present:

```text
scripts/configs/
|-- mamba_pretrain.yaml
`-- mamba_tracking.yaml
```

Use those only after checking that their dataset assumptions, model settings,
and paths match the experiment you intend to run.

## Running CLAS12 Downstream Experiments

Run commands from the repository root unless a script says otherwise.

### Adapter-Only Track Regression

```bash
python train/downstream/train_track_regression.py \
    --yaml_config scripts/configs/mamba_clas12_track_regression_adapteronly.yaml \
    --config clas12_track_regression_adapteronly_debug \
    --run_num run0 \
    --eventnumber 50000 \
    --train_batch_size 32
```

### Adapter-Only Momentum Regression

```bash
python train/downstream/train_momentum_regression.py \
    --yaml_config scripts/configs/mamba_clas12_momentum_adapteronly.yaml \
    --config clas12_momentum_adapteronly_debug \
    --run_num run0 \
    --eventnumber 50000 \
    --train_batch_size 32
```

### Track Regression With a Pretrained Backbone

The pretrained CLAS12 config expects the checkpoint path on the command line:

```bash
python train/downstream/train_track_regression.py \
    --yaml_config scripts/configs/mamba_clas12_track_regression_pretrained.yaml \
    --config clas12_track_regression_pretrained \
    --run_num run0 \
    --eventnumber 50000 \
    --train_batch_size 32 \
    --usepretrain \
    --pretrained_ckpt /absolute/path/to/backbone.ckpt
```

## Inherited FM4NPP Workflows

The repository still contains inherited FM4NPP pretraining and track-finding
entry points:

```bash
python -m train.pretrain.nppmamba.train_multi_gpu_mamba1 \
    --yaml_config scripts/configs/mamba_pretrain.yaml \
    --config mamba_5m \
    --run_num run0
```

```bash
python train/downstream/track_finding_trainer.py \
    --yaml_config scripts/configs/mamba_tracking.yaml \
    --config mamba_5m_downstream \
    --run_num run0
```

These commands may require additional path, dataset, or script updates before
they are useful for CLAS12-specific experiments.

## SLURM Scripts

Generic inherited SLURM scripts are available in `scripts/run/`:

```text
scripts/run/
|-- submit_mamba_pretrain.sh
|-- submit_downstream_mamba.sh
`-- train_mamba_direct.py
```

Before using them, update:

- cluster account and partition settings,
- GPU and memory requests,
- Python or conda environment path,
- YAML config path,
- checkpoint and output directories.

## Monitoring Runs

Logs and checkpoints are generally written under the configured
`downstream_dir`, `checkpoint_dir`, or pretraining checkpoint directory.

Useful checks:

```bash
tail -f /path/to/log/file.log
ls /path/to/checkpoints
nvidia-smi
```

## Troubleshooting

### Import Errors

Run from the repository root or set:

```bash
export PYTHONPATH=/absolute/path/to/CLAS12:$PYTHONPATH
```

### CUDA or Mamba Installation Errors

Check the installed PyTorch and CUDA combination:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
nvidia-smi
```

If `mamba-ssm` or `causal-conv1d` fails to install, use versions compatible with
your PyTorch and CUDA stack.

### Out-of-Memory Errors

Reduce one or more of:

```yaml
batch_size: 32
local_batch_size: 32
valid_batch_size: 32
limit_size: 10000
```

You can also pass smaller values through the training CLI:

```bash
--eventnumber 10000 --train_batch_size 16
```

### Missing Checkpoints

For pretrained runs, provide an absolute checkpoint path:

```bash
--usepretrain --pretrained_ckpt /absolute/path/to/backbone.ckpt
```

The current pretrained CLAS12 config intentionally does not hardcode a
checkpoint path.

## Attribution

This setup inherits structure and terminology from FM4NPP while the CLAS12
workflow is being developed. Preserve FM4NPP attribution when using inherited
components, and document CLAS12-specific assumptions in the relevant config or
script when they become stable.
