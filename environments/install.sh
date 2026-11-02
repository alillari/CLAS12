#!/usr/bin/env bash
# Run with a fresh Python 3.12 environment activated and CUDA 13.0 available.
set -euo pipefail
ENV_RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${CUDA_HOME:?Set CUDA_HOME to the CUDA 13.0 toolkit prefix}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=${MAX_JOBS:-2}
python - <<'PY'
import os, platform, shutil, subprocess, sys
if platform.system() != 'Linux' or platform.machine() not in {'x86_64','aarch64'}:
    raise SystemExit('This recipe supports Linux x86_64 and Linux aarch64')
if sys.version_info[:2] != (3,12):
    raise SystemExit('Activate a fresh Python 3.12 environment first')
libc, version = platform.libc_ver()
if libc != 'glibc' or tuple(int(x) for x in version.split('.')[:2]) < (2,34):
    raise SystemExit('Use glibc 2.34+ (for example Ubuntu 22.04+) or the supplied Docker image')
if sys.prefix == sys.base_prefix and not os.environ.get('CONDA_PREFIX'):
    raise SystemExit('Activate a Conda environment or virtualenv; do not install into system Python')
if not shutil.which('git') or not shutil.which(os.environ.get('CXX','c++')):
    raise SystemExit('Install Git and a C++ compiler before building the environment')
nvcc = subprocess.check_output([os.path.join(os.environ['CUDA_HOME'],'bin/nvcc'),'--version'],text=True)
if 'release 13.0,' not in nvcc:
    raise SystemExit('This pinned recipe requires the CUDA 13.0 compiler, not just a recent NVIDIA driver')
print(nvcc)
PY

CLAS12_ENV_CONSTRAINTS="$ENV_RECIPE_DIR/constraints-cu130.txt"
test -f "$CLAS12_ENV_CONSTRAINTS"
# Bootstrap CUDA PyTorch from its own index before resolving the complete
# project graph against PyPI. The second phase constrains all final versions.
python -m pip install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install --constraint "$CLAS12_ENV_CONSTRAINTS" -r "$ENV_RECIPE_DIR/requirements.txt"

# Container image builds have no GPU: pass this explicitly as a build argument.
# Native installs detect the actual GPU, independently of the CPU architecture.
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST=$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('No GPU visible. Set TORCH_CUDA_ARCH_LIST explicitly for an offline/image build.')
major, minor = torch.cuda.get_device_capability()
print(f'{major}.{minor}+PTX')
PY
)
  export TORCH_CUDA_ARCH_LIST
fi
python "$ENV_RECIPE_DIR/build_extensions.py"
python -m pip check
python -c 'import mamba_ssm, selective_scan_cuda, causal_conv1d_cuda'
python -m pip freeze --all > "$(python -c 'import sys; print(sys.prefix)')/clas12-pip-freeze.txt"
printf '\nInstalled. Run: python environments/verify.py --output /path/to/environment.json\n'
