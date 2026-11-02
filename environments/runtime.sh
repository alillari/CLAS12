# Source after activating the environment, including in each new job shell.
if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/nvcc" ]]; then
  export CUDA_HOME="$CONDA_PREFIX"
fi
if [[ ! -x "${CUDA_HOME:-}/bin/ptxas" ]]; then
  printf 'Set CUDA_HOME to the CUDA 13.0 toolkit prefix before sourcing runtime.sh\n' >&2
  return 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
unset TRITON_OVERRIDE_ARCH
export MPLBACKEND=Agg
