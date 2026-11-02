# Training environment reference

Start with the copy-paste commands in [SETUP](../SETUP.md). Native Conda and
Docker use the same Python installer. The Conda route supplies a fresh toolkit;
the container route starts from a public CUDA devel image. Neither relies on
the team's existing `pp_collision:alessio-latest` image.

## Host prerequisites

For a new Ubuntu machine with an NVIDIA driver already installed:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl ca-certificates
```

If Conda is absent, this pinned Miniforge bootstrap selects the native CPU
installer and verifies its published checksum. Choose an unused installation
prefix; do not run it over an existing Conda installation:

```bash
export CLAS12_MINIFORGE_FILE="Miniforge3-25.3.1-0-Linux-$(uname -m).sh"
export CLAS12_MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/download/25.3.1-0
curl -fL "$CLAS12_MINIFORGE_URL/$CLAS12_MINIFORGE_FILE" -o "/tmp/$CLAS12_MINIFORGE_FILE"
curl -fL "$CLAS12_MINIFORGE_URL/$CLAS12_MINIFORGE_FILE.sha256" -o "/tmp/$CLAS12_MINIFORGE_FILE.sha256"
(cd /tmp && sha256sum -c "$CLAS12_MINIFORGE_FILE.sha256")
bash "/tmp/$CLAS12_MINIFORGE_FILE" -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

Then follow the native block in [SETUP](../SETUP.md#a-fresh-native-conda-environment).
The Docker alternative needs [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)
and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host. Driver and Docker administration belong to the machine owner;
the repository installer creates a user environment and does not alter them.

## Runtime contract

| Component | Recipe | Purpose |
| --- | --- | --- |
| OS/CPU | Linux x86_64 or aarch64 | Native wheels and compiled libraries must match the host CPU. |
| Python | 3.12 | Common Python ABI for both platforms. |
| PyTorch | 2.9.1 from the `cu130` index | CUDA-enabled wheels on both CPU architectures. |
| Compiler toolkit | CUDA 13.0 | Builds extensions against the same CUDA major version as PyTorch. |
| Triton | 3.5.1 | Runs fused Mamba2 and Mamba3 SISO kernels. |
| Mamba | `e9594ce1c732d97440f0332fdc43170a2294dbfa` | Includes Mamba1, Mamba2 and Mamba3 source. |
| causal-conv1d | `4f6ae4e26ae5fe8af9372f8d312ab25cc4595223` | Compiled causal-convolution kernels. |
| Cosine scheduler | `12d03c07553aedd3d9e9155e2b3e31ce8c64081a` | Scheduler imported by the training entrypoints. |

[PyTorch's release instructions](https://pytorch.org/get-started/previous-versions/)
document the 2.9.1 CUDA 13 wheel index. NVIDIA specifies a minimum 580-series
driver for CUDA 13 in its [compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
`nvidia-smi` shows the driver's supported CUDA version, not the installed `nvcc`
version. `runtime.sh` selects the Conda toolkit when one is present, otherwise
uses `CUDA_HOME`, and points Triton at that toolkit's assembler.

`requirements.txt` pins the shared direct Python dependencies, including Plotly,
which the dataset loader imports even when voxelization is disabled. The
shared `constraints-cu130.txt` also pins the resolved Python dependencies; both
CPU architectures resolved the same versions. The installer applies it to
the complete project dependency graph after bootstrapping CUDA PyTorch from
its own wheel index. It records versions, not interchangeable x86/ARM
binary wheels. The upstream Mamba package declares TileLang, CUTLASS and TVM dependencies; they are
installed to satisfy its package metadata. CLAS12's Mamba3 integration executes
SISO only. Their installation does not establish MIMO support.
`apache-tvm-ffi` is pinned to 0.1.6 because 0.1.12 fails during TileLang 0.1.8
import, even though the declared dependency bounds allow that combination.
The upstream [compatibility issue](https://github.com/state-spaces/mamba/issues/986)
documents the same failure. Cached single-token decoding and MIMO remain outside
this recipe's GPU validation.

## GPU build choices

CPU architecture does not determine GPU architecture. The native installer
detects the first visible GPU; Docker builds cannot assume GPU access, so
`TORCH_CUDA_ARCH_LIST` must be supplied at build time. For example, `8.6+PTX`
targets an RTX 3050 Ti, while `12.1+PTX` targets GB10. Use a quoted list such as
`8.6;12.1+PTX` for a deliberately shared image. Do not use `--platform` CPU
emulation to claim ARM or x86 GPU validation.

The upstream build scripts hard-code a large GPU architecture list.
`build_extensions.py` fetches exact commits and replaces only that build list
with PyTorch's target-architecture flags. The kernel sources are unchanged.
This avoids compiling unused GPU targets on a laptop. A missing expected build
marker or a source-revision mismatch is fatal. The build records its source
commits, adjustment and flags in `clas12-kernel-build.json` under the environment
prefix. `MAX_JOBS` controls build parallelism; increase it only when sufficient
RAM is available.

The builder forces local source builds and uses `--no-build-isolation` so both
extensions compile against the selected CUDA PyTorch. For Mamba, it explicitly
sets `MAMBA_KEEP_CUDA_BUILD=TRUE`: upstream otherwise omits the selective-scan
extension needed by Mamba1. See the pinned upstream
[installation flags](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa).
The scheduler is fetched at an exact commit as well.

## Verification and provenance

`verify.py` requires a GPU by default and validates each requested backend with
finite loss/gradients, an optimizer step and strict checkpoint reload. Mamba2
must use its fused path. No CPU approximation counts as a successful GPU check.
It also invokes the actual pretraining, finding and momentum `--help` entrypoints.

To check a deliberately smaller backend set, specify it explicitly:

```bash
python environments/verify.py --backbones mamba1 mamba2 transformer \
  --output /tmp/clas12-selected-backends.json
```

The receipt lists exactly which backends passed. That command does not validate
Mamba3. `--imports-only` similarly produces a distinct `imports_only` status.
Failure receipts preserve the error and any earlier successful backend checks.

The installer writes `clas12-pip-freeze.txt` and kernel provenance under the
environment prefix. Keep the verifier's JSON with each run, along with the
source commit and resolved configuration. The freeze captures installed packages;
local build-directory URLs in it are provenance, not portable reinstall URLs.
Recreate kernels using the installer and its pinned upstream commits.

The Dockerfile pins the public CUDA base to a multi-architecture manifest digest
containing both amd64 and arm64 variants. For a Docker run also record the
resulting immutable image ID. APT packages are not a bit-for-bit image lock;
the resolved image ID identifies the actual built environment. A later image
rebuild still needs its own GPU preflight.

## Troubleshooting

- **`No CUDA GPU available`:** check the host driver and GPU visibility. Docker
  needs `--gpus all` and NVIDIA Container Toolkit. A Conda CUDA toolkit does not
  install or replace a host driver.
- **`sm_121a is not defined`:** source `environments/runtime.sh` in the actual
  job shell. It selects CUDA 13's `ptxas` for Triton and clears a stale
  `TRITON_OVERRIDE_ARCH`. Use a fresh per-environment/job `TRITON_CACHE_DIR`
  after changing compilers. The [upstream issue](https://github.com/triton-lang/triton/issues/10331)
  describes the GB10 assembler failure.
- **`nvcc` or build failures:** confirm `CUDA_HOME/bin/nvcc --version` reports
  13.0 and that a C++ compiler is available. Reduce `MAX_JOBS` for RAM pressure.
  Runtime-only CUDA images do not contain the compiler; use the supplied devel
  Dockerfile or Conda toolkit.
- **No matching wheel:** verify Linux, Python 3.12 and native CPU architecture.
  Do not install an x86 wheel into an ARM environment or drop CUDA PyTorch to
  satisfy a generic dependency resolver.
- **`selective_scan_cuda` / `causal_conv1d_cuda` import or ABI errors:** rebuild
  the extensions in a fresh environment after a PyTorch/CUDA change. Copying
  `.so` files from another machine is not a compatible installation method.
- **GPU out of resources:** a minimal preflight is not a guarantee that every
  width/state/chunk size fits the target GPU. Keep the requested architecture
  and test the real training configuration before a campaign.
- **PyTorch warns about GB10 capability 12.1 versus a maximum listed 12.0:**
  this pinned PyTorch build can emit that warning. Treat the actual kernel
  preflight and the tested matrix below as the validation evidence; the warning
  alone does not prove either failure or compatibility for untested operations.

## Validation matrix

Validated on 2026-11-02 with PyTorch `2.9.1+cu130`, Triton 3.5.1, the pinned
upstream kernels above, and TVM-FFI 0.1.6:

| Native CPU / GPU | Fresh environment | GPU preflight | Bounded onboarding sequence |
| --- | --- | --- | --- |
| Linux x86_64 / RTX 3050 Ti, SM 8.6, 4 GB | New Conda prefix with CUDA toolkit 13.0.3 | All four backends passed | Passed |
| Linux aarch64 / NVIDIA GB10, SM 12.1 | New Docker image from the public CUDA 13.0.1 devel base | All four backends passed | Passed |

Both environments compiled their own causal-convolution and selective-scan
extensions; no libraries were copied from the existing team environment. The
resolved Python versions matched on both CPU architectures, and the final
constraints passed resolver checks on each. Explicit import-only mode passed
without GPU access; the default verifier correctly failed in that same CPU-only
container.

The bounded sequence generated stats from 64 accepted training tracks,
pretrained a width-256/depth-12/k1 Mamba1 for ten updates, then trained both
frozen-backbone adapters for ten updates each. It reloaded/evaluated the adapters
and generated the pretraining, finding-validation and momentum diagnostic plots.
Validation/evaluation reused a tiny 16-sample slice: these checks establish
execution, not convergence, independent test accuracy or campaign performance.

The ARM Conda route and x86 Docker build were not independently executed;
they share the tested installer and version constraints. The Docker base
manifest was checked to contain both CPU architectures. Other GPUs, wider
models, distributed training and a different driver/compiler combination need
their own target-runtime checks. Existing prepared-image campaign records
remain in [Mamba support](../train/MAMBA_SUPPORT.md).
