# Mamba backbone and adapter selections

Pretraining and the point-classification, track-finding, and track-regression
trainers share the same backbone builder. Downstream Mamba blocks have an
independent selection. Existing launch module names remain valid.

```yaml
backbone_version: mamba3     # mamba1, mamba2, or mamba3
adapter_version: mamba2      # independent; default remains mamba2
```

`mambaversion` remains a supported legacy backbone setting. An explicit
`backbone_version` takes precedence. Omitting both keeps the historical
pretraining default of Mamba1 and downstream fallback of Mamba2; existing
downstream YAMLs continue to select their configured Mamba1 backbones.
Longformer and Linformer backbone selections also remain supported downstream.
Unknown versions fail immediately.

The adapter selection controls Mamba refinement and input-embedding layers.
It does not replace an attention decoder. For example, a tracking configuration with
`num_adapter_layers: 0` and `num_embedder_layers: 0` has no Mamba adapter layers,
regardless of `adapter_version`. Set the layer count explicitly when testing
Mamba3 refinement. Point classification retains its `use_attention_head` switch;
set it to `false` to select a Mamba refinement head.

## Block configuration

Backbone dimensions still use `embed_dim` and `num_layers_backbone`. Existing
top-level `d_state`, `d_conv`, `expand`, `headdim`, `ngroups`, and `chunk_size`
settings are forwarded to the backbone. A `backbone_` prefix takes precedence
over the equivalent top-level block option. Adapter overrides always use the
`adapter_` prefix and do not change the backbone.

For example, to use a pretrained Mamba3 backbone with a Mamba2 adapter:

```yaml
backbone_version: mamba3
embed_dim: 256
num_layers_backbone: 12
d_state: 128
expand: 2
headdim: 64
ngroups: 1
chunk_size: 64
is_mimo: false

adapter_version: mamba2
adapter_d_state: 64
adapter_headdim: 64
```

Match the backbone dimensions, embedding, target `klen`, and block settings to
the pretrained checkpoint. To test a Mamba3 adapter while keeping a Mamba1
backbone, change only the adapter selection and its block options:

```yaml
mambaversion: mamba1
adapter_version: mamba3
adapter_d_state: 64
adapter_headdim: 64
adapter_chunk_size: 64
adapter_rope_fraction: 0.5
adapter_is_mimo: false
```

Mamba3 also accepts `is_outproj_norm` and `rope_fraction` (with the same
backbone/adapter prefix conventions). This integration supports **SISO only**;
`is_mimo: true` is rejected. Mamba3 has no `d_conv`; the common constructor
argument is discarded. Mamba2-only options such as `use_mem_eff_path` are
rejected for Mamba3 rather than silently ignored. For Mamba2/3,
`embed_dim * expand` (or the adapter's corresponding width) must be divisible
by `headdim`, and the number of heads must be divisible by `ngroups`.

The regression head's historical width and layer count are unchanged. Its
`adapter_` block overrides apply to both refinement and input-embedding blocks;
their model widths remain those of their respective input paths. The same
rule applies to the classification and tracking heads.

## Pretraining

`scripts/configs/mamba_pretrain.yaml` contains `mamba_5m`, `mamba2_5m`, and
`mamba3_siso`. Dataset, statistics, and output paths must be set for the target
environment before using these inherited example recipes.

```bash
python -m train.pretrain.nppmamba.train_multi_gpu_mamba1 \
  --yaml_config scripts/configs/mamba_pretrain.yaml \
  --config mamba3_siso --run_num siso0 --root_dir /path/to/artifacts
```

The next-point objective and `klen * 3` output remain unchanged. Each backbone's
`return_z=True` returns `(None, accumulated_layer_features, positions)`, allowing
existing frozen-backbone adapters to use the same feature interface.

Before this integration, the configuration named `mamba2_5m` actually constructed
Mamba1. It now selects Mamba2 explicitly. To restore a checkpoint produced by
that old recipe, use a Mamba1 configuration matching its actual dimensions
(including its `d_state: 128`), and omit Mamba2-only options. A name on disk
cannot identify the architecture reliably; mismatched checkpoints are rejected.

## Mamba3 runtime

Mamba3 is optional. Existing Mamba1/2 usage does not require installing a new
package. Use an upstream `mamba-ssm` source revision containing
`mamba_ssm.modules.mamba3`, with its required dependencies. Upstream currently
requires Triton 3.5 or newer; consult and pin the upstream source revision and
runtime together in the target training environment:

- https://github.com/state-spaces/mamba
- https://github.com/state-spaces/mamba/blob/main/setup.py

When building an environment that also runs Mamba1, retain/build the
`selective_scan_cuda` extension as instructed upstream (`MAMBA_KEEP_CUDA_BUILD=TRUE`
for current source builds). Do not replace a working training environment
without validating all selected versions.

Mamba3 blocks keep FP32 parameters, execute their full-sequence kernel inside a
BF16 CUDA autocast boundary, and return the caller's dtype. This also covers
frozen-feature extraction and evaluation, where the existing trainer may not
use autocast. CPU construction and checkpoint loading are supported; CPU
forward execution raises a clear error. There is no approximate CPU Mamba3
fallback. Existing external normalization, residual additions, and dropout are
retained. Legacy custom initialization skips Mamba3 parameters, preserving
upstream SSM initialization, and its no-weight-decay flags are honored.

Local construction was checked with PyTorch 2.5.1+cu121, mamba-ssm
2.3.2.post1, and Triton 3.7.0.

CUDA validation passed on **2026-10-27** on a02c's NVIDIA GB10 (SM 12.1), using
the unchanged `pp_collision:alessio-latest` image (`bbc43e230b61`): PyTorch
`2.8.0a0+5228986c39.nv25.05`, CUDA 12.9, mamba-ssm 2.3.2.post1, causal-conv1d
1.6.2.post1, and Triton 3.7.1. All five CUDA tests ran without skips or failures,
including Mamba3 pretraining with state size 128 and mixed backbone/adapter
combinations at widths 256 and 512. Revalidation against the current pretraining
integration also exercised actual Mamba3 center/auxiliary training, width transfer,
validation, and checkpoint resume. The tests used isolated source copies and
caches, with no changes to the Docker image, primary checkout, or campaigns.
Execute the suite on other intended GPU/runtime combinations before a campaign.
Cached single-token decoding and MIMO are outside this integration's tested scope.

## Checkpoints and validation

Mamba1/2 parameter names and module nesting are unchanged. New checkpoints add
architecture metadata without changing the existing model-state keys. Loading
old checkpoints without this metadata remains supported; block-family checks
prevent a non-strict adapter load from discarding a different Mamba version.
New metadata also checks block options that may affect behavior without
changing weight shapes. Frozen-backbone adapter loading also checks the saved
backbone architecture when that metadata is available. Pretraining checkpoints
now save and restore scheduler state as well as model and optimizer state;
older files remain loadable. The current rank-based width-transfer and optional
offline W&B paths are preserved. Matrix-group scaling applies to all matrix
subgroups when Mamba3 adds no-decay groups. Optimizer parameter names are saved;
known older layouts are remapped with shape checks and ambiguity rejection.
Epoch-based iteration is retained; the data cursor is not restored exactly.

The repository's local Mamba2 implementation is retained for checkpoint
compatibility. Its manual fallback is a simplified EMA recurrence, not a
numerically equivalent reference implementation of the fused SSD path. Do not
use CPU fallback results to claim equivalence to official fused Mamba2. The
explicit Mamba2 `use_mem_eff_path` / `adapter_use_mem_eff_path` options control
that existing backend selection.

Run CPU construction, configuration, checkpoint, optimizer, and trainer tests:

```bash
python -m unittest train.downstream.tests.test_mamba_support -v
python -m unittest discover -s train/downstream/tests -v
python -m unittest discover -s train/pretrain/tests -v
```

Run real kernel tests on a CUDA-enabled target:

```bash
python -m unittest train.downstream.tests.test_mamba_support.MambaCUDATest train.pretrain.tests.test_mamba_integration.PretrainingCUDATest -v
```

The CUDA suite checks Mamba2/3 pretraining forward/backward and reload, plus
Mamba3 AdapterOnly, frozen features, and classification
and tracking heads. It also checks standard Mamba3 state size 128 and mixed
Mamba1/2/3 backbones and Mamba2/3 adapters, including a width-512 backbone with
a width-256 head. Mamba2 must use its fused path during CUDA validation.
The suite skips when CUDA is unavailable. Mamba3-specific tests also skip if
its optional dependencies are absent. No production training run is launched
by these tests.
