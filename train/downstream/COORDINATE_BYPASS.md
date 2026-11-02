# Momentum adapter coordinate bypass

Set `coordinate_bypass: true` in a regression model YAML and train with
`--usepretrain --pretrained_ckpt /path/to/backbone.ckpt` to supply both inputs:

```text
serialized input ── frozen backbone ── depth mix ── input_proj ─────────────┐
                └─ AdapterOnly embedder → input refinement → norm         │
                                                      → coordinate_proj ─┤
                                              hitwise sum ←──────────────┘
                                                 → shared Mamba → norm
                                                 → masked pooling → MLP
```

The coordinate branch calls the same input-processing code as AdapterOnly:
the positional embedder, any configured `mamba_embedder_layers`, and
`embedder_norm`. It then uses its own LayerNorm and linear projection into the
shared adapter width. With the current trainer, there are zero input refinement
layers, a 256-dimensional shared sequence, and one shared Mamba2 layer.
This combines both input modes before sequence refinement and pooling; there
is one momentum prediction and one regression loss.

The two projections followed by addition are equivalent to concatenating the
two normalized representations and applying a joint linear projection. The
sequence remains `(batch, hits, 256)`: no interleaving, second serialization,
or duplication of hits. No backbone changes or new dataset products are needed.

## Coordinate contract

The branch receives exactly the tokens that an AdapterOnly model would receive.
For `center_only` / `pos_only`, these are the loader's normalized `[eta, phi, r]`
coordinates (including its existing data scale), derived from source XYZ. They
are not source Cartesian values in centimeters. For `clas12_pos_plus_aux_v1` /
`pos_plus_aux`, the branch uses the complete packed center-plus-auxiliary token,
as AdapterOnly does.

Both paths use the same serialized tensor and hit mask. Padding is removed from
the coordinate input and frozen features before fusion, and fused padding is
zeroed before Mamba refinement. The backbone remains in evaluation mode under
`torch.no_grad()`; its weights do not receive gradients. The AdapterOnly input
branch and fusion projection follow their normal trainable/fixed-parameter
contracts, and the shared adapter is trained as usual.

## Matched configurations

`scripts/configs/mamba_clas12_momentum_adapteronly.yaml` contains three controls:

| Configuration | Backbone | Coordinate input |
| --- | --- | --- |
| `clas12_momentum_adapteronly_debug` | unused | AdapterOnly path |
| `clas12_momentum_pretrained` | frozen | through backbone only |
| `clas12_momentum_pretrained_coordinate_bypass` | frozen | backbone and AdapterOnly path |

The two pretrained presets inherit the same data, filtering, architecture,
optimization, and validation defaults. Their differences are the bypass flag
and output names. Before running, adjust the inherited backbone dimensions,
embedding, and serialization to match the selected pretrained checkpoint and
verify the data and target-statistic paths. These presets do not select a
checkpoint automatically.

```bash
python train/downstream/train_track_regression.py \
  --yaml_config scripts/configs/mamba_clas12_momentum_adapteronly.yaml \
  --config clas12_momentum_pretrained_coordinate_bypass \
  --usepretrain --pretrained_ckpt /path/to/backbone.ckpt \
  --eventnumber 50000 --train_batch_size 128 --seed 42
```

Use `clas12_momentum_pretrained` with the same arguments for the frozen-feature
control. Keep checkpoint, cohorts, seeds, label budget, optimizer steps, and
validation selection matched. Include the AdapterOnly control when assessing
whether frozen features add information beyond the direct input. The bypass
activates the existing AdapterOnly input modules and adds a projection
(`input_dim * 256 + 256` linear parameters plus `2 * input_dim` LayerNorm
parameters: 33,280 new parameters at input width 128); report parameter counts
as well as momentum metrics. Optional branch initialization preserves
the CPU random stream used by shared initialization and data sampling.

## Checkpoints and evaluation

The flag defaults to `false`, preserving baseline parameters, forward behavior,
and historical checkpoint loading. New checkpoints and training artifact
summaries record `coordinate_bypass`. Loading rejects baseline/bypass
architecture mismatches and incomplete bypass weights. Bypass checkpoints use
strict state loading. Historical baseline checkpoints without the flag remain
loadable.

Use the matching model YAML/config and `use_pretrained_backbone: true` with
the original backbone checkpoint in the normal regression evaluator. Its head
builder reads the same bypass flag. All training, validation, legacy momentum
wrappers, and standalone evaluation paths pass both aligned inputs to the head.

## Verification

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest \
  train.downstream.tests.test_coordinate_bypass -v
```

The focused tests exercise both input branches, equivalence to the actual
AdapterOnly input path, row alignment, padding, a real CPU Mamba optimizer step,
frozen-backbone gradients, validation, checkpoint compatibility, evaluator
construction, and matched recipe resolution. CPU checks establish functional
correctness using the repository's PyTorch Mamba fallback with fused kernels
disabled in the tests; a representative CUDA/data pilot and held-out physics evaluation
are still needed to measure the experiment's performance.
