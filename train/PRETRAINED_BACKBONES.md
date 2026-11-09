# Pretrained backbone selection

Select a backbone independently of the adapter recipe. The catalog at
`scripts/configs/pretrained_backbones.yaml` contains campaign 5 IDs and paths
relative to a checkpoint root. Existing checkpoint directories can stay where
they are; no renaming, symlinks, or per-model Python changes are needed.

From the repository root on a02c:

```bash
export CLAS12_PRETRAINED_ROOT=/home/alessio/ML-work/pretrained-FMs
python -m fm4npp.pretrained --list
python -m fm4npp.pretrained --backbone campaign5_m3
```

`--list` needs no checkpoint access or CUDA. Selecting an ID reads the checkpoint
on CPU, verifies its optional catalog SHA-256, and prints the resolved descriptor.
The models are `campaign5_m1` (64), `campaign5_m2` (128), `campaign5_m3` (256),
and `campaign5_m4` (512). These tier labels denote widths; their block family is
Mamba3. All four have 12 layers, state size 128, and a 20-output pretraining head.

## Single runs

The public track-finding, track-regression, and point-classification CLIs accept
`--backbone ID`, `--backbone-catalog FILE`, and `--backbone-root DIR`. The bundled
catalog is the default. A selection or an explicit `--pretrained_ckpt` enables
pretrained mode; `--no-pretrain` explicitly selects AdapterOnly.

For example, after configuring the data and output roots required by the recipe:

```bash
python train/downstream/train_track_finding.py \
  --yaml_config scripts/configs/clas12_adapter_recipes.yaml \
  --config clas12_track_finding_50k --backbone campaign5_m3
```

The same selector works with `train_track_regression.py` and the
`clas12_momentum_50k` profile, or `train_point_classification.py` and the
`hdf5_noise_50k_balanced` profile. Dataset locations, target statistics, adapter
architecture, optimization settings, and label budgets stay in those recipes.
The old interactive/notebook demos contain historical example paths; use the
public CLIs for catalog-based experiments.

The equivalent fields inside a named model YAML profile are:

```yaml
pretrained_backbone: campaign5_m3
# Optional; otherwise use the bundled catalog and CLAS12_PRETRAINED_ROOT.
backbone_catalog: /path/to/backbones.yaml
backbone_root: /path/to/pretrained-FMs
```

CLI selections override previous YAML selections, so a recipe can be reused for
another backbone or another machine. CLI catalog paths are relative to the
working directory; a catalog path stored in a model YAML is relative to that
YAML. Root flags/environment values are relative to the working directory.
Without either, a catalog's `root` is relative to the catalog file, defaulting to
its own directory. Checkpoint paths are relative to the resulting root. Environment
variables and `~` are supported; unset variables produce an error.

## Campaigns

Use all entries in a catalog, or repeat `--backbone` to select a subset. For example:

```bash
python train/downstream/campaign/build_track_finding_manifest.py \
  --backbone-catalog scripts/configs/pretrained_backbones.yaml \
  --backbone campaign5_m1 --backbone campaign5_m3 \
  --base-model-yaml scripts/configs/clas12_adapter_recipes.yaml \
  --base-model-config clas12_track_finding_50k \
  --campaign-name campaign5_finding \
  --artifact-root /path/to/results

python train/downstream/campaign/build_track_regression_manifest.py \
  --backbone-catalog scripts/configs/pretrained_backbones.yaml \
  --base-model-yaml scripts/configs/clas12_adapter_recipes.yaml \
  --pretrained-model-config clas12_momentum_50k \
  --campaign-name campaign5_momentum \
  --artifact-root /path/to/results
```

These commands create manifests; the existing campaign runners render and launch
them. Catalog resolution occurs once per model while building the manifest.
Each pretrained row carries its resolved config, checkpoint path, and SHA-256.
Rendered model YAMLs preserve that descriptor, so training and evaluation do not
need access to the original catalog. To relocate a campaign, rebuild its manifest
with the new root. Rendered configurations are snapshots of the selected files.

Checkpoint architecture/input settings replace stale backbone defaults from the
adapter recipe. Conflicting explicit campaign training overrides fail during
rendering. AdapterOnly rows keep their own settings. Legacy `--checkpoint-root`
discovery and metadata sidecars are also supported; see the campaign guide.

## Adding models and metadata

A catalog has this format; `defaults`, per-entry `config`, `root`, and `sha256`
are optional:

```yaml
version: 1
root: /path/to/checkpoint/root
defaults:
  embed_method: pos_only
  pe_method: nerf
  klen: 5
  use_sincos_target: true
backbones:
  my_model:
    checkpoint: arbitrary/run/training_checkpoints/ckpt_best.tar
    # config: {hilbert_angular_grid: unit_v2}
    # sha256: <expected SHA-256>
```

For current checkpoints, family, dimensions, block options, and Hilbert grid are
read from metadata; output width is checked against the weight tensor. Catalog
settings can fill gaps but cannot contradict saved metadata. Incomplete metadata
produces an error instead of guessing the embedding or target meaning. The
campaign 5 catalog supplies the position-only NeRF embedding and sine/cosine
target contract missing from those historical checkpoints.

The model factory supports `backbone_output_dim` and `use_sincos_target`; loading
campaign 5 preserves its full 20-output head under strict weight loading. This
does not add sine/cosine target generation to our pretraining dataset. Downstream
adapters consume the frozen hidden states, independently of that pretraining head.

New checkpoints produced by the current pretrainer save `backbone_config`, so
they can be selected through a catalog with only a path (and optional hash).
An explicit checkpoint path also works with metadata or a matching legacy
recipe. The old regression `MODEL2CKPT` dictionary has been removed: recipes no
longer select machine-specific checkpoints implicitly by their config name.

## Auxiliary CLAS12 inputs

Track finding, point classification (PID/noise), and momentum regression support
Mike's `pos_plus_aux` backbones. Training, validation and evaluation pack the same
serialized centers and geometry into `[eta, phi, r, s_r, s_phi, s_z, scalar]`.
Directions use the local radial/tangential/beam frame; length is divided by 45 cm
and pitch by 0.10 cm. Padding stays `-100` across every token column. Coordinate
exports and physics metrics continue to use the three center columns.

For an older auxiliary checkpoint, supply the missing input contract in its
catalog entry (along with the matching embedding and target settings):

```yaml
config:
  embed_method: pos_plus_aux
  pe_method: nerf
  input_layout: xyz
  input_representation: clas12_pos_plus_aux_v1
  pos_dim: 3
  aux_dim: 4
  aux_extra_feature: length   # length or pitch: four auxiliary columns
  klen: 5                   # match the checkpoint
  use_sincos_target: true    # match the checkpoint
```

Use `aux_extra_feature: both` with `aux_dim: 5` for an eight-column token
containing both length and pitch. Match the checkpoint's Hilbert grid as well.
These inputs require the published HDF5 geometry fields or aligned v7 token and
geometry sidecars; a dataset containing only centers cannot supply them.
New finding/classification adapter checkpoints save the input contract and
reject changes such as switching length to pitch when resuming or evaluating.
