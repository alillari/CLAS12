# CLAS12 backbone pretraining

Start with the [complete worked example](../WORKED_EXAMPLE.md) to pretrain one
backbone, fit both adapters, generate target stats and inspect evaluation plots.
The [backbone recipe reference](../BACKBONE_RECIPES.md) explains optimizer,
serialization, target, batch and width choices in detail.

## Choose a recipe

| YAML | Named profile example | Input and budget |
| --- | --- | --- |
| [Chinchilla sweep](../../scripts/configs/mamba_chinchilla_sweep.yaml) | `scale_w512_d12_chinchilla` | Center coordinates; six widths with width-dependent 100k–1.6M update budgets. Width 512 uses 200k. Portable environment paths. |
| [v6 center sweep](../../scripts/configs/mamba_v602_sweep.yaml) | `scale_w64_d12_v602` | Centers; all six widths use 100k updates. Edit machine-specific paths in a copy. |
| [v7 length sweep](../../scripts/configs/mamba_v701_aux_sweep.yaml) | `scale_w64_d12_v701aux` | Centers + local strip direction + length; edit paths in a copy. |
| [v7 pitch sweep](../../scripts/configs/mamba_v701_pitch_sweep.yaml) | `scale_w64_d12_v701pitch` | Centers + local strip direction + pitch; edit paths in a copy. |
| [Transformer](../../scripts/configs/transformer_pretrain.yaml) | See [Transformer support](../TRANSFORMER_SUPPORT.md) | Causal attention with independently selected NoPE/RoPE settings. |

The Mamba sweep profiles use depth 12, `d_state: 16`, convolution 4, expand 2
and `klen: 1`. Their default family is Mamba1. `klen` counts coordinate triples
to predict per token; it does not set sequence length. Equal width/depth across
families does not imply equal parameter count or compute.

## Prepare inputs and run

The legacy loaders accept 1–50 clusters/event, convert xyz to normalized
eta/phi/r, serialize with calibrated12 band-Hilbert ordering and construct
outward-band coordinate targets. Point cuts and target construction affect the
training population/objective; preserve them when comparing these recipes.

Their required split trees are `features_{pretrain,test}` and `seg_target_{pretrain,test}`. The trainer uses the `test` trees for
validation, so reserve separate source events for final downstream evaluation.
`reader_type: ragged_mmap` selects RaggedMmap input; `ragged_npy` is the legacy
values/offsets reader. Geometry recipes additionally need aligned
`cluster_geometry_context_target_{pretrain,test}` with the selected four
auxiliary features.

For the background-merged HDF5 skim, point `CLAS12_PRETRAIN_DATA` at the
published directory containing `manifest.json` and run the command below with
`--config scale_w64_d12_bgmerged_hdf5`. This width-64 position-only profile
uses all retained events in `train` and validates on `validation`. Each cluster
contributes one xyz token, including noise clusters. HDF5 has no 50-cluster
cut or 10-million-event limit in this profile. Longer events increase the
cost of the pairwise neighbor calculation. Measure batch size on your hardware.

Other pretraining profiles can opt in with `reader_type: hdf5` and
`limit_data: false`. The usual geometry options read direction, length and
pitch from cluster rows. The reader opens shards lazily in each worker and
only reads the requested fields. `CVTClusterEvents.read(i, fields)` in
`fm4npp.datasets.cvt_hdf5` also accepts event, particle and native reco
fields, sliced with each collection's own offsets. The loader requires h5py.

To repeat the real-event parity check with the single-part qualification skim:

```bash
CVT_HDF5_PILOT=/path/to/published/pilot-root \
  python -m unittest train.pretrain.tests.test_cvt_hdf5
```

Run the HDF5 profile from the repository root in the [CUDA environment](../../SETUP.md):

```bash
export CLAS12_ARTIFACT_ROOT=/absolute/path/to/result_deep_storage
python -m train.pretrain --data-root /absolute/path/to/published/cluster_skim
```

The default profile is `scale_w64_d12_bgmerged_hdf5`. Runs write to
`$CLAS12_ARTIFACT_ROOT/pretrain/<config>/<run_num>`. For another recipe,
pass `--yaml_config` and `--config`; `--data-root` overrides its data path.
The original module name remains available to existing launchers.
Only runs with `loss_reweight: true` need `loss_bin_pp.pkl` and
`loss_weight_pp.pkl` in `stat_dir`.

Only batches with eligible outward-neighbor targets advance the optimizer
and scheduler. Validation reports mean squared error over the valid target
coordinates in the evaluated batches. A validation set with no eligible
targets fails before checkpoint selection.
Resuming a checkpoint with the older batch-averaged metric starts a fresh
best-loss comparison at the next validation. Model and optimizer state resume
as usual. With chunking enabled, training takes a random contiguous window
of at most `len_chunk` cluster rows before ordering and target construction.

`run_num` names the output; it does not set the seed. The dataset module
initializes Torch's seed to 42. The documented command starts a single training process.

For intentional recipe changes, copy the chosen YAML to an external run/config
directory, edit the named profile and pass that copy via `--yaml_config`.
For the v6/v7 recipes, replace `data_root`, `stat_dir`, `checkpoint_dir` and any
other machine-specific paths. Preserve the chosen file with the run.

## Inspect outputs and use the backbone

Outputs under `<root_dir>/<config>/<run_num>/` include:

- `training_checkpoints/ckpt_best.tar`: lowest validation-MSE checkpoint.
- `training_checkpoints/ckpt.tar`: latest training checkpoint for resume.
- `config_<config>_run_<run_num>.csv`: training/validation loss and logged LR.

```bash
export PRETRAIN_RUN="$CLAS12_ARTIFACT_ROOT/pretrain/scale_w512_d12_chinchilla/seed42"
python train/pretrain/plot_pretraining_history.py \
  --log "$PRETRAIN_RUN/config_scale_w512_d12_chinchilla_run_seed42.csv" \
  --output-dir "$PRETRAIN_RUN/plots"
```

Check loss trends and the best validation point. Logged LR describes the
scheduler; with `use_mup`, rank > 1 parameter groups also receive width scaling
as explained in the recipe reference. Resume restores compatible optimizer and
scheduler state, but follows epoch-based data iteration rather than an exact
mid-epoch cursor. Optional W&B logging in this pretrainer is offline.

Pass `ckpt_best.tar` to a downstream run with matching family, width, depth,
state, convolution/expand settings where applicable, `klen`, positional encoding
and input representation. For supported four-column auxiliary inputs, match
the length/pitch choice and use the regression `clas12_pos_plus_aux_v1` feature
contract. Auxiliary support is specific to each downstream path; inspect its
[training options](../downstream/TRAINING_OPTIONS.md).

Select `backbone_version: mamba2` or `mamba3` only with valid family options;
Mamba3 currently supports SISO. Downstream `adapter_version` independently
selects the head family. See [Mamba support](../MAMBA_SUPPORT.md) and
[Transformer support](../TRANSFORMER_SUPPORT.md) for valid blocks, precision,
optimizer-state compatibility and checkpoint restrictions.
