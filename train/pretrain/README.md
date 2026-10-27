# CLAS12 pretraining

This pretraining path was first integrated on `integration-adapter-pretrain`
from Mike's `mike` branch at `38c82ec`, then combined with the current CLAS12
research baseline on `main`. Adapter-only and pretrained-adapter training retain
their established entrypoints and campaign configurations. See
[downstream campaigns](../downstream/campaign/README.md).

This integration preserves the existing pretraining experiment behavior. In
particular, the loader keeps **1–50 points per event**, calibrated 12-band
serialization, Mike's outward-band coordinate targets, and the existing
optimization, validation, and checkpoint behavior. The 50-point cut is an
intentional historical selection; its original anomaly diagnosis is not recorded
here. This branch does not revisit it or repair the inherited trainer.

## Recipes and execution

The imported recipes are unchanged from Mike:

| YAML | Example configuration | Input |
|---|---|---|
| `scripts/configs/mamba_v602_sweep.yaml` | `scale_w64_d12_v602` | Center coordinates |
| `scripts/configs/mamba_v701_aux_sweep.yaml` | `scale_w64_d12_v701aux` | Centers + local strip direction + length |
| `scripts/configs/mamba_v701_pitch_sweep.yaml` | `scale_w64_d12_v701pitch` | Centers + local strip direction + pitch |

Each file contains six widths. Use a CUDA environment with the repository's
Mamba dependencies. These recipes use `klen: 1`, `d_state: 16`, and the existing
optional width-dependent learning-rate rule. Historical absolute paths are
preserved in the source recipes; render a run configuration with your data and
statistics paths before running.

For example, from the repository root:

```bash
export CLAS12_ARTIFACT_ROOT=/path/to/result_deep_storage
export CLAS12_PRETRAIN_DATA=/path/to/mmap_canonical_loose_truthseg_event_v7_01
export CLAS12_PRETRAIN_STATS=/path/to/pretraining_stats
mkdir -p "$CLAS12_ARTIFACT_ROOT/pretrain/configs"

python - <<'PY'
import os
from pathlib import Path
import yaml

# PyYAML is already listed in requirements.txt.
name = 'scale_w64_d12_v701aux'
with open('scripts/configs/mamba_v701_aux_sweep.yaml') as stream:
    params = dict(yaml.safe_load(stream)[name])
params['data_root'] = os.environ['CLAS12_PRETRAIN_DATA']
params['stat_dir'] = os.environ['CLAS12_PRETRAIN_STATS']
out = Path(os.environ['CLAS12_ARTIFACT_ROOT']) / 'pretrain'
params['checkpoint_dir'] = str(out)
with (out / 'configs' / 'run.yaml').open('w') as stream:
    yaml.safe_dump({name: params}, stream, sort_keys=False)
PY

python -m train.pretrain.nppmamba.train_multi_gpu_mamba1 \
  --yaml_config "$CLAS12_ARTIFACT_ROOT/pretrain/configs/run.yaml" \
  --config scale_w64_d12_v701aux \
  --run_num run0 \
  --root_dir "$CLAS12_ARTIFACT_ROOT/pretrain"
```

The existing loader requires `features_{pretrain,test}`,
`seg_target_{pretrain,test}`, and `reg_target_{pretrain,test}` RaggedMmap folders.
Auxiliary recipes additionally require
`cluster_geometry_context_target_{pretrain,test}`. The statistics directory must
contain the existing `loss_bin_pp.pkl` and `loss_weight_pp.pkl` runtime inputs,
even when loss weighting is disabled. These datasets/statistics stay external to
the repository. The historical `test` folder is used for validation in this
trainer.

Checkpoints are written under
`<root_dir>/<config>/<run_num>/training_checkpoints/`. Pass the produced
`ckpt_best.tar` to the existing downstream pretrained workflow with a matching
backbone family, width, depth, `d_state`, `klen`, positional encoding, input
representation, and auxiliary feature choice. Use `clas12_pos_plus_aux_v1` for
the current regression auxiliary path. Sharing a branch does not add auxiliary
support to every downstream head or change existing adapter configuration.

## Backbone selection and checkpoint compatibility

The default for the imported recipes remains Mamba1. Set `backbone_version` to
`mamba2` or `mamba3` to select another backbone; downstream `adapter_version`
is independent. See [Mamba support](../MAMBA_SUPPORT.md) for block options,
checkpoint requirements, and GPU validation. Mamba3 currently supports SISO.

The existing rank-based width-transfer rule (`use_mup`) and optional offline
W&B logging are retained. Mamba3's no-decay groups are kept separate without
losing matrix-group learning-rate scaling. New checkpoints record parameter
names and scheduler state; known historical single-group, rank-based, and
Mamba3 no-decay optimizer layouts can be restored without dropping moments.
An ambiguous legacy mapping is rejected. Resume still follows the inherited
epoch-based data iteration; it does not restore an exact mid-epoch cursor.

## Integration boundary

The donor loader, legacy values/offsets reader, trainer, and three active recipe
files come from Mike. The shared backbone, embeddings, calibration, downstream
loaders, adapters, evaluators, and campaigns retain the current `main`
implementation (formerly `adapter-dev`).

The initial integration removed unsupported band-classification constructor
arguments and explicitly rejected that retired mode. Continuous pretraining
retains the same output shapes and model checkpoint keys. Subsequent Mamba
selection adds the optimizer/checkpoint compatibility described above. The
imported target selection, filtering, validation cap, width-transfer recipe,
and logging behavior are preserved. The entrypoint remains a single-process
CUDA workflow; multi-process correctness and exact data-cursor resume remain
separate follow-up work.

The existing strict shared auxiliary embedder remains in place: use the shipped
length or pitch recipes with four auxiliary columns. The historical `both`
experiment is outside this integration's supported recipes. Generated datasets,
plots, old CSV converters, and one-off sweep/tuning scripts were not imported.
