# CLAS12 pretraining

The shared research branch combines the downstream workflows from `adapter-dev`
with the active CLAS12 pretraining path from `mike` at `38c82ec`. Adapter-only and
pretrained-adapter training retain their existing entrypoints and campaign
configuration. See [downstream campaigns](../downstream/campaign/README.md).

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
from ruamel.yaml import YAML

yaml = YAML(typ='safe')
name = 'scale_w64_d12_v701aux'
with open('scripts/configs/mamba_v701_aux_sweep.yaml') as stream:
    params = dict(yaml.load(stream)[name])
params['data_root'] = os.environ['CLAS12_PRETRAIN_DATA']
params['stat_dir'] = os.environ['CLAS12_PRETRAIN_STATS']
out = Path(os.environ['CLAS12_ARTIFACT_ROOT']) / 'pretrain'
params['checkpoint_dir'] = str(out)
with (out / 'configs' / 'run.yaml').open('w') as stream:
    yaml.dump({name: params}, stream)
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

## Integration boundary

The donor loader, legacy values/offsets reader, trainer, and three active recipe
files come from Mike. The shared backbone, embeddings, calibration, downstream
loaders, adapters, evaluators, and campaigns remain those of `adapter-dev`.

The only trainer compatibility change removes unsupported band-classification
constructor arguments and rejects that retired mode explicitly. Continuous
pretraining keeps the same output shapes and checkpoint state keys. No scheduler,
resume, distributed, target-selection, validation, filtering, or optimizer repair
is included. The imported entrypoint should be treated as the existing
single-process CUDA workflow; multi-process correctness and exact resume remain
separate follow-up work.

The existing strict shared auxiliary embedder remains in place: use the shipped
length or pitch recipes with four auxiliary columns. The historical `both`
experiment is outside this integration's supported recipes. Generated datasets,
plots, old CSV converters, and one-off sweep/tuning scripts were not imported.
