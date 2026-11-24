# M4 adapter environment (November 2026 five-model comparison)

This branch recreates the code that trained the fifteen M4 adapter fits in Alessio's
verification package (`mike_m4_verification_20261124`, 100,000 labels, 30,000 updates,
seed 42).

- The repository tree at this branch's base is commit `692aa26` ("Add Mike Frank's
  geometric CVT next-cluster target", the first commit of PR #33). Every file under
  `fm4npp/`, `train/`, `scripts/`, `dev_scripts/` and `docs/` is byte-identical to the
  package's execution snapshots `software/ed04691c76eb` (plain η and geometric θ noise and
  finding), `software/d20c0693dfd6` (their momentum) and `software/b2114add08d5` (plain θ).
- The files in this folder are the launch helpers and recipes those snapshots added:
  `run_matched.py`, `run_downstream.py`, `calibrate_finding.py` and `recipes/`, copied from
  `software/b2114add08d5`. Its `run_matched.py` is a superset of the earlier copy
  (adds `--task` and head-only support); for the `plain` and `adopted` arms it launches
  the same commands.

## Running

From the repository root, inside the GPU container, with these mounts (as on a02c):

| Mount | Contents |
|---|---|
| `/cvt` | production CVT HDF5 skim (`manifest.json` plan `1adb071f…`) |
| `/backbones` | `mike_hdf5_chinchilla_klen1/…/ckpt_best.tar` (plain η) and `mike_adopted_m4_20261123/ckpt_best.tar` (geometric θ) |
| `/results` | empty writable output directory, one per run |

```bash
export PYTHONPATH=$PWD
python3 experiments/m4_adapter_env/run_matched.py adopted --probe          # preflight only
python3 experiments/m4_adapter_env/run_matched.py adopted --task finding    # one task
# tuned finding cutoffs (validation search, then test), as in the package's calibration/
M4_FINDING_FIT=/results/finding M4_FINDING_SCORE=/results/calibration \
    python3 experiments/m4_adapter_env/calibrate_finding.py
```

Arms: `plain` (η), `adopted` (geometric θ) and `plain_theta`; the last reads the
backbone-only export `plain_theta/plain_theta_m4.tar` from the package's `backbones/`, which
is the file its original heads loaded (`run_downstream.py` registered it the same way).

The preflight checks the skim manifest, the backbone SHA-256 and that downstream tokens
equal the pretraining tokens for eight events. Results, settings and per-fit receipts of
the original runs are in the package (`FIT_SETTINGS.md`, `REPLAY.md`, `provenance/`).

## Layer mixing study

`--feature-mix` selects how the heads combine the frozen backbone's layers
(`train/downstream/feature_mix.py`):

| Value | Input to the head |
|---|---|
| `raw` (default) | learned average of the raw layer outputs; reproduces the original fits |
| `standardize` | learned average after standardizing each layer's channels |
| `last` | the final layer through the backbone's own final norm (what pretraining read) |
 The motivation: the geometric θ M4's residual stream grows
from ~130 to ~11,600 across its layers, with one near-constant direction added at layer 7,
so the raw average is dominated by layers 7-12 and the mixed feature has an effective
dimension of ~8 against ~19 for the plain backbones.

```bash
python3 experiments/m4_adapter_env/run_matched.py adopted --task finding --feature-mix standardize
```
