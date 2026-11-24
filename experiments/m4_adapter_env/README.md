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
python3 experiments/m4_adapter_env/run_matched.py adopted --probe          # preflight only
python3 experiments/m4_adapter_env/run_matched.py adopted --task finding    # one task
```

The preflight checks the skim manifest, the backbone SHA-256 and that downstream tokens
equal the pretraining tokens for eight events. Results, settings and per-fit receipts of
the original runs are in the package (`FIT_SETTINGS.md`, `REPLAY.md`, `provenance/`).
