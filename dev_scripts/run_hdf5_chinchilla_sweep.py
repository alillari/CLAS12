"""
Chinchilla sweep m1-m4 on the background-merged HDF5 data.

Model (fixed, same as the Optuna search): Mamba-3, positions only, sin/cos phi
target, kNNN = 5, unit_v2 Hilbert grid, dropout 0. Hyperparameters from the
Optuna best trial at m3; learning rate transferred to other widths by muP.

Per width:
  N            parameter count, measured by building the model
  D = 2 x 20 N twice the Chinchilla-optimal training tokens (1 token = 1 cluster)
  total_steps  D / (mean clusters per training event * batch size)
  warmup       same fraction of total_steps as in the Optuna trials (283 / 5000)
Snapshots are kept 5 times per run (for dev_scripts/loss_by_category.py).

Run from the repo root (in tmux):
  python3 dev_scripts/run_hdf5_chinchilla_sweep.py --dry-run   # print the plan only
  python3 dev_scripts/run_hdf5_chinchilla_sweep.py             # run m1 -> m4 in order
  python3 dev_scripts/run_hdf5_chinchilla_sweep.py --klen 1    # kNNN = 1 sweep, after its Optuna study
Widths that already finished (marked <log>.done) are skipped, so re-running resumes.
"""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import h5py

REPO = "/workspace/PP_collision"
sys.path.insert(0, REPO)
DATA_ROOT = f"{REPO}/data/cvt_cluster_hdf5_20261103_002"
STATS = f"{REPO}/data/stats"
CKPT_ROOT = f"{REPO}/checkpoints"
LOG_DIR = f"{REPO}/sweep_logs"

_cli = argparse.ArgumentParser()
_cli.add_argument("--dry-run", action="store_true", help="print the plan and write the YAML only")
_cli.add_argument("--klen", type=int, default=5, help="kNNN; uses that kNNN's Optuna study")
ARGS = _cli.parse_args()

WIDTHS = {"m1": 64, "m2": 128, "m3": 256, "m4": 512}
DEPTH, KLEN = 12, ARGS.klen
TOKENS_PER_PARAM = 20
# 2x the Chinchilla-optimal tokens: these backbones feed the downstream adapters,
# so they train past the compute-optimal point. Same multiple for every width.
BUDGET_MULT = 2

# kNNN = 5 keeps its original names (that sweep has already run).
_tag = "" if KLEN == 5 else f"_klen{KLEN}"
YAML = f"{REPO}/scripts/configs/hdf5_chinchilla_sweep{_tag}.yaml"
RUN_NUM = f"hdf5_chinchilla{_tag}"
NAME = lambda m, w: f"hdf5_chinchilla{_tag}_{m}_w{w}"

OPTUNA_STEPS = 5000
if KLEN == 5:
    # Optuna best trial (m3, 5,000-step trials), as used for the kNNN = 5 sweep
    BEST = dict(max_lr=0.00109418308196948, local_batch_size=64, warmup_steps=283,
                weight_decay=0.0003621987805395929, grad_clip_value=0.3649169792089614)
else:
    import optuna
    _study = optuna.load_study(
        study_name=f"clas12_hdf5_mamba3_pos_sincos_klen{KLEN}",
        storage=f"sqlite:///{REPO}/sweep_analysis/optuna_study_hdf5_mamba3_pos_sincos_klen{KLEN}.db")
    BEST = dict(_study.best_params)
    print(f"Optuna kNNN = {KLEN} best (val {_study.best_value:.6g}, "
          f"{sum(t.value is not None for t in _study.trials)} complete trials): {BEST}")
WARMUP_FRAC = BEST["warmup_steps"] / OPTUNA_STEPS


def mean_clusters_per_event():
    m = json.loads(Path(DATA_ROOT, "manifest.json").read_text())
    clusters = events = 0
    for part in m["splits"]["train"]:
        with h5py.File(Path(DATA_ROOT, part["path"]), "r") as f:
            off = f["clusters/offsets"]
            clusters += int(off[-1]) - int(off[0]); events += len(off) - 1
    return clusters / events, events


def block(name, width, total, warmup, n_eval, snap):
    lr = BEST["max_lr"]
    return f"""{name}:
  data_root: {DATA_ROOT}
  stat_dir: {STATS}
  checkpoint_dir: {CKPT_ROOT}
  reader_type: hdf5
  data_version: cvt-clusters-h5-v1
  limit_data: false
  limit_size: 10000000
  n_data_idx: 10000000
  data_fraction: 1.0

  backbone_version: mamba3
  embed_dim: {width}
  num_layers_backbone: {DEPTH}
  d_state: 128
  expand: 2
  headdim: 64
  ngroups: 1
  chunk_size: 64
  is_mimo: false
  base_dim: 256
  init_std: 0.025
  dropout: 0.0

  embed_method: pos_only
  pe_method: nerf
  use_aux_features: false
  return_knn_aux_target: false
  use_sincos_target: true
  klen: {KLEN}
  hilbert_angular_grid: unit_v2

  group_size: 8
  nleave: 1e10
  order: EPR
  len_chunk: 512
  portion: 0.2
  chunk_training: false
  return_reg_test: false
  return_dict: false
  max_gt_classes: 150
  voxelize: false
  space_filling_order: true
  space_filling_curve: z
  rep_aaai: false
  nexttoken: false
  loss_reweight: false
  ablate_loss_scale: false
  save_checkpoint: true
  snapshot_every: {snap}

  use_mup: true
  mup_base_width: 256
  batch_size: {BEST['local_batch_size']}
  local_batch_size: {BEST['local_batch_size']}
  valid_batch_size: 16
  local_valid_batch_size: 16
  max_val_batches: 100
  max_epochs: 10000
  total_steps: {total}
  warmup_steps: {warmup}
  n_eval_steps: {n_eval}
  max_lr: {lr}
  min_lr: {lr / 10}
  weight_decay: {BEST['weight_decay']}
  grad_clip_value: {BEST['grad_clip_value']}

  num_data_workers: 8
  log_to_screen: true
  use_wandb: false
  save_version: {name}
"""


def plan():
    import torch
    from fm4npp.utils import YParams
    from fm4npp.models.mamba_config import build_backbone
    tok, n_events = mean_clusters_per_event()
    bs = BEST["local_batch_size"]
    rows, text = [], []
    for m, w in WIDTHS.items():
        name = NAME(m, w)
        # count parameters with the exact model config
        tmp = Path(YAML + ".count")
        tmp.write_text(block(name, w, 1, 1, 1, 0))
        n_params = sum(p.numel() for p in build_backbone(YParams(str(tmp), name)).parameters())
        tmp.unlink()
        total = math.ceil(BUDGET_MULT * TOKENS_PER_PARAM * n_params / (tok * bs))
        warmup = max(1, round(WARMUP_FRAC * total))
        n_eval = max(50, total // 20)
        snap = n_eval * 4
        text.append(block(name, w, total, warmup, n_eval, snap))
        rows.append((m, w, name, n_params, total, warmup, n_eval, total * bs))
    Path(YAML).write_text("\n".join(text))
    print(f"mean clusters/event (train, all {n_events:,} events): {tok:.1f}; batch {bs}; "
          f"budget {BUDGET_MULT} x {TOKENS_PER_PARAM} tokens/param\n")
    print(f"{'model':>5} {'width':>6} {'params':>12} {'steps':>9} {'warmup':>7} {'eval every':>11} {'events used':>12} {'epochs':>7}")
    for m, w, _, n, t, wu, ev, used in rows:
        print(f"{m:>5} {w:>6} {n:>12,} {t:>9,} {wu:>7,} {ev:>11,} {used:>12,} {used / n_events:>7.3f}")
    print(f"\nWrote {YAML}")
    return rows


def finished(log):
    return os.path.exists(log + ".done")


def run(rows):
    os.makedirs(LOG_DIR, exist_ok=True)
    for m, w, name, *_ in rows:
        log = f"{LOG_DIR}/{name}.log"
        if finished(log):
            print(f"[{m}] already finished, skipping ({log})"); continue
        cmd = ["python", "-m", "train.pretrain.nppmamba.train_multi_gpu_mamba1",
               f"--yaml_config={YAML}", f"--config={name}", f"--run_num={RUN_NUM}", f"--root_dir={CKPT_ROOT}"]
        print(f"[{m}] starting {time.strftime('%H:%M')}  log: {log}", flush=True)
        for attempt in (1, 2):  # retry once if the GPU was not released yet
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc == 0 or "No CUDA GPUs are available" not in open(log).read():
                break
            time.sleep(10)
        print(f"[{m}] {'finished' if rc == 0 else f'FAILED (exit {rc})'} {time.strftime('%H:%M')}", flush=True)
        if rc != 0:
            sys.exit(f"Stopping: {m} failed, see {log}")
        Path(log + ".done").touch()
        time.sleep(10)


if __name__ == "__main__":
    rows = plan()
    if not ARGS.dry_run:
        run(rows)