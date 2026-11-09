"""
Optuna search for the background-merged HDF5 pretraining baseline, at m3
(width 256, depth 12). Fixed model: Mamba-3, positions only, sin/cos phi
target, kNNN (default 5, --klen), unit_v2 Hilbert grid, dropout 0. Searched: learning rate,
warmup, weight decay, gradient clip, and batch size in a narrow window.

Each trial writes a self-contained config, runs the real trainer as a
subprocess, and reports the final validation loss. The validation set is the
same fixed slice of the validation split for every trial.

Run from the repo root:
    python3 dev_scripts/optuna_search_hdf5.py --n-trials 25            # kNNN = 5
    python3 dev_scripts/optuna_search_hdf5.py --n-trials 25 --klen 1   # kNNN = 1
Resume by running the same command again (the study is stored in SQLite).
"""
import argparse, os, re, subprocess, time
import optuna

REPO = "/workspace/PP_collision"
DATA_ROOT = f"{REPO}/data/cvt_cluster_hdf5_20261103_002"
STATS = f"{REPO}/data/stats"
CKPT_ROOT = f"{REPO}/checkpoints"
STUDY_DIR = f"{REPO}/sweep_analysis"
_cli = argparse.ArgumentParser()
_cli.add_argument("--n-trials", type=int, default=25)
_cli.add_argument("--klen", type=int, default=5, help="kNNN; each value gets its own study")
ARGS = _cli.parse_args()

WIDTH, DEPTH, KLEN = 256, 12, ARGS.klen
# kNNN = 5 keeps its original names so the finished study stays readable.
_tag = "" if KLEN == 5 else f"_klen{KLEN}"
GEN_YAML = f"{REPO}/scripts/configs/_optuna_hdf5_trial{_tag}.yaml"
STUDY_NAME = f"clas12_hdf5_mamba3_pos_sincos_klen{KLEN}"
STUDY_DB = f"{STUDY_DIR}/optuna_study_hdf5_mamba3_pos_sincos_klen{KLEN}.db"
RUN_NUM = f"optuna_hdf5{_tag}"
PREFIX = f"hdf5_m3{_tag}_trial"
TRIAL_STEPS = 5000
BATCH_SIZES = [32, 48, 64]          # narrow window around the old bs64
VAL_BATCH, VAL_BATCHES = 16, 100    # 1,600 fixed validation events per evaluation

os.makedirs(STUDY_DIR, exist_ok=True)


def trial_yaml(name, lr, bs, warmup, wd, clip):
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

  # model: Mamba-3 SISO, bf16 inside the blocks
  backbone_version: mamba3
  embed_dim: {WIDTH}
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

  # input and target: positions only, sin/cos phi, kNNN = 5
  embed_method: pos_only
  pe_method: nerf
  use_aux_features: false
  return_knn_aux_target: false
  use_sincos_target: true
  klen: {KLEN}
  hilbert_angular_grid: unit_v2

  # legacy loader settings (unchanged from every earlier sweep)
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

  # optimization
  use_mup: true
  mup_base_width: 256
  batch_size: {bs}
  local_batch_size: {bs}
  valid_batch_size: {VAL_BATCH}
  local_valid_batch_size: {VAL_BATCH}
  max_val_batches: {VAL_BATCHES}
  max_epochs: 10000
  total_steps: {TRIAL_STEPS}
  warmup_steps: {warmup}
  n_eval_steps: {TRIAL_STEPS // 5}
  max_lr: {lr}
  min_lr: {lr / 10}
  weight_decay: {wd}
  grad_clip_value: {clip}

  num_data_workers: 8
  log_to_screen: true
  use_wandb: false
  save_version: {name}
"""


def objective(trial):
    lr = trial.suggest_float("max_lr", 1e-4, 1e-2, log=True)
    bs = trial.suggest_categorical("local_batch_size", BATCH_SIZES)
    warmup = trial.suggest_int("warmup_steps", 50, 1000, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    clip = trial.suggest_float("grad_clip_value", 0.05, 2.0, log=True)

    name = f"{PREFIX}_{trial.number}"
    with open(GEN_YAML, "w") as f:
        f.write(trial_yaml(name, lr, bs, warmup, wd, clip))
    log_path = f"{REPO}/sweep_logs/{name}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = ["python", "-m", "train.pretrain.nppmamba.train_multi_gpu_mamba1",
           f"--yaml_config={GEN_YAML}", f"--config={name}",
           f"--run_num={RUN_NUM}", f"--root_dir={CKPT_ROOT}"]

    # Retry once if the GPU has not been released yet by the previous trial.
    for attempt in (1, 2):
        with open(log_path, "w") as logf:
            result = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
        if result.returncode == 0 or "No CUDA GPUs are available" not in open(log_path).read():
            break
        print(f"[{trial.number}] CUDA not ready, retrying in 10 s")
        time.sleep(10)
    time.sleep(10)

    if result.returncode != 0:
        print(f"[{trial.number}] FAILED (see {log_path})")
        raise optuna.TrialPruned()
    vals = re.findall(r"Val loss\s*=\s*([0-9.eE+-]+)", open(log_path).read())
    if not vals:
        print(f"[{trial.number}] no val loss in log")
        raise optuna.TrialPruned()
    val = float(vals[-1])
    print(f"[{trial.number}] lr={lr:.2e} bs={bs} warmup={warmup} wd={wd:.2e} clip={clip:.3f} -> val={val:.6g}")
    return val


if __name__ == "__main__":
    args = ARGS
    study = optuna.create_study(study_name=STUDY_NAME, storage=f"sqlite:///{STUDY_DB}",
                                direction="minimize", load_if_exists=True)
    study.optimize(objective, n_trials=args.n_trials)
    print("\n=== BEST TRIAL ===")
    print(f"val_loss: {study.best_value:.6g}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nStudy: {STUDY_DB}")