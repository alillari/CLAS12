"""
Optuna hyperparameter search over OPTIMIZATION hyperparameters only
(learning rate, batch size, warmup, dropout, weight decay, grad clip) --
NOT model architecture (width/depth), which is being explored deliberately
as its own grid elsewhere. This search finds one good training recipe at a
FIXED representative architecture (w256/d12), matching the paper's own
approach of tuning once on their m3 and reusing it everywhere.

Each trial: samples a hyperparameter set, writes a temp config, runs a SHORT
training run (few thousand steps) via the real trainer as a subprocess,
reads the final val loss, reports it to Optuna. Optuna uses this history to
pick smarter values on later trials (not brute-force grid search).

Study is saved to a SQLite file, so the search can be paused/resumed and
inspected later.

Run: python3 dev_scripts/optuna_search.py --n-trials 30
"""
import argparse, os, subprocess, re, sys
import optuna

REPO = "/workspace/PP_collision"
YAML = f"{REPO}/scripts/configs/mamba_pretrain.yaml"
GEN_YAML = f"{REPO}/scripts/configs/_optuna_trial.yaml"
CKPT_ROOT = f"{REPO}/checkpoints"
STUDY_DB = f"{REPO}/sweep_analysis/optuna_study.db"
DATA_ROOT = f"{REPO}/data/mmap_v4"

FIXED_WIDTH = 256
FIXED_DEPTH = 12
TRIAL_STEPS = 5000       # short trials for fast search
TRIAL_DATA_FRACTION = 0.20   # a moderate slice, fast but representative

os.makedirs(os.path.dirname(STUDY_DB), exist_ok=True)


def get_default_anchor():
    """Copy the *default anchor block out of the main yaml (same approach as
    run_grid_sweep.sh) so trial configs are self-contained."""
    lines = open(YAML).read().splitlines()
    out, capturing = [], False
    for ln in lines:
        if not capturing and "&default" in ln:
            capturing = True
            out.append(ln); continue
        if capturing:
            if ln and not ln[0].isspace() and "&default" not in ln:
                break
            out.append(ln)
    return "\n".join(out)


DEFAULT_ANCHOR = get_default_anchor()


def objective(trial):
    # --- sample the optimization hyperparameters ---
    max_lr = trial.suggest_float("max_lr", 1e-5, 1e-2, log=True)
    local_batch_size = trial.suggest_categorical("local_batch_size", [8, 16, 32, 64, 128])
    warmup_steps = trial.suggest_int("warmup_steps", 50, 1000, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    grad_clip_value = trial.suggest_float("grad_clip_value", 0.05, 2.0, log=True)

    name = f"optuna_trial_{trial.number}"
    block = f"""
{name}:
  <<: *default
  data_root: {DATA_ROOT}
  reader_type: ragged_npy
  stat_dir:  {REPO}/data/stats
  checkpoint_dir: {CKPT_ROOT}
  klen: 1
  embed_method: pos_only
  voxelize: false
  space_filling_order: true
  rep_aaai: false
  nexttoken: false
  ablate_loss_scale: false
  data_fraction: {TRIAL_DATA_FRACTION}
  embed_dim: {FIXED_WIDTH}
  num_layers_backbone: {FIXED_DEPTH}
  d_state: 16
  d_conv: 4
  expand: 2
  batch_size: 128
  local_batch_size: {local_batch_size}
  valid_batch_size: 32
  local_valid_batch_size: 2
  max_val_batches: 200
  warmup_steps: {warmup_steps}
  total_steps: {TRIAL_STEPS}
  max_lr: {max_lr}
  min_lr: {max_lr/10}
  weight_decay: {weight_decay}
  grad_clip_value: {grad_clip_value}
  dropout: {dropout}
  n_eval_steps: {max(TRIAL_STEPS//5, 100)}
  save_version: {name}
"""
    with open(GEN_YAML, "w") as f:
        f.write(DEFAULT_ANCHOR + "\n" + block)

    log_path = f"{REPO}/sweep_logs/{name}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [
        "python", "-m", "train.pretrain.nppmamba.train_multi_gpu_mamba1",
        f"--yaml_config={GEN_YAML}", f"--config={name}",
        "--run_num=optuna", f"--root_dir={CKPT_ROOT}",
    ]
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"[trial {trial.number}] FAILED (see {log_path})")
        raise optuna.TrialPruned()

    # parse the final val loss out of the log
    text = open(log_path).read()
    matches = re.findall(r"Val loss\s*=\s*([0-9.eE+-]+)", text)
    if not matches:
        print(f"[trial {trial.number}] no val loss found in log")
        raise optuna.TrialPruned()
    final_val = float(matches[-1])
    print(f"[trial {trial.number}] lr={max_lr:.2e} bs={local_batch_size} warmup={warmup_steps} "
          f"dropout={dropout:.3f} wd={weight_decay:.2e} clip={grad_clip_value:.3f} "
          f"-> val_loss={final_val:.6g}")
    return final_val


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=30)
    args = p.parse_args()

    study = optuna.create_study(
        study_name="clas12_optim_hparams",
        storage=f"sqlite:///{STUDY_DB}",
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== BEST TRIAL ===")
    print(f"val_loss: {study.best_value:.6g}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nFull study saved to: {STUDY_DB}")
    print("Resume anytime by re-running this script (load_if_exists=True).")