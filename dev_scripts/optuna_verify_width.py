"""
muP VERIFICATION search at any width.

Purpose: confirm the base recipe tuned at w256 transfers correctly to other
widths via muP. NOT a blind re-search -- it varies each hyperparameter in a
NARROW band around the w256 base recipe. If the best lr_mult lands near 1.0,
muP transfer worked at that width.

use_mup: true is set, so the trainer auto-scales the matrix-group LR by
(width / 256). We pass the BASE max_lr (w256 scale); muP does the rest.

NOTE for widths BELOW 256 (m1=64, m2=128): the divisor is < 1, so muP
*increases* the matrix LR (e.g. w64 -> base_lr x4). If those trials diverge
(NaN / pruned), that is itself a finding, not a bug.

Run:
    python3 dev_scripts/optuna_verify_width.py --width 512 --n-trials 30
"""
import argparse, os, subprocess, re, sys
import optuna

REPO = "/workspace/PP_collision"
YAML = f"{REPO}/scripts/configs/mamba_pretrain.yaml"
CKPT_ROOT = f"{REPO}/checkpoints"
DATA_ROOT = f"{REPO}/data/mmap_v4"

FIXED_WIDTH = None      # set from --width
GEN_YAML = None         # set per-width
STUDY_DB = None         # set per-width
FIXED_DEPTH = 12
MUP_BASE_WIDTH = 256
TRIAL_STEPS = 15000
TRIAL_DATA_FRACTION = 1.0

# --- BASE RECIPE from the w256 100-trial search ---
BASE = {
    "max_lr": 0.005654656285806099,
    "local_batch_size": 128,
    "warmup_steps": 53,
    "dropout": 0.0903784154249514,
    "weight_decay": 0.0842571876326289,
    "grad_clip_value": 1.8025125913594795,
}


def get_default_anchor():
    lines = open(YAML).read().splitlines()
    out, capturing = [], False
    for ln in lines:
        if not capturing and "&default" in ln:
            capturing = True; out.append(ln); continue
        if capturing:
            if ln and not ln[0].isspace() and "&default" not in ln:
                break
            out.append(ln)
    return "\n".join(out)


def objective(trial):
    lr_mult = trial.suggest_float("lr_mult", 0.5, 2.0, log=True)
    max_lr = BASE["max_lr"] * lr_mult
    local_batch_size = trial.suggest_categorical("local_batch_size", [64, 128])
    warmup_steps = trial.suggest_int("warmup_steps", 30, 120, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.15)
    weight_decay = trial.suggest_float("weight_decay", 0.04, 0.15, log=True)
    grad_clip_value = trial.suggest_float("grad_clip_value", 1.0, 2.5, log=True)

    name = f"verify_w{FIXED_WIDTH}_trial_{trial.number}"
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
  use_mup: true
  mup_base_width: {MUP_BASE_WIDTH}
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
        f.write(get_default_anchor() + "\n" + block)

    log_path = f"{REPO}/sweep_logs/{name}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = ["python", "-m", "train.pretrain.nppmamba.train_multi_gpu_mamba1",
           f"--yaml_config={GEN_YAML}", f"--config={name}",
           f"--run_num=verify_w{FIXED_WIDTH}", f"--root_dir={CKPT_ROOT}"]
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"[trial {trial.number}] FAILED (see {log_path})")
        raise optuna.TrialPruned()

    text = open(log_path).read()
    matches = re.findall(r"Val loss\s*=\s*([0-9.eE+-]+)", text)
    if not matches:
        print(f"[trial {trial.number}] no val loss in log")
        raise optuna.TrialPruned()
    final_val = float(matches[-1])
    print(f"[trial {trial.number}] lr_mult={lr_mult:.3f} (max_lr={max_lr:.2e}) "
          f"bs={local_batch_size} -> val_loss={final_val:.6g}")
    return final_val


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, required=True,
                   help="model width to verify (64,128,256,512,1024,1536)")
    p.add_argument("--n-trials", type=int, default=30)
    args = p.parse_args()

    FIXED_WIDTH = args.width
    GEN_YAML = f"{REPO}/scripts/configs/_optuna_verify_w{FIXED_WIDTH}.yaml"
    STUDY_DB = f"{REPO}/sweep_analysis/optuna_verify_w{FIXED_WIDTH}.db"
    os.makedirs(os.path.dirname(STUDY_DB), exist_ok=True)

    ratio = FIXED_WIDTH / MUP_BASE_WIDTH
    direction = ("matrix LR REDUCED" if ratio > 1 else
                 "matrix LR INCREASED" if ratio < 1 else "no scaling (base width)")
    print(f"=== muP verification at width {FIXED_WIDTH} (base {MUP_BASE_WIDTH}) ===")
    print(f"    muP matrix-LR divisor = {ratio:.4f}  ({direction})")
    print(f"    base max_lr = {BASE['max_lr']:.4e}; searching lr_mult in [0.5, 2.0]")

    study = optuna.create_study(
        study_name=f"clas12_verify_w{FIXED_WIDTH}",
        storage=f"sqlite:///{STUDY_DB}",
        direction="minimize", load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print(f"\n=== BEST TRIAL (width {FIXED_WIDTH}) ===")
    print(f"val_loss: {study.best_value:.6g}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    best_mult = study.best_params.get("lr_mult", None)
    if best_mult is not None:
        print(f"\n>>> muP VERDICT at w{FIXED_WIDTH}: best lr multiplier = {best_mult:.3f}")
        if 0.7 <= best_mult <= 1.4:
            print("    -> CLOSE to 1.0: muP transfer worked at this width.")
        else:
            print("    -> FAR from 1.0: muP transfer imperfect at this width.")
        if best_mult <= 0.55 or best_mult >= 1.9:
            print("    !! best multiplier sits at the edge of the search range --")
            print("       true optimum may lie outside [0.5, 2.0]; consider widening.")
    print(f"\nStudy: {STUDY_DB}")
    