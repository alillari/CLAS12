"""
Optuna hyperparameter search for the two sincos pretraining models, at a
fixed m3 (w256/d12) architecture -- same approach as the original
optuna_search.py: each trial writes a self-contained temp config, runs the
real trainer as a subprocess, parses the final val loss from the log,
reports it to Optuna.

Two completely separate studies (Model A: pos+sincos, no aux -- Model B:
pos+sincos+aux pitch, ŝ²-weighted loss), each with its own study DB, trial
name prefix, and run_num. This is deliberate: an earlier m6 study once
loaded a w256 checkpoint because trial names/run_num collided across two
different studies. Giving each model fully distinct names/paths prevents
that from ever happening again, rather than relying on care at launch time.

Run ONE model at a time (GPU contention):
    python3 dev_scripts/optuna_search_sincos.py --model a --n-trials 30
    python3 dev_scripts/optuna_search_sincos.py --model b --n-trials 30
"""
import argparse, os, subprocess, re, sys, time
import optuna

REPO = "/workspace/PP_collision"
YAML = f"{REPO}/scripts/configs/mamba_pretrain.yaml"
GEN_YAML = f"{REPO}/scripts/configs/_optuna_sincos_trial.yaml"
CKPT_ROOT = f"{REPO}/checkpoints"
STUDY_DIR = f"{REPO}/sweep_analysis"
DATA_ROOT = f"{REPO}/data/mmap_canonical_loose_truthseg_event_v7_01"

FIXED_WIDTH = 256
FIXED_DEPTH = 12
TRIAL_STEPS = 5000
TRIAL_DATA_FRACTION = 0.20

os.makedirs(STUDY_DIR, exist_ok=True)

# Fixed, non-searched settings per model -- everything confirmed in the
# model_configs_preview.yaml review. Only the aux-related lines differ.
MODEL_CONFIGS = {
    "a": {
        "label": "pos + sincos, no aux",
        "study_db": "{STUDY_DIR}/optuna_study_sincos_modelA_klen{klen}.db",
        "study_name": "clas12_sincos_modelA_klen{klen}",
        "trial_prefix": "sincos_a_klen{klen}_trial",
        "run_num": "optuna_sincos_a_klen{klen}",
        "extra_lines": """  embed_method: pos_only
  use_aux_features: false
  return_knn_aux_target: false
  use_sincos_target: true""",
    },
    "b": {
        "label": "pos + sincos + aux pitch",
        "study_db": "{STUDY_DIR}/optuna_study_sincos_modelB_klen{klen}.db",
        "study_name": "clas12_sincos_modelB_klen{klen}",
        "trial_prefix": "sincos_b_klen{klen}_trial",
        "run_num": "optuna_sincos_b_klen{klen}",
        "extra_lines": """  embed_method: pos_plus_aux
  pos_dim: 3
  aux_dim: 4
  use_aux_features: true
  aux_extra_feature: pitch
  return_knn_aux_target: true
  use_sincos_target: true""",
    },
}


def get_default_anchor():
    """Copy the &default anchor block out of the main yaml (same approach
    as every sweep config this whole session), so trial configs are
    self-contained."""
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


def make_objective(model_key, klen):
    raw_cfg = MODEL_CONFIGS[model_key]
    cfg = dict(raw_cfg)
    cfg["study_db"] = raw_cfg["study_db"].format(STUDY_DIR=STUDY_DIR, klen=klen)
    cfg["study_name"] = raw_cfg["study_name"].format(klen=klen)
    cfg["trial_prefix"] = raw_cfg["trial_prefix"].format(klen=klen)
    cfg["run_num"] = raw_cfg["run_num"].format(klen=klen)

    def objective(trial):
        max_lr = trial.suggest_float("max_lr", 1e-5, 1e-2, log=True)
        local_batch_size = trial.suggest_categorical("local_batch_size", [8, 16, 32, 64, 128])
        warmup_steps = trial.suggest_int("warmup_steps", 50, 1000, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
        grad_clip_value = trial.suggest_float("grad_clip_value", 0.05, 2.0, log=True)

        name = f"{cfg['trial_prefix']}_{trial.number}"
        block = f"""
{name}:
  <<: *default
  data_root: {DATA_ROOT}
  reader_type: ragged_mmap
  stat_dir: {REPO}/data/stats
  checkpoint_dir: {CKPT_ROOT}
  klen: {klen}
{cfg['extra_lines']}
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
            f"--run_num={cfg['run_num']}", f"--root_dir={CKPT_ROOT}",
        ]
        # [FIX] trial 0 (real, 5+ min training) succeeded; trials 1/2
        # (5 sec each) failed instantly on "No CUDA GPUs are available",
        # even though nvidia-smi showed the GPU idle and `torch.cuda.
        # is_available()` returned True moments later in the same shell.
        # This points at the GPU needing a brief moment to actually
        # release after a subprocess exits (plausibly specific to this
        # machine's unified-memory architecture) -- launching the next
        # subprocess immediately can race that. Retry once after a short
        # cooldown before giving up on a trial for a CUDA-availability
        # failure specifically (a real training failure still fails fast).
        MAX_ATTEMPTS = 2
        COOLDOWN_SECONDS = 10
        result = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            with open(log_path, "w") as logf:
                result = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                break
            log_text = open(log_path).read()
            if "No CUDA GPUs are available" in log_text and attempt < MAX_ATTEMPTS:
                print(f"[{model_key}:{trial.number}] CUDA not ready yet (attempt {attempt}/{MAX_ATTEMPTS}), "
                      f"waiting {COOLDOWN_SECONDS}s and retrying...")
                time.sleep(COOLDOWN_SECONDS)
                continue
            break

        if result.returncode != 0:
            print(f"[{model_key}:{trial.number}] FAILED (see {log_path})")
            raise optuna.TrialPruned()

        # cooldown after every successful trial too, so the NEXT trial
        # doesn't immediately race the same release-timing issue
        time.sleep(COOLDOWN_SECONDS)

        text = open(log_path).read()
        matches = re.findall(r"Val loss\s*=\s*([0-9.eE+-]+)", text)
        if not matches:
            print(f"[{model_key}:{trial.number}] no val loss found in log")
            raise optuna.TrialPruned()
        final_val = float(matches[-1])
        print(f"[{model_key}:{trial.number}] lr={max_lr:.2e} bs={local_batch_size} warmup={warmup_steps} "
              f"dropout={dropout:.3f} wd={weight_decay:.2e} clip={grad_clip_value:.3f} "
              f"-> val_loss={final_val:.6g}")
        return final_val

    return objective


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["a", "b"], required=True,
                   help="a = pos+sincos no aux, b = pos+sincos+aux pitch")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--klen", type=int, default=1, choices=[1, 2, 3, 5],
                   help="kNNN neighbor count. klen=1 was already searched; "
                        "klen=2/3 get their own fully separate study, since "
                        "the optimal recipe may genuinely differ when "
                        "predicting more neighbors at once (never tested).")
    args = p.parse_args()

    raw_cfg = MODEL_CONFIGS[args.model]
    study_db = raw_cfg["study_db"].format(STUDY_DIR=STUDY_DIR, klen=args.klen)
    study_name = raw_cfg["study_name"].format(klen=args.klen)
    print(f"=== Model {args.model.upper()}: {raw_cfg['label']}  (klen={args.klen}) ===")
    print(f"study db: {study_db}\n")

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_db}",
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(make_objective(args.model, args.klen), n_trials=args.n_trials)

    print("\n=== BEST TRIAL ===")
    print(f"val_loss: {study.best_value:.6g}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nFull study saved to: {study_db}")
    print("Resume anytime by re-running this script with the same --model (load_if_exists=True).")