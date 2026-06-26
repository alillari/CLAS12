"""
Analyze the CLAS12 k-sweep (real vs straight-track, k=1..5).

Reads the per-run global CSV logs (split,step,loss,lr) written by the trainer,
produces a standard set of ML-research plots, and prints a summary table.

CSV location pattern:
  <CKPT_ROOT>/<config>/sweep/config_<config>_run_sweep.csv

Run inside the container (needs matplotlib/pandas; install if missing:
  pip install matplotlib pandas):
    python3 analyze_sweep.py
Outputs PNGs to <OUT_DIR>.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

CKPT_ROOT = "/workspace/PP_collision/checkpoints"
OUT_DIR = "/workspace/PP_collision/sweep_analysis"
KS = [1, 2, 3, 4, 5]
RUN_NUM = "sweep"

os.makedirs(OUT_DIR, exist_ok=True)

# Colorblind-friendly: real = blues, straight = oranges, k sets shade
def shade(base_rgb, k, n=5):
    # darker as k increases
    f = 0.35 + 0.65 * (k - 1) / (n - 1)
    return tuple(c * f for c in base_rgb)

REAL_BASE = (0.12, 0.40, 0.85)
STRAIGHT_BASE = (0.95, 0.55, 0.10)

# ============================================================
# LOAD ALL RUNS
# ============================================================

def csv_path(config):
    return os.path.join(CKPT_ROOT, config, RUN_NUM,
                        f"config_{config}_run_{RUN_NUM}.csv")

runs = {}   # (kind, k) -> DataFrame
missing = []
for kind in ["real", "straight"]:
    for k in KS:
        config = f"sweep_{kind}_k{k}"
        p = csv_path(config)
        if not os.path.exists(p):
            missing.append((config, p))
            continue
        df = pd.read_csv(p)
        runs[(kind, k)] = df

print(f"Loaded {len(runs)} runs.")
if missing:
    print("MISSING (run may not have finished yet):")
    for c, p in missing:
        print(f"  {c}: {p}")

if not runs:
    raise SystemExit("No run CSVs found — check CKPT_ROOT / RUN_NUM / that the sweep has produced output.")

def get(kind, k, split):
    df = runs.get((kind, k))
    if df is None:
        return None
    sub = df[df["split"] == split].sort_values("step")
    return sub

# ============================================================
# PLOT 1: Training loss curves (all runs, log-y)
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))
for (kind, k), df in runs.items():
    sub = get(kind, k, "train")
    if sub is None or len(sub) == 0:
        continue
    base = REAL_BASE if kind == "real" else STRAIGHT_BASE
    ax.plot(sub["step"], sub["loss"], color=shade(base, k),
            label=f"{kind} k={k}", linewidth=1.3, alpha=0.9)
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("training loss (log scale)")
ax.set_title("Training loss curves — all runs")
ax.legend(ncol=2, fontsize=8)
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_train_loss_curves.png"), dpi=150)
plt.close(fig)

# ============================================================
# PLOT 2: Validation loss curves (all runs, log-y)
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))
for (kind, k), df in runs.items():
    sub = get(kind, k, "val")
    if sub is None or len(sub) == 0:
        continue
    base = REAL_BASE if kind == "real" else STRAIGHT_BASE
    ax.plot(sub["step"], sub["loss"], color=shade(base, k),
            marker="o", markersize=3, label=f"{kind} k={k}", linewidth=1.3)
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("validation loss (log scale)")
ax.set_title("Validation loss curves — all runs")
ax.legend(ncol=2, fontsize=8)
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_val_loss_curves.png"), dpi=150)
plt.close(fig)

# ============================================================
# Helper: final & best val loss per run
# ============================================================

def final_val(kind, k):
    sub = get(kind, k, "val")
    if sub is None or len(sub) == 0:
        return np.nan
    return sub["loss"].iloc[-1]

def best_val(kind, k):
    sub = get(kind, k, "val")
    if sub is None or len(sub) == 0:
        return np.nan
    return sub["loss"].min()

def final_train(kind, k):
    sub = get(kind, k, "train")
    if sub is None or len(sub) == 0:
        return np.nan
    # average last few train points to smooth per-batch noise
    return sub["loss"].tail(10).mean()

# ============================================================
# PLOT 3: HEADLINE — final val loss vs k, real vs straight
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))
real_finals = [final_val("real", k) for k in KS]
straight_finals = [final_val("straight", k) for k in KS]
ax.plot(KS, real_finals, "o-", color=REAL_BASE, linewidth=2, markersize=8, label="real data")
ax.plot(KS, straight_finals, "s-", color=STRAIGHT_BASE, linewidth=2, markersize=8, label="straight tracks (baseline)")
ax.set_xlabel("k (number of neighbors predicted)")
ax.set_ylabel("final validation loss")
ax.set_title("Final loss vs k — real vs straight-track baseline\n(headline: gap = structure model must learn beyond trivial)")
ax.set_xticks(KS)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_HEADLINE_final_loss_vs_k.png"), dpi=150)
plt.close(fig)

# ============================================================
# PLOT 4: Real-minus-straight gap vs k
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))
gap = [r - s for r, s in zip(real_finals, straight_finals)]
ratio = [r / s if s and not np.isnan(s) else np.nan for r, s in zip(real_finals, straight_finals)]
ax.bar([k - 0.0 for k in KS], gap, width=0.5, color="purple", alpha=0.7)
ax.set_xlabel("k")
ax.set_ylabel("real loss − straight loss (absolute gap)")
ax.set_title("Excess loss on real data vs straight baseline, per k\n(how much harder real physics is than trivial straight tracks)")
ax.set_xticks(KS)
ax.grid(True, alpha=0.3, axis="y")
# annotate ratio on top
for k, g, rt in zip(KS, gap, ratio):
    if not np.isnan(g):
        ax.text(k, g, f"{rt:.1f}x" if not np.isnan(rt) else "", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_real_minus_straight_gap.png"), dpi=150)
plt.close(fig)

# ============================================================
# PLOT 5: Learning rate schedule (sanity)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
sub = get("real", 1, "train")
if sub is not None and "lr" in sub:
    ax.plot(sub["step"], sub["lr"], color="black", linewidth=1.5)
ax.set_xlabel("training step")
ax.set_ylabel("learning rate")
ax.set_title("Learning rate schedule (warmup + cosine) — sanity check")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_lr_schedule.png"), dpi=150)
plt.close(fig)

# ============================================================
# PLOT 6: Train vs val gap per run (overfitting check)
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))
width = 0.35
x = np.arange(len(KS))
for i, kind in enumerate(["real", "straight"]):
    ft = [final_train(kind, k) for k in KS]
    fv = [final_val(kind, k) for k in KS]
    base = REAL_BASE if kind == "real" else STRAIGHT_BASE
    offset = (i - 0.5) * width
    ax.bar(x + offset, fv, width, color=base, alpha=0.9, label=f"{kind} val")
    ax.plot(x + offset, ft, "k_", markersize=14, markeredgewidth=2,
            label=f"{kind} train" if i == 0 else None)
ax.set_xticks(x)
ax.set_xticklabels([f"k={k}" for k in KS])
ax.set_ylabel("final loss")
ax.set_title("Train (dash) vs Val (bar) final loss — overfitting check")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "06_train_val_gap.png"), dpi=150)
plt.close(fig)

# ============================================================
# PLOT 7: Convergence speed (steps to reach a loss threshold)
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))
threshold = 0.01
for kind in ["real", "straight"]:
    steps_to = []
    for k in KS:
        sub = get(kind, k, "train")
        if sub is None or len(sub) == 0:
            steps_to.append(np.nan); continue
        below = sub[sub["loss"] <= threshold]
        steps_to.append(below["step"].iloc[0] if len(below) else np.nan)
    base = REAL_BASE if kind == "real" else STRAIGHT_BASE
    ax.plot(KS, steps_to, "o-", color=base, linewidth=2, markersize=8, label=kind)
ax.set_xlabel("k")
ax.set_ylabel(f"steps to reach train loss < {threshold}")
ax.set_title(f"Convergence speed (steps to cross {threshold})")
ax.set_xticks(KS)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "07_convergence_speed.png"), dpi=150)
plt.close(fig)

# ============================================================
# SUMMARY TABLE (printed + saved csv)
# ============================================================

rows = []
for kind in ["real", "straight"]:
    for k in KS:
        rows.append({
            "kind": kind,
            "k": k,
            "final_train": round(final_train(kind, k), 6),
            "final_val": round(final_val(kind, k), 6),
            "best_val": round(best_val(kind, k), 6),
        })
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)

print("\n=== SUMMARY TABLE ===")
print(summary.to_string(index=False))

print("\n=== real vs straight (final val) ===")
for k in KS:
    r, s = final_val("real", k), final_val("straight", k)
    if not (np.isnan(r) or np.isnan(s)):
        print(f"  k={k}: real={r:.5f}  straight={s:.5f}  gap={r-s:+.5f}  ratio={r/s:.2f}x")

print(f"\nPlots + summary saved to: {OUT_DIR}")