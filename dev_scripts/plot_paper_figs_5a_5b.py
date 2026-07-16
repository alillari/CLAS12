"""
Reproduce FM4NPP Figure 5(a) and 5(b) style plots from the depth-12 sweep
(scale_w{W}_d12_n{N} runs):

  Fig 5(a): validation MSE vs MODEL PARAMETERS (log-log), at full (100%) data.
            One point per width, power-law fit overlaid.

  Fig 5(b): validation MSE vs TRAINING SPACEPOINT COUNT (log-log), at one
            fixed model width (see FIXED_WIDTH below -- edit to change it).
            One point per data fraction, power-law fit overlaid.

Run: python3 dev_scripts/plot_paper_figs_5a_5b.py   (from /workspace/PP_collision)
"""
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.models.mambagpt import Mamba1GPT

# ============================================================
# EDIT THIS to change which width Figure 5(b) is fixed at:
FIXED_WIDTH = 256
# ============================================================

CKPT_ROOT = "/workspace/PP_collision/checkpoints"
RUN_NUM = "sweep"
DEPTH = 12
OUT_DIR = "/workspace/PP_collision/sweep_analysis"

def final_val_loss(run_dir):
    name = os.path.basename(run_dir)
    csv = f"{run_dir}/{RUN_NUM}/config_{name}_run_{RUN_NUM}.csv"
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv, header=None, names=["split", "step", "loss", "lr"])
    v = df[df["split"] == "val"].sort_values("step")
    return float(v["loss"].iloc[-1]) if len(v) else None

def param_count(width, depth):
    m = Mamba1GPT(embed_dim=width, num_layers=depth, d_state=16, klen=1,
                  dropout=0.1, embed_method='pos_only', pe_method='nerf',
                  band_classification=False, n_bands=6)
    return sum(p.numel() for p in m.parameters())

# --- discover all depth-12 runs (exclude old small-data sweep, n<20000) ---
pat = re.compile(rf"scale_w(\d+)_d{DEPTH}_n(\d+)$")
runs = {}  # (width, n) -> loss
for d in sorted(glob.glob(f"{CKPT_ROOT}/scale_w*_d{DEPTH}_n*")):
    m = pat.match(os.path.basename(d))
    if not m:
        continue
    w, n = int(m.group(1)), int(m.group(2))
    if n < 20000:
        continue
    loss = final_val_loss(d)
    if loss is not None:
        runs[(w, n)] = loss

widths = sorted({w for w, _ in runs})
max_n = max(n for _, n in runs)  # treat the largest n seen as "100% data"
print(f"Widths found: {widths}")
print(f"Max (100%) event count: {max_n}")

def power_law_fit(x, y):
    logx, logy = np.log10(x), np.log10(y)
    b, c = np.polyfit(logx, logy, 1)
    a = 10 ** (-c / b)
    return a, b

# ============================================================
# Figure 5(a): loss vs params, at full data
# ============================================================
pts_a = [(w, runs[(w, max_n)]) for w in widths if (w, max_n) in runs]
if len(pts_a) < 2:
    print("WARNING: not enough full-data points for Fig 5(a).")
else:
    ws, losses_a = zip(*pts_a)
    params = np.array([param_count(w, DEPTH) for w in ws], dtype=float)
    losses_a = np.array(losses_a, dtype=float)
    a, b = power_law_fit(params, losses_a)
    fit_x = np.logspace(np.log10(params.min())*0.98, np.log10(params.max())*1.02, 100)
    fit_y = (fit_x / a) ** b

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(fit_x, fit_y, "k--", lw=1.5, label=fr"$L=(M/{a:.3g})^{{{b:.3f}}}$")
    ax.scatter(params, losses_a, s=70, color="#2a9d8f", zorder=5, edgecolors="black", linewidths=0.5)
    for w, x, y in zip(ws, params, losses_a):
        ax.annotate(f"w{w}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Model Parameters (M)"); ax.set_ylabel("MSE (L)")
    ax.set_title(f"(a) Model scaling  (full data, n={max_n:,})")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout()
    out_a = f"{OUT_DIR}/fig5a_model_scaling.png"
    fig.savefig(out_a, dpi=150)
    print(f"Fig 5(a) fit: L = (M/{a:.4g})^{b:.4f}")
    print(f"Saved: {out_a}")

# ============================================================
# Figure 5(b): loss vs data size, at FIXED_WIDTH
# ============================================================
pts_b = [(n, loss) for (w, n), loss in runs.items() if w == FIXED_WIDTH]
if len(pts_b) < 2:
    print(f"WARNING: not enough data points at width {FIXED_WIDTH} for Fig 5(b).")
else:
    pts_b.sort()
    ns, losses_b = zip(*pts_b)
    ns = np.array(ns, dtype=float); losses_b = np.array(losses_b, dtype=float)
    a2, b2 = power_law_fit(ns, losses_b)
    fit_x2 = np.logspace(np.log10(ns.min())*0.98, np.log10(ns.max())*1.02, 100)
    fit_y2 = (fit_x2 / a2) ** b2

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(fit_x2, fit_y2, "k--", lw=1.5, label=fr"$L=(D/{a2:.3g})^{{{b2:.3f}}}$")
    ax.scatter(ns, losses_b, s=70, color="#e76f51", zorder=5, edgecolors="black", linewidths=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Spacepoint / Event Count (D)"); ax.set_ylabel("MSE (L)")
    ax.set_title(f"(b) Data scaling  (width={FIXED_WIDTH})")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout()
    out_b = f"{OUT_DIR}/fig5b_data_scaling.png"
    fig.savefig(out_b, dpi=150)
    print(f"Fig 5(b) fit: L = (D/{a2:.4g})^{b2:.4f}")
    print(f"Saved: {out_b}")