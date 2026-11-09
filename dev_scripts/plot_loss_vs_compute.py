"""
Loss vs. compute (FLOPs) curves, one line per width, for both the original
flat-100k-steps sweep and the Chinchilla-corrected sweep, side by side.

Compute follows the standard approximation used throughout the scaling-law
literature (Kaplan et al. 2020, Hoffmann et al. 2022 "Chinchilla"):
    C ~= 6 * N_params * D_tokens_seen
(6x accounts for forward + backward pass; "tokens" = points/hits processed,
directly analogous to word tokens in an LLM, since the backbone consumes
the point cloud as a sequence).

Each width's FULL logged trajectory is plotted (not just the final value),
so you can see the curve bend and where it's actually heading -- its
"end point" -- not just where it currently sits.

Run: python3 dev_scripts/plot_loss_vs_compute.py
"""
import os, glob, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.models.mambagpt import Mamba1GPT

# ============================================================
OLD_SWEEP_DIR = "/workspace/PP_collision/checkpoints/Unbiased_Cluster_Full_Sweep_2026-08-13"
OLD_RUN_NUM = "sweep"
OLD_CONFIG_PATTERN = "scale_w{w}_d12_n39553933"

CHINCHILLA_ROOT = "/workspace/PP_collision/checkpoints"
CHINCHILLA_RUN_NUM = "sweep"
CHINCHILLA_CONFIG_PATTERN = "scale_w{w}_d12_chinchilla"

DEPTH = 12
BATCH_SIZE = 64
AVG_POINTS_PER_EVENT = 35.2
WIDTHS = [64, 128, 256, 512, 1024, 1536]
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5', 1536: 'm6'}
COLORS = {64: '#264653', 128: '#2a9d8f', 256: '#8ab17d', 512: '#e9c46a',
         1024: '#f4a261', 1536: '#e76f51'}
OUT = "/workspace/PP_collision/fig_loss_vs_compute.png"
# ============================================================


def find_chinchilla_sweep_dir():
    candidates = sorted(glob.glob(f"{CHINCHILLA_ROOT}/Chinchilla_Sweep_*"))
    if not candidates:
        raise SystemExit("No Chinchilla_Sweep_* folder found.")
    return candidates[-1]


def param_count(width, depth, klen=1):
    m = Mamba1GPT(embed_dim=width, num_layers=depth, d_state=16, klen=klen,
                  dropout=0.1, embed_method='pos_only', pe_method='nerf')
    return sum(p.numel() for p in m.parameters())


def read_curve(run_dir, config_name, run_num, n_params):
    """Returns (compute_array, loss_array) for every logged val point."""
    csv = f"{run_dir}/{config_name}/{run_num}/config_{config_name}_run_{run_num}.csv"
    if not os.path.exists(csv):
        return None, None
    df = pd.read_csv(csv, header=None, names=["split", "step", "loss", "lr"])
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df.dropna(subset=["step", "loss"])
    v = df[df["split"] == "val"].sort_values("step")
    if len(v) == 0:
        return None, None
    tokens_seen = v["step"].values * BATCH_SIZE * AVG_POINTS_PER_EVENT
    compute = 6.0 * n_params * tokens_seen
    return compute, v["loss"].values


def plot_panel(ax, sweep_dir, config_pattern, run_num, title):
    any_data = False
    for w in WIDTHS:
        n_params = param_count(w, DEPTH)
        cfg = config_pattern.format(w=w)
        compute, loss = read_curve(sweep_dir, cfg, run_num, n_params)
        if compute is None:
            print(f"  [{title}] w{w}: no data yet -- skipped")
            continue
        any_data = True
        color = COLORS[w]
        ax.plot(compute, loss, color=color, lw=1.6, alpha=0.85,
               label=f"{M_LABEL[w]} (w{w})")
        # mark the endpoint clearly
        ax.scatter([compute[-1]], [loss[-1]], color=color, s=60,
                  zorder=5, edgecolors="black", linewidths=0.8)
        print(f"  [{title}] {M_LABEL[w]}: {len(loss)} points, "
              f"final compute={compute[-1]:.3e} FLOPs, final loss={loss[-1]:.6g}")
    if not any_data:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Compute (FLOPs, ~6 x N x D)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")


chinchilla_dir = find_chinchilla_sweep_dir()
print(f"Chinchilla sweep dir: {chinchilla_dir}\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)

print("Original (flat 100k steps):")
plot_panel(axes[0], OLD_SWEEP_DIR, OLD_CONFIG_PATTERN, OLD_RUN_NUM,
          "Original (flat 100k steps)")

print("\nChinchilla-corrected:")
plot_panel(axes[1], chinchilla_dir, CHINCHILLA_CONFIG_PATTERN, CHINCHILLA_RUN_NUM,
          "Chinchilla-corrected")

axes[0].set_ylabel("Val loss (MSE)")
fig.suptitle("Loss vs. compute, per width -- dot marks each curve's current end point", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT}")