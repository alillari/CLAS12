"""
3D surface: width x depth (the "floor") vs final validation loss (the "height").

Scans checkpoints/ for all scale_w{W}_d{D}_n* runs (globs the event count, so it
works regardless of what the real filtered count was), reads each one's final
val loss from its CSV, arranges into a width x depth grid, and plots a 3D surface.

When a data-fraction axis is added later, this script's glob-based discovery
will still find runs -- just re-run per fraction, or extend to facet by fraction.

Run: python3 dev_scripts/plot_3d_scaling_surface.py   (from /workspace/PP_collision)
"""
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

CKPT_ROOT = "/workspace/PP_collision/checkpoints"
RUN_NUM = "sweep"
OUT = "/workspace/PP_collision/sweep_analysis/scaling_surface_3d.png"

WIDTHS = [64, 128, 256, 512, 1024, 1536]
DEPTHS = [6, 12, 18, 24]

def find_run_dir(w, d):
    """Find the checkpoints/scale_w{w}_d{d}_n<count> dir regardless of count."""
    matches = glob.glob(f"{CKPT_ROOT}/scale_w{w}_d{d}_n*")
    return matches[0] if matches else None

def final_val_loss(run_dir):
    name = os.path.basename(run_dir)
    csv_path = f"{run_dir}/{RUN_NUM}/config_{name}_run_{RUN_NUM}.csv"
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, header=None, names=["split", "step", "loss", "lr"])
    v = df[df["split"] == "val"].sort_values("step")
    return float(v["loss"].iloc[-1]) if len(v) else None

# build the grid
Z = np.full((len(WIDTHS), len(DEPTHS)), np.nan)
n_events_seen = None
print("Reading grid:")
for i, w in enumerate(WIDTHS):
    for j, d in enumerate(DEPTHS):
        run_dir = find_run_dir(w, d)
        if run_dir is None:
            print(f"  w={w:>5} d={d:>3}: MISSING")
            continue
        loss = final_val_loss(run_dir)
        if loss is None:
            print(f"  w={w:>5} d={d:>3}: no val row in {run_dir}")
            continue
        Z[i, j] = loss
        m = re.search(r'_n(\d+)$', os.path.basename(run_dir))
        if m: n_events_seen = m.group(1)
        print(f"  w={w:>5} d={d:>3}: val loss {loss:.6g}   ({os.path.basename(run_dir)})")

if np.isnan(Z).all():
    raise SystemExit("No runs found -- check checkpoints/ and naming.")

# --- 3D surface: x = log10(width), y = depth, z = val loss ---
logW = np.log10(WIDTHS)
X, Y = np.meshgrid(logW, DEPTHS, indexing='ij')   # shape (len(WIDTHS), len(DEPTHS))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# mask NaNs for plotting (surface needs finite data; use masked array for safety)
Zm = np.ma.masked_invalid(Z)
surf = ax.plot_surface(X, Y, Zm, cmap="viridis", edgecolor="k", linewidth=0.3, alpha=0.9)

# overlay the actual data points for clarity
for i, w in enumerate(WIDTHS):
    for j, d in enumerate(DEPTHS):
        if not np.isnan(Z[i, j]):
            ax.scatter(logW[i], d, Z[i, j], color="red", s=25, depthshade=False)

ax.set_xlabel("model width (log10)")
ax.set_xticks(logW)
ax.set_xticklabels([str(w) for w in WIDTHS])
ax.set_ylabel("depth (layers)")
ax.set_yticks(DEPTHS)
ax.set_zlabel("final validation loss")
title = "Width x Depth scaling surface"
if n_events_seen:
    title += f"  (n={n_events_seen} events)"
ax.set_title(title)
fig.colorbar(surf, ax=ax, shrink=0.6, label="val loss")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"\nSaved: {OUT}")
