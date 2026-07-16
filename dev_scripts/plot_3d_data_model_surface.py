"""
3D surface: model width x dataset size (the "floor") vs final validation loss ("height").

Discovers all scale_w{W}_d12_n{N} runs by globbing checkpoints/ -- reads the width
and the true event count straight from each run's name (no hardcoding), pulls the
final val loss from each CSV, and plots the surface.

Run: python3 dev_scripts/plot_3d_data_model_surface.py   (from /workspace/PP_collision)
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
OUT = "/workspace/PP_collision/sweep_analysis/scaling_surface_data_vs_model.png"
DEPTH = 6   # this sweep held depth fixed

def final_val_loss(run_dir):
    name = os.path.basename(run_dir)
    csv = f"{run_dir}/{RUN_NUM}/config_{name}_run_{RUN_NUM}.csv"
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv, header=None, names=["split", "step", "loss", "lr"])
    v = df[df["split"] == "val"].sort_values("step")
    return float(v["loss"].iloc[-1]) if len(v) else None

# --- discover runs: scale_w{W}_d12_n{N} ---
runs = {}   # (width, nevents) -> loss
pat = re.compile(rf"scale_w(\d+)_d{DEPTH}_n(\d+)$")
print("Discovered runs:")
for d in sorted(glob.glob(f"{CKPT_ROOT}/scale_w*_d{DEPTH}_n*")):
    m = pat.match(os.path.basename(d))
    if not m:
        continue
    w, n = int(m.group(1)), int(m.group(2))
    # exclude the OLD small-data sweep (n~8906/8916) if present
    if n < 20000:
        continue
    loss = final_val_loss(d)
    if loss is None:
        print(f"  w={w:<5} n={n:<9} -- no val row, skipped")
        continue
    runs[(w, n)] = loss
    print(f"  w={w:<5} n={n:<9} val loss {loss:.6g}")

if not runs:
    raise SystemExit("No runs found. Check checkpoints/ and naming.")

widths = sorted({w for w, _ in runs})
nevents = sorted({n for _, n in runs})
print(f"\nGrid: {len(widths)} widths x {len(nevents)} dataset sizes")

Z = np.full((len(widths), len(nevents)), np.nan)
for i, w in enumerate(widths):
    for j, n in enumerate(nevents):
        if (w, n) in runs:
            Z[i, j] = runs[(w, n)]

missing = int(np.isnan(Z).sum())
if missing:
    print(f"NOTE: {missing} grid cells missing (will show as gaps).")

# --- 3D surface: x = log10(width), y = log10(events), z = val loss ---
logW = np.log10(widths)
logN = np.log10(nevents)
X, Y = np.meshgrid(logW, logN, indexing="ij")

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
Zm = np.ma.masked_invalid(Z)
surf = ax.plot_surface(X, Y, Zm, cmap="viridis", edgecolor="k", linewidth=0.3, alpha=0.9)

# actual measured points
for i, w in enumerate(widths):
    for j, n in enumerate(nevents):
        if not np.isnan(Z[i, j]):
            ax.scatter(logW[i], logN[j], Z[i, j], color="red", s=25, depthshade=False)

ax.set_xlabel("model width")
ax.set_xticks(logW); ax.set_xticklabels([str(w) for w in widths])
ax.set_ylabel("training events")
ax.set_yticks(logN)
ax.set_yticklabels([f"{n/1000:.0f}k" if n < 1e6 else f"{n/1e6:.2f}M" for n in nevents])
ax.set_zlabel("final validation loss")
ax.set_title(f"Scaling surface: model size x dataset size (depth {DEPTH})")
ax.invert_yaxis()
fig.colorbar(surf, ax=ax, shrink=0.6, label="val loss")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.tight_layout(); fig.savefig(OUT, dpi=150)
print(f"\nSaved: {OUT}")