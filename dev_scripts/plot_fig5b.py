"""
FM4NPP Figure 5(b) style plot: validation MSE vs training-set size (log-log)
with a fitted power law.

Fixed model (m3 / w256, depth 12), swept data fraction. Reads each run's
val loss from its CSV; completeness is judged from the run's checkpoint
(`iters`), which is the reliable record -- the printed logs lag behind.

Run: python3 dev_scripts/plot_fig5b.py     (from /workspace/PP_collision)
"""
import os, glob, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ============================================================
SWEEP_DIR   = "/workspace/PP_collision/checkpoints/Data_Fraction_Test_1_2026-07-22"
RUN_NUM     = "sweep"
WIDTH       = 256
DEPTH       = 12
TOTAL_STEPS = 100000
# Plot x-axis as spacepoints (events x ~18 hits) instead of events?
USE_SPACEPOINTS = False
HITS_PER_EVENT  = 18
# Supply/override a loss by event count, e.g. {5483352: 6.4e-05}
MANUAL_LOSSES = {}
FIT_COMPLETED_ONLY = True
OUT = f"{SWEEP_DIR}/fig5b_data_scaling.png"
# ============================================================

TOTAL_EVENTS = 5483352


def latest_val(run_dir):
    name = os.path.basename(run_dir)
    csv = f"{run_dir}/{RUN_NUM}/config_{name}_run_{RUN_NUM}.csv"
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv, header=None, names=["split", "step", "loss", "lr"])
    v = df[df["split"] == "val"].sort_values("step")
    return float(v["loss"].iloc[-1]) if len(v) else None


def ckpt_iters(run_dir):
    ck = f"{run_dir}/{RUN_NUM}/training_checkpoints/ckpt.tar"
    if not os.path.exists(ck):
        return None
    try:
        return int(torch.load(ck, map_location="cpu", weights_only=False)["iters"])
    except Exception:
        return None


rows = []
pat = re.compile(rf"scale_w{WIDTH}_d{DEPTH}_n(\d+)$")
for d in sorted(glob.glob(f"{SWEEP_DIR}/scale_w{WIDTH}_d{DEPTH}_n*")):
    m = pat.match(os.path.basename(d))
    if not m:
        continue
    n = int(m.group(1))
    loss = MANUAL_LOSSES.get(n, latest_val(d))
    if loss is None:
        print(f"  n={n}: no val data -- skipped")
        continue
    it = ckpt_iters(d)
    rows.append(dict(n=n, loss=loss, iters=it,
                     done=(it is not None and it >= TOTAL_STEPS)))

for n, loss in MANUAL_LOSSES.items():
    if not any(r["n"] == n for r in rows):
        rows.append(dict(n=n, loss=loss, iters=None, done=False))

if len(rows) < 2:
    raise SystemExit("Need at least 2 runs with val data to plot.")

rows.sort(key=lambda r: r["n"])
print(f"{'events':>10} {'frac':>7} {'val loss':>12} {'iters':>9}  status")
for r in rows:
    frac = 100.0 * r["n"] / TOTAL_EVENTS
    print(f"{r['n']:>10,} {frac:>6.1f}% {r['loss']:>12.6g} "
          f"{(r['iters'] if r['iters'] is not None else '-'):>9}  "
          f"{'complete' if r['done'] else 'IN PROGRESS'}")

fit_rows = [r for r in rows if r["done"]] if FIT_COMPLETED_ONLY else rows
if len(fit_rows) < 2:
    print("\nNot enough completed runs to fit; fitting all points.")
    fit_rows = rows

scale = HITS_PER_EVENT if USE_SPACEPOINTS else 1
xlabel = "Spacepoint Count (D)" if USE_SPACEPOINTS else "Training Events (D)"

X = np.array([r["n"] * scale for r in fit_rows], float)
L = np.array([r["loss"] for r in fit_rows], float)
b, c = np.polyfit(np.log10(X), np.log10(L), 1)
a = 10 ** (-c / b)

allX = np.array([r["n"] * scale for r in rows], float)
fx = np.logspace(np.log10(allX.min()) * 0.98, np.log10(allX.max()) * 1.02, 200)
fy = (fx / a) ** b

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(fx, fy, "k--", lw=1.4, label=fr"$L=(D/{a:.3g})^{{{b:.3f}}}$")

done = [r for r in rows if r["done"]]
prog = [r for r in rows if not r["done"]]
if done:
    ax.scatter([r["n"] * scale for r in done], [r["loss"] for r in done],
               s=70, color="#e76f51", zorder=5, edgecolors="black",
               linewidths=0.6, label="completed")
if prog:
    ax.scatter([r["n"] * scale for r in prog], [r["loss"] for r in prog],
               s=80, facecolors="none", edgecolors="#264653",
               linewidths=1.8, zorder=5, label="in progress")

for r in rows:
    frac = 100.0 * r["n"] / TOTAL_EVENTS
    lbl = f"{frac:.0f}%" if frac >= 1 else f"{frac:.1f}%"
    ax.annotate(lbl, (r["n"] * scale, r["loss"]), textcoords="offset points",
                xytext=(7, 6), fontsize=9)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(xlabel)
ax.set_ylabel("MSE (L)")
ax.set_title(f"(b) Data scaling  (m3: width {WIDTH}, depth {DEPTH})")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"\nFit (on {len(fit_rows)} points): L = (D / {a:.4g})^{b:.4f}")
print(f"Saved: {OUT}")
