"""
FM4NPP Figure 5(a) style plot: validation MSE vs model parameters (log-log)
with a fitted power law, built from a sweep folder.

Reads each run's latest val loss from its CSV. Runs that have not yet reached
total_steps are still plotted, but as OPEN markers with a note -- an unfinished
run's loss is not comparable to a converged one, so the distinction matters.

Edit SWEEP_DIR for a different sweep. Use MANUAL_LOSSES to override/supply a
value by width (e.g. {1536: 6.4e-05}).

Run: python3 dev_scripts/plot_fig5a.py     (from /workspace/PP_collision)
"""
import os, glob, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.models.mambagpt import Mamba1GPT

# ============================================================
SWEEP_DIR = "/workspace/PP_collision/checkpoints/LR_Full_Test_1_2026-07-20"
RUN_NUM   = "sweep"
DEPTH     = 12
# Supply/override a width's loss manually, e.g. {1536: 6.4e-05}
MANUAL_LOSSES = {}
# Fit the power law using only fully-converged runs (recommended)
FIT_COMPLETED_ONLY = True
OUT = f"{SWEEP_DIR}/fig5a_model_scaling.png"
# ============================================================

M_LABEL = {64:'m1', 128:'m2', 256:'m3', 512:'m4', 1024:'m5', 1536:'m6'}


def read_run(run_dir):
    """Return (latest_val_loss, last_step, total_steps or None)."""
    name = os.path.basename(run_dir)
    csv = f"{run_dir}/{RUN_NUM}/config_{name}_run_{RUN_NUM}.csv"
    if not os.path.exists(csv):
        return None, None, None
    df = pd.read_csv(csv, header=None, names=["split", "step", "loss", "lr"])
    v = df[df["split"] == "val"].sort_values("step")
    if len(v) == 0:
        return None, None, None
    return float(v["loss"].iloc[-1]), int(v["step"].iloc[-1]), None


def total_steps_from_yaml(width):
    y = f"{SWEEP_DIR}/_grid_generated.yaml"
    if not os.path.exists(y):
        return None
    txt = open(y).read()
    m = re.search(r"total_steps:\s*(\d+)", txt)
    return int(m.group(1)) if m else None


def param_count(width, depth):
    m = Mamba1GPT(embed_dim=width, num_layers=depth, d_state=16, klen=1,
                  dropout=0.1, embed_method='pos_only', pe_method='nerf')
    return sum(p.numel() for p in m.parameters())


rows = []
pat = re.compile(rf"scale_w(\d+)_d{DEPTH}_n(\d+)$")
for d in sorted(glob.glob(f"{SWEEP_DIR}/scale_w*_d{DEPTH}_n*")):
    m = pat.match(os.path.basename(d))
    if not m:
        continue
    w = int(m.group(1))
    loss, last_step, _ = read_run(d)
    if w in MANUAL_LOSSES:
        loss, last_step = MANUAL_LOSSES[w], last_step
    if loss is None:
        print(f"  w{w}: no val data yet -- skipped")
        continue
    tot = total_steps_from_yaml(w)
    done = (tot is not None and last_step is not None and last_step >= tot)
    rows.append(dict(width=w, loss=loss, last_step=last_step, total=tot, done=done))

for w, loss in MANUAL_LOSSES.items():
    if not any(r["width"] == w for r in rows):
        rows.append(dict(width=w, loss=loss, last_step=None, total=None, done=False))

if len(rows) < 2:
    raise SystemExit("Need at least 2 runs with val data to plot.")

rows.sort(key=lambda r: r["width"])
print(f"{'width':>6} {'m':>3} {'params':>12} {'val loss':>12} {'steps':>16}  status")
for r in rows:
    r["params"] = param_count(r["width"], DEPTH)
    step_s = f"{r['last_step']}/{r['total']}" if r["last_step"] and r["total"] else "-"
    print(f"{r['width']:>6} {M_LABEL.get(r['width'],'?'):>3} {r['params']:>12,} "
          f"{r['loss']:>12.6g} {step_s:>16}  {'complete' if r['done'] else 'IN PROGRESS'}")

fit_rows = [r for r in rows if r["done"]] if FIT_COMPLETED_ONLY else rows
if len(fit_rows) < 2:
    print("\nNot enough completed runs to fit; fitting all points instead.")
    fit_rows = rows

P = np.array([r["params"] for r in fit_rows], float)
L = np.array([r["loss"] for r in fit_rows], float)
b, c = np.polyfit(np.log10(P), np.log10(L), 1)
a = 10 ** (-c / b)

allP = np.array([r["params"] for r in rows], float)
fx = np.logspace(np.log10(allP.min()) * 0.98, np.log10(allP.max()) * 1.02, 200)
fy = (fx / a) ** b

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(fx, fy, "k--", lw=1.4, label=fr"$L=(M/{a:.3g})^{{{b:.3f}}}$")

done = [r for r in rows if r["done"]]
prog = [r for r in rows if not r["done"]]
if done:
    ax.scatter([r["params"] for r in done], [r["loss"] for r in done],
               s=70, color="#2a9d8f", zorder=5, edgecolors="black",
               linewidths=0.6, label="completed")
if prog:
    ax.scatter([r["params"] for r in prog], [r["loss"] for r in prog],
               s=80, facecolors="none", edgecolors="#e76f51",
               linewidths=1.8, zorder=5, label="in progress (not converged)")

for r in rows:
    ax.annotate(M_LABEL.get(r["width"], f"w{r['width']}"),
                (r["params"], r["loss"]), textcoords="offset points",
                xytext=(7, 6), fontsize=10)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Model Parameters (M)")
ax.set_ylabel("MSE (L)")
ax.set_title("(a) Model scaling")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"\nFit (on {len(fit_rows)} points): L = (M / {a:.4g})^{b:.4f}")
print(f"Saved: {OUT}")