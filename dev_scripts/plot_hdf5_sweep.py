"""
Loss vs. parameters for the background-merged HDF5 sweeps (Mamba-3, positions
only, sin/cos phi, unit_v2 Hilbert, 2x Chinchilla budgets, m1-m4). Same
conventions as the earlier scaling plots: hollow circle = width hasn't reached
its step budget yet, filled = complete; power-law fit on completed points only.

Parameter counts and step budgets are read from each sweep's generated YAML, so
they are exactly what the runs used. A series whose YAML does not exist yet
(e.g. the kNNN = 1 sweep before it is generated) is skipped.

Run: python3 dev_scripts/plot_hdf5_sweep.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/workspace/PP_collision"
sys.path.insert(0, REPO)
from fm4npp.utils import YParams
from fm4npp.models.mamba_config import build_backbone

CKPT_ROOT = f"{REPO}/checkpoints"
WIDTHS = {"m1": 64, "m2": 128, "m3": 256, "m4": 512}
FIT_COMPLETED_ONLY = True
OUT = f"{REPO}/fig_hdf5_sweep.png"

SERIES = [
    dict(label="kNNN = 5", color="#D55E00",
         yaml=f"{REPO}/scripts/configs/hdf5_chinchilla_sweep.yaml", run_num="hdf5_chinchilla",
         config=lambda m, w: f"hdf5_chinchilla_{m}_w{w}"),
    dict(label="kNNN = 1", color="#0072B2",
         yaml=f"{REPO}/scripts/configs/hdf5_chinchilla_sweep_klen1.yaml", run_num="hdf5_chinchilla_klen1",
         config=lambda m, w: f"hdf5_chinchilla_klen1_{m}_w{w}"),
]


def read_run(config, run_num):
    csv = f"{CKPT_ROOT}/{config}/{run_num}/config_{config}_run_{run_num}.csv"
    if not os.path.exists(csv):
        return None, None
    df = pd.read_csv(csv, header=None, names=["split", "step", "loss", "lr"])
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    v = df.dropna(subset=["step", "loss"])
    v = v[v["split"] == "val"].sort_values("step")
    if len(v) == 0:
        return None, None
    return float(v["loss"].min()), int(v["step"].iloc[-1])


def collect(s):
    rows = []
    if not os.path.exists(s["yaml"]):
        print(f"  {s['label']}: {s['yaml']} not generated yet -- skipped")
        return rows
    for m, w in WIDTHS.items():
        cfg = s["config"](m, w)
        loss, last = read_run(cfg, s["run_num"])
        if loss is None:
            print(f"  {s['label']} {m}: not started yet -- skipped")
            continue
        p = YParams(s["yaml"], cfg)
        params = sum(t.numel() for t in build_backbone(p).parameters())
        total = int(p.total_steps)
        rows.append(dict(m=m, width=w, loss=loss, last=last, total=total,
                         done=last >= total, params=params))
    return rows


def fit_and_plot(ax, rows, color, label):
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["width"])
    fit_rows = [r for r in rows if r["done"]] if FIT_COMPLETED_ONLY else rows
    done = [r for r in rows if r["done"]]
    prog = [r for r in rows if not r["done"]]
    if done:
        ax.scatter([r["params"] for r in done], [r["loss"] for r in done], s=75, color=color,
                   zorder=5, edgecolors="black", linewidths=0.7, label=f"{label} (completed)")
    if prog:
        ax.scatter([r["params"] for r in prog], [r["loss"] for r in prog], s=90, facecolors="none",
                   edgecolors=color, linewidths=2.0, zorder=5, label=f"{label} (in progress)")
    for r in rows:
        ax.annotate(r["m"], (r["params"], r["loss"]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color=color)
    if len(fit_rows) >= 2:
        P = np.array([r["params"] for r in fit_rows], float)
        L = np.array([r["loss"] for r in fit_rows], float)
        b, c = np.polyfit(np.log10(P), np.log10(L), 1)
        a = 10 ** (-c / b)
        allP = np.array([r["params"] for r in rows], float)
        fx = np.logspace(np.log10(allP.min()) * 0.98, np.log10(allP.max()) * 1.02, 200)
        ax.plot(fx, (fx / a) ** b, "--", color=color, lw=1.4,
                label=fr"{label}: $L=(M/{a:.3g})^{{{b:.3f}}}$")
        print(f"{label} fit (on {len(fit_rows)} completed points): L = (M / {a:.4g})^{b:.4f}")


fig, ax = plt.subplots(figsize=(9, 7))
for s in SERIES:
    print(f"\n{s['label']}:")
    rows = collect(s)
    for r in rows:
        print(f"  {r['m']:>3}  params={r['params']:>12,}  best val loss={r['loss']:.6g}  "
              f"steps={r['last']:,}/{r['total']:,}  {'complete' if r['done'] else 'IN PROGRESS'}")
    fit_and_plot(ax, rows, s["color"], s["label"])

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Model Parameters (M)")
ax.set_ylabel("Validation loss (L)")
ax.set_title("Background-merged data: Mamba-3, positions only, sin/cos phi (m1-m4, 2x Chinchilla)\n"
             "(kNNN = 1 and 5 losses average over different targets -- compare trends, not absolute values)",
             fontsize=10)
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"\nSaved: {OUT}")