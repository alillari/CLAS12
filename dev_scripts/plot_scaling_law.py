"""
Scaling-law plot in the form of FM4NPP Figure 5(a):
  x = model parameters (log scale)
  y = final validation MSE (log scale)
  + fitted power-law line  L = (M / a)^b

Reads final val loss from each scale_m*_reg run's CSV and computes each model's
actual parameter count by instantiating it. Skips any run whose checkpoint/CSV
is missing.

Run INSIDE container (after the two pip installs):
    python3 plot_scaling_law.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.utils import YParams
from fm4npp.models.mambagpt import Mamba1GPT

YAML = "/workspace/PP_collision/scripts/configs/mamba_pretrain.yaml"
CKPT_ROOT = "/workspace/PP_collision/checkpoints"
RUN_NUM = "sweep"
OUT = "/workspace/PP_collision/sweep_analysis/scaling_law.png"

# (label, config_name)
MODELS = [
    ("m1", "scale_m1_reg"),
    ("m2", "scale_m2_reg"),
    ("m3", "scale_m3_reg"),
    ("m4", "scale_m4_reg"),
    ("m5", "scale_m5_reg"),
    ("m6", "scale_m6_reg"),
]

def csv_path(cfg):
    return f"{CKPT_ROOT}/{cfg}/{RUN_NUM}/config_{cfg}_run_{RUN_NUM}.csv"

def final_val_loss(cfg):
    p = csv_path(cfg)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, header=None, names=["split", "step", "loss", "lr"])
    v = df[df["split"] == "val"].sort_values("step")
    return float(v["loss"].iloc[-1]) if len(v) else None

def param_count(cfg):
    prm = YParams(os.path.abspath(YAML), cfg)
    m = Mamba1GPT(embed_dim=prm.embed_dim, num_layers=prm.num_layers_backbone,
                  d_state=prm.d_state, klen=prm.klen, dropout=prm.dropout,
                  embed_method=prm.embed_method, pe_method=prm.pe_method,
                  band_classification=getattr(prm, 'band_classification', False),
                  n_bands=getattr(prm, 'n_bands', 6))
    return sum(p.numel() for p in m.parameters())

labels, params, losses = [], [], []
print("Reading scaling runs:")
for label, cfg in MODELS:
    loss = final_val_loss(cfg)
    if loss is None:
        print(f"  SKIP {label} ({cfg}): no CSV / no val row")
        continue
    n = param_count(cfg)
    labels.append(label); params.append(n); losses.append(loss)
    print(f"  {label}: {n:,} params, val MSE {loss:.6g}")

if len(params) < 2:
    raise SystemExit("Need at least 2 completed runs to plot a scaling curve.")

params = np.array(params, dtype=float)
losses = np.array(losses, dtype=float)

# power-law fit in log-log space: log L = b*log M + c  ->  L = (M/a)^b
logM, logL = np.log10(params), np.log10(losses)
b, c = np.polyfit(logM, logL, 1)
# express as L = (M / a)^b  =>  a = 10^(-c/b)
a = 10 ** (-c / b)
fit_x = np.logspace(np.log10(params.min())*0.98, np.log10(params.max())*1.02, 100)
fit_y = (fit_x / a) ** b

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(fit_x, fit_y, "k--", lw=1.5,
        label=fr"$L = (M/{a:.3g})^{{{b:.3f}}}$")
ax.scatter(params, losses, s=70, color="#2a9d8f", zorder=5, edgecolors="black", linewidths=0.5)
for label, x, y in zip(labels, params, losses):
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=10)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Model Parameters (M)")
ax.set_ylabel("MSE (L)")
ax.set_title("Neural scaling: validation MSE vs model size")
ax.grid(True, which="both", alpha=0.25)
ax.legend()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.tight_layout(); fig.savefig(OUT, dpi=150)
print(f"\nFit: L = (M / {a:.4g})^{b:.4f}")
print(f"Saved: {OUT}")