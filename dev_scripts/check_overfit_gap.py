"""
Check train-vs-val loss gap across the m1-m6 scaling sweep, to test whether
increasing validation loss with model size is driven by overfitting.

For each size, reads the final train loss and final val loss from its CSV,
prints both plus the ratio (val/train) -- a growing ratio with width would
indicate overfitting worsens as models get bigger.

Run: python3 dev_scripts/check_overfit_gap.py   (from /workspace/PP_collision)
"""
import os
import pandas as pd

CKPT_ROOT = "/workspace/PP_collision/checkpoints"
RUN_NUM = "sweep"
MODELS = [
    ("m1", "scale_m1_reg"), ("m2", "scale_m2_reg"), ("m3", "scale_m3_reg"),
    ("m4", "scale_m4_reg"), ("m5", "scale_m5_reg"), ("m6", "scale_m6_reg"),
]

def csv_path(cfg):
    return f"{CKPT_ROOT}/{cfg}/{RUN_NUM}/config_{cfg}_run_{RUN_NUM}.csv"

print(f"{'model':<6}{'final_train':>14}{'final_val':>14}{'val/train':>12}")
for label, cfg in MODELS:
    p = csv_path(cfg)
    if not os.path.exists(p):
        print(f"{label:<6} MISSING ({p})")
        continue
    df = pd.read_csv(p, header=None, names=["split", "step", "loss", "lr"])
    train = df[df["split"] == "train"].sort_values("step")
    val = df[df["split"] == "val"].sort_values("step")
    if len(train) == 0 or len(val) == 0:
        print(f"{label:<6} incomplete data")
        continue
    ft = float(train["loss"].iloc[-1])
    fv = float(val["loss"].iloc[-1])
    print(f"{label:<6}{ft:>14.6g}{fv:>14.6g}{fv/ft:>12.2f}")
    