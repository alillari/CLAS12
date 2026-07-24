"""
Sanity-check whether a trained model's low loss is REAL or degenerate.
Loads a trained checkpoint, runs a few test events, and prints predicted vs
actual next-neighbor coordinates (unnormalized) side by side, plus summary
stats that would reveal a degenerate/collapsed solution.

Usage:
    python3 dev_scripts/inspect_trial_predictions.py <config_name> <run_num>
Example (trial 0 of the base search):
    python3 dev_scripts/inspect_trial_predictions.py optuna_trial_0 optuna
"""
import sys, os
sys.path.insert(0, "/workspace/PP_collision")
import numpy as np
import torch
from fm4npp.utils import YParams
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset
from fm4npp.models.mambagpt import Mamba1GPT

GEN_YAML = "/workspace/PP_collision/scripts/configs/_optuna_trial.yaml"
CKPT_ROOT = "/workspace/PP_collision/checkpoints"

cfg = sys.argv[1] if len(sys.argv) > 1 else "optuna_trial_0"
run = sys.argv[2] if len(sys.argv) > 2 else "optuna"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = f"{CKPT_ROOT}/{cfg}/{run}/training_checkpoints/ckpt_best.tar"
print(f"Checkpoint: {ckpt_path}")
if not os.path.exists(ckpt_path):
    raise SystemExit(f"No checkpoint at {ckpt_path} -- check config/run names.")

params = YParams(os.path.abspath(GEN_YAML), cfg)
model = Mamba1GPT(embed_dim=params.embed_dim, num_layers=params.num_layers_backbone,
                  d_state=params.d_state, klen=params.klen, dropout=params.dropout,
                  embed_method=params.embed_method, pe_method=params.pe_method).to(device)
ck = torch.load(ckpt_path, map_location=device, weights_only=False)
st = ck.get("model_state", ck.get("model_state_dict", ck))
model.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
model.eval()

ds = TPCBatchDataset(data_root=params.data_root, version='dataset_name', split='test',
                     reader_type=getattr(params, 'reader_type', 'ragged_mmap'),
                     num_pred_points=params.klen, group_size=params.group_size,
                     normalize=True, nleave=params.nleave, chunk_training=False,
                     train=False, order=params.order)

def unnorm(a): return ds.apply_unnorm(a.clone())

all_pred, all_act = [], []
with torch.no_grad():
    for i in range(5):  # inspect 5 events
        sp, seg, knn = ds[i]
        out = model(sp.unsqueeze(0).to(device)).squeeze(0).cpu()
        N = sp.shape[0]
        actual = knn.reshape(N, 3)
        valid = (actual != -100).all(dim=1)
        pred = out.reshape(N, -1)[:, :3]  # first 3 = eta,phi,r prediction (k=1)
        act_r = unnorm(actual[valid]); pred_r = unnorm(pred[valid])
        all_pred.append(pred_r); all_act.append(act_r)
        if i == 0:
            print(f"\n--- Event 0: predicted vs actual (unnormalized eta, phi, r) ---")
            for j in range(min(valid.sum().item(), 8)):
                p = pred_r[j]; a = act_r[j]
                print(f"  pred [{p[0]:+.3f} {p[1]:+.3f} {p[2]:6.2f}]   "
                      f"actual [{a[0]:+.3f} {a[1]:+.3f} {a[2]:6.2f}]")

P = torch.cat(all_pred); A = torch.cat(all_act)
print(f"\n--- Summary over {P.shape[0]} valid predictions ---")
for idx, nm in [(0,'eta'),(1,'phi'),(2,'r')]:
    pe, ae = P[:,idx], A[:,idx]
    mae = (pe-ae).abs().mean().item()
    print(f"  {nm}: pred mean={pe.mean():+.3f} std={pe.std():.3f} | "
          f"actual mean={ae.mean():+.3f} std={ae.std():.3f} | MAE={mae:.4f}")

print("\n--- Degeneracy checks ---")
print(f"  prediction std across all points (eta,phi,r): "
      f"{P[:,0].std():.4f}, {P[:,1].std():.4f}, {P[:,2].std():.4f}")
print("  (if any std ~0, the model is outputting a constant = degenerate)")
print(f"  unique r predictions (rounded to 0.1cm): {len(torch.unique((P[:,2]*10).round()))}")
print("  (should be several distinct band values, not 1)")
