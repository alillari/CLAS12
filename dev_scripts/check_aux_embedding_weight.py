"""
Checks whether a pretrained pos_plus_aux backbone learned to downweight the
aux input (sx,sy,sz,pitch/length) relative to position.

Because EmbedderPosPlusAux applies its own LayerNorm to each branch right
before summing them:
    pos_embed = pos_norm(embed(pos))
    aux_embed = aux_norm(aux_proj(aux))
    out = pos_embed + aux_embed
LayerNorm renormalizes each branch to unit variance BEFORE applying its own
learned gamma (the .weight parameter) -- so whatever raw scale aux_proj
produces gets erased by the normalization. The ONLY thing that determines
each branch's actual contribution to the summed token is pos_norm.weight
vs aux_norm.weight. Comparing their magnitudes directly answers "did the
network learn to downweight aux" with no approximation involved.

Also does a real empirical check as a cross-check: runs actual held-out
data through the embedder and measures each branch's real output RMS
before the sum, in case anything subtle (per-sample variance collapse,
etc.) makes the gamma comparison alone misleading.

Run: python3 dev_scripts/check_aux_embedding_weight.py
"""
import sys
import torch

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.models.mambagpt import Mamba1GPT
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset

WIDTHS = [64, 128, 256, 512, 1024, 1536]
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5', 1536: 'm6'}
DEPTH = 12

CKPT_DIR = "/workspace/PP_collision/checkpoints/v7_01_Pitch_Full_Sweep_2026-09-17"
CKPT_PATTERN = "scale_w{w}_d12_v701pitch"
DATA_ROOT = "/workspace/PP_collision/data/mmap_canonical_loose_truthseg_event_v7_01"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path, embed_dim):
    model = Mamba1GPT(embed_dim=embed_dim, num_layers=DEPTH, d_state=16, klen=1,
                      dropout=0.1, embed_method='pos_plus_aux', pe_method='nerf',
                      pos_dim=3, aux_dim=4).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    st = ck.get("model_state", ck.get("model_state_dict", ck))
    model.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    model.eval()
    return model


print(f"{'model':>6} {'gamma pos (RMS)':>16} {'gamma aux (RMS)':>16} {'aux/pos ratio':>14}   "
      f"{'empirical pos RMS':>18} {'empirical aux RMS':>18} {'emp. ratio':>11}")

ds = TPCBatchDataset(
    data_root=DATA_ROOT, version="dataset_name", split="test", reader_type="ragged_mmap",
    num_pred_points=1, group_size=8, normalize=True, nleave=1e6,
    chunk_training=False, train=False, order="REP",
    use_aux_features=True, aux_extra_feature="pitch")

for w in WIDTHS:
    ckpt = f"{CKPT_DIR}/{CKPT_PATTERN.format(w=w)}/sweep/training_checkpoints/ckpt_best.tar"
    import os
    if not os.path.exists(ckpt):
        print(f"{M_LABEL[w]:>6}  checkpoint not found -- skipped")
        continue

    model = load_model(ckpt, w)
    embedder = model.embedder

    # --- weight-based check ---
    gamma_pos = embedder.pos_norm.weight.detach()
    gamma_aux = embedder.aux_norm.weight.detach()
    rms_pos = gamma_pos.pow(2).mean().sqrt().item()
    rms_aux = gamma_aux.pow(2).mean().sqrt().item()
    ratio = rms_aux / rms_pos if rms_pos > 0 else float('nan')

    # --- empirical check: real forward pass, both branches captured ---
    captured = {}
    h1 = embedder.pos_norm.register_forward_hook(lambda m, i, o: captured.__setitem__('pos', o.detach()))
    h2 = embedder.aux_norm.register_forward_hook(lambda m, i, o: captured.__setitem__('aux', o.detach()))

    sp, seg, _ = ds[0]  # (N, 7): x,y,z,sx,sy,sz,length/pitch
    with torch.no_grad():
        model(sp.unsqueeze(0).to(device))
    h1.remove(); h2.remove()

    emp_pos_rms = captured['pos'].pow(2).mean().sqrt().item()
    emp_aux_rms = captured['aux'].pow(2).mean().sqrt().item()
    emp_ratio = emp_aux_rms / emp_pos_rms if emp_pos_rms > 0 else float('nan')

    print(f"{M_LABEL[w]:>6} {rms_pos:>16.4f} {rms_aux:>16.4f} {ratio:>14.4f}   "
          f"{emp_pos_rms:>18.4f} {emp_aux_rms:>18.4f} {emp_ratio:>11.4f}")

    del model
    torch.cuda.empty_cache()

print("\nratio ~1.0 -> aux contributes comparably to position")
print("ratio << 1.0 -> aux was downweighted relative to position")
print("ratio >> 1.0 -> aux actually dominates over position")