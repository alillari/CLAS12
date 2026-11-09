"""
Test: does the frozen backbone's per-layer "feature" (return_z=True) carry
more usable momentum signal as the CUMULATIVE hidden state (x after the
residual add) than as the ISOLATED per-layer delta (z before the residual
add, what the code currently returns)?

Does NOT modify any real pretraining code or checkpoint -- builds a local
copy of Mamba1GPT with a `feature_mode` toggle ('delta' = current behavior,
'cumulative' = proposed), loads a real checkpoint's weights into both, and
compares closed-form ridge-regression R^2 (raw px/py/pz) per layer for each
mode. Same evaluation approach as probes/train_linear_probe.py and this
session's earlier probe_momentum.py.

Uses mmap_v4 (has real momentum ground truth) and a checkpoint trained on
it, since the current mmap_cluster_full's reg_target is degenerate (no real
momentum yet -- confirmed via NaN ystd earlier this session) and the real
momentum dataset (v5_03) isn't available locally yet.

Run (from /workspace/PP_collision):
    python3 dev_scripts/test_feature_mode_probe.py \
        --ckpt checkpoints/LR_Full_Test_1_2026-07-20/scale_w256_d12_n5483352/sweep/training_checkpoints/ckpt_best.tar \
        --n-events 3000
"""
import argparse, sys, copy
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.models.mamba2 import Mamba2
from fm4npp.models.embed import *
from fm4npp.models.rmsnorm import RMSNorm
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset, cartesian_to_polar_batched

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class Mamba1GPT_Probe(nn.Module):
    """Local copy of Mamba1GPT with a feature_mode toggle. Only the return_z
    feature-collection behavior differs; everything else (weights, forward
    math) is identical to the real model -- loading a real checkpoint's
    state_dict into this class is exact, not approximate."""

    def __init__(self, embed_dim, num_layers, d_state, d_conv, expand, klen,
                 dropout, embed_method, pe_method, feature_mode='delta'):
        super().__init__()
        assert feature_mode in ('delta', 'cumulative')
        self.feature_mode = feature_mode
        self.embed_dim = embed_dim
        if embed_method == 'concat':
            Embedder = EmbedderConcat
        elif embed_method == 'pos_only':
            Embedder = EmbedderPosOnly
        else:
            Embedder = EmbedderAdd
        self.embedder = Embedder(pe_method=pe_method, embed_dim=embed_dim, learnable_projection=False)
        self.mamba_layers = nn.ModuleList(
            [nn.Sequential(RMSNorm(embed_dim),
                            Mamba(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand),
                            nn.Dropout(dropout))
             for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(embed_dim, klen * 3)
        self.norm = RMSNorm(embed_dim)

    def change_maskval(self, x, init_val=-100, target_val=0):
        out = x.clone()
        out[out == init_val] = target_val
        return out

    def forward(self, x, return_z=False):
        x = self.change_maskval(x)
        x, pos = self.embedder(x)
        feature = []
        for layer in self.mamba_layers:
            z = layer(x)
            if self.feature_mode == 'delta':
                feature.append(z)          # CURRENT real behavior
                x = z + x
            else:
                x = z + x
                feature.append(x)          # PROPOSED: cumulative state
        if return_z:
            return None, feature, pos
        x = self.norm(x)
        return self.output_layer(x)


def ridge_fit_eval(F_train, Y_train, F_test, Y_test, alpha=1e-4):
    """Closed-form ridge regression, same approach as train_linear_probe.py."""
    X = F_train.numpy()
    Y = Y_train.numpy()
    d = X.shape[1]
    W = np.linalg.solve(X.T @ X + alpha * np.eye(d), X.T @ Y)

    Xte = F_test.numpy()
    Yte = Y_test.numpy()
    pred = Xte @ W
    sse = ((pred - Yte) ** 2).sum()
    baseline = Y_train.numpy().mean(axis=0)
    sse_baseline = ((Yte - baseline) ** 2).sum()
    r2 = 1 - sse / sse_baseline if sse_baseline > 0 else float('nan')
    return r2


def masked_mean_pool(feat_layer, mask):
    """feat_layer: (N, D) for one event, one layer. mask: (N,) bool valid."""
    if mask.sum() == 0:
        return feat_layer.mean(dim=0)
    return feat_layer[mask].mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="/workspace/PP_collision/data/mmap_v4")
    ap.add_argument("--n-events", type=int, default=3000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--ridge-alpha", type=float, default=1e-4)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    st = ck.get("model_state", ck.get("model_state_dict", ck))
    st = {k.replace("module.", ""): v for k, v in st.items()}

    # infer architecture from the checkpoint itself
    embed_dim = st["output_layer.weight"].shape[1]
    num_layers = len([k for k in st if k.endswith("mamba_layers.0.1.in_proj.weight")]) and \
                 max(int(k.split(".")[1]) for k in st if k.startswith("mamba_layers.")) + 1
    print(f"Inferred from checkpoint: embed_dim={embed_dim}, num_layers={num_layers}")

    ds = TPCBatchDataset(
        data_root=args.data_root, version="dataset_name", split="pretrain",
        reader_type="ragged_npy", num_pred_points=1, group_size=1,
        normalize=True, nleave=1e6, chunk_training=False, train=True,
        order="clas12_band_hilbert_order",
    )
    n = min(args.n_events, len(ds))
    idx = np.random.RandomState(42).choice(len(ds), size=n, replace=False)
    n_test = int(n * args.test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    results = {}
    for mode in ["delta", "cumulative"]:
        print(f"\n=== mode: {mode} ===")
        model = Mamba1GPT_Probe(
            embed_dim=embed_dim, num_layers=num_layers, d_state=16, d_conv=4,
            expand=2, klen=1, dropout=0.0, embed_method="pos_only",
            pe_method="nerf", feature_mode=mode,
        ).to(device)
        model.load_state_dict(st, strict=False)
        model.eval()

        def extract(index_list):
            F_layers = [[] for _ in range(num_layers)]
            Y = []
            with torch.no_grad():
                for i in index_list:
                    sp, seg, _ = ds[i]
                    real = ds.idxlist[i]
                    reg = torch.from_numpy(np.copy(ds.memmap_reg_target[real])).float()
                    mom = reg[:, :3]
                    feat = torch.from_numpy(np.copy(ds.memmap_feature[real])).unsqueeze(0)
                    nf = ds.apply_norm(cartesian_to_polar_batched(feat)) if ds.normalize else cartesian_to_polar_batched(feat)
                    ind = nf[..., -1].argsort(dim=1).squeeze()
                    nf = nf[:, ind]
                    mom = mom[ind]
                    _, layers, _ = model(nf.to(device), return_z=True)
                    mask = torch.ones(nf.shape[1], dtype=torch.bool)
                    for li, layer_feat in enumerate(layers):
                        pooled = masked_mean_pool(layer_feat.squeeze(0).cpu(), mask)
                        F_layers[li].append(pooled)
                    Y.append(mom.mean(dim=0))
            F_layers = [torch.stack(f) for f in F_layers]
            Y = torch.stack(Y)
            return F_layers, Y

        F_train, Y_train = extract(train_idx)
        F_test, Y_test = extract(test_idx)
        ymean, ystd = Y_train.mean(0), Y_train.std(0).clamp(min=1e-6)
        Y_train_n = (Y_train - ymean) / ystd
        Y_test_n = (Y_test - ymean) / ystd

        r2_per_layer = []
        for li in range(num_layers):
            # [FIX] standardize features (zero mean, unit variance, fit on
            # train only) before ridge fitting. Without this, "cumulative"
            # mode's growing-with-depth scale (residual accumulation) versus
            # "delta" mode's bounded scale makes the tiny ridge_alpha=1e-4
            # numerically ill-conditioned for cumulative -- producing huge,
            # unstable weights and catastrophic (e.g. R2=-2878) held-out
            # predictions that reflect a broken linear solve, not a real
            # signal difference. Standardizing puts both modes on equal
            # footing so the comparison actually tests the hypothesis.
            fmean = F_train[li].mean(dim=0, keepdim=True)
            fstd = F_train[li].std(dim=0, keepdim=True).clamp(min=1e-6)
            F_train_std = (F_train[li] - fmean) / fstd
            F_test_std = (F_test[li] - fmean) / fstd
            r2 = ridge_fit_eval(F_train_std, Y_train_n, F_test_std, Y_test_n, alpha=args.ridge_alpha)
            r2_per_layer.append(r2)
            print(f"  layer {li:>2}: R2 = {r2:.4f}")
        results[mode] = r2_per_layer

    print("\n" + "=" * 60)
    print(f"{'layer':>6} {'delta R2':>12} {'cumulative R2':>15} {'diff':>10}")
    for li in range(num_layers):
        d, c = results["delta"][li], results["cumulative"][li]
        print(f"{li:>6} {d:>12.4f} {c:>15.4f} {c-d:>+10.4f}")
    print(f"\n{'BEST':>6} {max(results['delta']):>12.4f} {max(results['cumulative']):>15.4f}")


if __name__ == "__main__":
    main()