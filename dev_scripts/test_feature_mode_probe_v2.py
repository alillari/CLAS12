"""
Three-way test: does the LEARNED WEIGHTED-LAYER-MIXTURE benefit from
normalizing each layer's readout before mixing -- the same fix ELMo's
ScalarMix (do_layer_norm=True) uses, for exactly the same reason (layer
activations have different distributions/scales, so the raw magnitude can
dominate a learned combination independent of actual informativeness).

Three feature_mode variants of the SAME checkpoint's weights:
  'delta'                -- original bug: isolated per-layer delta (z)
  'cumulative'            -- current fix: full accumulated state (x)
  'normalized_cumulative' -- x, RMS-normalized (parameter-free, no new
                             trainable weights) before being added to the
                             readout list. The REAL x used for continued
                             computation into the next layer is completely
                             untouched -- only the copy going into `feature`
                             is normalized. Same safety guarantee as the
                             original fix: doesn't change the real forward
                             pass, doesn't need retraining, testable against
                             ANY existing checkpoint immediately.

Uses mmap_v4 (has real momentum ground truth) and a real checkpoint, same
methodology as the earlier delta-vs-cumulative test (closed-form ridge
regression, standardized features, held-out R^2 per layer).

Run (from /workspace/PP_collision):
    python3 dev_scripts/test_feature_mode_probe_v2.py \
        --ckpt checkpoints/LR_Full_Test_1_2026-07-20/scale_w256_d12_n5483352/sweep/training_checkpoints/ckpt_best.tar \
        --n-events 3000
"""
import argparse, sys
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
    """Local copy of Mamba1GPT with a feature_mode toggle. Only the
    return_z feature-collection behavior differs; the real forward math
    (the x that continues through subsequent layers) is IDENTICAL across
    all three modes -- loading a real checkpoint's state_dict into this
    class is exact, not approximate, for any mode."""

    def __init__(self, embed_dim, num_layers, d_state, d_conv, expand, klen,
                 dropout, embed_method, pe_method, feature_mode='cumulative'):
        super().__init__()
        assert feature_mode in ('delta', 'cumulative', 'normalized_cumulative')
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

    @staticmethod
    def _rms_normalize(t, eps=1e-6):
        # Parameter-free RMS normalization -- same spirit as RMSNorm's
        # normalization step, but with no learnable scale, so it needs no
        # training and is safe to apply to any existing checkpoint's
        # activations immediately. Matches ELMo's do_layer_norm intent:
        # equalize each layer's readout scale before a learned mixture
        # combines them, so no single layer dominates purely by magnitude.
        rms = t.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)
        return t / rms

    def forward(self, x, return_z=False):
        x = self.change_maskval(x)
        x, pos = self.embedder(x)
        feature = []
        for layer in self.mamba_layers:
            z = layer(x)
            x = z + x  # the REAL forward pass -- identical across all 3 modes
            if self.feature_mode == 'delta':
                feature.append(z)
            elif self.feature_mode == 'cumulative':
                feature.append(x)
            else:  # normalized_cumulative
                feature.append(self._rms_normalize(x))
        if return_z:
            return None, feature, pos
        x = self.norm(x)
        return self.output_layer(x)


def ridge_fit_eval(F_train, Y_train, F_test, Y_test, alpha=1e-4):
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
    return 1 - sse / sse_baseline if sse_baseline > 0 else float('nan')


def masked_mean_pool(feat_layer, mask):
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

    embed_dim = st["output_layer.weight"].shape[1]
    num_layers = max(int(k.split(".")[1]) for k in st if k.startswith("mamba_layers.")) + 1
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
    for mode in ["delta", "cumulative", "normalized_cumulative"]:
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
            fmean = F_train[li].mean(dim=0, keepdim=True)
            fstd = F_train[li].std(dim=0, keepdim=True).clamp(min=1e-6)
            F_train_std = (F_train[li] - fmean) / fstd
            F_test_std = (F_test[li] - fmean) / fstd
            r2 = ridge_fit_eval(F_train_std, Y_train_n, F_test_std, Y_test_n, alpha=args.ridge_alpha)
            r2_per_layer.append(r2)
            print(f"  layer {li:>2}: R2 = {r2:.4f}")
        results[mode] = r2_per_layer

    print("\n" + "=" * 80)
    print(f"{'layer':>6} {'delta R2':>12} {'cumulative R2':>15} {'normalized R2':>15}")
    for li in range(num_layers):
        d, c, n_ = results["delta"][li], results["cumulative"][li], results["normalized_cumulative"][li]
        print(f"{li:>6} {d:>12.4f} {c:>15.4f} {n_:>15.4f}")
    print(f"\n{'BEST':>6} {max(results['delta']):>12.4f} "
          f"{max(results['cumulative']):>15.4f} {max(results['normalized_cumulative']):>15.4f}")


if __name__ == "__main__":
    main()