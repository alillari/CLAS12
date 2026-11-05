"""
Where does the pretraining loss come from? Evaluation only: labels are read
here to sort predictions into categories, never in training.

For each checkpoint, runs the model on a fixed sample of TRAIN events and a
fixed sample of VALIDATION events through the exact training pipeline (same
loader, ordering, kNNN targets and loss formula), and splits the per-slot loss
by what the query and its target are:

  signal -> same particle    query owned by a primary particle, target on that particle
  signal -> other particle   target owned by a different primary particle
  signal -> background       target is a pure-background cluster
  signal -> ambiguous        target is contested / has no primary owner
  background query           query is a pure-background cluster
  contested query            query has a primary contribution but no majority owner
  no-primary query           query has no primary owner and is not pure background

Memorization check: train and validation use different background files. If the
model memorizes reused background, background-query loss on TRAIN events falls
below VALIDATION and the gap widens across snapshots (train with
`snapshot_every: <steps>` in the config to keep them).

Run from the repo root:
  python3 dev_scripts/loss_by_category.py --yaml <run yaml> --config <name> \
      --ckpt <ckpt.tar> [<ckpt_step2000.tar> ...] [--n-events 2000]
Writes loss_by_category.csv next to the first checkpoint.
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, ".")
from fm4npp.utils import YParams
from fm4npp.hilbert import LEGACY_HILBERT_GRID
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset
from fm4npp.models.mamba_config import build_backbone

CATS = ["signal -> same particle", "signal -> other particle", "signal -> background",
        "signal -> ambiguous", "background query", "contested query", "no-primary query"]
# clusters/ownership: 0 pure background, 1 one primary has a strict majority,
# 2 primary contribution without a majority (contested), 3 no primary, not pure background
OWN_BG, OWN_OWNED, OWN_CONTESTED, OWN_NONE = 0, 1, 2, 3


def make_dataset(p, split, train):
    ds = TPCBatchDataset(
        p.data_root, version=p.data_version, reader_type="hdf5", split=split, train=train,
        normalize=True, group_size=p.group_size, limit_data=False, nleave=p.nleave, order=p.order,
        num_pred_points=p.klen, chunk_training=False, voxelize=getattr(p, "voxelize", True),
        space_filling_order=getattr(p, "space_filling_order", None),
        use_aux_features=getattr(p, "use_aux_features", False),
        aux_extra_feature=getattr(p, "aux_extra_feature", "length"),
        hilbert_angular_grid=getattr(p, "hilbert_angular_grid", LEGACY_HILBERT_GRID),
        return_knn_aux_target=getattr(p, "return_knn_aux_target", False))
    # Carry ownership through the loader's reordering by packing it into the
    # per-cluster label it already reorders: label = (primary_segment + 1) * 4 + ownership.
    ds.fields = ds.fields + ["clusters.ownership"]
    read = ds.hdf5.read

    def read_with_ownership(i, fields):
        rows = read(i, fields)
        seg = rows["clusters.primary_segment"].astype(np.int64)
        own = rows["clusters.ownership"].astype(np.int64)
        if own.size and own.max() > 3:
            raise ValueError(f"unexpected ownership value {own.max()}")
        rows["clusters.primary_segment"] = (seg + 1) * 4 + own
        return rows
    ds.hdf5.read = read_with_ownership
    return ds


def slot_errors(p, pred, kn):
    """Per-slot error, identical to the trainer's loss. pred (N, k*out), kn (N, k*w)."""
    n, k = kn.shape[0], p.klen
    if getattr(p, "use_sincos_target", False):
        aux = getattr(p, "aux_dim", 4) if getattr(p, "return_knn_aux_target", False) else 0
        t = kn.reshape(n, k, 3 + aux)
        q = pred.reshape(n, k, 4)
        phi = t[..., 1] * (2 * torch.pi) - torch.pi
        e_eta = (q[..., 0] - t[..., 0]) ** 2
        e_phi = ((q[..., 1] - torch.sin(phi)) ** 2 + (q[..., 2] - torch.cos(phi)) ** 2) / (2 * torch.pi) ** 2
        e_r = (q[..., 3] - t[..., 2]) ** 2
        if aux >= 3:
            err = t[..., 3] ** 2 * e_r + t[..., 4] ** 2 * e_phi + t[..., 5] ** 2 * e_eta
        else:
            err = (e_r + e_phi + e_eta) / 3
        return err, t[..., :3], t[..., 0] != -100
    t = kn.reshape(n, k, 3)
    return ((pred.reshape(n, k, 3) - t) ** 2).mean(-1), t, t[..., 0] != -100


def category(q_own, q_seg, t_own, t_seg):
    if q_own == OWN_BG:
        return "background query"
    if q_own == OWN_CONTESTED:
        return "contested query"
    if q_own == OWN_NONE:
        return "no-primary query"
    if t_own == OWN_OWNED:
        return "signal -> same particle" if t_seg == q_seg else "signal -> other particle"
    if t_own == OWN_BG:
        return "signal -> background"
    return "signal -> ambiguous"


@torch.no_grad()
def evaluate(model, p, ds, idx, dev):
    tot, cnt = defaultdict(float), defaultdict(int)
    for i in idx:
        g, lab, kn = ds[int(i)]
        pred = model(g.unsqueeze(0).to(dev))[0].float().cpu()
        err, tgt, valid = slot_errors(p, pred, kn)
        own, seg = (lab % 4).numpy(), (lab // 4 - 1).numpy()
        where = {tuple(row): j for j, row in enumerate(g[:, :3].tolist())}
        for a, m in zip(*np.nonzero(valid.numpy())):
            j = where.get(tuple(tgt[a, m].tolist()))
            if j is None:
                continue
            c = category(own[a], seg[a], own[j], seg[j])
            tot[c] += float(err[a, m]); cnt[c] += 1
    return tot, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n-events", type=int, default=2000)
    args = ap.parse_args()

    p = YParams(os.path.abspath(args.yaml), args.config)
    dev = "cuda"
    rng = np.random.default_rng(0)  # same events for every checkpoint
    data = {}
    for split, train in (("train", True), ("validation", False)):
        ds = make_dataset(p, split, train)
        data[split] = (ds, np.sort(rng.choice(len(ds), args.n_events, replace=False)))

    out_csv = os.path.join(os.path.dirname(os.path.abspath(args.ckpt[0])), "loss_by_category.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "step", "split", "category", "n", "mean_loss", "share_of_predictions", "share_of_loss"])
        for ck in args.ckpt:
            state = torch.load(ck, map_location="cpu", weights_only=False)
            model = build_backbone(p).to(dev)
            model.load_state_dict({k.removeprefix("module."): v for k, v in state["model_state"].items()})
            model.eval()
            res = {s: evaluate(model, p, ds, idx, dev) for s, (ds, idx) in data.items()}
            step = state.get("iters", "?")
            print(f"\n=== {os.path.basename(ck)}  (step {step}, {args.n_events} events per split) ===")
            print(f"{'category':26s} {'share of preds':>15s} {'share of loss':>14s} "
                  f"{'loss train':>11s} {'loss val':>10s} {'val/train':>10s}")
            for c in CATS:
                row = []
                for s in ("train", "validation"):
                    tot, cnt = res[s]
                    n_all, l_all = sum(cnt.values()), sum(tot.values())
                    mean = tot[c] / cnt[c] if cnt[c] else float("nan")
                    row.append((cnt[c] / max(n_all, 1), tot[c] / max(l_all, 1e-30), mean))
                    w.writerow([os.path.basename(ck), step, s, c, cnt[c], mean, row[-1][0], row[-1][1]])
                (sp_t, sl_t, m_t), (_, _, m_v) = row
                print(f"{c:26s} {100 * sp_t:14.1f}% {100 * sl_t:13.1f}% {m_t:11.3e} {m_v:10.3e} {m_v / m_t:10.3f}")
            del model
            torch.cuda.empty_cache()
    print(f"\nSaved {out_csv}")
    print("val/train > 1 and growing over snapshots, mostly on the background rows, means "
          "the model is fitting the reused training background rather than general structure.")


if __name__ == "__main__":
    main()
