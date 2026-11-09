"""
Descriptive statistics for the CVT strip pretraining handoff (v1).

Answers the questions that drive the pretraining design:
  1. Tokens per event            -> sequence-length cap / memory
  2. Radius per (detector,layer) -> do identity layers order radially, and how
                                    much radius spread is there inside a layer
  3. Strip orientation per layer -> which layers are along-z vs. around-phi strips
  4. Strips per cluster          -> clusters rebuilt as runs of consecutive strip
                                    numbers in the same (detector,layer,sector)
  5. Spacing within a cluster    -> size of the +-1 strip target ambiguity
  6. Energy per detector         -> raw and log1p ranges (BST/BMT scales differ)
  7. Pitch / strip length        -> per layer
  8. Discreteness                -> unique strip addresses vs. token count

Only numpy needed. Arrays are memory-mapped; per-cluster stats use a sample.

Run (host or container):
    python3 strip_dataset_stats.py /path/to/strip_pretraining_handoff_v1 [--sample-events 20000]
"""
import argparse
from pathlib import Path
import numpy as np

DET_NAME = {2: "BST", 1: "BMT"}
# feature columns
E, CX, CY, CZ, X1, Y1, Z1, X2, Y2, Z2, PITCH = range(11)
# identity columns
DET, LAYER, SECTOR, STRIP, HIT_ID = range(5)


def pct(a, qs=(0, 5, 25, 50, 75, 95, 99, 100)):
    v = np.percentile(a, qs)
    return "  ".join(f"p{q}={x:.4g}" for q, x in zip(qs, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--sample-events", type=int, default=20000)
    args = ap.parse_args()

    p = args.root / "data" / "train"
    feats = np.load(p / "features.npy", mmap_mode="r")
    ident = np.load(p / "identity.npy", mmap_mode="r")
    offs = np.load(p / "offsets.npy", mmap_mode="r")
    n_events = len(offs) - 1
    print(f"{n_events:,} events, {len(feats):,} tokens\n")

    # ---------- 1. tokens per event (all events) ----------
    tpe = np.diff(np.asarray(offs))
    print("=== 1. Tokens per event (all events) ===")
    print(pct(tpe))
    for cap in (50, 100, 150, 200, 261, 300, 400):
        print(f"  events > {cap:>3}: {np.mean(tpe > cap)*100:6.2f}%")
    print()

    # sample of events for the heavier stats
    n_s = min(args.sample_events, n_events)
    hi = int(offs[n_s])
    F = np.asarray(feats[:hi], dtype=np.float64)
    I = np.asarray(ident[:hi])
    print(f"(sections 2-8 use the first {n_s:,} events, {hi:,} tokens)\n")

    r = np.hypot(F[:, CX], F[:, CY])
    dvec = F[:, [X2, Y2, Z2]] - F[:, [X1, Y1, Z1]]
    length = np.linalg.norm(dvec, axis=1)
    frac_z = np.abs(dvec[:, 2]) / np.maximum(length, 1e-12)

    # ---------- 2/3/7. per (detector, layer) ----------
    print("=== 2/3/7. Per detector/layer: radius, orientation, pitch, length ===")
    print(f"{'det':>4} {'lay':>3} {'tokens':>9} {'r_mean':>8} {'r_min':>8} {'r_max':>8} "
          f"{'r_std':>7} {'along_z':>8} {'pitch_cm':>9} {'len_cm':>8}")
    order_check = []
    for det in (2, 1):
        for lay in range(1, 7):
            m = (I[:, DET] == det) & (I[:, LAYER] == lay)
            if not m.any():
                print(f"{DET_NAME[det]:>4} {lay:>3}   (no tokens)")
                continue
            rr = r[m]
            order_check.append((DET_NAME[det], lay, rr.mean()))
            print(f"{DET_NAME[det]:>4} {lay:>3} {m.sum():>9,} {rr.mean():>8.3f} {rr.min():>8.3f} "
                  f"{rr.max():>8.3f} {rr.std():>7.3f} {np.median(frac_z[m]):>8.2f} "
                  f"{np.median(F[m, PITCH]):>9.4f} {np.median(length[m]):>8.2f}")
    radii = [x[2] for x in order_check]
    print("\n  BST 1-6 then BMT 1-6 strictly increasing in mean radius:",
          all(a < b for a, b in zip(radii, radii[1:])))
    print("  along_z = median |dz|/|strip vector|: ~1 -> strip runs along z, ~0 -> runs around phi\n")

    # ---------- 4/5. rebuilt clusters ----------
    print("=== 4/5. Rebuilt clusters (consecutive strips, same detector/layer/sector) ===")
    sizes = {2: [], 1: []}
    spacing = {2: [], 1: []}
    for e in range(n_s):
        lo, hi_e = int(offs[e]), int(offs[e + 1])
        ie = I[lo:hi_e]
        fe = F[lo:hi_e]
        key_order = np.lexsort((ie[:, STRIP], ie[:, SECTOR], ie[:, LAYER], ie[:, DET]))
        ie, fe = ie[key_order], fe[key_order]
        same_mod = ((ie[1:, DET] == ie[:-1, DET]) & (ie[1:, LAYER] == ie[:-1, LAYER])
                    & (ie[1:, SECTOR] == ie[:-1, SECTOR]))
        adjacent = same_mod & (ie[1:, STRIP] - ie[:-1, STRIP] == 1)
        # spacing between adjacent strips in the same cluster
        if adjacent.any():
            d = np.linalg.norm(fe[1:, [CX, CY, CZ]][adjacent] - fe[:-1, [CX, CY, CZ]][adjacent], axis=1)
            for det in (2, 1):
                spacing[det].extend(d[ie[1:, DET][adjacent] == det].tolist())
        # cluster sizes: break wherever the next token is not an adjacent strip
        breaks = np.concatenate(([0], np.where(~adjacent)[0] + 1, [len(ie)]))
        for a, b in zip(breaks[:-1], breaks[1:]):
            sizes[int(ie[a, DET])].append(b - a)
    for det in (2, 1):
        s = np.array(sizes[det])
        if len(s) == 0:
            continue
        counts = np.bincount(s)
        dist = "  ".join(f"{k}:{counts[k]/len(s)*100:.1f}%" for k in range(1, min(len(counts), 8)))
        print(f"  {DET_NAME[det]} clusters: {len(s):,}  mean size {s.mean():.2f}  [{dist}]"
              + (f"  >=8: {np.mean(s >= 8)*100:.1f}%" if len(counts) > 8 else ""))
        if spacing[det]:
            print(f"     adjacent-strip center spacing (cm): {pct(np.array(spacing[det]), (5, 50, 95))}")
    print("  (spacing = size of the +-1 strip ambiguity in the kNNN target)\n")

    # ---------- 6. energy ----------
    print("=== 6. Energy per detector (native units -- scales differ) ===")
    for det in (2, 1):
        m = I[:, DET] == det
        en = F[m, E]
        print(f"  {DET_NAME[det]} raw:   {pct(en, (0, 1, 50, 99, 100))}   (<=0: {np.mean(en <= 0)*100:.2f}%)")
        print(f"  {DET_NAME[det]} log1p: mean={np.log1p(np.clip(en, 0, None)).mean():.4g} "
              f"std={np.log1p(np.clip(en, 0, None)).std():.4g}")
    print()

    # ---------- 8. discreteness ----------
    print("=== 8. Discreteness ===")
    addr = I[:, [DET, LAYER, SECTOR, STRIP]]
    uniq_addr = np.unique(addr, axis=0)
    uniq_pos = np.unique(np.round(F[:, [CX, CY, CZ]], 4), axis=0)
    print(f"  tokens: {len(I):,}  unique strip addresses: {len(uniq_addr):,}  "
          f"unique center positions: {len(uniq_pos):,}")
    print("  (addresses ~= positions -> each strip always reports the same point)")


if __name__ == "__main__":
    main()