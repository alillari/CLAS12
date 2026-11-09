"""
Verify the truth-link chain in the reconstructed, background-merged HIPO:
    cluster -> HitN_ID -> Hits.ID -> raw adc row -> MC::True (hitn)

Checks, per detector (BST, BMT), over the first N events:
  A. Hits.ID indexing: does adc row (ID-1) [1-based] or (ID) [0-based] have
     the same (sector, layer, strip) as the Hits row?
  B. MC::True hitn: does adc row (hitn-1) carry a signal order flag (0 or 20)?
  C. Geometric test of the link (the BST fix): for each signal hit inside a
     cluster, distance from its MC::True average position to the cluster's
     strip line (x1,y1,z1)-(x2,y2,z2). Correct links -> sub-mm; wrong -> cm.
     MC positions are tried both as cm and as mm (/10); the small one is real.
  D. Cluster composition: pure signal / pure background / mixed, and the
     number of distinct truth tracks per signal cluster.

Run: python3 dev_scripts/verify_truth_link.py [path] [n_events]
"""
import sys
from collections import Counter
import numpy as np
import hipopy.hipopy as hp

PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspace/PP_collision/data/merged_recon.hipo"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
DETS = {"BST": 2, "BMT": 1}
SIGNAL_ORDERS = (0, 20)
BG_ORDERS = (10, 30)

f = hp.open(PATH, mode="r")
for b in ["MC::True"] + [f"{d}::{k}" for d in DETS for k in ("adc", "Hits", "Clusters")]:
    f.readBank(b)


def col(ev, bank, name, dtype=float):
    v = ev.get(f"{bank}_{name}")
    return np.array([], dtype) if v is None else np.asarray(v).ravel().astype(dtype)


def point_line_dist(p, a, b):
    d = b - a
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    d = d / np.where(n > 0, n, 1)
    v = p - a
    return np.linalg.norm(v - (v * d).sum(-1, keepdims=True) * d, axis=-1)


stats = {d: Counter() for d in DETS}
dist_cm = {d: [] for d in DETS}
dist_mm = {d: [] for d in DETS}
ntracks = {d: Counter() for d in DETS}

for i, ev in enumerate(f):
    if i >= N:
        break
    mdet = col(ev, "MC::True", "detector", int)
    mhit = col(ev, "MC::True", "hitn", int)
    mtid = col(ev, "MC::True", "tid", int)
    mpos = np.stack([col(ev, "MC::True", c) for c in ("avgX", "avgY", "avgZ")], -1) if len(mdet) else np.zeros((0, 3))

    for d, code in DETS.items():
        a_sec = col(ev, f"{d}::adc", "sector", int)
        a_lay = col(ev, f"{d}::adc", "layer", int)
        a_str = col(ev, f"{d}::adc", "component", int)
        a_ord = col(ev, f"{d}::adc", "order", int)
        h_id = col(ev, f"{d}::Hits", "ID", int)
        h_sec = col(ev, f"{d}::Hits", "sector", int)
        h_lay = col(ev, f"{d}::Hits", "layer", int)
        h_str = col(ev, f"{d}::Hits", "strip", int)
        na = len(a_ord)

        # A. Hits.ID -> adc row
        for tag, idx in (("1-based", h_id - 1), ("0-based", h_id)):
            ok = (idx >= 0) & (idx < na)
            same = np.zeros(len(h_id), bool)
            same[ok] = (a_sec[idx[ok]] == h_sec[ok]) & (a_lay[idx[ok]] == h_lay[ok]) & (a_str[idx[ok]] == h_str[ok])
            stats[d][f"A: Hits.ID {tag} -> same sector/layer/strip"] += int(same.sum())
        stats[d]["A: Hits rows"] += len(h_id)

        # B. MC::True hitn -> signal adc row
        sel = mdet == code
        hn, tid, pos = mhit[sel], mtid[sel], mpos[sel]
        ok = (hn >= 1) & (hn <= na)
        stats[d]["B: MC rows"] += len(hn)
        stats[d]["B: hitn-1 is a signal adc row"] += int(np.isin(a_ord[hn[ok] - 1], SIGNAL_ORDERS).sum())

        # map adc row (0-based) -> (tid, pos), assuming hitn is 1-based adc row
        truth_tid = {int(h) - 1: int(t) for h, t in zip(hn, tid)}
        truth_pos = {int(h) - 1: p for h, p in zip(hn, pos)}

        # C/D. clusters
        c_hits = np.stack([col(ev, f"{d}::Clusters", f"Hit{k}_ID", int) for k in range(1, 6)], -1) if len(col(ev, f"{d}::Clusters", "ID")) else np.zeros((0, 5), int)
        ends = [np.stack([col(ev, f"{d}::Clusters", c) for c in cs], -1) for cs in (("x1", "y1", "z1"), ("x2", "y2", "z2"))] if len(c_hits) else [np.zeros((0, 3))] * 2
        for ci in range(len(c_hits)):
            ids = c_hits[ci][c_hits[ci] > 0]
            rows = ids - 1                       # assumes Hits.ID is a 1-based adc row (check A)
            rows = rows[(rows >= 0) & (rows < na)]
            if len(rows) == 0:
                stats[d]["D: clusters with no resolvable hits"] += 1
                continue
            is_sig = np.isin(a_ord[rows], SIGNAL_ORDERS)
            if is_sig.all():
                stats[d]["D: pure signal clusters"] += 1
            elif not is_sig.any():
                stats[d]["D: pure background clusters"] += 1
            else:
                stats[d]["D: mixed clusters"] += 1
            tids = {truth_tid[r] for r in rows[is_sig] if r in truth_tid}
            if is_sig.any():
                ntracks[d][len(tids)] += 1
            for r in rows[is_sig]:
                if r in truth_pos:
                    p = truth_pos[r]
                    dist_cm[d].append(point_line_dist(p, ends[0][ci], ends[1][ci]))
                    dist_mm[d].append(point_line_dist(p / 10.0, ends[0][ci], ends[1][ci]))

print(f"Truth-link check over {min(i, N)} events of {PATH}\n")
for d in DETS:
    s = stats[d]
    print(f"=== {d} ===")
    for k in sorted(s):
        print(f"  {k}: {s[k]}")
    if s["A: Hits rows"]:
        for tag in ("1-based", "0-based"):
            print(f"  -> A {tag} match rate: {s[f'A: Hits.ID {tag} -> same sector/layer/strip']/s['A: Hits rows']*100:.1f}%")
    if s["B: MC rows"]:
        print(f"  -> B signal-row rate: {s['B: hitn-1 is a signal adc row']/s['B: MC rows']*100:.1f}%")
    print("  distinct truth tracks per signal cluster: " + ", ".join(f"{k}:{v}" for k, v in sorted(ntracks[d].items())))
    for tag, arr in (("MC as cm", dist_cm[d]), ("MC as mm (/10)", dist_mm[d])):
        a = np.asarray(arr)
        if len(a):
            q = np.percentile(a, [50, 90, 99])
            print(f"  C: truth-to-strip-line distance, {tag}: median {q[0]:.4f} cm, p90 {q[1]:.4f}, p99 {q[2]:.4f} (n={len(a)})")
    print()