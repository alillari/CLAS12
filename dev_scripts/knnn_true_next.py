"""
kNNN target study on the reconstructed, background-merged HIPO.

For every signal cluster (query) whose track has at least one cluster further
out, report as a function of k:
  1. kNNN rule (as in training): slot m = nearest cluster >= m layers out.
     % of queries where ANY of slots 1..k is on the SAME truth track.
  2. Plain k nearest clusters in later layers: % with >= 1 SAME-track cluster.
  3. Plain k nearest clusters in later layers: % with >= 1 SIGNAL cluster
     (any track, i.e. not background).

Truth chain (verified with verify_truth_link.py): cluster HitN_ID -> 1-based
row of {BST,BMT}::adc -> `order` flag (0/20 signal, 10/30 background) and
MC::True hitn (1-based adc row) -> tid.

Layers: band = layer for BST (1-6), 6 + layer for BMT (7-12); radius increases
strictly in this order. Distances: Euclidean in normalized (eta, phi, r) with
the fixed bounds eta [-2.5, 1.5], phi [-pi, pi], r [6, 23] cm, no phi
wraparound -- same convention as the pretraining pipeline.

Run:  python3 dev_scripts/knnn_true_next.py [path] [max_events (0 = all)] [k_max]
Out:  fig_knnn_true_next.png, knnn_true_next.csv in the repo root
"""
import sys
import numpy as np

K_MAX = 10
N_SLOTS = 11   # 12 layers -> at most 11 depth slots
SIGNAL_ORDERS = (0, 20)
DETS = (("BST", 2, 0), ("BMT", 1, 6))   # name, MC::True detector code, band offset
SKIP_EVENTS = {(11, 1798)}               # Alessio: no background overlay in this event
ETA_LIM, R_LIM = (-2.5, 1.5), (6.0, 23.0)


def normalize(cx, cy, cz):
    rho = np.hypot(cx, cy)
    eta = np.arcsinh(cz / np.where(rho > 0, rho, 1e-9))
    phi = np.arctan2(cy, cx)
    return np.stack([(eta - ETA_LIM[0]) / (ETA_LIM[1] - ETA_LIM[0]),
                     (phi + np.pi) / (2 * np.pi),
                     (rho - R_LIM[0]) / (R_LIM[1] - R_LIM[0])], -1)


def event_counts(pos, band, tid, is_sig, k_max=K_MAX):
    """Per-event tallies. pos (N,3) normalized; band (N,); tid (N,) (-1 = none);
    is_sig (N,) bool. Returns dict of count arrays over k = 1..k_max, the number
    of eligible queries, and slot-1 hits per query band."""
    out = {"rule_same": np.zeros(k_max), "knn_same": np.zeros(k_max),
           "knn_sig": np.zeros(k_max), "eligible": 0, "no_later": 0,
           "slot1_by_band": {},
           # per-slot outcome of the kNNN rule, slots 1..11:
           # [same track, other signal track, background, no candidate]
           "slot_outcome": np.zeros((N_SLOTS, 4))}
    for q in np.where(is_sig & (tid >= 0))[0]:
        later = np.where(band > band[q])[0]
        same_later = later[tid[later] == tid[q]]
        if len(same_later) == 0:
            out["no_later"] += 1
            continue
        out["eligible"] += 1
        d = np.linalg.norm(pos[later] - pos[q], axis=1)
        order = later[np.argsort(d, kind="stable")]
        same_k = np.cumsum(tid[order[:k_max]] == tid[q]) > 0
        sig_k = np.cumsum(is_sig[order[:k_max]]) > 0
        n = len(same_k)
        out["knn_same"][:n] += same_k
        out["knn_sig"][:n] += sig_k
        if n < k_max:                       # fewer than k candidates: stays at last value
            out["knn_same"][n:] += same_k[-1]
            out["knn_sig"][n:] += sig_k[-1]
        hit = False
        for m in range(1, max(k_max, N_SLOTS) + 1):
            cand = np.where(band >= band[q] + m)[0]
            if m <= N_SLOTS and len(cand) == 0:
                out["slot_outcome"][m - 1, 3] += 1
            if len(cand):
                j = cand[np.argmin(np.linalg.norm(pos[cand] - pos[q], axis=1))]
                if m <= N_SLOTS:
                    cat = 0 if tid[j] == tid[q] else (1 if is_sig[j] else 2)
                    out["slot_outcome"][m - 1, cat] += 1
                hit = hit or (tid[j] == tid[q])
                if m == 1:
                    b = int(band[q])
                    s = out["slot1_by_band"].setdefault(b, [0, 0])
                    s[0] += int(tid[j] == tid[q]); s[1] += 1
            if m <= k_max:
                out["rule_same"][m - 1] += hit
    return out


def read_events(path, max_events):
    import hipopy.hipopy as hp
    f = hp.open(path, mode="r")
    for b in ("RUN::config", "MC::True", "BST::adc", "BMT::adc", "BST::Clusters", "BMT::Clusters"):
        f.readBank(b)

    def col(ev, bank, name, dtype=float):
        v = ev.get(f"{bank}_{name}")
        return np.array([], dtype) if v is None else np.asarray(v).ravel().astype(dtype)

    for i, ev in enumerate(f):
        if max_events and i >= max_events:
            break
        run, evn = col(ev, "RUN::config", "run", int), col(ev, "RUN::config", "event", int)
        if len(run) and (int(run[0]), int(evn[0])) in SKIP_EVENTS:
            continue
        mdet = col(ev, "MC::True", "detector", int)
        mhit = col(ev, "MC::True", "hitn", int)
        mtid = col(ev, "MC::True", "tid", int)
        P, B, T, S = [], [], [], []
        for name, code, off in DETS:
            order = col(ev, f"{name}::adc", "order", int)
            cid = col(ev, f"{name}::Clusters", "ID", int)
            if len(cid) == 0:
                continue
            truth = {int(h) - 1: int(t) for h, t in zip(mhit[mdet == code], mtid[mdet == code])}
            hits = np.stack([col(ev, f"{name}::Clusters", f"Hit{k}_ID", int) for k in range(1, 6)], -1)
            for ci in range(len(cid)):
                rows = hits[ci][hits[ci] > 0] - 1
                rows = rows[(rows >= 0) & (rows < len(order))]
                sig_rows = [r for r in rows if order[r] in SIGNAL_ORDERS]
                tids = [truth[r] for r in sig_rows if r in truth]
                S.append(len(sig_rows) > 0)
                T.append(max(set(tids), key=tids.count) if tids else -1)
            P.append(normalize(col(ev, f"{name}::Clusters", "cx"),
                               col(ev, f"{name}::Clusters", "cy"),
                               col(ev, f"{name}::Clusters", "cz")))
            B.append(col(ev, f"{name}::Clusters", "layer", int) + off)
        if not P:
            continue
        yield np.concatenate(P), np.concatenate(B), np.array(T), np.array(S, bool)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/PP_collision/data/merged_recon.hipo"
    max_events = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    k_max = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    RULE_MAX = 11   # 12 layers: slot m needs a cluster >= m layers out, so m <= 11
    tot = {"rule_same": np.zeros(k_max), "knn_same": np.zeros(k_max), "knn_sig": np.zeros(k_max),
           "eligible": 0, "no_later": 0, "slot1_by_band": {}, "slot_outcome": np.zeros((N_SLOTS, 4))}
    n_ev = 0
    for pos, band, tid, sig in read_events(path, max_events):
        r = event_counts(pos, band, tid, sig, k_max=k_max)
        for k in ("rule_same", "knn_same", "knn_sig"):
            tot[k] += r[k]
        tot["eligible"] += r["eligible"]; tot["no_later"] += r["no_later"]
        tot["slot_outcome"] += r["slot_outcome"]
        for b, (h, n) in r["slot1_by_band"].items():
            s = tot["slot1_by_band"].setdefault(b, [0, 0]); s[0] += h; s[1] += n
        n_ev += 1
        if n_ev % 1000 == 0:
            print(f"  {n_ev} events...")

    e = max(tot["eligible"], 1)
    pct = {k: 100 * tot[k] / e for k in ("rule_same", "knn_same", "knn_sig")}
    print(f"\n{n_ev} events, {tot['eligible']:,} signal queries with a later cluster on their track "
          f"({tot['no_later']:,} more had none and are excluded)\n")
    print(f"{'k':>3} {'kNNN rule, same track':>22} {'k nearest, same track':>22} {'k nearest, any signal':>22}")
    for k in range(k_max):
        rule = f"{pct['rule_same'][k]:>21.1f}%" if k < RULE_MAX else f"{'n/a':>22}"
        print(f"{k+1:>3} {rule} {pct['knn_same'][k]:>21.1f}% {pct['knn_sig'][k]:>21.1f}%")
    names = {b: (f"BST{b}" if b <= 6 else f"BMT{b-6}") for b in range(1, 13)}
    print("\nkNNN slot 1 lands on the same track, by query layer:")
    for b in sorted(tot["slot1_by_band"]):
        h, n = tot["slot1_by_band"][b]
        print(f"  {names.get(b, b):>5} -> next: {100*h/n:5.1f}%  (n={n:,})")

    out_dir = "/workspace/PP_collision"
    with open(f"{out_dir}/knnn_true_next.csv", "w") as fh:
        fh.write("k,knnn_rule_same_track_pct,k_nearest_same_track_pct,k_nearest_any_signal_pct\n")
        for k in range(k_max):
            rule = f"{pct['rule_same'][k]:.3f}" if k < RULE_MAX else ""
            fh.write(f"{k+1},{rule},{pct['knn_same'][k]:.3f},{pct['knn_sig'][k]:.3f}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = np.arange(1, k_max + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rk = ks[ks <= RULE_MAX]
    ax.plot(rk, pct["rule_same"][:len(rk)], "o-", ms=4, color="#2a6f97",
            label="kNNN rule (slot m = nearest \u2265 m layers out), same track (max k = 11)")
    ax.plot(ks, pct["knn_same"], "s-", ms=3, color="#e76f51", label="k nearest later clusters, same track")
    ax.plot(ks, pct["knn_sig"], "^-", ms=3, color="#8ab17d", label="k nearest later clusters, any signal track")
    ax.set_xlabel("k")
    ax.set_ylabel("% of signal queries with a true hit among the k targets")
    ax.set_ylim(0, 100)
    ax.set_title(f"How often the kNNN targets contain a real next hit\n"
                 f"background-merged sample, {n_ev:,} events, {tot['eligible']:,} signal queries", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_knnn_true_next.png", dpi=150)
    so = tot["slot_outcome"]
    sp = 100 * so / max(tot["eligible"], 1)
    print("\nPer slot (kNNN rule), % of signal queries -- each slot on its own:")
    print(f"{'slot':>4} {'same track':>11} {'other track':>12} {'background':>11} {'no candidate':>13} {'same | slot exists':>19}")
    for m in range(N_SLOTS):
        exists = so[m, :3].sum()
        cond = 100 * so[m, 0] / exists if exists else float("nan")
        print(f"{m+1:>4} {sp[m,0]:>10.1f}% {sp[m,1]:>11.1f}% {sp[m,2]:>10.1f}% {sp[m,3]:>12.1f}% {cond:>18.1f}%")
    with open(f"{out_dir}/knnn_per_slot.csv", "w") as fh:
        fh.write("slot,same_track_pct,other_track_pct,background_pct,no_candidate_pct\n")
        for m in range(N_SLOTS):
            fh.write(f"{m+1},{sp[m,0]:.3f},{sp[m,1]:.3f},{sp[m,2]:.3f},{sp[m,3]:.3f}\n")

    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    slots = np.arange(1, N_SLOTS + 1)
    bottom = np.zeros(N_SLOTS)
    for idx, (label, col) in enumerate((("same track", "#2a6f97"), ("other signal track", "#e9c46a"),
                                        ("background", "#e76f51"), ("no candidate (fewer layers left)", "#cccccc"))):
        ax2.bar(slots, sp[:, idx], 0.7, bottom=bottom, label=label, color=col)
        bottom += sp[:, idx]
    ax2.set_xticks(slots)
    ax2.set_xlabel("kNNN slot m (target = nearest cluster \u2265 m layers out)")
    ax2.set_ylabel("% of signal queries")
    ax2.set_ylim(0, 100)
    ax2.set_title(f"What each kNNN slot's target is, per slot\n"
                  f"background-merged sample, {n_ev:,} events, {tot['eligible']:,} signal queries", fontsize=11)
    ax2.legend(fontsize=8.5, loc="upper right")
    fig2.tight_layout()
    fig2.savefig(f"{out_dir}/fig_knnn_per_slot.png", dpi=150)

    print(f"\nSaved: {out_dir}/fig_knnn_true_next.png, fig_knnn_per_slot.png, knnn_true_next.csv, knnn_per_slot.csv")


if __name__ == "__main__":
    main()