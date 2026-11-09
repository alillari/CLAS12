"""
Sequence-length stats on the background-merged recon HIPO, and what they
imply for the compute budget, dataset size and background reuse.

Part 1 (data): clusters per event (BST::Clusters + BMT::Clusters), split into
  signal-containing vs pure-background clusters (via Hit*_ID -> adc row -> order).
Part 2 (model sizes): parameter count per width m1-m5 (instantiates Mamba1GPT
  from the repo; falls back to an analytic Mamba-1 count if the import fails).
Part 3 (budget): Chinchilla-optimal tokens D = 20*N, events needed = D / tokens
  per event, training FLOPs = 6*N*D, for the old data (35.2 tokens/event) and
  the new merged data. Also the batch size that keeps tokens/step equal to the
  old bs64.
Part 4 (background reuse): for each assumed size of the distinct background
  pool, how many times each background event is reused to reach the events
  needed per width. Only the first background slot reaches the CVT, so one
  background event is consumed per signal event.

Run: python3 dev_scripts/compute_budget_bg.py [path] [max_events] [bg_pool sizes...]
  e.g. python3 dev_scripts/compute_budget_bg.py data/merged_recon.hipo 0 10000 50000 200000
"""
import sys, math
import numpy as np
import hipopy.hipopy as hp

PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspace/PP_collision/data/merged_recon.hipo"
MAXEV = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BG_POOLS = [int(x) for x in sys.argv[3:]] or [10_000, 50_000, 100_000, 500_000]

DETS = ("BST", "BMT")
SIGNAL_ORDERS = (0, 20)
WIDTHS = [64, 128, 256, 512, 1024]
M_LABEL = {64: "m1", 128: "m2", 256: "m3", 512: "m4", 1024: "m5"}
DEPTH, D_STATE = 12, 16
OLD_TOKENS_PER_EVENT = 35.2
OLD_BS = 64
TOKENS_PER_PARAM = 20


def col(ev, bank, name, dtype=float):
    v = ev.get(f"{bank}_{name}")
    return np.array([], dtype) if v is None else np.asarray(v).ravel().astype(dtype)


# ---------------- Part 1: data ----------------
f = hp.open(PATH, mode="r")
for d in DETS:
    f.readBank(f"{d}::adc")
    f.readBank(f"{d}::Clusters")

n_tot, n_sig, n_bg = [], [], []
per_det = {d: [] for d in DETS}
for i, ev in enumerate(f):
    if MAXEV and i >= MAXEV:
        break
    t = s = b = 0
    for d in DETS:
        order = col(ev, f"{d}::adc", "order", int)
        cid = col(ev, f"{d}::Clusters", "ID", int)
        per_det[d].append(len(cid))
        if len(cid) == 0:
            continue
        hits = np.stack([col(ev, f"{d}::Clusters", f"Hit{k}_ID", int) for k in range(1, 6)], -1)
        for row in hits:
            r = row[row > 0] - 1
            r = r[(r >= 0) & (r < len(order))]
            t += 1
            if len(r) and np.isin(order[r], SIGNAL_ORDERS).any():
                s += 1
            else:
                b += 1
    n_tot.append(t); n_sig.append(s); n_bg.append(b)

n_tot, n_sig, n_bg = map(np.asarray, (n_tot, n_sig, n_bg))
mean_tok = n_tot.mean()
mean_sig = n_sig.mean()
print(f"=== Part 1: tokens (clusters) per event, {len(n_tot)} events ===")
q = [50, 90, 99, 99.9, 100]
print(f"{'':<22}{'mean':>8}" + "".join(f"{'p'+str(x):>8}" for x in q))
for name, a in (("all clusters", n_tot), ("signal-containing", n_sig), ("pure background", n_bg),
                ("BST", np.asarray(per_det["BST"])), ("BMT", np.asarray(per_det["BMT"]))):
    print(f"{name:<22}{a.mean():>8.1f}" + "".join(f"{np.percentile(a, x):>8.0f}" for x in q))
print(f"background fraction of tokens: {n_bg.sum()/n_tot.sum()*100:.1f}%")
print(f"tokens/event vs old data: {mean_tok:.1f} vs {OLD_TOKENS_PER_EVENT} ({mean_tok/OLD_TOKENS_PER_EVENT:.2f}x)")


# ---------------- Part 2: parameter counts ----------------
def analytic_mamba1_params(d):
    di = 2 * d
    dt_rank = math.ceil(d / 16)
    per_layer = (d * 2 * di          # in_proj
                 + di * 4 + di       # depthwise conv (k=4) + bias
                 + di * (dt_rank + 2 * D_STATE)  # x_proj
                 + dt_rank * di + di # dt_proj + bias
                 + di * D_STATE + di # A_log, D
                 + di * d            # out_proj
                 + d)                # norm
    return DEPTH * per_layer


params = {}
try:
    sys.path.insert(0, "/workspace/PP_collision")
    from fm4npp.models.mambagpt import Mamba1GPT
    for w in WIDTHS:
        m = Mamba1GPT(embed_dim=w, num_layers=DEPTH, d_state=D_STATE, klen=5, dropout=0.0,
                      embed_method="pos_plus_aux", pe_method="nerf", pos_dim=3, aux_dim=4,
                      use_sincos_target=True)
        params[w] = sum(p.numel() for p in m.parameters())
        del m
    src = "Mamba1GPT (repo, model B config, kNNN=5)"
except Exception as e:
    params = {w: analytic_mamba1_params(w) for w in WIDTHS}
    src = f"analytic Mamba-1 backbone only (repo import failed: {type(e).__name__})"

print(f"\n=== Part 2: parameters, from {src} ===")
for w in WIDTHS:
    print(f"  {M_LABEL[w]} (w{w}): {params[w]/1e6:8.2f} M")


# ---------------- Part 3: budget ----------------
print(f"\n=== Part 3: Chinchilla-optimal budget (D = {TOKENS_PER_PARAM}*N, FLOPs = 6*N*D) ===")
print("FLOPs and token counts are the same for old and new data at fixed N; what changes is the number of events.")
hdr = f"{'model':>6}{'tokens D':>12}{'FLOPs':>11}{'events (old)':>14}{'events (new)':>14}{'events (new, signal tokens only)':>34}"
print(hdr)
need = {}
for w in WIDTHS:
    N = params[w]
    D = TOKENS_PER_PARAM * N
    need[w] = D / mean_tok
    print(f"{M_LABEL[w]:>6}{D:>12.3g}{6*N*D:>11.3g}{D/OLD_TOKENS_PER_EVENT:>14,.0f}{need[w]:>14,.0f}{D/mean_sig:>34,.0f}")
print("'signal tokens only' = events needed if only signal-query tokens count toward D (background masked from the loss);"
      " input FLOPs per event are unchanged.")

bs_equiv = OLD_BS * OLD_TOKENS_PER_EVENT / mean_tok
print(f"\nTokens/step at old bs{OLD_BS}: {OLD_BS*OLD_TOKENS_PER_EVENT:,.0f}. Same tokens/step on new data: bs ~{bs_equiv:.1f}.")
print(f"Longest event: {n_tot.max()} tokens (old max-length settings should cover this).")


# ---------------- Part 4: background reuse ----------------
print("\n=== Part 4: background reuse (one background event per signal event reaches the CVT) ===")
print(f"{'model':>6}{'events needed':>15}" + "".join(f"{'pool '+format(p, ','):>16}" for p in BG_POOLS))
for w in WIDTHS:
    print(f"{M_LABEL[w]:>6}{need[w]:>15,.0f}" + "".join(f"{need[w]/p:>15.1f}x" for p in BG_POOLS))
print("Each entry = times each background event appears in training at the Chinchilla-optimal event count.")
print(f"Background is {n_bg.sum()/n_tot.sum()*100:.0f}% of tokens, so that share of the input is repeated at that rate.")