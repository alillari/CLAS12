"""
Probe the background-merged HIPO before writing the real analysis.
Reports, over the first N events:
  1. which CVT/MC banks actually hold rows (and how many per event)
  2. the distinct `order` values in BST::adc / BMT::adc (background flag)
  3. how MC::True rows (detector, hitn) line up with hit IDs and adc rows

Run: python3 dev_scripts/probe_merged_hipo.py [path] [n_events]
"""
import sys
from collections import Counter, defaultdict
import numpy as np
import hipopy.hipopy as hp

PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspace/PP_collision/data/merged_raw.hipo"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200

# bank -> a column used to count rows
BANKS = {
    "RUN::config": "event",
    "MC::Particle": "pid",
    "MC::True": "hitn",
    "BST::adc": "order",
    "BMT::adc": "order",
    "BST::Hits": "ID",
    "BMT::Hits": "ID",
    "BSTRec::Hits": "ID",
    "BMTRec::Hits": "ID",
    "BST::Clusters": "ID",
    "BMT::Clusters": "ID",
    "BSTRec::Clusters": "ID",
    "BMTRec::Clusters": "ID",
    "CVTRec::Tracks": "ID",
}

f = hp.open(PATH, mode="r")
for b in BANKS:
    f.readBank(b)


def col(ev, bank, name):
    """Return a column as a numpy array, empty if the bank is absent."""
    v = ev.get(f"{bank}_{name}")
    if v is None:
        return np.array([])
    return np.asarray(v).ravel()


rows = defaultdict(list)
orders = {"BST::adc": Counter(), "BMT::adc": Counter()}
mc_det = Counter()
link = defaultdict(lambda: Counter())
printed_keys = False

for i, ev in enumerate(f):
    if i >= N:
        break
    if not printed_keys:
        ks = sorted(k for k in ev.keys() if any(k.startswith(b + "_") for b in BANKS))
        print(f"Example event keys ({len(ks)} shown, first 25):")
        for k in ks[:25]:
            print("   ", k)
        print()
        printed_keys = True

    for b, c in BANKS.items():
        rows[b].append(len(col(ev, b, c)))

    for b in orders:
        for o in col(ev, b, "order"):
            orders[b][int(o)] += 1

    det = col(ev, "MC::True", "detector").astype(int)
    hitn = col(ev, "MC::True", "hitn").astype(int)
    for d in det:
        mc_det[int(d)] += 1

    # MC::True hitn vs adc rows / hit IDs, per detector (BMT=1, BST=2)
    for d, name in ((1, "BMT"), (2, "BST")):
        h = hitn[det == d] if len(det) else np.array([], int)
        if len(h) == 0:
            continue
        n_adc = len(col(ev, f"{name}::adc", "order"))
        for hits_bank in (f"{name}Rec::Hits", f"{name}::Hits"):
            ids = col(ev, hits_bank, "ID").astype(int)
            if len(ids):
                link[name][f"hitn in {hits_bank}.ID"] += int(np.isin(h, ids).sum())
        link[name]["MC::True rows"] += len(h)
        link[name]["hitn <= adc rows (1-based)"] += int(((h >= 1) & (h <= n_adc)).sum())
        link[name]["hitn < adc rows (0-based)"] += int(((h >= 0) & (h < n_adc)).sum())
        link[name]["min hitn"] = min(link[name].get("min hitn", 10**9), int(h.min()))

print(f"=== 1. Rows per event over {min(i, N)} events ===")
print(f"{'bank':<20} {'events with rows':>17} {'mean rows':>10} {'max rows':>9}")
for b in BANKS:
    r = np.array(rows[b])
    print(f"{b:<20} {np.mean(r > 0)*100:>16.1f}% {r.mean():>10.1f} {r.max():>9}")

print("\n=== 2. Distinct `order` values in adc banks (value: count) ===")
for b, c in orders.items():
    print(f"  {b}: " + ", ".join(f"{k}:{v}" for k, v in sorted(c.items())))

print("\n=== 3. MC::True ===")
print("  rows by detector code: " + ", ".join(f"{k}:{v}" for k, v in sorted(mc_det.items())))
for name, c in link.items():
    print(f"  {name}:")
    for k, v in c.items():
        print(f"     {k}: {v}")