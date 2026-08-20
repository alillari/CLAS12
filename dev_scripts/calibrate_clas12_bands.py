"""
Calibrate CLAS12 CVT band boundaries automatically from real data, replacing
the hardcoded 6-value CLAS12_LAYER_RADII_RAW in hilbert.py (which silently
drops 3 of BMT's 6 layers).

Method:
  1. Pool raw (unnormalized) radii from a large sample of events.
  2. Fine histogram + peak detection (same approach as check_radius_histogram.py).
  3. Merge peaks separated by "small" gaps (SVT's internal stereo/quadruplet
     structure) into one band; keep peaks separated by "large" gaps (real
     region/layer boundaries) as separate bands. Split point is found
     automatically as the largest relative jump in sorted gap sizes.
  4. Output BAND BOUNDARIES (the cut points between bands, placed at the
     midpoint of each retained large gap) -- not centers+delta. Boundaries
     are what assign_clas12_layer will use with torch.bucketize, which
     assigns every real value to exactly one band with no failure mode.

Run (from /workspace/PP_collision):
    python3 dev_scripts/calibrate_clas12_bands.py --n-events 5000
"""
import argparse
import json
import numpy as np
from mmap_ninja import RaggedMmap
from scipy.signal import find_peaks

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="/workspace/PP_collision/data/mmap_event_cluster_v5_01")
ap.add_argument("--split", default="pretrain")
ap.add_argument("--n-events", type=int, default=5000)
ap.add_argument("--n-bands", type=int, default=None,
                 help="expected number of physical bands (e.g. 12 = 6 active SVT "
                      "U/V layers + 6 BMT layers). If given, boundaries are the "
                      "(n_bands-1) LARGEST gaps between detected peaks -- robust "
                      "even with 3+ gap-size tiers (fine intra-layer spread, "
                      "U/V-within-region, inter-region/BMT), which a single "
                      "relative-jump threshold cannot separate correctly. If "
                      "omitted, falls back to the largest-relative-jump auto-split "
                      "(only correct for a clean 2-tier gap structure).")
ap.add_argument("--out", default="/workspace/PP_collision/dev_scripts/clas12_band_calibration.json")
args = ap.parse_args()

f = RaggedMmap(f"{args.data_root}/features_{args.split}")
n_total = len(f)
rng = np.random.RandomState(42)
idx = rng.choice(n_total, size=min(args.n_events, n_total), replace=False)

all_r = []
for i in idx:
    ev = np.array(f[i])
    all_r.extend(np.sqrt(ev[:, 0]**2 + ev[:, 1]**2).tolist())
all_r = np.array(all_r)
print(f"sampled {len(idx):,} events, {len(all_r):,} points")
print(f"radius range: {all_r.min():.4f} - {all_r.max():.4f} cm")

# --- fine histogram + peak detection ---
bins = np.linspace(all_r.min() - 0.05, all_r.max() + 0.05, 2000)
counts, edges = np.histogram(all_r, bins=bins)
centers_h = (edges[:-1] + edges[1:]) / 2
peaks, _ = find_peaks(counts, height=counts.max() * 0.02, distance=5)
peak_radii = np.sort(centers_h[peaks])
print(f"\ndetected {len(peak_radii)} raw peaks")

# --- gap analysis: find which gaps are real band boundaries ---
gaps = np.diff(peak_radii)

if args.n_bands is not None:
    # Take the (n_bands - 1) LARGEST gaps as real boundaries, regardless of
    # how many gap-size tiers exist. Robust to 3+ tiers (fine intra-layer
    # spread from stereo angle, U/V-within-region, inter-region/BMT) --
    # a single relative-jump threshold only finds ONE split point and can
    # pick the wrong one when there's more than 2 tiers (verified: it did,
    # on this exact data -- see the n_bands=None fallback path below).
    n_boundaries_needed = args.n_bands - 1
    assert n_boundaries_needed <= len(gaps), (
        f"requested {args.n_bands} bands needs {n_boundaries_needed} boundary "
        f"gaps, but only {len(gaps)} gaps exist between the {len(peak_radii)} "
        f"detected peaks -- lower --n-bands or check peak detection."
    )
    gap_order = np.argsort(gaps)[::-1]  # largest first
    is_boundary = np.zeros(len(gaps), dtype=bool)
    is_boundary[gap_order[:n_boundaries_needed]] = True
    split_threshold = None
    print(f"n_bands={args.n_bands} requested -> taking the {n_boundaries_needed} "
          f"largest of {len(gaps)} gaps as real boundaries")
    smallest_kept = gaps[is_boundary].min()
    largest_dropped = gaps[~is_boundary].max() if (~is_boundary).any() else None
    print(f"  smallest boundary gap kept: {smallest_kept:.4f} cm")
    if largest_dropped is not None:
        print(f"  largest non-boundary gap dropped: {largest_dropped:.4f} cm")
        if largest_dropped >= smallest_kept:
            print(f"  WARNING: dropped gap >= kept gap -- n_bands may not match "
                  f"the data's real structure, double check peak_radii above.")
else:
    # fallback: single largest-relative-jump split (only correct for a clean
    # 2-tier gap structure -- prefer --n-bands when the physical count is known)
    sorted_gaps = np.sort(gaps)
    ratios = sorted_gaps[1:] / sorted_gaps[:-1]
    split_i = np.argmax(ratios)
    split_threshold = (sorted_gaps[split_i] + sorted_gaps[split_i + 1]) / 2
    print(f"no --n-bands given, using auto relative-jump split: {split_threshold:.4f} cm "
          f"(jump {sorted_gaps[split_i]:.4f} -> {sorted_gaps[split_i+1]:.4f} cm, "
          f"ratio {ratios[split_i]:.2f}x) -- WARNING: only correct for a clean "
          f"2-tier gap structure; pass --n-bands if you know the physical count.")
    is_boundary = gaps >= split_threshold

# --- merge peaks separated by non-boundary gaps into bands ---
bands = [[peak_radii[0]]]
for i in range(1, len(peak_radii)):
    if not is_boundary[i - 1]:
        bands[-1].append(peak_radii[i])
    else:
        bands.append([peak_radii[i]])

band_centers = [float(np.mean(b)) for b in bands]
band_mins = [float(np.min(b)) for b in bands]
band_maxs = [float(np.max(b)) for b in bands]
print(f"\n{len(bands)} merged bands:")
for i, (c, lo, hi) in enumerate(zip(band_centers, band_mins, band_maxs)):
    width = hi - lo
    print(f"  band {i}: center={c:.4f} cm  span=[{lo:.4f}, {hi:.4f}]  "
          f"width={width:.4f} cm  ({len(bands[i])} merged sub-peak(s))")

# --- boundaries: midpoint of the gap between consecutive band envelopes ---
boundaries = []
for i in range(len(bands) - 1):
    boundaries.append((band_maxs[i] + band_mins[i + 1]) / 2.0)
print(f"\n{len(boundaries)} band boundaries (raw cm):")
for b in boundaries:
    print(f"  {b:.4f}")

# --- smallest real (large-tier) gap: for the kNNN r_threshold ---
large_gaps = [band_mins[i+1] - band_maxs[i] for i in range(len(bands) - 1)]
min_large_gap = min(large_gaps)
suggested_r_threshold_raw = min_large_gap / 2.0
print(f"\nsmallest inter-band gap: {min_large_gap:.4f} cm")
print(f"suggested kNNN r_threshold (half of it): {suggested_r_threshold_raw:.4f} cm")

# --- save calibration ---
calibration = {
    "n_events_sampled": int(len(idx)),
    "n_points_sampled": int(len(all_r)),
    "band_centers_raw_cm": band_centers,
    "band_boundaries_raw_cm": boundaries,
    "band_widths_raw_cm": [hi - lo for lo, hi in zip(band_mins, band_maxs)],
    "min_inter_band_gap_raw_cm": float(min_large_gap),
    "suggested_r_threshold_raw_cm": float(suggested_r_threshold_raw),
}
with open(args.out, "w") as fh:
    json.dump(calibration, fh, indent=2)
print(f"\nSaved calibration: {args.out}")