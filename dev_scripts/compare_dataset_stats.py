"""
Basic dataset statistics comparison: old (mmap_cluster_full) vs v6_02
(target-restricted). Checks whether the label/grouping structure itself
differs between datasets -- noise rate, points/event, tracks/event,
average track size -- BEFORE trusting a downstream probe comparison as a
pure backbone-quality signal. If these differ meaningfully, a same/diff
track-ID probe could be measuring a genuinely different task difficulty
between datasets, independent of any real representation-quality gap.

Run: python3 dev_scripts/compare_dataset_stats.py
"""
import sys
import numpy as np

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset

N_EVENTS = 2000
NOISE_LABEL = -1

DATASETS = {
    "OLD (mmap_cluster_full)": "/workspace/PP_collision/data/mmap_cluster_full",
    "V6_02 (target-restricted)": "/workspace/PP_collision/mmap_canonical_loose_truthseg_event_v6_02",
}


def get_ds(data_root, split="test"):
    return TPCBatchDataset(
        data_root=data_root, version="dataset_name", split=split,
        reader_type="ragged_mmap", num_pred_points=1, group_size=8,
        normalize=True, nleave=1e6, chunk_training=False,
        train=(split == "pretrain"), order="REP")


for label, data_root in DATASETS.items():
    print(f"\n=== {label} ===")
    ds = get_ds(data_root)
    n = min(N_EVENTS, len(ds))

    points_per_event = []
    noise_frac_per_event = []
    tracks_per_event = []
    track_sizes = []

    for i in range(n):
        _, seg, _ = ds[i]
        seg_np = seg.long().numpy()
        n_points = len(seg_np)
        points_per_event.append(n_points)

        is_noise = seg_np == NOISE_LABEL
        noise_frac_per_event.append(is_noise.sum() / n_points if n_points > 0 else 0)

        real_ids = seg_np[~is_noise]
        unique_tracks, counts = np.unique(real_ids, return_counts=True)
        tracks_per_event.append(len(unique_tracks))
        track_sizes.extend(counts.tolist())

    points_per_event = np.array(points_per_event)
    noise_frac_per_event = np.array(noise_frac_per_event)
    tracks_per_event = np.array(tracks_per_event)
    track_sizes = np.array(track_sizes)

    print(f"  events checked: {n}")
    print(f"  points/event:   mean={points_per_event.mean():.2f}  median={np.median(points_per_event):.1f}  "
          f"min={points_per_event.min()}  max={points_per_event.max()}")
    print(f"  noise fraction: mean={noise_frac_per_event.mean()*100:.2f}%  median={np.median(noise_frac_per_event)*100:.2f}%")
    print(f"  tracks/event:   mean={tracks_per_event.mean():.2f}  median={np.median(tracks_per_event):.1f}  "
          f"min={tracks_per_event.min()}  max={tracks_per_event.max()}")
    print(f"  track size (points/track): mean={track_sizes.mean():.2f}  median={np.median(track_sizes):.1f}  "
          f"min={track_sizes.min()}  max={track_sizes.max()}")