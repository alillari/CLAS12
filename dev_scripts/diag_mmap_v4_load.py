"""
Minimal diagnostic: construct TPCBatchDataset against mmap_v4 with nothing
else going on, to isolate exactly where "0 events" comes from -- reader
returning nothing at all, vs. filter_data discarding everything.

Run: python3 dev_scripts/diag_mmap_v4_load.py
"""
import sys
sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset

print("Constructing TPCBatchDataset against mmap_v4...")
ds = TPCBatchDataset(
    data_root="/workspace/PP_collision/data/mmap_v4",
    version="dataset_name",
    split="pretrain",
    reader_type="ragged_npy",
    num_pred_points=1,
    group_size=1,
    normalize=True,
    nleave=1e6,
    chunk_training=False,
    train=True,
    order="clas12_band_hilbert_order",
)
print(f"\nlen(ds) = {len(ds)}")

# inspect raw reader state directly, before/independent of any filtering
for attr in ["idxlist", "memmap_feature", "toolong", "tooshort"]:
    if hasattr(ds, attr):
        val = getattr(ds, attr)
        try:
            print(f"ds.{attr}: length/size = {len(val)}")
        except TypeError:
            print(f"ds.{attr}: {val!r} (no len)")

if hasattr(ds, "memmap_feature"):
    try:
        print(f"\nraw reader total items (before filtering): {len(ds.memmap_feature)}")
    except Exception as e:
        print(f"could not get raw reader length: {e}")