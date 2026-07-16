"""
Report the number of filtered pretrain events for a given data_root and fraction.
Reads the actual data -- no hardcoded counts. Used by the sweep runner to bake
the true event count into each run's name.

Supports both real mmap_ninja.RaggedMmap directories and Alessio's
values.npy+offsets.npy format (see dev_scripts/ragged_npy_reader.py) --
auto-detects which one a folder actually is.

Usage:
    python3 dev_scripts/count_events.py <data_root> <fraction>
Example:
    python3 dev_scripts/count_events.py /workspace/PP_collision/data/mmap_v4 0.01
Prints a single integer: the number of events that fraction corresponds to.
"""
import sys, os
sys.path.insert(0, "/workspace/PP_collision")
sys.path.insert(0, "/workspace/PP_collision/dev_scripts")

def open_reader(folder):
    """Auto-detect: real RaggedMmap (has a 'data' file + shapes_are_flat.ninja)
    vs. Alessio's values.npy+offsets.npy format."""
    if os.path.exists(os.path.join(folder, "values.npy")) and os.path.exists(os.path.join(folder, "offsets.npy")):
        from ragged_npy_reader import RaggedNpyReader
        return RaggedNpyReader(folder)
    else:
        from mmap_ninja import RaggedMmap
        return RaggedMmap(folder)

def filtered_count(data_root, split="pretrain", low_thr=1, high_thr=100):
    mm = open_reader(os.path.join(data_root, f"features_{split}"))
    n = 0
    for i in range(len(mm)):
        L = mm[i].shape[0]
        if low_thr <= L <= high_thr:
            n += 1
    return n

if __name__ == "__main__":
    data_root = sys.argv[1]
    fraction = float(sys.argv[2])
    total = filtered_count(data_root)
    if total == 0:
        print(f"[count_events.py] WARNING: total filtered count is 0 for {data_root} "
              f"-- check reader/path.", file=sys.stderr)
    used = max(1, int(round(total * fraction)))
    print(used)