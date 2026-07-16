"""
Minimal drop-in reader for Alessio's "values.npy + offsets.npy" ragged format
(NOT a real mmap_ninja.RaggedMmap -- see WARNING_NOT_RAGGEDMMAP.txt in each folder).

Exposes the same len()/__getitem__(i) interface TPCBatchDataset already uses
for self.memmap_feature / self.memmap_seg_target / self.memmap_reg_target, so
it's a drop-in replacement requiring only an import + constructor swap.

Convention confirmed from data:
  offsets.npy: int64, length = n_events + 1, cumulative row offsets, offsets[0]=0
  values.npy:  float32/int64, shape (total_rows, n_cols) or (total_rows,)
  event i's data = values[offsets[i] : offsets[i+1]]

Usage (mirrors RaggedMmap(os.path.join(data_root, 'features_{split}'))):
    from dev_scripts.ragged_npy_reader import RaggedNpyReader
    mm = RaggedNpyReader(os.path.join(data_root, 'features_pretrain'))
    mm[i]        # -> np.ndarray, this event's rows
    len(mm)      # -> number of events
"""
import os
import numpy as np


class RaggedNpyReader:
    def __init__(self, folder):
        self.folder = folder
        offsets_path = os.path.join(folder, "offsets.npy")
        values_path = os.path.join(folder, "values.npy")
        if not os.path.exists(offsets_path) or not os.path.exists(values_path):
            raise FileNotFoundError(
                f"Expected offsets.npy and values.npy in {folder}, "
                f"found: {os.listdir(folder) if os.path.isdir(folder) else 'NOT A DIR'}"
            )
        self.offsets = np.load(offsets_path)
        # mmap_mode='r' so the (potentially multi-GB) values array is NOT
        # loaded into RAM -- rows are read from disk on demand, same spirit
        # as RaggedMmap's memory-mapped design.
        self.values = np.load(values_path, mmap_mode="r")

        if self.offsets[0] != 0:
            raise ValueError(f"{offsets_path}: expected offsets[0] == 0, got {self.offsets[0]}")
        if self.offsets[-1] != self.values.shape[0]:
            raise ValueError(
                f"{offsets_path}: offsets[-1]={self.offsets[-1]} != "
                f"values.shape[0]={self.values.shape[0]} -- data may be corrupt/mismatched."
            )
        self.n = len(self.offsets) - 1

    def __len__(self):
        return self.n

    def lengths(self):
        """All event lengths, computed instantly from offsets -- no data reads.
        Enables fast filtering (vs reading every event to check its shape)."""
        import numpy as np
        return np.diff(self.offsets)

    def __getitem__(self, i):
        if i < 0 or i >= self.n:
            raise IndexError(f"index {i} out of range for {self.n} events")
        start, end = self.offsets[i], self.offsets[i + 1]
        # np.array(...) copies out of the memmap into a normal in-memory array,
        # matching what callers expect (they index/slice/convert to torch after).
        return np.array(self.values[start:end])