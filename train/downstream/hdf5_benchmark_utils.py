"""Shared serialization and cohort identity checks for HDF5 benchmarks."""
import json
import numpy as np


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def write_json(path, data):
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def cohort(dataset):
    rows = []
    for key in dataset.idxlist:
        index, segment = key if isinstance(key, tuple) else (key, None)
        rows.append(dict(dataset.hdf5_source.identity(int(index)),
                         source_event_index=int(index), segment_label=segment))
    return rows


def event_keys(rows):
    return {(r["source_id"], r["original_ordinal"]) for r in rows}
