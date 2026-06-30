import json
from pathlib import Path


REGRESSION_TARGET_COLUMNS = (
    "mc_entrance_px",
    "mc_entrance_py",
    "mc_entrance_pz",
    "mc_vx",
    "mc_vy",
    "mc_vz",
    "mc_energy",
)


def regression_column_indices(task):
    if task in ("mom", "momentum"):
        return (0, 1, 2)
    if task in ("3vtx", "3vertex"):
        return (3, 4, 5)
    if task in ("Zvtx", "Zvertex", "zvtx", "zvertex"):
        return (5,)
    raise ValueError(f"Unknown regression task: {task}")


def load_regression_target_stats(path, task):
    path = Path(path)
    with path.open() as stream:
        stats = json.load(stream)

    columns = tuple(stats["columns"])
    if columns != REGRESSION_TARGET_COLUMNS:
        raise ValueError(
            f"Unexpected regression columns in {path}: {columns}; "
            f"expected {REGRESSION_TARGET_COLUMNS}"
        )

    indices = regression_column_indices(task)
    mean = stats["mean"]
    std = stats["std"]
    if len(mean) != len(columns) or len(std) != len(columns):
        raise ValueError(f"Malformed regression statistics in {path}")

    return {
        "path": str(path),
        "indices": indices,
        "columns": [columns[index] for index in indices],
        "mean": [mean[index] for index in indices],
        "std": [std[index] for index in indices],
    }
