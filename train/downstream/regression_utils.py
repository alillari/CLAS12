import json
from pathlib import Path

import numpy as np
import torch


REGRESSION_TARGET_COLUMNS = (
    "mc_entrance_px",
    "mc_entrance_py",
    "mc_entrance_pz",
    "mc_vx",
    "mc_vy",
    "mc_vz",
    "mc_energy",
)

TASK_ALIASES = {
    "momentum": "mom",
}

TARGET_COLUMNS_BY_TASK = {
    "mom": ("mc_entrance_px", "mc_entrance_py", "mc_entrance_pz"),
    "3vtx": ("mc_vx", "mc_vy", "mc_vz"),
    "zvtx": ("mc_vz",),
    "pt_phi_eta": ("mc_entrance_pt", "mc_entrance_phi", "mc_entrance_eta"),
}

ANGULAR_INDICES_BY_TASK = {
    "pt_phi_eta": (1,),
}


def canonical_regression_task(task):
    task = str(task)
    lowered = task.lower()
    if task in ("3vertex", "3vtx"):
        return "3vtx"
    if task in ("Zvtx", "Zvertex", "zvtx", "zvertex"):
        return "zvtx"
    return TASK_ALIASES.get(lowered, lowered)


def regression_column_indices(task):
    task = canonical_regression_task(task)
    if task == "mom":
        return (0, 1, 2)
    if task == "3vtx":
        return (3, 4, 5)
    if task == "zvtx":
        return (5,)
    raise ValueError(f"Unknown regression task: {task}")


def regression_target_columns(task):
    task = canonical_regression_task(task)
    try:
        return TARGET_COLUMNS_BY_TASK[task]
    except KeyError as exc:
        raise ValueError(f"Unknown regression task: {task}") from exc


def regression_output_dim(task):
    return len(regression_target_columns(task))


def regression_angular_indices(task):
    return ANGULAR_INDICES_BY_TASK.get(canonical_regression_task(task), ())


def _torch_pt_phi_eta(px, py, pz, eps=1.0e-12):
    pt = torch.sqrt(px * px + py * py)
    phi = torch.atan2(py, px)
    eta = torch.asinh(torch.where(pt > eps, pz / pt, torch.full_like(pt, float("nan"))))
    return torch.stack((pt, phi, eta), dim=-1)


def _numpy_pt_phi_eta(px, py, pz, eps=1.0e-12):
    pt = np.hypot(px, py)
    phi = np.arctan2(py, px)
    eta = np.arcsinh(np.divide(pz, pt, out=np.full_like(pz, np.nan, dtype=float), where=pt > eps))
    return np.stack((pt, phi, eta), axis=-1)


def transform_regression_target_torch(reg, task):
    task = canonical_regression_task(task)
    if task in {"mom", "3vtx", "zvtx"}:
        return reg[..., list(regression_column_indices(task))]
    if task == "pt_phi_eta":
        return _torch_pt_phi_eta(reg[..., 0], reg[..., 1], reg[..., 2])
    raise ValueError(f"Unknown regression task: {task}")


def transform_regression_target_numpy(reg, task):
    task = canonical_regression_task(task)
    reg = np.asarray(reg)
    if task in {"mom", "3vtx", "zvtx"}:
        return reg[..., list(regression_column_indices(task))]
    if task == "pt_phi_eta":
        return _numpy_pt_phi_eta(reg[..., 0], reg[..., 1], reg[..., 2])
    raise ValueError(f"Unknown regression task: {task}")


def target_to_cartesian_numpy(target, task):
    task = canonical_regression_task(task)
    target = np.asarray(target, dtype=float)
    if task == "mom":
        return target
    if task == "pt_phi_eta":
        pt, phi, eta = np.moveaxis(target, -1, 0)
        return np.stack((pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)), axis=-1)
    raise ValueError(
        f"Task {task!r} cannot be converted to Cartesian momentum for evaluation"
    )


def load_regression_target_stats(path, task):
    path = Path(path)
    with path.open() as stream:
        stats = json.load(stream)

    columns = tuple(stats["columns"])
    expected_columns = regression_target_columns(task)
    if columns == REGRESSION_TARGET_COLUMNS and canonical_regression_task(task) in {"mom", "3vtx", "zvtx"}:
        indices = regression_column_indices(task)
        selected_columns = [columns[index] for index in indices]
        mean = stats["mean"]
        std = stats["std"]
        if len(mean) != len(columns) or len(std) != len(columns):
            raise ValueError(f"Malformed regression statistics in {path}")
        selected_mean = [mean[index] for index in indices]
        selected_std = [std[index] for index in indices]
    elif columns == expected_columns:
        selected_columns = list(columns)
        selected_mean = stats["mean"]
        selected_std = stats["std"]
        if len(selected_mean) != len(columns) or len(selected_std) != len(columns):
            raise ValueError(f"Malformed regression statistics in {path}")
    else:
        raise ValueError(
            f"Unexpected regression columns in {path}: {columns}; "
            f"expected {expected_columns}"
        )

    return {
        "path": str(path),
        "task": canonical_regression_task(task),
        "columns": selected_columns,
        "mean": selected_mean,
        "std": selected_std,
        "angular_indices": list(regression_angular_indices(task)),
    }
