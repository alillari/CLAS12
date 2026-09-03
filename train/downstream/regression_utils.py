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
    "pt_phi_eta": (
        "mc_entrance_pt",
        "mc_entrance_cosphi",
        "mc_entrance_sinphi",
        "mc_entrance_eta",
    ),
    "p_phi_theta": (
        "mc_entrance_p",
        "mc_entrance_cosphi",
        "mc_entrance_sinphi",
        "mc_entrance_theta",
    ),
}

UNSTANDARDIZED_COLUMNS_BY_TASK = {
    "pt_phi_eta": ("mc_entrance_cosphi", "mc_entrance_sinphi"),
    "p_phi_theta": ("mc_entrance_cosphi", "mc_entrance_sinphi"),
}

PHI_PAIR_INDICES_BY_TASK = {
    "pt_phi_eta": ((1, 2),),
    "p_phi_theta": ((1, 2),),
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
    return ()


def regression_unstandardized_columns(task):
    return UNSTANDARDIZED_COLUMNS_BY_TASK.get(canonical_regression_task(task), ())


def regression_phi_pairs(task):
    return PHI_PAIR_INDICES_BY_TASK.get(canonical_regression_task(task), ())


def _torch_pt_phi_eta(px, py, pz, eps=1.0e-12):
    pt = torch.sqrt(px * px + py * py)
    phi = torch.atan2(py, px)
    cosphi = torch.cos(phi)
    sinphi = torch.sin(phi)
    eta = torch.asinh(torch.where(pt > eps, pz / pt, torch.full_like(pt, float("nan"))))
    return torch.stack((pt, cosphi, sinphi, eta), dim=-1)


def _numpy_pt_phi_eta(px, py, pz, eps=1.0e-12):
    pt = np.hypot(px, py)
    phi = np.arctan2(py, px)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    eta = np.arcsinh(np.divide(pz, pt, out=np.full_like(pz, np.nan, dtype=float), where=pt > eps))
    return np.stack((pt, cosphi, sinphi, eta), axis=-1)


def _torch_p_phi_theta(px, py, pz, eps=1.0e-12):
    pt = torch.sqrt(px * px + py * py)
    p = torch.sqrt(pt * pt + pz * pz)
    phi = torch.atan2(py, px)
    cosphi = torch.cos(phi)
    sinphi = torch.sin(phi)
    theta = torch.atan2(pt, pz)
    theta = torch.where(p > eps, theta, torch.full_like(theta, float("nan")))
    return torch.stack((p, cosphi, sinphi, theta), dim=-1)


def _numpy_p_phi_theta(px, py, pz, eps=1.0e-12):
    pt = np.hypot(px, py)
    p = np.sqrt(pt * pt + pz * pz)
    phi = np.arctan2(py, px)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    theta = np.arctan2(pt, pz)
    theta = np.where(p > eps, theta, np.nan)
    return np.stack((p, cosphi, sinphi, theta), axis=-1)


def transform_regression_target_torch(reg, task):
    task = canonical_regression_task(task)
    if task in {"mom", "3vtx", "zvtx"}:
        return reg[..., list(regression_column_indices(task))]
    if task == "pt_phi_eta":
        return _torch_pt_phi_eta(reg[..., 0], reg[..., 1], reg[..., 2])
    if task == "p_phi_theta":
        return _torch_p_phi_theta(reg[..., 0], reg[..., 1], reg[..., 2])
    raise ValueError(f"Unknown regression task: {task}")


def transform_regression_target_numpy(reg, task):
    task = canonical_regression_task(task)
    reg = np.asarray(reg)
    if task in {"mom", "3vtx", "zvtx"}:
        return reg[..., list(regression_column_indices(task))]
    if task == "pt_phi_eta":
        return _numpy_pt_phi_eta(reg[..., 0], reg[..., 1], reg[..., 2])
    if task == "p_phi_theta":
        return _numpy_p_phi_theta(reg[..., 0], reg[..., 1], reg[..., 2])
    raise ValueError(f"Unknown regression task: {task}")


def project_phi_pair_numpy(cosphi, sinphi, eps=1.0e-12):
    radius = np.sqrt(cosphi * cosphi + sinphi * sinphi + eps)
    return cosphi / radius, sinphi / radius


def target_to_cartesian_numpy(target, task):
    task = canonical_regression_task(task)
    target = np.asarray(target, dtype=float)
    if task == "mom":
        return target
    if task == "pt_phi_eta":
        pt, cosphi, sinphi, eta = np.moveaxis(target, -1, 0)
        cosphi, sinphi = project_phi_pair_numpy(cosphi, sinphi)
        return np.stack((pt * cosphi, pt * sinphi, pt * np.sinh(eta)), axis=-1)
    if task == "p_phi_theta":
        p, cosphi, sinphi, theta = np.moveaxis(target, -1, 0)
        cosphi, sinphi = project_phi_pair_numpy(cosphi, sinphi)
        pt = p * np.sin(theta)
        return np.stack((pt * cosphi, pt * sinphi, p * np.cos(theta)), axis=-1)
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

    unstandardized = set(regression_unstandardized_columns(task))
    selected_mean = [
        0.0 if column in unstandardized else value
        for column, value in zip(selected_columns, selected_mean)
    ]
    selected_std = [
        1.0 if column in unstandardized else value
        for column, value in zip(selected_columns, selected_std)
    ]

    return {
        "path": str(path),
        "task": canonical_regression_task(task),
        "columns": selected_columns,
        "mean": selected_mean,
        "std": selected_std,
        "angular_indices": list(regression_angular_indices(task)),
        "phi_pairs": [list(pair) for pair in regression_phi_pairs(task)],
    }
