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


def load_regression_loss_reference_stats(path, task, momentum_residual="absolute"):
    """Load the opt-in physical-resolution reference for ``p_phi_theta``.

    This is deliberately separate from target standardization statistics.  The
    latter describe the MC-label distribution; this artifact records a matched
    conventional-reconstruction residual scale in physical units.
    """
    path = Path(path)
    with path.open() as stream:
        stats = json.load(stream)

    expected_task = "p_phi_theta"
    if canonical_regression_task(task) != expected_task:
        raise ValueError(
            "Physical-resolution losses are currently defined only for "
            f"{expected_task!r}, not {canonical_regression_task(task)!r}"
        )
    if momentum_residual not in {"absolute", "relative"}:
        raise ValueError(f"Unknown momentum residual mode {momentum_residual!r}")
    if stats.get("schema") != "clas12_regression_loss_reference_v1":
        raise ValueError(
            f"Unexpected regression loss-reference schema in {path}: "
            f"{stats.get('schema')!r}"
        )
    if canonical_regression_task(stats.get("task", "")) != expected_task:
        raise ValueError(
            f"Loss-reference task in {path} is {stats.get('task')!r}, "
            f"expected {expected_task!r}"
        )

    residuals = stats.get("residuals")
    if not isinstance(residuals, dict):
        raise ValueError(f"Loss-reference file {path} has no residuals object")

    momentum_key = (
        ("p_scale_gev", "p_absolute_gev", "GeV")
        if momentum_residual == "absolute"
        else ("p_scale_relative", "p_relative", "fraction")
    )
    keys = {
        momentum_key[0]: (momentum_key[1], momentum_key[2]),
        "theta_scale_rad": ("theta_rad", "rad"),
        "phi_scale_rad": ("phi_rad_wrapped", "rad"),
    }
    loaded = {}
    for output_name, (name, unit) in keys.items():
        payload = residuals.get(name)
        if not isinstance(payload, dict):
            raise ValueError(f"Loss-reference file {path} is missing residual {name!r}")
        if payload.get("unit") != unit:
            raise ValueError(
                f"Loss-reference residual {name!r} in {path} has unit "
                f"{payload.get('unit')!r}, expected {unit!r}"
            )
        try:
            value = float(payload["central_width_68"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Loss-reference residual {name!r} in {path} lacks a finite "
                "central_width_68"
            ) from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Loss-reference residual {name!r} in {path} must have a "
                f"positive finite central_width_68, got {value!r}"
            )
        loaded[output_name] = value

    try:
        momentum_scale = float(stats["target_momentum_scale_to_gev"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Loss-reference file {path} lacks target_momentum_scale_to_gev"
        ) from exc
    if not np.isfinite(momentum_scale) or momentum_scale <= 0.0:
        raise ValueError(
            f"Loss-reference file {path} has invalid target_momentum_scale_to_gev "
            f"{momentum_scale!r}"
        )
    loaded.update({
        "path": str(path),
        "target_momentum_scale_to_gev": momentum_scale,
        "reference_method": stats.get("reference_method"),
    })
    return loaded
