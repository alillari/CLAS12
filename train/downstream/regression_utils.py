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

SINGLE_TARGET_TASKS = frozenset({"p", "theta", "phi"})

TARGET_COLUMNS_BY_TASK = {
    # p uses log momentum in the stored reg_target momentum unit. Predicting
    # log(p) makes p_pred = exp(output) positive without a clamped gradient.
    "p": ("mc_entrance_log_p",),
    "theta": ("mc_entrance_theta",),
    "phi": ("mc_entrance_phi",),
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
    "p": ("mc_entrance_log_p",),
    "theta": ("mc_entrance_theta",),
    "phi": ("mc_entrance_phi",),
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
    return (0,) if canonical_regression_task(task) == "phi" else ()


def resolve_regression_loss(task, loss=None):
    """Keep single-target objectives in log momentum / physical radians."""
    single = canonical_regression_task(task) in SINGLE_TARGET_TASKS
    loss = (loss or ("huber" if single else "mse")).lower()
    if single and loss != "huber":
        raise ValueError(f"task={task!r} requires regression_loss: huber")
    return loss


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
    if task in SINGLE_TARGET_TASKS:
        px, py, pz = reg[..., 0], reg[..., 1], reg[..., 2]
        pt = torch.hypot(px, py)
        p = torch.hypot(pt, pz)
        if task == "p":
            value = torch.log(torch.where(p > 0, p, torch.full_like(p, float("nan"))))
        elif task == "theta":
            value = torch.where(p > 0, torch.atan2(pt, pz), torch.full_like(p, float("nan")))
        else:
            value = torch.where(pt > 0, torch.atan2(py, px), torch.full_like(pt, float("nan")))
        return value.unsqueeze(-1)
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
    if task in SINGLE_TARGET_TASKS:
        px, py, pz = reg[..., 0], reg[..., 1], reg[..., 2]
        pt = np.hypot(px, py)
        p = np.hypot(pt, pz)
        if task == "p":
            value = np.log(np.where(p > 0, p, np.nan))
        elif task == "theta":
            value = np.where(p > 0, np.arctan2(pt, pz), np.nan)
        else:
            value = np.where(pt > 0, np.arctan2(py, px), np.nan)
        return value[..., None]
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


def single_target_to_physical_numpy(target, task):
    """Decode a one-output head; momentum keeps the input data's units."""
    task = canonical_regression_task(task)
    target = np.asarray(target, dtype=float)
    if task not in SINGLE_TARGET_TASKS or target.shape[-1] != 1:
        raise ValueError("Expected a single-target task and final dimension 1")
    if task == "p":
        return np.exp(target)
    if task == "phi":
        return np.arctan2(np.sin(target), np.cos(target))
    return target.copy()


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
def validate_regression_training(params):
    """Reject retired experimental recipes before opening data or training."""
    task = canonical_regression_task(params.get("task", "mom"))
    if task in SINGLE_TARGET_TASKS:
        raise ValueError(
            "Single-target momentum training is retired; use task=mom. "
            "Historical scalar checkpoints remain evaluable."
        )
    if params.get("input_representation", "center_only") not in {"center_only", "clas12_pos_plus_aux_v1"}:
        raise ValueError("Use center_only or clas12_pos_plus_aux_v1; endpoint-geometry training is retired.")
    if resolve_regression_loss(task, params.get("regression_loss")) not in {"mse", "mae", "huber"}:
        raise ValueError("Use regression_loss=mae, mse or huber; physical-resolution losses are retired.")
    if params.get("checkpoint_selection", "validation_loss") != "validation_loss":
        raise ValueError(
            "Training selects by validation_loss; physics checkpoint selection is retired. "
            "Physics metrics remain evaluation diagnostics."
        )
