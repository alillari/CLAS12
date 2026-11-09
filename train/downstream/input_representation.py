"""Pack serialized CLAS12 centers and geometry into the backbone's tokens."""

from collections.abc import Mapping
import math

import torch

from fm4npp.models.mamba_config import backbone_version, value


AUX_REPRESENTATION = "clas12_pos_plus_aux_v1"


def input_config(params):
    representation = value(params, "input_representation", "center_only")
    config = {"input_representation": representation}
    if representation == AUX_REPRESENTATION:
        config.update(pos_dim=int(value(params, "pos_dim", 3)),
                      aux_dim=int(value(params, "aux_dim", 4)),
                      aux_extra_feature=value(params, "aux_extra_feature", "length"))
    return config


def validate_input_representation(params):
    config = input_config(params)
    representation = config["input_representation"]
    if representation not in {"center_only", AUX_REPRESENTATION}:
        raise ValueError(f"Unknown input_representation: {representation!r}")
    if representation == "center_only":
        if value(params, "embed_method") == "pos_plus_aux":
            raise ValueError("embed_method='pos_plus_aux' requires input_representation='clas12_pos_plus_aux_v1'")
        return
    if value(params, "embed_method") != "pos_plus_aux":
        raise ValueError("clas12_pos_plus_aux_v1 requires embed_method='pos_plus_aux'")
    if config["pos_dim"] != 3:
        raise ValueError("clas12_pos_plus_aux_v1 requires pos_dim=3")
    extra = config["aux_extra_feature"]
    if extra not in {"length", "pitch", "both"}:
        raise ValueError("aux_extra_feature must be one of 'length', 'pitch', or 'both'")
    expected = 5 if extra == "both" else 4
    if config["aux_dim"] != expected:
        raise ValueError(f"aux_extra_feature={extra!r} produces {expected} auxiliary columns, "
                         f"but aux_dim={config['aux_dim']}")
    if value(params, "input_layout", "xyz") not in {"xyz", "etaphr"}:
        raise ValueError("clas12_pos_plus_aux_v1 requires input_layout='xyz' or 'etaphr'")
    if backbone_version(params) not in {"mamba1", "mamba2", "mamba3", "transformer"}:
        raise ValueError("clas12_pos_plus_aux_v1 requires a Mamba or TransformerGPT backbone")


def representation_input(points, batch, params):
    """Keep centers separate for metrics; pack geometry only for model calls.

    Geometry has eleven columns: endpoints (6), Cartesian strip direction (3),
    length in cm, and pitch in cm. Rows already share the centers' serialization.
    """
    if value(params, "input_representation", "center_only") != AUX_REPRESENTATION:
        return points
    if points.ndim != 3 or points.size(-1) != 3:
        raise ValueError("clas12_pos_plus_aux_v1 expects normalized center points with three columns (B, N, 3)")
    if not isinstance(batch, Mapping) or "geometry_context" not in batch:
        raise KeyError("clas12_pos_plus_aux_v1 requires geometry_context from HDF5 or v7 sidecars")
    geometry = batch["geometry_context"]
    if geometry.ndim != 3 or geometry.shape[:2] != points.shape[:2] or geometry.size(-1) != 11:
        raise ValueError("geometry_context must be row-aligned (B, N, 11) with serialized points; "
                         f"got points={tuple(points.shape)}, geometry={tuple(geometry.shape)}")
    geometry = geometry.to(device=points.device, dtype=points.dtype)
    # Equivalent to projecting raw x/y onto their local cylindrical frame.
    phi = points[..., 1] * (2.0 * math.pi) - math.pi
    sx, sy, sz = geometry[..., 6:9].unbind(-1)
    direction = torch.stack((sx * torch.cos(phi) + sy * torch.sin(phi),
                             -sx * torch.sin(phi) + sy * torch.cos(phi), sz), dim=-1)
    extra = value(params, "aux_extra_feature", "length")
    length, pitch = geometry[..., 9:10] / 45.0, geometry[..., 10:11] / 0.10
    if extra == "length":
        scalar = length
    elif extra == "pitch":
        scalar = pitch
    elif extra == "both":
        scalar = torch.cat((length, pitch), dim=-1)
    else:
        raise ValueError(f"Unexpected aux_extra_feature={extra!r}")
    packed = torch.cat((points, direction, scalar), dim=-1)
    expected = int(value(params, "pos_dim", 3)) + int(value(params, "aux_dim", 4))
    if packed.size(-1) != expected:
        raise ValueError(f"Packed CLAS12 token has width {packed.size(-1)}, but pos_dim + aux_dim is {expected}; "
                         "check aux_extra_feature and aux_dim")
    return packed.masked_fill((points[..., 0] == -100).unsqueeze(-1), -100.)


def validate_checkpoint_inputs(checkpoint, params):
    saved = checkpoint.get("input_config")
    if saved is None:
        raw = checkpoint.get("training_config", checkpoint.get("params", {}))
        saved = input_config({**raw.get("params", {}), **raw})
    current = input_config(params)
    if saved != current:
        raise ValueError(f"Checkpoint input representation differs: saved={saved}, requested={current}")
