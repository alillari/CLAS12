"""Validate saved momentum units independently of the evaluation configuration."""

from collections.abc import Mapping
import math


def validate_hdf5_momentum_checkpoint_units(checkpoint):
    """Require consistent, explicit GeV evidence before loading HDF5 weights.

    A fresh HDF5 stats file cannot establish the units of a checkpoint's saved
    normalizer (or a scalar log-momentum output). Older checkpoints may keep
    their unit declaration in vars(YParams), including its nested params copy.
    """
    params = checkpoint.get("params") or {}
    if not isinstance(params, Mapping) or not isinstance(params.get("params", {}), Mapping):
        raise ValueError("Checkpoint momentum units cannot be read from malformed saved params")
    saved = {**params.get("params", {}), **params}
    declarations = []
    for name, contract in (("data_contract", checkpoint.get("data_contract")),
                           ("params.data_contract", saved.get("data_contract"))):
        if contract is not None:
            if not isinstance(contract, Mapping):
                raise ValueError(f"Checkpoint momentum units: malformed {name}")
            if "momentum_unit" in contract:
                unit = contract["momentum_unit"]
                declarations.append((name + ".momentum_unit", unit, unit == "GeV/c"))
    for name, config in (("physics_checkpoint_config", checkpoint.get("physics_checkpoint_config")),
                         ("params.physics_checkpoint", saved.get("physics_checkpoint"))):
        if config is not None:
            if not isinstance(config, Mapping):
                raise ValueError(f"Checkpoint momentum units: malformed {name}")
            if "momentum_scale_to_gev" in config:
                value = config["momentum_scale_to_gev"]
                try:
                    scale = float(value)
                    valid = not isinstance(value, bool) and math.isfinite(scale) and scale == 1.0
                except (TypeError, ValueError, OverflowError):
                    valid = False
                declarations.append((name + ".momentum_scale_to_gev", value, valid))
    if not declarations:
        raise ValueError(
            "Checkpoint momentum units are unknown; HDF5 momentum requires saved GeV/c units. "
            "Use a checkpoint with verified training-unit metadata, not a replacement stats file."
        )
    for name, value, valid in declarations:
        if not valid:
            raise ValueError(
                f"Checkpoint momentum unit mismatch: {name}={value!r}; HDF5 momentum requires "
                "a checkpoint trained in GeV/c (scale 1). Evaluate MeV checkpoints with their "
                "original mmap configuration or use a GeV-trained checkpoint."
            )
