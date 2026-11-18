"""Initialize fresh momentum adapters without overwriting Mamba2 dynamics."""

import math

import torch

from fm4npp.models.mamba2 import Mamba2
from fm4npp.models.mamba_config import legacy_initialization_parameters


def initialize_regression_adapter(model, ssm_initialization="constructor"):
    """Preserve Mamba2 dt_bias, A_log and D while initializing the other weights.

    ``legacy`` reproduces the historical initializer, including its overwrites.
    Both modes consume identical random draws, keeping every other initial
    parameter and the subsequent RNG stream equal for paired comparisons.
    Existing Mamba3/Transformer protected-initialization rules still apply.
    Apply this only to a fresh head, before loading any trained checkpoint.
    """
    if ssm_initialization not in {"legacy", "constructor"}:
        raise ValueError(
            "adapter_ssm_initialization must be 'legacy' or 'constructor'; "
            f"got {ssm_initialization!r}"
        )
    parameters = list(legacy_initialization_parameters(model))
    ssm_ids = {
        id(getattr(block, name))
        for block in model.modules() if isinstance(block, Mamba2)
        for name in ("dt_bias", "A_log", "D")
    }
    preserved = {
        name: param.detach().clone()
        for name, param in parameters
        if ssm_initialization == "constructor" and id(param) in ssm_ids
    }
    with torch.no_grad():
        for name, param in parameters:
            if "A" in name:
                torch.nn.init.uniform_(param, -0.1 / 3, 0.1 / 3)
            elif "D" in name:
                torch.nn.init.normal_(param, mean=0.1, std=0.02)
            elif "conv1d.weight" in name:
                torch.nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="linear")
            elif "out_proj.weight" in name or "in_proj.weight" in name:
                torch.nn.init.xavier_uniform_(param, gain=1.0 / math.sqrt(3))
            elif "norm.weight" in name:
                torch.nn.init.ones_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
        for name, param in parameters:
            if name in preserved:
                param.copy_(preserved[name])
