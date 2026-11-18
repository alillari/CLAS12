"""Optional learning-rate control for the pretrained layer mixture."""
import math

from fm4npp.models.mamba_config import optimizer_parameters


def feature_mixing_lr_ratio(params):
    value = float(getattr(params, "adapter_feature_mixing_lr_ratio", 1.0))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("adapter_feature_mixing_lr_ratio must be finite and positive")
    return value


def regression_optimizer_parameters(model, params):
    ratio = feature_mixing_lr_ratio(params)
    if ratio == 1.0:
        return optimizer_parameters(model)
    head = getattr(model, "module", model)
    mixer = getattr(head, "weighted_avg_weights", None)
    if mixer is None or not mixer.requires_grad:
        raise ValueError("A feature-mixing LR multiplier requires trainable layer-mixture weights")
    groups = [
        {"params": [p for p in model.parameters() if p is not mixer], "lr_ratio": 1.0},
        {"params": [mixer], "lr_ratio": ratio},
    ]
    return optimizer_parameters(model, groups)


def apply_regression_lr_ratios(optimizer):
    """Apply ratios before each update, after the scheduler set its base LR.

    Group zero always has ratio one. Repeating this call is idempotent, including
    during epoch-based scheduling. Historical optimizers have no ratio metadata
    and are left untouched.
    """
    if not any("lr_ratio" in group for group in optimizer.param_groups):
        return
    if optimizer.param_groups[0].get("lr_ratio", 1.0) != 1.0:
        raise ValueError("Regression optimizer's first group must use the base learning rate")
    base = optimizer.param_groups[0]["lr"]
    for group in optimizer.param_groups:
        group["lr"] = base * group.get("lr_ratio", 1.0)


def validate_regression_optimizer_resume(checkpoint, params):
    saved = float(checkpoint.get("adapter_feature_mixing_lr_ratio", 1.0))
    expected = feature_mixing_lr_ratio(params)
    if saved != expected:
        raise ValueError(
            "Cannot resume with a different adapter_feature_mixing_lr_ratio: "
            f"checkpoint={saved}, current={expected}"
        )
