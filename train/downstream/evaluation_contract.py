"""Entrance-only contract shared by regression evaluation and campaigns."""

import warnings


EVALUATION_CONTRACT = "clas12_momentum_entrance_v1"
CVT_REFERENCE_DEFINITION = (
    "CVT::Tracks p/theta with CVT::Trajectory entrance phi=atan2(cy,cx); "
    "no DOCA phi0 or swingback"
)


def configure_entrance_evaluation(config):
    """Migrate old rendered YAMLs without permitting DOCA evaluation."""
    comparison = config.get("comparison_truth", "mctrue_inner_hit")
    if comparison not in {"mctrue_inner_hit", "mctrue_swingback_doca"}:
        raise ValueError(f"Unsupported comparison_truth: {comparison!r}")
    if comparison == "mctrue_swingback_doca" or config.get("swingback_enabled", False):
        warnings.warn(
            "DOCA/swingback evaluation has been retired. Using MC::True and "
            "adapter outputs at the entrance, with CVT::Trajectory entrance phi.",
            UserWarning, stacklevel=2,
        )
    for key in list(config):
        if key.startswith("swingback_") or key == "write_unswung_diagnostics":
            config.pop(key)
    config.update({
        "evaluation_contract": EVALUATION_CONTRACT,
        "comparison_truth": "mctrue_inner_hit",
        "swingback_enabled": False,
    })
    return config


def is_entrance_evaluation(payload):
    return (
        payload.get("evaluation_contract") == EVALUATION_CONTRACT
        and payload.get("comparison_truth") == "mctrue_inner_hit"
        and not payload.get("swingback_enabled", False)
    )


def require_entrance_evaluation(payload, source):
    if not is_entrance_evaluation(payload):
        raise ValueError(
            f"{source} contains legacy or unversioned momentum evaluation. "
            "Reevaluate the existing adapter checkpoint with the entrance-only "
            "evaluator, then collate again; retraining is not required."
        )
