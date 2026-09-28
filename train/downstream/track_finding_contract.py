"""Configuration guards for the supported track-finding background policy."""

from collections.abc import Mapping


SIGNAL_ONLY = "signal_only"


def validate_track_finding_modes(config: Mapping, *, source: str = "configuration") -> None:
    """Reject retired modes, including legacy checkpoint parameter containers.

    Missing fields retain the historical signal-only/native defaults. Explicit
    unsupported values must never be silently replaced by those defaults.
    """
    for key, supported in (("track_target_mode", SIGNAL_ONLY), ("noise_attribution_mode", "native")):
        if key in config and config[key] != supported:
            raise ValueError(
                f"{source}: unsupported {key}={config[key]!r}; only {supported!r} "
                "is supported. Aggregate background-query training and "
                "truth-assisted noise attribution have been removed."
            )
    nested = config.get("params")
    if isinstance(nested, Mapping):
        validate_track_finding_modes(nested, source=f"{source}.params")
