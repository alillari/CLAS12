"""Configuration guards for the supported track-finding background policy."""

from collections.abc import Mapping


SIGNAL_ONLY = "signal_only"
FINDING_METRICS_VERSION = 2


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

def decoder_settings(params, head=None):
    # These settings can change predictions without changing state-dict shapes.
    defaults = {"softmax_mask": False, "do_masked_attn": True, "num_heads_decoder": 4,
                "embed_method": "add", "pe_method": "nerf", "use_lora": False}
    settings = {key: params.get(key, default) for key, default in defaults.items()}
    if head is not None:
        settings.update(softmax_mask=head.softmax_mask, do_masked_attn=head.do_masked_attn,
                        num_heads_decoder=head.num_heads, embed_method=head.embed_method, pe_method=head.pe_method)
    if settings["use_lora"]:
        settings["lora_alpha"] = params.get("lora_alpha", 16)
    return settings


def validate_checkpoint_forward(checkpoint, current_decoder, use_pretrained, backbone_sha256):
    saved = checkpoint.get("forward_config")
    if saved is None:
        # Historical checkpoints saved vars(YParams). The resolved attributes
        # take precedence over the original nested configuration.
        attributes = checkpoint.get("params", {})
        old = {**attributes.get("params", {}), **attributes}
        if "pretrained_ckpt" not in old:
            raise ValueError("Cannot verify the input mode of this legacy track-finding checkpoint")
        saved = {"use_pretrained_backbone": old["pretrained_ckpt"] is not None,
                 "decoder": decoder_settings(old)}
    if saved["use_pretrained_backbone"] != use_pretrained:
        raise ValueError("Track-finding checkpoint input mode differs from requested use_pretrained_backbone")
    if use_pretrained:
        if not backbone_sha256:
            raise ValueError("Frozen track finding requires a loaded pretrained backbone checkpoint")
        if "forward_config" in checkpoint and saved["backbone_sha256"] != backbone_sha256:
            raise ValueError("Track-finding checkpoint was trained with different pretrained backbone weights")
    if saved["decoder"] != current_decoder:
        raise ValueError(f"Track-finding decoder mismatch: saved {saved['decoder']!r}, requested {current_decoder!r}")
