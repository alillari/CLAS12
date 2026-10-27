"""Shared construction and checkpoint contracts for backbones and adapters.

Factories return blocks directly, preserving all legacy state-dict prefixes.
New selections are independent: backbone_version (legacy: mambaversion) and
adapter_version. The latter controls Mamba layers, not attention decoders.
"""

from collections.abc import Mapping

import torch


MAMBA_VERSIONS = {"mamba1", "mamba2", "mamba3"}
BACKBONE_VERSIONS = MAMBA_VERSIONS | {"longformer", "linformer"}


def value(params, key, default=None):
    return params.get(key, default) if isinstance(params, Mapping) else getattr(params, key, default)


def backbone_version(params, default="mamba2"):
    version = value(params, "backbone_version", value(params, "mambaversion", default))
    if version not in BACKBONE_VERSIONS:
        raise ValueError(f"Unknown backbone_version {version!r}; choose {sorted(BACKBONE_VERSIONS)}")
    return version


def adapter_options(params):
    version = value(params, "adapter_version", "mamba2")
    if version not in MAMBA_VERSIONS:
        raise ValueError(f"Unknown adapter_version {version!r}; choose {sorted(MAMBA_VERSIONS)}")
    return {"adapter_version": version, "adapter_block_kwargs": block_options(params, "adapter")}


def block_options(params, prefix):
    names = ("d_state", "d_conv", "expand", "headdim", "ngroups", "chunk_size",
             "use_mem_eff_path", "rope_fraction", "is_mimo", "is_outproj_norm")
    result = {}
    for name in names:
        item = value(params, f"{prefix}_{name}")
        if item is None and prefix == "backbone":
            item = value(params, name)
        if item is not None:
            result[name] = item
    return result


def build_mamba_block(version, *, d_model, d_state=64, d_conv=4, expand=2, block_kwargs=None):
    if version not in MAMBA_VERSIONS:
        raise ValueError(f"Unknown Mamba version {version!r}")
    options = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    options.update(block_kwargs or {})
    if version == "mamba1":
        from mamba_ssm import Mamba
        cls = Mamba
        allowed = {"d_model", "d_state", "d_conv", "expand"}
    elif version == "mamba2":
        from fm4npp.models.mamba2 import Mamba2
        cls = Mamba2
        allowed = {"d_model", "d_state", "d_conv", "expand", "headdim", "ngroups",
                   "chunk_size", "use_mem_eff_path"}
    else:
        try:
            from fm4npp.models.mamba3 import Mamba3SISO
        except ImportError as exc:
            raise ImportError(
                "Mamba3 requires a mamba-ssm source installation with Mamba3 and "
                "its Triton dependencies; see train/MAMBA_SUPPORT.md"
            ) from exc
        cls = Mamba3SISO
        # Mamba3 has no convolution. Keep the shared call signature, but never
        # pass Mamba2-only options into upstream's permissive **kwargs.
        options.pop("d_conv")
        allowed = {"d_model", "d_state", "expand", "headdim", "ngroups", "chunk_size",
                   "rope_fraction", "is_mimo", "is_outproj_norm"}
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported {version} block options: {sorted(unknown)}")
    if version in {"mamba2", "mamba3"}:
        inner = options["d_model"] * options["expand"]
        head = options.get("headdim", 64)
        if head <= 0 or inner % head:
            raise ValueError(f"{version}: d_model * expand ({inner}) must be divisible by headdim ({head})")
        groups = options.get("ngroups", 1)
        if groups <= 0 or (inner // head) % groups:
            raise ValueError(f"{version}: number of heads must be divisible by ngroups ({groups})")
    block = cls(**options)
    config = {"version": version, **options}
    for name in ("headdim", "ngroups", "chunk_size"):
        if version != "mamba1":
            config.setdefault(name, getattr(block, name, 1))
    if version == "mamba2":
        config.setdefault("use_mem_eff_path", True)
    if version == "mamba3":
        config.setdefault("rope_fraction", 0.5)
        config.setdefault("is_mimo", False)
        config.setdefault("is_outproj_norm", False)
    block._mamba_config = config
    return block


def build_backbone(params, default="mamba2"):
    # Import lazily so config inspection does not initialize CUDA extensions.
    from fm4npp.models.mambagpt import Mamba1GPT, MambaGPT, Mamba3GPT
    version = backbone_version(params, default)
    common = dict(embed_dim=value(params, "embed_dim", 512),
                  num_layers=value(params, "num_layers_backbone", 12),
                  klen=value(params, "klen", 10), dropout=value(params, "dropout", 0.2),
                  embed_method=value(params, "embed_method", "add"),
                  pe_method=value(params, "pe_method", "nerf"))
    if version in MAMBA_VERSIONS:
        options = block_options(params, "backbone")
        d_state = options.pop("d_state", 16 if version == "mamba1" else 128 if version == "mamba3" else 64)
        d_conv = options.pop("d_conv", 4)
        expand = options.pop("expand", 2)
        cls = {"mamba1": Mamba1GPT, "mamba2": MambaGPT, "mamba3": Mamba3GPT}[version]
        return cls(**common, d_state=d_state, d_conv=d_conv, expand=expand,
                   pos_dim=value(params, "pos_dim", 3), aux_dim=value(params, "aux_dim", 4),
                   block_kwargs=options)
    common["num_heads"] = value(params, "num_heads_backbone", 4)
    common["mlp_ratio"] = value(params, "mlp_ratio", 2.0)
    if version == "longformer":
        from fm4npp.models.longformer_gpt import LongformerGPT
        model = LongformerGPT(**common, window_size=value(params, "window_size", 256))
    else:
        from fm4npp.models.linformer_gpt import LinformerGPT
        model = LinformerGPT(**common, seq_len=value(params, "seq_len", 512),
                             proj_dim=value(params, "proj_dim", 256))
    model.backbone_version = version
    return model


def legacy_initialization_parameters(model):
    """Retain legacy initialization exactly, except inside new Mamba3 blocks."""
    return ((name, param) for name, param in model.named_parameters()
            if not getattr(param, "_preserve_mamba_init", False))


def uses_mamba3(model):
    return any(getattr(p, "_preserve_mamba_init", False) for p in model.parameters())


def optimizer_parameters(model, groups=None):
    """Keep old optimizer layouts; honor upstream no-decay flags for Mamba3."""
    if not uses_mamba3(model):
        return model.parameters() if groups is None else groups
    groups = [{"params": list(model.parameters())}] if groups is None else groups
    result = []
    for group in groups:
        params = list(group["params"])
        for no_decay in (False, True):
            selected = [p for p in params if bool(getattr(p, "_no_weight_decay", False)) == no_decay]
            if selected:
                item = {**group, "params": selected}
                if no_decay:
                    item["weight_decay"] = 0.0
                result.append(item)
    return result


def architecture_metadata(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    return {
        "version": getattr(model, "backbone_version", getattr(model, "adapter_version", None)),
        "blocks": {name: module._mamba_config for name, module in model.named_modules()
                   if hasattr(module, "_mamba_config")},
    }


def validate_checkpoint_architecture(checkpoint, model, kind, *, backbone=None):
    """Validate new metadata, and reject legacy block-family mismatches.

    Metadata is additive; historical checkpoints without it remain supported.
    In particular, never allow strict=False to silently discard another SSM.
    """
    expected = architecture_metadata(model)
    saved = checkpoint.get(f"{kind}_architecture")
    if saved is not None and saved != expected:
        raise ValueError(f"Checkpoint {kind} architecture does not match current configuration: "
                         f"checkpoint={saved}, current={expected}")
    if kind == "adapter" and backbone is not None:
        saved_backbone = checkpoint.get("backbone_architecture")
        if saved_backbone is not None and saved_backbone != architecture_metadata(backbone):
            raise ValueError("Adapter checkpoint was trained with a different backbone architecture")
    state = checkpoint.get("model_state" if kind == "backbone" else "model_state_dict", {})
    state = {k.removeprefix("module."): v for k, v in state.items()}
    for name, config in expected["blocks"].items():
        prefix = name + "."
        keys = [k[len(prefix):] for k in state if k.startswith(prefix)]
        if not keys:
            raise ValueError(f"Checkpoint is missing {kind} Mamba block {name!r}")
        inferred = ("mamba3" if "B_bias" in keys else
                    "mamba2" if "lin_B.weight" in keys else "mamba1")
        if config["version"] != inferred:
            raise ValueError(f"Checkpoint {kind} block {name!r} is {inferred}, "
                             f"configured as {config['version']}")
    # Catch the reverse case too: a zero-layer/attention head must not silently
    # discard saved Mamba layers under the historical non-strict loader.
    markers = (".B_bias", ".lin_B.weight", ".A_log")
    actual_blocks = {k[:-len(marker)] for k in state for marker in markers if k.endswith(marker)}
    if any(name not in expected["blocks"] for name in actual_blocks):
        raise ValueError(f"Checkpoint contains Mamba blocks absent from the configured {kind}")
