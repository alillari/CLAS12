"""Portable pretrained-backbone selection shared by training and evaluation.

Catalogs describe locations and missing historical input settings, never adapter
recipes. Checkpoint metadata/tensor dimensions remain authoritative. Reading a
catalog does not import Mamba kernels or require CUDA.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re

from ruamel.yaml import YAML


DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "scripts/configs/pretrained_backbones.yaml"
BLOCK_KEYS = ("d_state", "d_conv", "expand", "headdim", "ngroups", "chunk_size",
              "use_mem_eff_path", "rope_fraction", "is_mimo", "is_outproj_norm")
MODEL_KEYS = ("backbone_version", "embed_dim", "num_layers_backbone", "klen",
              "backbone_output_dim", "embed_method", "pe_method", "pos_dim", "aux_dim",
              "dropout", "num_heads_backbone", "backbone_sequence_encoding",
              "backbone_rope_theta", "backbone_mlp_ratio", "mlp_ratio", "window_size",
              "seq_len", "proj_dim", *BLOCK_KEYS)
INPUT_KEYS = ("input_layout", "input_representation", "serialization", "clas12_band_config",
              "hilbert_angular_grid", "use_sincos_target", "voxelize", "space_filling_order",
              "space_filling_curve", "order", "rep_aaai", "nexttoken", "aux_extra_feature")
CONFIG_KEYS = frozenset((*MODEL_KEYS, *INPUT_KEYS))
FAMILY_BLOCK_KEYS = {
    "mamba1": {"d_state", "d_conv", "expand"},
    "mamba2": {"d_state", "d_conv", "expand", "headdim", "ngroups", "chunk_size", "use_mem_eff_path"},
    "mamba3": {"d_state", "expand", "headdim", "ngroups", "chunk_size", "rope_fraction", "is_mimo", "is_outproj_norm"},
}


def _mapping(params):
    if isinstance(params, Mapping):
        return params
    return {**getattr(params, "params", {}), **{key: val for key, val in vars(params).items()
                                                if key != "params" and not key.startswith("_")}}


def _update(params, values):
    if isinstance(params, dict):
        params.update(values)
    elif hasattr(params, "update_params"):
        params.update_params(values)
    else:
        for key, value in values.items():
            setattr(params, key, value)


def _clear_unused_block_options(params, config):
    allowed = FAMILY_BLOCK_KEYS.get(config["backbone_version"])
    if allowed is None:
        return
    for name in set(BLOCK_KEYS) - allowed:
        for key in (name, f"backbone_{name}"):
            if isinstance(params, dict):
                params.pop(key, None)
            else:
                getattr(params, "params", {}).pop(key, None)
                if hasattr(params, key):
                    delattr(params, key)


def _config(values):
    result = dict(values or {})
    if "mambaversion" in result:
        result.setdefault("backbone_version", result.pop("mambaversion"))
    for key in BLOCK_KEYS:
        if f"backbone_{key}" in result:
            result[key] = result.pop(f"backbone_{key}")
    unknown = result.keys() - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown backbone config fields: {sorted(unknown)}; keep task/adapter settings in the recipe")
    return result


def _merge_contract(left, right, source):
    conflicts = {key: (left[key], val) for key, val in right.items()
                 if key in left and left[key] != val}
    if conflicts:
        raise ValueError(f"Conflicting backbone settings in {source}: {conflicts}")
    return {**left, **right}


def _path(value, base):
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in {value!r}; set it or pass --backbone-root")
    path = Path(expanded)
    return (path if path.is_absolute() else Path(base) / path).resolve()


def read_catalog(path=None):
    path = _path(path or DEFAULT_CATALOG, Path.cwd())
    with path.open() as stream:
        data = YAML(typ="safe").load(stream)
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("backbones"), dict):
        raise ValueError(f"{path}: expected version: 1 and a backbones mapping")
    for name in data["backbones"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(name)):
            raise ValueError(f"Unsafe backbone identifier: {name!r}")
    return path, data


def catalog_selections(catalog=None, root=None, names=None):
    path, data = read_catalog(catalog)
    names = list(data["backbones"]) if not names else list(dict.fromkeys(names))
    missing = set(names) - data["backbones"].keys()
    if missing:
        raise KeyError(f"Unknown backbone(s) in {path}: {sorted(missing)}")
    external_root = root or os.environ.get("CLAS12_PRETRAINED_ROOT")
    base = _path(external_root, Path.cwd()) if external_root else _path(data.get("root", "."), path.parent)
    defaults = _config(data.get("defaults", {}))
    selections = []
    for name in names:
        entry = dict(data["backbones"][name])
        selections.append({"name": name, "checkpoint": str(_path(entry["checkpoint"], base)),
                           "config": {**defaults, **_config(entry.get("config", {}))},
                           "sha256": entry.get("sha256"), "catalog": str(path)})
    return selections


def checkpoint_config(checkpoint):
    """Read known metadata and unambiguous dimensions; never guess token meaning."""
    result = _config(checkpoint.get("backbone_config", {}))
    architecture = checkpoint.get("backbone_architecture") or {}
    result = _merge_contract(result, _config(architecture.get("input_embedding", {})), "checkpoint embedding")
    blocks = list(architecture.get("blocks", {}).values())
    inferred = {}
    if architecture.get("version"):
        inferred["backbone_version"] = architecture["version"]
    if blocks:
        first = blocks[0]
        if any(block != first for block in blocks):
            raise ValueError("Nonuniform backbone blocks need a matching model factory")
        inferred.update(embed_dim=first["d_model"], num_layers_backbone=len(blocks))
        inferred.update({key: first[key] for key in BLOCK_KEYS if key in first})
        for old, new in (("num_heads", "num_heads_backbone"), ("sequence_encoding", "backbone_sequence_encoding"),
                         ("rope_theta", "backbone_rope_theta"), ("mlp_ratio", "backbone_mlp_ratio")):
            if old in first:
                inferred[new] = first[old]
        if first.get("version") == "transformer" and "dropout" in first:
            inferred["dropout"] = first["dropout"]
    state = checkpoint.get("model_state", {})
    state = {key.removeprefix("module."): val for key, val in state.items()}
    head = state.get("output_layer.weight")
    if head is not None:
        inferred = _merge_contract(inferred, {"embed_dim": int(head.shape[1]),
                                            "backbone_output_dim": int(head.shape[0])}, "checkpoint tensors")
    if "hilbert_angular_grid" in checkpoint:
        inferred["hilbert_angular_grid"] = checkpoint["hilbert_angular_grid"]
    return _merge_contract(result, inferred, "checkpoint metadata")


def load_backbone_checkpoint(path, *, map_location="cpu", expected_sha256=None):
    """Hash and deserialize the same file; reject changed or substituted weights."""
    import torch
    path = Path(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        sha256 = digest.hexdigest()
        if expected_sha256 and expected_sha256 != sha256:
            raise ValueError(f"Backbone SHA-256 mismatch: {path}")
        stream.seek(0)
        # Same trusted-checkpoint format as the historical trainer loaders.
        checkpoint = torch.load(stream, map_location=map_location, weights_only=False)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Backbone checkpoint changed while being read: {path}")
    return checkpoint, sha256


def load_pretrained_checkpoint(path, params, *, map_location="cpu"):
    selection = _mapping(params).get("pretrained_backbone") if params is not None else None
    expected = selection.get("sha256") if isinstance(selection, Mapping) else None
    return load_backbone_checkpoint(path, map_location=map_location, expected_sha256=expected)


def resolve_backbone(selection, *, catalog=None, root=None, fallback=None):
    """Return a JSON-serializable, verified descriptor for a catalog ID or mapping."""
    if isinstance(selection, str):
        descriptor = catalog_selections(catalog, root, [selection])[0]
    else:
        descriptor = dict(selection)
    path = _path(descriptor["checkpoint"], Path.cwd())
    checkpoint, sha256 = load_backbone_checkpoint(path, expected_sha256=descriptor.get("sha256"))
    if not isinstance(checkpoint, Mapping) or "model_state" not in checkpoint:
        raise ValueError(f"{path}: expected a pretraining checkpoint with model_state")
    saved = checkpoint_config(checkpoint)
    known = _merge_contract(saved, _config(descriptor.get("config", {})), "catalog/descriptor")
    config = {**_config(fallback or {}), **known}
    allowed = FAMILY_BLOCK_KEYS.get(config.get("backbone_version"))
    if allowed is not None:
        for key in set(BLOCK_KEYS) - allowed:
            if key in known and key != "d_conv":
                raise ValueError(f"Unsupported {config['backbone_version']} block option: {key}")
            config.pop(key, None)
    required = {"backbone_version", "embed_dim", "num_layers_backbone", "embed_method", "pe_method", "klen"}
    missing = {key for key in required if config.get(key) is None}
    if missing:
        raise ValueError(f"{path}: missing backbone settings {sorted(missing)}; provide config in a backbone catalog")
    for key in ("embed_dim", "num_layers_backbone", "klen"):
        if isinstance(config[key], bool) or int(config[key]) != config[key] or int(config[key]) <= 0:
            raise ValueError(f"Backbone {key} must be a positive integer")
        config[key] = int(config[key])
    if "use_sincos_target" in config and not isinstance(config["use_sincos_target"], bool):
        raise ValueError("use_sincos_target must be a boolean")
    from fm4npp.models.mamba_config import backbone_version
    backbone_version(config)
    if "backbone_output_dim" in config:
        expected = int(config["klen"]) * (4 if config.get("use_sincos_target", False) else 3)
        if config["backbone_output_dim"] != expected:
            raise ValueError(f"{path}: output width {config['backbone_output_dim']} disagrees with klen/use_sincos_target ({expected}); "
                             "declare the pretraining target settings in the backbone catalog")
    return {"name": descriptor.get("name", path.parent.name), "checkpoint": str(path),
            "config": config, "sha256": sha256,
            **({"catalog": descriptor["catalog"]} if descriptor.get("catalog") else {})}


def configure_pretrained_backbone(params, *, enabled=None, checkpoint=None, backbone=None, catalog=None, root=None):
    """Resolve before datasets/models; checkpoint config replaces recipe defaults.

    Explicit AdapterOnly mode clears selection. Legacy checkpoint paths retain
    recipe values where historical checkpoints carry no metadata.
    """
    values = _mapping(params)
    selection = values.get("pretrained_backbone")
    path = values.get("pretrained_ckpt")
    if backbone is not None:
        selection, path = backbone, checkpoint
    elif checkpoint is not None:
        path = checkpoint
        if not isinstance(selection, Mapping) or _path(selection["checkpoint"], Path.cwd()) != _path(path, Path.cwd()):
            selection = None
    if enabled is False or (enabled is None and not selection and not path):
        _update(params, {"pretrained_ckpt": None, "pretrained_backbone": None})
        return None
    if not selection and not path:
        raise ValueError("Pretrained mode requires --backbone, --pretrained_ckpt, or pretrained_backbone in YAML")
    fallback = {key: val for key, val in values.items() if key in CONFIG_KEYS or key == "mambaversion" or
                key in {f"backbone_{name}" for name in BLOCK_KEYS}}
    selection = selection or {"checkpoint": path}
    catalog_path = catalog or values.get("backbone_catalog")
    if catalog is None and catalog_path and hasattr(params, "_yaml_filename"):
        catalog_path = _path(catalog_path, Path(params._yaml_filename).parent)
    resolved = resolve_backbone(selection, catalog=catalog_path,
                                root=root or values.get("backbone_root"), fallback=fallback)
    if path and _path(path, Path.cwd()) != Path(resolved["checkpoint"]):
        raise ValueError("pretrained_ckpt and pretrained_backbone select different checkpoints")
    config = resolved["config"]
    _clear_unused_block_options(params, config)
    updates = {**config, "base_dim": config["embed_dim"], "mambaversion": config["backbone_version"],
               "pretrained_ckpt": resolved["checkpoint"], "pretrained_backbone": resolved}
    # Prefix aliases take precedence in the factory; replace stale recipe aliases.
    updates.update({f"backbone_{key}": config[key] for key in BLOCK_KEYS if key in config and f"backbone_{key}" in values})
    _update(params, updates)
    return resolved


def apply_backbone_to_run(params, run, *overrides):
    """Render a resolved campaign row without permitting conflicting overrides."""
    resolved = run.get("pretrained_backbone")
    if not resolved or not run.get("use_pretrained_backbone", True):
        return
    config = resolved["config"]
    for override in overrides:
        relevant = {key: val for key, val in override.items() if key in CONFIG_KEYS or key == "mambaversion" or
                    key in {f"backbone_{name}" for name in BLOCK_KEYS}}
        _merge_contract(config, _config(relevant), "campaign training_overrides")
        if "base_dim" in override and override["base_dim"] != config["embed_dim"]:
            raise ValueError("Campaign base_dim override conflicts with pretrained embedding width")
    updates = {**config, "base_dim": config["embed_dim"], "mambaversion": config["backbone_version"],
               "pretrained_ckpt": resolved["checkpoint"], "pretrained_backbone": resolved}
    updates.update({f"backbone_{key}": config[key] for key in BLOCK_KEYS if key in config and f"backbone_{key}" in params})
    _clear_unused_block_options(params, config)
    params.update(updates)


def export_backbone_config(params, model):
    """Persist the factory's effective architecture and only its input contract."""
    model = getattr(model, "module", model)
    result = dict(model._backbone_config)
    values = _mapping(params)
    result.update({key: values[key] for key in INPUT_KEYS if key in values})
    return result


def add_backbone_arguments(parser, *, multiple=False):
    parser.add_argument("--backbone", action="append" if multiple else "store",
                        help="Backbone catalog ID (repeat to select several campaign models); implies pretrained mode.")
    parser.add_argument("--backbone-catalog", help="Portable backbone catalog YAML (defaults to the bundled catalog).")
    parser.add_argument("--backbone-root", help="Checkpoint root; overrides CLAS12_PRETRAINED_ROOT/catalog root.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backbone_arguments(parser)
    parser.add_argument("--list", action="store_true", help="List IDs without opening checkpoint files.")
    args = parser.parse_args()
    if args.list:
        path, data = read_catalog(args.backbone_catalog)
        print(json.dumps({"catalog": str(path), "backbones": list(data["backbones"])}, indent=2))
    elif args.backbone:
        print(json.dumps(resolve_backbone(args.backbone, catalog=args.backbone_catalog, root=args.backbone_root), indent=2))
    else:
        parser.error("specify --list or --backbone ID")


if __name__ == "__main__":
    main()
