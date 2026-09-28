"""Typed YAML values and recursive campaign configuration overrides."""
from collections.abc import Mapping
from copy import deepcopy
from ruamel.yaml import YAML


def deep_merge(base, overrides):
    result = deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def parse_value(value):
    if value.lower() == 'none':
        return None
    try:
        return YAML(typ='safe').load(value)
    except Exception as exc:
        raise ValueError(f'Invalid YAML override value: {value!r}') from exc


def assign_override(overrides, key, value):
    """Apply in command order; mappings merge, lists/scalars replace.

    Traversing a scalar is rejected rather than creating an ignored literal key.
    """
    parts = key.split('.')
    if any(not part or part.strip() != part for part in parts):
        raise ValueError(f'Invalid dotted override key: {key!r}')
    target = overrides
    for part in parts[:-1]:
        target.setdefault(part, {})
        if not isinstance(target[part], dict):
            raise ValueError(f'Cannot traverse non-mapping override {part!r} in {key!r}')
        target = target[part]
    leaf = parts[-1]
    if isinstance(value, Mapping) and isinstance(target.get(leaf), Mapping):
        target[leaf] = deep_merge(target[leaf], value)
    else:
        target[leaf] = deepcopy(value)
