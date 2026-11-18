"""Explicit, checkpointed numerical controls for regression feature encoding."""
from numbers import Integral

import torch
import torch.nn.functional as F

from fm4npp.models.mamba_config import value


def regression_encoding_config(params):
    config = {}
    for name in ('sequence_padding', 'batch_padding'):
        item = value(params, 'backbone_' + name, 0)
        if isinstance(item, bool) or not isinstance(item, Integral) or item < 0:
            raise ValueError(f'backbone_{name} must be a nonnegative integer')
        config[name] = int(item)
    precision = value(params, 'regression_matmul_precision', 'default')
    if precision not in {'default', 'highest'}:
        raise ValueError("regression_matmul_precision must be 'default' or 'highest'")
    config['matmul_precision'] = precision
    feature_kind = value(params, 'backbone_feature_kind', 'accumulated_states')
    if feature_kind not in {'accumulated_states', 'residual_updates'}:
        raise ValueError("backbone_feature_kind must be 'accumulated_states' or 'residual_updates'")
    config['feature_kind'] = feature_kind
    return config


def configure_regression_precision(params):
    """Leave global defaults untouched unless the caller explicitly opts in."""
    if regression_encoding_config(params)['matmul_precision'] == 'highest':
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction'):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def _capture_features(model, points, feature_kind):
    if feature_kind == 'accumulated_states':
        return torch.stack(model(points, return_z=True)[1])
    backbone = getattr(model, 'module', model)
    layers = getattr(backbone, 'mamba_layers', None)
    if layers is None or not len(layers):
        raise ValueError('Residual-update features require a backbone exposing mamba_layers')
    updates = []
    handles = []
    try:
        for layer in layers:
            handles.append(layer.register_forward_hook(
                lambda module, inputs, output: updates.append(output)))
        model(points, return_z=True)
    finally:
        for handle in handles:
            handle.remove()
    if len(updates) != len(layers):
        raise ValueError('Expected one residual update per backbone layer')
    # Capture before addition: subtracting rounded states can erase small updates.
    return torch.stack(updates)


def regression_backbone_features(model, points, params):
    """Return [layer,batch,token,channel] features in the original sample order.

    Canonical batches split larger evaluation batches and pad shorter ones.
    Padding uses the input sentinel and is removed before the adapter sees it.
    Callers own eval/no_grad mode, as with an ordinary backbone forward.
    """
    config = regression_encoding_config(params)
    size, length = points.shape[:2]
    if not size or not length:
        raise ValueError('Backbone encoding requires a nonempty batch and sequence')
    sequence_size = config['sequence_padding'] or length
    if length > sequence_size:
        raise ValueError(f'Input sequence length {length} exceeds backbone_sequence_padding={sequence_size}')
    batch_size = config['batch_padding'] or size
    features = []
    for begin in range(0, size, batch_size):
        part = points[begin:begin+batch_size]
        count = part.shape[0]
        if sequence_size != length or batch_size != count:
            part = F.pad(part, (0, 0, 0, sequence_size-length, 0, batch_size-count), value=-100.)
        hidden = _capture_features(model, part, config['feature_kind'])
        features.append(hidden[:, :count, :length])
    return features[0] if len(features) == 1 else torch.cat(features, dim=1)


def validate_regression_encoding(checkpoint, params):
    saved = checkpoint.get('backbone_encoding', regression_encoding_config({}))
    if isinstance(saved, dict):
        # Existing regression checkpoints used accumulated states before the
        # extraction policy became explicit metadata.
        saved = {'feature_kind': 'accumulated_states', **saved}
    current = regression_encoding_config(params)
    if saved != current:
        raise ValueError(f'Adapter checkpoint backbone encoding mismatch: saved {saved}, requested {current}')
