"""Resolve and record the momentum head's dropout independently of its backbone."""

import torch

from fm4npp.models.mamba_config import value


def regression_dropout(params):
    return float(value(params, 'downstream_dropout', value(params, 'dropout', 0.0)))


def regression_dropout_config(model):
    return {name.removeprefix('module.'): module.p for name, module in model.named_modules()
            if isinstance(module, torch.nn.Dropout)}


def validate_regression_dropout_resume(checkpoint, model, params):
    """Reject a regularization change before loading training state.

    Inference does not call this check: dropout is disabled by model.eval().
    Architectural checks, including Transformer block dropout, remain separate.
    """
    saved = checkpoint.get('adapter_dropout_config')
    if saved is not None:
        if saved != regression_dropout_config(model):
            raise ValueError('Adapter dropout mismatch when resuming training')
        return
    old = checkpoint.get('params', {})
    old = {**old.get('params', {}), **old}
    # Historical regression used backbone dropout for its output MLP, even if
    # downstream_dropout was explicitly present in the YAML.
    if 'dropout' in old and regression_dropout(params) != float(old['dropout']):
        raise ValueError('Legacy adapter dropout mismatch: resume with downstream_dropout '
                         'equal to the saved dropout, or start a new training run')
