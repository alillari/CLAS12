"""Validation sample controls independent of the training label budget."""
DEFAULT_VALIDATION_SAMPLES = 50000
DEFAULT_VALIDATION_BATCH_SIZE = 128


def configure_validation(params):
    """Resolve in place before constructing loaders; keep the existing test root.

    Explicit max_val_batches must not silently shrink the requested sample.
    The dataset can contain fewer accepted tracks; evaluation reports actual counts.
    """
    params.setdefault('limit_test_data', True)
    params.setdefault('limit_test_size', DEFAULT_VALIDATION_SAMPLES)
    params.setdefault('valid_batch_size', DEFAULT_VALIDATION_BATCH_SIZE)
    params.setdefault('max_val_batches', None)
    for key in ('limit_test_size', 'valid_batch_size'):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not isinstance(params['limit_test_data'], bool):
        raise ValueError('limit_test_data must be true or false')
    cap = params['max_val_batches']
    if cap is not None:
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError('max_val_batches must be null or a positive integer')
        if not params['limit_test_data'] or cap * params['valid_batch_size'] < params['limit_test_size']:
            raise ValueError('max_val_batches would truncate the fixed validation sample. '
                             'Increase/remove it or explicitly reduce limit_test_size.')
    params['drop_last_test'] = False
    return params
