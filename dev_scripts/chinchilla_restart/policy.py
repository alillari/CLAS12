"""Historical rank-based width transfer, with native SSM decay exemptions."""


def parameter_groups(model, weight_decay):
    groups = []
    for matrix in (False, True):
        for no_decay in (False, True):
            named = [(name, p) for name, p in model.named_parameters()
                     if p.requires_grad and (p.ndim > 1) == matrix
                     and bool(getattr(p, '_no_weight_decay', False)) == no_decay]
            if named:
                groups.append({'params': [p for _, p in named], 'matrix': matrix,
                               'weight_decay': 0.0 if no_decay else weight_decay})
    return groups


def scale_matrix_rates(optimizer, ratio):
    # Called once AFTER scheduler.step(), exactly as in the historical trainer.
    for group in optimizer.param_groups:
        if group['matrix']:
            group['lr'] /= ratio


def group_audit(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [{'matrix': g['matrix'], 'weight_decay': g['weight_decay'],
             'lr': g['lr'], 'parameter_names': [names[id(p)] for p in g['params']],
             'parameter_count': sum(p.numel() for p in g['params'])}
            for g in optimizer.param_groups]
