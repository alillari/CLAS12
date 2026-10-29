"""Auditable algorithmic compute estimates; not hardware instruction counts.

MAC=2 FLOPs. The dense-only estimate includes every Linear projection and the
frozen NeRF projection. The second estimate adds causal convolutions and a
canonical recurrent-arithmetic model (Mamba1 Euler, Mamba2 scalar-A Euler,
Mamba3 two-input trapezoidal update). Forward+backward is estimated as 3x
trainable forward arithmetic. Nonlinearities, normalization, optimizer work,
parallel-scan overhead, chunk/tile padding and kernel recomputation are excluded.
Both estimates are reported, so a ranking need not rest on one SSM convention.
"""


def describe(model, version):
    import torch

    dense_forward = 0
    projections = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            cost = 2 * module.weight.numel()
            dense_forward += cost
            projections.append({'name': name, 'shape': list(module.weight.shape), 'forward_flops_per_token': cost})
    frozen_projection = model.embedder.embed.projection
    assert not frozen_projection.requires_grad
    frozen_forward = 2 * frozen_projection.numel()
    conv_forward = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            conv_forward += 2 * module.weight.numel()
    ssm_forward = 0
    blocks = []
    for module in model.modules():
        if not hasattr(module, '_mamba_config'):
            continue
        config = module._mamba_config
        inner = config['d_model'] * config['expand']
        state = config['d_state']
        if version == 'mamba1':
            # delta*A + delta*B*x + state update + C contraction; skip/gate.
            cost = 7 * inner * state + 3 * inner
        elif version == 'mamba2':
            heads = inner // config['headdim']
            cost = 5 * inner * state + heads * state + heads + 3 * inner
        elif version == 'mamba3':
            heads = inner // config['headdim']
            cost = 8 * inner * state + 6 * heads * state * config['rope_fraction'] + 3 * inner
        else:
            raise ValueError(version)
        blocks.append({'config': config, 'canonical_ssm_forward_flops_per_token': cost})
        ssm_forward += cost
    dense_training = 3 * dense_forward + frozen_forward
    estimated_training = dense_training + 3 * (conv_forward + ssm_forward)
    return {'version': version, 'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'dense_trainable_forward_flops_per_token': dense_forward,
            'frozen_embedding_forward_flops_per_token': frozen_forward,
            'conv_forward_flops_per_token': conv_forward,
            'canonical_ssm_forward_flops_per_token': ssm_forward,
            'dense_training_flops_per_padded_token': dense_training,
            'estimated_training_flops_per_padded_token': estimated_training,
            'projections': projections, 'blocks': blocks,
            'convention': 'MAC=2; trainable forward+backward approx 3x forward; frozen embedding forward only',
            'excluded': ['nonlinearities and normalization', 'optimizer/audit/data/validation work',
                         'parallel-scan overhead', 'kernel tile/chunk padding', 'backward recomputation'],
            'status': 'algorithmic estimate, not measured total executed GPU FLOPs'}
