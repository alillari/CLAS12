"""Replay only sampler indices and event lengths, without loading event payloads."""

import numpy as np
import torch


def replay(lengths, batch_size, steps, seed=42):
    lengths = np.asarray(lengths)
    batches = len(lengths) // batch_size
    if not batches:
        raise ValueError('No complete training batch')
    generator = torch.Generator().manual_seed(seed)
    # Persistent-worker DataLoader draws its base seed once, before RandomSampler.
    torch.empty((), dtype=torch.int64).random_(generator=generator).item()
    padded, valid = [], []
    remaining = steps
    while remaining:
        order = torch.randperm(len(lengths), generator=generator).numpy()
        take = min(remaining, batches)
        values = lengths[order[:take * batch_size]].reshape(take, batch_size)
        padded.append(values.max(1).astype(np.int64) * batch_size)
        valid.append(values.sum(1).astype(np.int64))
        remaining -= take
        if remaining:
            # RandomSampler also draws a final permutation when remainder==0.
            torch.randperm(len(lengths), generator=generator)
    return np.concatenate(padded), np.concatenate(valid)
