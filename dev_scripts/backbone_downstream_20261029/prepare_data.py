"""Build immutable, event-disjoint downstream cohorts and train-only statistics."""
import hashlib
import json
from pathlib import Path
import numpy as np
from mmap_ninja import RaggedMmap

DATA = Path('/home/alessio/ML-work/HIPO_processing/multi_particle_storage/big-data/mmap_canonical_loose_truthseg_event_v6_02')
OUT = Path('/results')
SIZES = {'finding': (50000, 5000, 10000), 'momentum': (50000, 10000, 50000)}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def select(task, role, count):
    split = 'pretrain' if role == 'train' else 'test'
    start, stop = (0, None) if role == 'train' else ((10000, 100000) if role == 'validation' else (100000, None))
    features = RaggedMmap(str(DATA / ('features_' + split)))
    labels = RaggedMmap(str(DATA / ('seg_target_' + split)))
    regression = RaggedMmap(str(DATA / ('reg_target_' + split)))
    assert len(features) == len(labels) == len(regression)
    keys, lengths, targets = [], [], []
    digest = hashlib.sha256()
    for event in range(start, min(stop or len(features), len(features))):
        x, y, r = np.asarray(features[event]), np.asarray(labels[event]), np.asarray(regression[event])
        assert x.ndim == 2 and x.shape[1] == 3 and y.shape[0] == r.shape[0] == x.shape[0], event
        if task == 'finding':
            signal = np.unique(y[y >= 0])
            if not (1 <= len(x) <= 512) or len(signal) > 16 or len(signal) == 0:
                continue
            accepted = [(event, len(x), None)]
        else:
            accepted = []
            for label in sorted(int(v) for v in np.unique(y) if v >= 0):
                mask = y == label
                length = int(mask.sum())
                if not 12 <= length <= 100:
                    continue
                values = np.asarray(r[mask, :3], dtype=np.float64)
                # Never substitute, skip or repair an invalid accepted truth target.
                assert np.isfinite(values).all(), (event, label)
                accepted.append(((event, label), length, values.mean(axis=0)))
        if accepted:
            for array in (x, y, r):
                digest.update(str((array.shape, array.dtype.str)).encode())
                digest.update(np.ascontiguousarray(array).tobytes())
        for key, length, target in accepted:
            keys.append(key); lengths.append(length)
            if target is not None: targets.append(target)
            if len(keys) == count: break
        if len(keys) == count: break
    assert len(keys) == count, (task, role, len(keys), count)
    keys, lengths = np.asarray(keys, dtype=np.int64), np.asarray(lengths, dtype=np.int64)
    events = keys if keys.ndim == 1 else keys[:, 0]
    np.savez(OUT / f'{task}_{role}.npz', keys=keys, lengths=lengths)
    result = {'samples': len(keys), 'source_split': split, 'first_source_event': int(events.min()),
              'last_source_event': int(events.max()), 'source_events': len(np.unique(events)),
              'keys_sha256': hashlib.sha256(keys.tobytes()).hexdigest(),
              'lengths_sha256': hashlib.sha256(lengths.tobytes()).hexdigest(),
              'selected_source_content_sha256': digest.hexdigest()}
    print(task, role, result, flush=True)
    return result, np.asarray(targets)


def main():
    report = {'data_root': str(DATA), 'cohorts': {},
              'split_contract': 'Existing pretrain source split for training; test event IDs [10000,100000) for validation, [100000,infinity) for evaluation. No source event shared between downstream validation/evaluation. Early test events reserved from downstream use.'}
    for task, sizes in SIZES.items():
        report['cohorts'][task] = {}
        for role, count in zip(('train', 'validation', 'evaluation'), sizes):
            result, targets = select(task, role, count)
            report['cohorts'][task][role] = result
            if task == 'momentum' and role == 'train':
                std = targets.std(axis=0)
                assert np.isfinite(std).all() and (std > 0).all()
                write(OUT / 'momentum_stats.json', {'version': 2, 'task': 'mom', 'split': 'pretrain',
                      'columns': ['mc_entrance_px', 'mc_entrance_py', 'mc_entrance_pz'],
                      'mean': targets.mean(axis=0).tolist(), 'std': std.tolist(),
                      'count': [len(targets)] * 3, 'selected_events': len(targets),
                      'cohort_keys_sha256': result['keys_sha256'], 'data_root': str(DATA)})
        assert report['cohorts'][task]['validation']['last_source_event'] < 100000
        assert report['cohorts'][task]['evaluation']['first_source_event'] >= 100000
    report['metadata_sha256'] = hashlib.sha256((DATA / 'metadata.json').read_bytes()).hexdigest()
    write(OUT / 'data_ready.json', report)


if __name__ == '__main__': main()
