"""Revalidate the retained source and build a fresh 1k-label experiment bundle."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmap_ninja import RaggedMmap

HERE = Path(__file__).resolve().parent
OUT = Path('/results')
DATA = Path('/data')
PRIOR = Path('/prior_cohorts')


def read(p): return json.loads(Path(p).read_text())
def write(p, d): Path(p).write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def audit_cohort(role, expected):
    with np.load(PRIOR / f'momentum_{role}.npz') as f:
        keys, lengths = f['keys'], f['lengths']
    assert len(keys) == expected['samples']
    assert hashlib.sha256(keys.tobytes()).hexdigest() == expected['keys_sha256']
    assert hashlib.sha256(lengths.tobytes()).hexdigest() == expected['lengths_sha256']
    assert len(np.unique(keys, axis=0)) == len(keys)
    split = 'pretrain' if role == 'train' else 'test'
    arrays = [RaggedMmap(str(DATA / (kind + '_' + split)))
              for kind in ('features', 'seg_target', 'reg_target')]
    digest = hashlib.sha256()
    seen = set(); current = None
    for (event, label), length in zip(keys, lengths):
        event, label = int(event), int(label)
        if event != current:
            x, y, r = [np.asarray(a[event]) for a in arrays]
            current = event
        assert x.ndim == 2 and x.shape[1] == 3 and len(x) == len(y) == len(r)
        if event not in seen:
            for a in (x, y, r):
                digest.update(str((a.shape, a.dtype.str)).encode())
                digest.update(np.ascontiguousarray(a).tobytes())
            seen.add(event)
        mask = y == label
        assert mask.sum() == length and 12 <= length <= 100
        assert np.isfinite(r[mask, :3]).all()
    assert digest.hexdigest() == expected['selected_source_content_sha256'], role
    return keys, lengths


def main():
    reference = read(HERE / 'reference.json')
    prior_ready = read(PRIOR / 'data_ready.json')
    assert file_sha(DATA / 'metadata.json') == prior_ready['metadata_sha256']
    source_changes = []
    primary_differences = []
    for name, original in reference['source_manifest'].items():
        important = name.startswith(('fm4npp/', 'train/', 'scripts/configs/')) or name in (
            'dev_scripts/backbone_downstream_20261029/run.py',
            'dev_scripts/backbone_pretraining_comparison/recipe.yaml')
        if not important: continue
        local = ROOT / name
        if not local.exists() or file_sha(local) != original['sha256']: source_changes.append(name)
        primary = Path('/primary') / name
        if not primary.exists() or file_sha(primary) != file_sha(local): primary_differences.append(name)
    assert not source_changes, source_changes
    roles = {role: audit_cohort(role, prior_ready['cohorts']['momentum'][role])
             for role in ('train', 'validation', 'evaluation')}
    selected = np.sort(np.random.default_rng(42).choice(len(roles['train'][0]), size=1000, replace=False))
    keys, lengths = (a[selected] for a in roles['train'])
    np.savez(OUT / 'momentum_train.npz', keys=keys, lengths=lengths)
    np.savez(OUT / 'subset_indices.npz', indices=selected)
    for role in ('validation', 'evaluation'):
        shutil.copyfile(PRIOR / f'momentum_{role}.npz', OUT / f'momentum_{role}.npz')
    assert not set(roles['validation'][0][:, 0]) & set(roles['evaluation'][0][:, 0])
    labels = RaggedMmap(str(DATA / 'seg_target_pretrain'))
    regression = RaggedMmap(str(DATA / 'reg_target_pretrain'))
    targets = []
    for event, label in keys:
        y, r = np.asarray(labels[int(event)]), np.asarray(regression[int(event)])
        values = np.asarray(r[y == label, :3], dtype=np.float64)
        assert len(values) >= 12 and np.isfinite(values).all()
        targets.append(values.mean(axis=0))
    targets = np.asarray(targets)
    mean, std = targets.mean(axis=0), targets.std(axis=0)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
    train_sha = hashlib.sha256(keys.tobytes()).hexdigest()
    stats = {'version': 2, 'task': 'mom', 'split': 'pretrain',
        'columns': ['mc_entrance_px', 'mc_entrance_py', 'mc_entrance_pz'],
        'mean': mean.tolist(), 'std': std.tolist(), 'count': [1000] * 3,
        'selected_tracks': 1000, 'selected_events': len(np.unique(keys[:, 0])),
        'cohort_keys_sha256': train_sha, 'data_root': reference['data_path'],
        'statistics_scope': 'Only the selected 1000 training tracks; no 50k-label target statistics reused'}
    write(OUT / 'momentum_stats.json', stats)
    state = torch.load('/old_initial/initial_head.pth', map_location='cpu', weights_only=True)
    assert state_sha(state) == reference['one_layer_initial_sha256']
    normalizer = {'target_normalizer.mean': torch.as_tensor(mean, dtype=torch.float32),
        'target_normalizer.std': torch.as_tensor(std, dtype=torch.float32),
        'target_normalizer.constant': torch.zeros(3, dtype=torch.bool)}
    assert set(normalizer) == {k for k in state if k.startswith('target_normalizer.')}
    initial = {}
    for layers in (0, 1):
        arm = {k: normalizer.get(k, v).clone() for k, v in state.items()
               if layers or not k.startswith('mamba_layers.')}
        for k, v in arm.items():
            if not k.startswith('target_normalizer.'): assert torch.equal(v, state[k])
        torch.save(arm, OUT / f'initial_head_{layers}.pth')
        initial[str(layers)] = {'initial_head_sha256': state_sha(arm),
            'shared_initial_head_sha256': state_sha({k: v for k, v in arm.items() if not k.startswith('mamba_layers.')}),
            'normalization_replaced_from_1k_only': True, 'all_other_initial_weights_unchanged': True}
    checkpoints = {}
    for version in ('mamba1', 'mamba2', 'mamba3'):
        if version == 'mamba2':
            checkpoints[version] = {'verification': 'Deferred to pre-pilot gate after the checkpoint relay completes'}
            continue
        path = Path(f'/pretrained/{version}/training/{version}_chinchilla_w512_d12_k1/seed42/training_checkpoints/ckpt_best.tar')
        checkpoints[version] = {'sha256': file_sha(path), 'bytes': path.stat().st_size}
    ready = {'complete': True, 'training_tracks': 1000, 'training_source_events': stats['selected_events'],
        'train_keys_sha256': train_sha, 'validation_keys_sha256': prior_ready['cohorts']['momentum']['validation']['keys_sha256'],
        'evaluation_keys_sha256': prior_ready['cohorts']['momentum']['evaluation']['keys_sha256'],
        'validation_tracks': 10000, 'evaluation_tracks': 50000,
        'subset_policy': 'Seed42 numpy.default_rng choice without replacement from original 50k cohort; original order retained',
        'statistics': stats, 'metadata_sha256': prior_ready['metadata_sha256'],
        'previous_selected_data_content_revalidated': True, 'scientific_source_changes_since_50k': source_changes,
        'remote_primary_source_differences': primary_differences,
        'source_policy': 'Use the verified immutable local campaign snapshot; do not alter the remote primary checkout',
        'initial_heads': initial, 'checkpoints': checkpoints,
        'label_budget_note': '1k training labels; retain 10k validation labels for checkpoint selection and 50k independent evaluation labels',
        'cache_policy': 'Fresh experiment-local compiler caches; no deleted dataset/feature cache is required'}
    write(OUT / 'data_ready.json', ready)
    print(json.dumps(ready, indent=2))


if __name__ == '__main__': main()
