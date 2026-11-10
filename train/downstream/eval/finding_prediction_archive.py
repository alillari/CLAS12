"""Validated, CPU-only inputs shared by finding calibration and diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from fm4npp.datasets.track_eligibility import PRIMARY_TRACK_POLICY, eligible_primary_labels
from train.downstream import track_finding_metrics as metrics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Create a new result; never replace an earlier scientific artifact."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def probability(value, name):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be finite and in [0, 1]')
    return value


def vector(value, name, length=None, integer=False):
    value = np.asarray(value)
    if value.ndim != 1 or (length is not None and len(value) != length):
        raise ValueError(f'{name} must be a one-dimensional array of length {length}')
    if integer and value.dtype.kind not in 'iu':
        raise ValueError(f'{name} must contain integers')
    return value


def signed_labels(value, name, length=None):
    """Normalize integer labels without unsigned wraparound or float promotion."""
    value = vector(value, name, length, integer=True)
    if value.dtype.kind == 'u' and np.any(value > np.uint64(np.iinfo(np.int64).max)):
        raise ValueError(f'{name} must fit in signed int64')
    return value.astype(np.int64, copy=False)


@dataclass
class PredictionArchive:
    arrays: dict
    receipt: dict

    @property
    def n_events(self):
        return len(self.arrays['offsets']) - 1

    def event_slice(self, event):
        return slice(int(self.arrays['offsets'][event]), int(self.arrays['offsets'][event + 1]))

    def event(self, event):
        sl = self.event_slice(event)
        return {key: value[sl] for key, value in self.arrays.items()
                if key not in ('offsets', 'source_id', 'original_ordinal')}

    def physical_keys(self):
        if 'source_id' not in self.arrays or 'original_ordinal' not in self.arrays:
            raise ValueError('Physical event identities (source_id and original_ordinal) are required')
        return list(zip(self.arrays['source_id'].tolist(), self.arrays['original_ordinal'].tolist()))

    def final_predictions(self, threshold=None):
        if threshold is None:
            return np.where(self.arrays['pred'] >= 0, self.arrays['pred'], -1)
        threshold = probability(threshold, 'assignment threshold')
        if 'scores' not in self.arrays or 'signal' not in self.arrays:
            raise ValueError('Thresholding requires raw scores and the native signal gate')
        gate = self.arrays['signal'] & (self.arrays['pred'] >= 0) & (self.arrays['scores'] > threshold)
        return np.where(gate, self.arrays['pred'], -1)


def load_archive(path, *, metadata=None, prediction_key='pred', truth_policy='as_saved', require_scores=False):
    """Read full-event NPZ arrays, optionally joining an explicitly aligned sidecar.

    Metadata must carry the exact same original truth and offsets. Duplicate
    fields must agree; a sidecar never silently overrides prediction contents.
    """
    if truth_policy not in ('as_saved', PRIMARY_TRACK_POLICY):
        raise ValueError('Unknown truth policy')
    paths = [Path(path)] + ([Path(metadata)] if metadata else [])
    with np.load(paths[0], allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    if metadata:
        with np.load(paths[1], allow_pickle=False) as sidecar:
            for key in ('offsets', 'truth'):
                if key not in raw or key not in sidecar or not np.array_equal(raw[key], sidecar[key]):
                    raise ValueError(f'Metadata {key} alignment mismatch')
            for key in sidecar.files:
                if key in raw and not np.array_equal(raw[key], sidecar[key]):
                    raise ValueError(f'Metadata field mismatch: {key}')
                raw[key] = sidecar[key]
    truth = signed_labels(raw['truth'], 'truth')
    if np.any(truth < -1):
        raise ValueError('Truth background must use -1')
    offsets = vector(raw['offsets'], 'offsets', integer=True)
    if (len(offsets) == 0 or offsets[0] != 0 or offsets[-1] != len(truth)
            or np.any(offsets[1:] < offsets[:-1])):
        raise ValueError('Invalid event offsets')
    arrays = dict(truth=truth.copy(), offsets=offsets,
                  pred=signed_labels(raw[prediction_key], prediction_key, len(truth)))
    n_events = len(offsets) - 1
    for key in ('source_id', 'original_ordinal'):
        if key in raw:
            arrays[key] = vector(raw[key], key, n_events, integer=True)
    if ('source_id' in arrays) != ('original_ordinal' in arrays):
        raise ValueError('Supply source_id and original_ordinal together')
    if 'source_id' in arrays and len(set(zip(arrays['source_id'], arrays['original_ordinal']))) != n_events:
        raise ValueError('Duplicate physical events')
    if 'point_ids' in raw:
        arrays['point_ids'] = signed_labels(raw['point_ids'], 'point_ids', len(truth))
        if np.any(arrays['point_ids'] < 0):
            raise ValueError('Original point_ids must be nonnegative')
    if 'provenance' in raw:
        arrays['provenance'] = vector(raw['provenance'], 'provenance', len(truth), integer=True)
    raw_provenance = ('background_only', 'ownership', 'origin_noise', 'purity')
    if 'provenance' in raw and any(key in raw for key in raw_provenance):
        raise ValueError('Supply provenance codes or raw provenance fields, not both')
    if any(key in raw for key in raw_provenance[1:]) and 'background_only' not in raw:
        raise ValueError('Raw provenance fields require background_only')
    for key in raw_provenance:
        if key in raw:
            vector(raw[key], key, len(truth))
    if 'provenance' not in arrays and 'background_only' in raw:
        arrays['provenance'] = metrics.cluster_provenance_from_hdf5(**{
            key: raw.get(key) for key in ('background_only', 'ownership', 'origin_noise', 'purity')})
    if 'provenance' in arrays and not np.isin(arrays['provenance'], list(metrics.ClusterProvenance)).all():
        raise ValueError('Invalid cluster provenance codes')
    if require_scores:
        scores = vector(raw['scores'], 'scores', len(truth))
        signal = vector(raw['signal'], 'signal', len(truth))
        if signal.dtype.kind != 'b':
            raise ValueError('The native signal gate must be boolean')
        active = signal & (arrays['pred'] >= 0)
        if (scores.dtype.kind not in 'fiu' or not np.isfinite(scores).all()
                or np.any((scores < -1) | (scores > 1))
                or np.any((scores[active] < 0) | (scores[active] > 1))):
            raise ValueError('Invalid assignment scores; active candidates must lie in [0, 1]')
        arrays.update(scores=scores, signal=signal)
    for key, shape in (('xyz', (len(truth), 3)), ('endpoints', (len(truth), 6))):
        if key in raw:
            value = np.asarray(raw[key])
            if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
                raise ValueError(f'Invalid {key} geometry')
            arrays[key] = value.astype(np.float64, copy=False)
    for key in ('detector', 'layer'):
        if key in raw:
            arrays[key] = vector(raw[key], key, len(truth), integer=True)
    for e in range(n_events):
        sl = slice(int(offsets[e]), int(offsets[e + 1]))
        ids = arrays['point_ids'][sl] if 'point_ids' in arrays else np.arange(sl.stop - sl.start)
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f'Duplicate original point IDs in event {e}')
        if truth_policy == PRIMARY_TRACK_POLICY:
            if 'point_ids' in arrays and not np.array_equal(np.sort(ids), np.arange(len(ids))):
                raise ValueError('Truth-policy projection requires full-event point permutations')
            arrays['truth'][sl] = eligible_primary_labels(truth[sl])
    if 'point_ids' not in arrays:
        arrays['point_ids'] = np.concatenate([np.arange(int(b-a)) for a,b in zip(offsets, offsets[1:])]) if n_events else np.empty(0, dtype=np.int64)
    if 'provenance' in arrays and np.any((arrays['truth'] >= 0) & (arrays['provenance'] == metrics.ClusterProvenance.OVERLAY_ONLY)):
        raise ValueError('Eligible MC truth cannot be verified pure overlay')
    receipt = dict(inputs=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in paths],
                   truth_policy=truth_policy, n_events=n_events, n_points=len(truth), prediction_key=prediction_key,
                   original_point_ids_supplied='point_ids' in raw,
                   archive_reader_sha256=sha256(__file__), numpy_version=np.__version__,
                   metrics_source_sha256=sha256(metrics.__file__))
    return PredictionArchive(arrays, receipt)


def align_to(reference, other):
    """Return point indices into other, aligned to reference's physical events."""
    if reference.receipt['truth_policy'] != other.receipt['truth_policy']:
        raise ValueError('Truth policies differ')
    if not all(x.receipt['original_point_ids_supplied'] for x in (reference, other)):
        raise ValueError('Paired comparisons require original point_ids in every input')
    first, second = reference.physical_keys(), other.physical_keys()
    if set(first) != set(second):
        raise ValueError('Paired comparisons require the same physical event cohort')
    lookup = {key: i for i, key in enumerate(second)}
    indices = []
    for e, key in enumerate(first):
        a = reference.event(e); j = lookup[key]; sl = other.event_slice(j); b = other.event(j)
        order = np.argsort(b['point_ids']); sorted_ids = b['point_ids'][order]
        if not np.array_equal(np.sort(a['point_ids']), sorted_ids):
            raise ValueError(f'Point membership mismatch in physical event {key}')
        ix = order[np.searchsorted(sorted_ids, a['point_ids'])]
        if not np.array_equal(a['truth'], b['truth'][ix]):
            raise ValueError(f'Truth alignment mismatch in physical event {key}')
        for field in ('provenance', 'xyz', 'endpoints', 'detector', 'layer'):
            if field in a and field in b and not np.array_equal(a[field], b[field][ix]):
                raise ValueError(f'{field} alignment mismatch in physical event {key}')
        indices.extend((sl.start + ix).tolist())
    return np.asarray(indices, dtype=np.int64)
