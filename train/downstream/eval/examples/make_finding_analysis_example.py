"""Generate a tiny synthetic, geometry-free example for the analysis CLIs."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    common = dict(truth=np.array([0]*10 + [-1]*2), pred=np.array([7]*12),
        offsets=np.array([0,12]), scores=np.array([.9]*9 + [.4,.9,.6]),
        signal=np.ones(12,dtype=bool), point_ids=np.arange(12),
        source_id=np.array([21]), provenance=np.array([1]*10 + [2]*2))
    for name, ordinal in [('validation',1), ('test',2)]:
        np.savez_compressed(args.output_dir / f'{name}.npz', **common, original_ordinal=np.array([ordinal]))
    baseline = {**common, 'pred': np.array([7]*10 + [-1]*2), 'original_ordinal': np.array([2])}
    np.savez_compressed(args.output_dir / 'reference.npz', **baseline)
    manifest = dict(truth_policy='primary_mc_min4_v1',
        baseline=dict(name='Reference reconstruction',kind='final',path='reference.npz'),
        models=[dict(name='Example adapter',kind='raw',path='test.npz',assignment_threshold=.2)])
    (args.output_dir / 'comparison.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Wrote synthetic example inputs to {args.output_dir}')


if __name__ == '__main__':
    main()
