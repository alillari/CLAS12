"""Hash every file read from the six training/validation RaggedMmap trees."""

import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/home/alessio/ML-work/HIPO_processing/multi_particle_storage/big-data/mmap_canonical_loose_truthseg_event_v6_02')
OUT = Path('/results/data_identity.json')
started = time.monotonic()
files = {}
for split in ['pretrain', 'test']:
    for kind in ['features', 'seg_target', 'reg_target']:
        directory = ROOT / (kind + '_' + split)
        assert directory.is_dir(), directory
        for path in sorted(directory.rglob('*')):
            assert not path.is_symlink(), path
            if not path.is_file():
                continue
            stat = path.stat()
            value = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    value.update(block)
            after = path.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
            files[str(path.relative_to(ROOT))] = {'size': stat.st_size, 'sha256': value.hexdigest()}
            print('HASHED', path.relative_to(ROOT), stat.st_size, flush=True)
fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
report = {'root': str(ROOT), 'files': files, 'sha256': fingerprint,
          'file_count': len(files), 'bytes': sum(v['size'] for v in files.values()),
          'elapsed_seconds': time.monotonic() - started}
OUT.write_text(json.dumps(report, indent=2) + '\n')
print('DATA_IDENTITY', json.dumps({k: v for k, v in report.items() if k != 'files'}), flush=True)
