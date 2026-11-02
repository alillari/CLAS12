"""Copy the unchanged best checkpoint through the existing SSH aliases."""
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE = '/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/mamba2-chinchilla-w512-d12-k1-20261027/output/training/mamba2_chinchilla_w512_d12_k1/seed42/training_checkpoints/ckpt_best.tar'
DESTINATION = '/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/momentum-1k-checkpoint-import-a02c-20261029/output/training/mamba2_chinchilla_w512_d12_k1/seed42/training_checkpoints/ckpt_best.tar'
LOCAL = Path('/tmp/clas12-momentum-1k-mamba2-20261029/ckpt_best.tar')


def remote_sha(alias, path):
    return subprocess.check_output(['ssh', '-o', 'ClearAllForwardings=yes', alias,
        'sha256sum', path], text=True).split()[0]


def main():
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    before = remote_sha('alessio-aux', SOURCE)
    base = ['rsync', '-a', '--partial', '--checksum', '-e', 'ssh -o ClearAllForwardings=yes']
    subprocess.run(base + ['alessio-aux:' + SOURCE, str(LOCAL)], check=True)
    h = hashlib.sha256()
    with LOCAL.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    assert h.hexdigest() == before
    subprocess.run(base + [str(LOCAL), 'alessio-edge:' + DESTINATION], check=True)
    after = remote_sha('alessio-edge', DESTINATION)
    assert after == before
    receipt = {'complete': True, 'source_host': 'a518', 'training_host': 'a02c',
        'source': SOURCE, 'destination': DESTINATION, 'sha256': before, 'bytes': LOCAL.stat().st_size}
    Path(__file__).with_name('checkpoint_import.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
