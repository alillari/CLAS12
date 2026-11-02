"""Download existing output artifacts through the established SSH aliases."""
import argparse
from pathlib import Path
import subprocess


def download(alias, remote, local):
    local.mkdir(parents=True, exist_ok=True)
    command = ['rsync', '-az', '--partial', '-e', 'ssh -o ClearAllForwardings=yes']
    for pattern in ('triton/', 'inductor/', 'cache/', 'matplotlib/', '*.pth', '*.pt', '*.tar', '*.png', '*.pdf'):
        command.extend(['--exclude', pattern])
    command.extend([alias + ':' + remote + '/', str(local) + '/'])
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    base = args.root.resolve() / 'backbone_refinement'
    jobs = '/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/'
    for host, alias in [('a02c', 'alessio-edge'), ('a518', 'alessio-aux')]:
        download(alias, jobs + f'momentum-backbone-refinement-{host}-20261029/output', base / host)
        download(alias, jobs + f'momentum-backbone-refinement-pilot-{host}-20261029/output', base / 'pilots' / host)
        download(alias, jobs + f'momentum-refined-head-initial-{host}-20261029/output', base / 'initialization' / host)


if __name__ == '__main__': main()
