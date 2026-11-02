"""Copy existing 1k experiment reports/histories/predictions for local analysis."""
import argparse
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); args = p.parse_args()
    jobs = '/home/alessio/ML-work/result_deep_storage/edge_mcp/jobs/'
    for job, folder in [('momentum-1k-a02c-20261029', 'a02c'), ('momentum-1k-prepare-a02c-20261029', 'preparation')]:
        target = args.root / folder; target.mkdir(parents=True, exist_ok=True)
        command = ['rsync', '-az', '--partial', '-e', 'ssh -o ClearAllForwardings=yes']
        for pattern in ('triton/', 'inductor/', 'cache/', 'matplotlib/', '*.pth', '*.pt', '*.tar', '*.png', '*.pdf'):
            command.extend(['--exclude', pattern])
        command.extend(['alessio-edge:' + jobs + job + '/output/', str(target) + '/'])
        subprocess.run(command, check=True)


if __name__ == '__main__': main()
