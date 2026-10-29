"""Detached sequential queue: both gates pass before either production fit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from dev_scripts.transformer_pretraining_20261029.train import source_digest


def save(path, value):
    p=Path(path); tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(p)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase', choices=['pilot','train'], required=True)
    args=parser.parse_args(); root=Path('/results')
    results=[]
    for encoding in ['nope','rope']:
        out=root/encoding; out.mkdir(exist_ok=True)
        save(root/'status.json', {'phase':args.phase,'active':encoding,'completed':[r['encoding'] for r in results]})
        command=[sys.executable, '-m', 'dev_scripts.transformer_pretraining_20261029.train',
                 '--encoding',encoding,'--phase',args.phase,'--output',str(out)]
        with (out/'stdout.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        save(out/'child_result.json', {'exit_code':result.returncode,'command':command})
        if result.returncode: raise RuntimeError(f'{encoding} {args.phase} failed; see {out}/stdout.log')
        results.append(json.loads((out/'completed.json').read_text()))
    for key in ['initial_state_sha256','embedding_sha256','sampler_generator_sha256','source_digest']:
        assert results[0][key] == results[1][key], key
    if args.phase=='pilot':
        save(root/'pilot_complete.json',{'passed':True,'source_digest':source_digest(),
              'matched_initial_state':results[0]['initial_state_sha256'],
              'steps_per_arm':2000,'cohorts_match_all_mambas':True,'arms':results})
    else:
        subprocess.run([sys.executable,'-m','dev_scripts.transformer_pretraining_20261029.report',
                        '--root',str(root)],check=True)
        save(root/'completed.json',{'complete':True,'source_digest':source_digest(),
              'scope':'pretraining only','arms':[{'encoding':r['encoding'],'steps':r['final_step'],
              'best_validation_mse':r['best_validation_mse']} for r in results]})
    save(root/'status.json',{'phase':'complete','scope':args.phase,'completed':['nope','rope']})


if __name__=='__main__': main()
