"""Run the three matched downstream tasks once the full theta-distance backbone is selected."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from run_matched import PLAN, PRETRAIN, configure, preflight, run

ROOT=Path('/results')
UPSTREAM=Path('/pretraining-run')
ARM='plain_theta'
TASKS=('noise','finding','momentum')


def write(path, data):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def sha(file):
    digest=hashlib.sha256()
    with file.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4<<20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def register():
    import torch
    while True:
        try:
            receipt=json.loads((UPSTREAM/'complete.json').read_text())
            break
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(30)
    assert receipt['plan_id']==PLAN and receipt['final_step']==126746
    native=UPSTREAM/'pretraining/hdf5_chinchilla_klen1_m4_w512/hdf5_chinchilla_klen1/training_checkpoints/ckpt_best.tar'
    assert sha(native)==receipt['original_best_sha256']
    ckpt=torch.load(native,map_location='cpu',weights_only=False)
    assert ckpt['iters']==receipt['selected_step'] and ckpt['best_loss']==receipt['best_loss']
    original=json.loads((UPSTREAM/'resolved.json').read_text())['hdf5_chinchilla_klen1_m4_w512']
    assert original['coord_system']=='theta' and original['dropout']==0 and original['weight_decay']==0.04015310456820763
    import run_matched
    desc=run_matched.Y.load(run_matched.RECIPES/'finding_100000.yaml')
    compatible=dict(next(iter(desc.values()))['pretrained_backbone']['config'],coord_system='theta')
    # Persist architecture and token contract. The full historical training recipe
    # is retained separately from the downstream heads' dropout and runtime setup.
    for key in ('dropout','space_filling_order','order'):
        compatible.pop(key,None)
    payload={k:ckpt[k] for k in ('model_state','backbone_architecture','hilbert_angular_grid','validation_metric','iters','best_loss')}
    payload.update(backbone_config=compatible,pretraining_recipe=original,
                   pretraining_objective=dict(coord_system='theta',knn_rule='distance',use_sincos_target=True))
    path=ROOT/'backbone'
    path.mkdir(exist_ok=True)
    tmp=path/'plain_theta_m4.tmp'
    torch.save(payload,tmp)
    target=path/'plain_theta_m4.tar'
    tmp.replace(target)
    digest=sha(target)
    write(path/'receipt.json',dict(checkpoint=target.name,sha256=digest,raw_best_sha256=receipt['original_best_sha256'],
          selected_step=ckpt['iters'],final_step=receipt['final_step'],best_loss=float(ckpt['best_loss']),plan_id=PLAN,
          original_recipe=original,input_contract=compatible,objective=payload['pretraining_objective']))
    return target,digest


def main():
    write(ROOT/'progress.json',dict(current_stage='awaiting_pretraining',completed_stages=[],total_stages=3))
    checkpoint,digest=register()
    PRETRAIN[ARM]=('theta',str(checkpoint),digest)
    configs={task:configure(ARM,task) for task in TASKS}
    preflight(ARM,configs)
    done=[]
    for task in TASKS:
        write(ROOT/'progress.json',dict(current_stage=task,completed_stages=done,total_stages=3))
        folder,model,path=configs[task]
        run(task,model,path,SimpleNamespace(arm=ARM))
        if task=='finding':
            subprocess.run([sys.executable,'-u','calibrate_finding.py'],check=True,
                           env=dict(os.environ,M4_FINDING_FIT=str(folder),M4_FINDING_SCORE='/results/calibration'))
        done.append(task)
    write(ROOT/'complete.json',dict(arm=ARM,tasks=done,backbone_sha256=digest,seed=42))
    write(ROOT/'progress.json',dict(current_stage='complete',completed_stages=done,total_stages=3,status='succeeded'))


if __name__=='__main__':
    main()
