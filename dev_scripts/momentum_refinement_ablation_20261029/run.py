"""Compare the existing no-refinement AdapterOnly control with one Mamba2 layer."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev_scripts.backbone_downstream_20261029 import run as common

OUT = Path('/results')
PRIOR = Path('/prior')
CONTROL = PRIOR / 'runs/momentum_adapteronly'


def wait_for_gpu():
    while not (PRIOR / 'runner_result.json').exists():
        common.write(OUT/'phase.json', {'phase': 'waiting_for_existing_campaign',
            'prior_job': 'backbone-downstream-no-mamba-a518-20261029'})
        time.sleep(30)
    result = json.loads((PRIOR/'runner_result.json').read_text())
    assert result['exit_code'] == 0, 'Prior campaign failed; preserve outputs and investigate before this test'
    assert json.loads((PRIOR/'completed.json').read_text())['complete']
    while True:
        r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                           text=True, capture_output=True)
        if r.returncode == 0 and not r.stdout.strip(): break
        common.write(OUT/'phase.json', {'phase': 'waiting_for_idle_gpu', 'processes': r.stdout.strip()})
        time.sleep(30)


def child(stage, run_dir, pilot):
    cmd = [sys.executable, str(Path(__file__).resolve()), '--stage', stage, '--run-dir', str(run_dir)]
    if pilot: cmd.append('--pilot')
    common.write(OUT/'phase.json', {'phase': ('pilot_' if pilot else 'full_')+stage})
    with (run_dir/(stage+'.stdout.log')).open('w') as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    assert r.returncode == 0, f'{stage} failed with {r.returncode}; inspect {run_dir}/{stage}.stdout.log'


def worker():
    OUT.mkdir(exist_ok=True)
    wait_for_gpu()
    control_training = json.loads((CONTROL/'training_complete.json').read_text())
    control_eval = json.loads((CONTROL/'evaluation_complete.json').read_text())
    assert control_training['final_step'] == control_training['finite_gradient_updates'] == 30000
    assert control_training['head_mamba_blocks'] == 0 and control_training['strict_head_reload']
    assert control_eval['actual_samples'] == 50000
    for pilot in (True, False):
        root = OUT/('pilot' if pilot else 'with_mamba2')
        root.mkdir(exist_ok=True)
        for stage in ('train', 'eval'): child(stage, root, pilot)
        training = json.loads((root/'training_complete.json').read_text())
        assert training['head_mamba_blocks'] == 1
        assert training['shared_initial_head_sha256'] == control_training['initial_head_sha256'], 'Common head initialization changed'
        assert training['head_parameters'] == 664744 and training['head_trainable_parameters'] == 632488
        if pilot:
            common.write(OUT/'pilot_complete.json', {'complete': True, 'training': training,
                'evaluation': json.loads((root/'evaluation_complete.json').read_text())})
    treatment_training = json.loads((OUT/'with_mamba2/training_complete.json').read_text())
    treatment_eval = json.loads((OUT/'with_mamba2/evaluation_complete.json').read_text())
    for key in ('train_keys_sha256','validation_keys_sha256','final_step','train_samples','validation_samples'):
        assert treatment_training[key] == control_training[key], key
    for key in ('evaluation_keys_sha256','truth_sha256','actual_samples','adapter_valid_samples'):
        assert treatment_eval[key] == control_eval[key], key
    control_config = json.loads((CONTROL/'resolved.json').read_text())
    treatment_config = json.loads((OUT/'with_mamba2/resolved.json').read_text())
    for key in ('batch_size','limit_size','limit_test_size','seed','max_optimizer_steps','max_lr','min_lr',
                'warmup_steps','scheduler_first_cycle_steps','adapter_weight_decay','grad_clip_value',
                'dropout','regression_loss','pooling','embed_dim','num_embedder_layers'):
        assert control_config[key] == treatment_config[key], key
    comparison = {'complete': True, 'host': 'a518', 'seed': 42,
        'scope': 'AdapterOnly momentum; 0 versus 1 Mamba2 refinement block. Same examples, budgets, common initial weights and FP32 execution. Added block also adds capacity.',
        'without_mamba': {'source_job': 'backbone-downstream-no-mamba-a518-20261029',
            'training': control_training, 'evaluation': control_eval,
            'metrics': json.loads((CONTROL/'evaluation/summary.json').read_text())},
        'with_mamba2': {'training': treatment_training, 'evaluation': treatment_eval,
            'metrics': json.loads((OUT/'with_mamba2/evaluation/summary.json').read_text())}}
    common.write(OUT/'comparison.json', comparison)
    common.write(OUT/'completed.json', {'complete': True, 'comparison': 'comparison.json',
        'matched_data_and_shared_initialization': True, 'total_parameters': {'without': 232720, 'with': 664744}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('train','eval'))
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.stage:
        common.MOMENTUM_REFINEMENT_LAYERS = 1
        common.torch.set_num_threads(4)
        if args.stage == 'train': common.train('momentum','adapteronly',args.run_dir,args.pilot)
        else: common.evaluate('momentum','adapteronly',args.run_dir,args.pilot)
    else: worker()


if __name__ == '__main__': main()
