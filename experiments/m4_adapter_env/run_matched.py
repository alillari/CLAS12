"""Train both released M4 backbones through the same production adapter entry points."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from ruamel.yaml import YAML

ROOT = Path('/results')
RECIPES = Path(__file__).resolve().parent / 'recipes'
SKIM = Path('/cvt')
PLAN = '1adb071fbb72eb24b7518f836c64e22c83a67df39685b5cee8f56fe5b458dcf2'
PRETRAIN = {
    'plain': ('eta', 'mike_hdf5_chinchilla_klen1/hdf5_chinchilla_klen1_m4_w512/hdf5_chinchilla_klen1/training_checkpoints/ckpt_best.tar',
              'a6ea644b035cc9cb851824b7999a5d7895d45e973ad52e87fdc99bfebf00ed61'),
    'adopted': ('theta', 'mike_adopted_m4_20261123/ckpt_best.tar',
                '7740ceb44b0349701121997093d01f920e59612c9626f4e547740ddc618cc26c'),
    # backbone-only export the plain theta heads loaded (package backbones/plain_theta/)
    'plain_theta': ('theta', 'plain_theta/plain_theta_m4.tar',
                    '75c09f2e315cbed891a5d4d3f811ce1f11a7cc838136fa631d4ee60fc203dcff'),
}
Y = YAML(typ='safe')
TASKS = {'noise': 'point_classification', 'finding': 'track_finding', 'momentum': 'track_regression'}


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def configure(arm, task, feature_mix='raw'):
    angle, filename, digest = PRETRAIN[arm]
    checkpoint = str(Path('/backbones') / filename)
    d = ROOT / task
    d.mkdir(exist_ok=True)
    model = next(iter(Y.load(RECIPES / f'{task}_100000.yaml').values()))
    if task == 'noise':
        descriptor = json.loads((RECIPES / 'noise_training.json').read_text())['training_config']['pretrained_backbone']
    else:
        descriptor = model['pretrained_backbone']
    descriptor = dict(name=arm, checkpoint=checkpoint, sha256=digest, config=dict(descriptor['config'], coord_system=angle))
    model.update(descriptor['config'])
    model.update(pretrained_backbone=descriptor, pretrained_ckpt=checkpoint, data_root=str(SKIM),
                 data_root_train=str(SKIM), data_root_test=str(SKIM), stat_dir='/cvt/stats',
                 checkpoint_dir=str(d / 'checkpoints'), downstream_dir=str(d), limit_size=100000)
    if feature_mix != 'raw':
        model['feature_mix'] = feature_mix  # train/downstream/feature_mix.py
    if task == 'momentum':
        model['regression_target_stats'] = str(RECIPES / 'momentum_100000_stats.json')
    if task == 'finding':
        model['validation_selection_metric'] = 'dice'
        model['save_validation_checkpoints'] = False  # best checkpoint is retained
    model_path = d / 'model.yaml'
    save(model_path, {task: model})
    return d, model, model_path


def run(task, model, model_path, args):
    path = ROOT / task
    batch = model['local_batch_size']
    frozen = bool(model.get('pretrained_ckpt'))
    train = [f'train/downstream/train_{TASKS[task]}.py', '--yaml_config', model_path, '--config', task,
             '--run_num', '42', '--root_dir', path / 'runtime', '--usepretrain' if frozen else '--no-pretrain', '--eventnumber', 100000,
             '--train_batch_size', batch, '--seed', 42]
    if task != 'noise':
        train += ['--checkpoint_dir', path / 'checkpoints', '--checkpoint_file_name', 'adapter.pth',
                  '--log_file_name', 'training.log', '--artifact_summary', path / 'training_run.json',
                  '--resolved_config_path', path / 'resolved_config.json']
    if task == 'noise':
        train += ['--task', 'nid']
    def execute(command):
        argv = [sys.executable, '-u', *map(str, command)]
        print('Executing', argv, flush=True)
        subprocess.run(argv, check=True)
    execute(train)
    if task == 'noise':
        receipt = json.loads((path / 'checkpoints/training_run.json').read_text())
        assert receipt['optimizer_steps'] == 30000
        assert receipt['training_contract']['mode'] == ('frozen' if frozen else 'adapteronly')
        assert receipt['training_contract']['backbone_sha256'] == (PRETRAIN[args.arm][2] if frozen else None)
        evaluate = ['train/downstream/eval_point_classification.py', '--yaml_config', model_path, '--config', task,
                    '--checkpoint', receipt['checkpoint'],
                    '--task', 'nid', '--validation_events', 5000, '--eventnumber', 10000, '--eval_batch_size', 32,
                    '--root_dir', path / 'evaluation']
        if frozen:
            evaluate += ['--pretrained_ckpt', model['pretrained_ckpt']]
    else:
        receipt = json.loads((path / 'training_run.json').read_text())
        assert receipt['usepretrain'] == frozen
        assert receipt['final_step'] == 30000, receipt['final_step']
        analysis = Y.load(RECIPES / f'{task}_analysis.yaml')['analysis']
        analysis.update(analysis_tag=f'{args.arm}_{task}', model_yaml=str(model_path), model_config=task,
                        use_pretrained_backbone=frozen,
                        output_dir=str(path / 'evaluation'), pretrained_checkpoint=model['pretrained_ckpt'],
                        checkpoint=receipt['checkpoint'], training_log=str(path / 'checkpoints/training.log'),
                        evaluation_split='test', run_name=f'{args.arm}_{task}', run_num='42')
        if task == 'momentum':
            analysis['target_momentum_scale_to_gev'] = 1.0
        analysis_path = path / 'analysis.yaml'
        save(analysis_path, dict(analysis=analysis))
        evaluate = [f'train/downstream/eval/evaluate_{TASKS[task]}.py', '--analysis-config', analysis_path]
    execute(evaluate)
    assert (path / 'evaluation/summary.json').is_file()


def preflight(arm, configs):
    import torch
    import importlib.metadata
    from fm4npp.pretrained import resolve_backbone, load_pretrained_checkpoint
    from fm4npp.models.mamba_config import build_backbone
    from fm4npp.datasets.dataset import TPCBatchDataset
    from fm4npp.datasets.dataset_pretrain import TPCBatchDataset as Pretrain
    assert torch.cuda.is_available()
    manifest = json.loads((SKIM / 'manifest.json').read_text())
    assert manifest['plan_id'] == PLAN and manifest['status'] == 'complete'
    stats = json.loads((RECIPES / 'momentum_100000_stats.json').read_text())
    assert stats['count'] == [100000]*3 and stats['data_contract']['plan_id'] == PLAN
    selected = resolve_backbone(configs['finding'][1]['pretrained_backbone'])
    net = build_backbone(selected['config'])
    payload, sha = load_pretrained_checkpoint(selected['checkpoint'], {'pretrained_backbone': selected})
    state = {k.removeprefix('module.'): v for k,v in payload['model_state'].items()}
    net.load_state_dict(state, strict=True)
    net.cuda().eval()
    common = dict(split='validation', reader_type='hdf5', hilbert_angular_grid='unit_v2', coord_system=PRETRAIN[arm][0], num_pred_points=1)
    upstream = Pretrain(SKIM, train=False, knn_rule='geometric' if arm == 'adopted' else 'distance', **common)
    downstream = TPCBatchDataset(SKIM, train=False, normalize=True, limit_data=True, limit_size=16, voxelize=False,
                                 return_dict=True, compute_neighbors=False, serialization='clas12_band_hilbert', **common)
    source_rows = {int(raw): idx for idx, raw in enumerate(upstream.idxlist)}
    for i in range(8):
        x = upstream[source_rows[int(downstream.idxlist[i])]][0]
        torch.testing.assert_close(downstream[i]['points'], x, rtol=0, atol=0)
        if len(x):
            with torch.inference_mode():
                pred = net(x[None].cuda())
            assert pred.shape == (1, len(x), 4) and torch.isfinite(pred).all()
    save(ROOT / 'preflight.json', dict(arm=arm, plan_id=PLAN, backbone_sha256=sha, coord_system=PRETRAIN[arm][0],
         events_checked=8, torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
         dependencies={k:importlib.metadata.version(k) for k in ['numpy','h5py','triton']},
         manifest_sha256=hashlib.sha256((SKIM / 'manifest.json').read_bytes()).hexdigest()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm', choices=PRETRAIN)
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--task', choices=TASKS, help='Run one task with the same matched configuration')
    ap.add_argument('--feature-mix', default='raw', choices=('raw', 'standardize', 'last'),
                    help="How the adapter combines the backbone's layers; raw reproduces the original fits")
    args = ap.parse_args()
    configs = {task: configure(args.arm, task, args.feature_mix) for task in TASKS}
    preflight(args.arm, configs)
    if args.probe:
        return
    completed = []
    requested = [args.task] if args.task else list(configs)
    for task in requested:
        _, model, path = configs[task]
        save(ROOT / 'progress.json', dict(current_stage=task, completed_stages=completed, total_stages=len(requested), status='running'))
        run(task, model, path, args)
        completed.append(task)
    save(ROOT / 'progress.json', dict(status='succeeded', completed_stages=completed, total_stages=len(requested)))


if __name__ == '__main__':
    main()
