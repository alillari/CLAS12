#!/usr/bin/env python3
"""Exercise project imports and real CUDA kernels; save a machine-readable receipt."""

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(args, report):
    import torch

    report.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), python=sys.version,
                  python_executable=sys.executable)
    for module in ['numpy', 'scipy', 'yaml', 'ruamel.yaml', 'sklearn', 'einops',
                   'mmap_ninja', 'plotly', 'matplotlib', 'pandas', 'optuna', 'wandb',
                   'cosine_annealing_warmup', 'mamba_ssm', 'selective_scan_cuda',
                   'causal_conv1d_cuda']:
        importlib.import_module(module)
    for script in ['train/downstream/train_track_finding.py',
                   'train/downstream/train_track_regression.py',
                   'train/pretrain/nppmamba/train_multi_gpu_mamba1.py']:
        subprocess.run([sys.executable, str(ROOT/script), '--help'], cwd=ROOT,
                       env={**os.environ, 'PYTHONPATH': str(ROOT)}, check=True,
                       stdout=subprocess.DEVNULL)
    report['imports_and_cli'] = 'passed'
    if args.imports_only:
        report['status'] = 'imports_only'
        return
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU available. Imports alone do not validate a training environment.')
    report['gpu'] = torch.cuda.get_device_name(0)
    report['compute_capability'] = list(torch.cuda.get_device_capability(0))
    report['bf16_supported'] = torch.cuda.is_bf16_supported()
    report['ptxas'] = os.environ.get('TRITON_PTXAS_PATH')
    from fm4npp.models.mamba_config import build_backbone

    torch.set_num_threads(2)
    torch.manual_seed(42)
    for version in args.backbones:
        print(f'Checking {version}; the first run may compile/autotune GPU kernels...', flush=True)
        if version == 'mamba3' and not report['bf16_supported']:
            raise RuntimeError('The selected Mamba3 path requires a BF16-capable GPU')
        params = dict(backbone_version=version, embed_dim=64, num_layers_backbone=1,
                      d_state=16 if version == 'mamba1' else 128 if version == 'mamba3' else 64,
                      d_conv=4, expand=2, klen=1, dropout=0., embed_method='pos_only',
                      pe_method='nerf', num_heads_backbone=4)
        model = build_backbone(params).cuda()
        if version == 'mamba2' and not all(layer[1].use_mem_eff_path for layer in model.mamba_layers):
            raise RuntimeError('Mamba2 verification must use fused kernels, not the CPU fallback')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        points = torch.rand(1, 65, 3, device='cuda')
        prediction = model(points)
        loss = prediction.float().square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f'{version} produced nonfinite loss/gradients or no gradients')
        optimizer.step()
        with tempfile.TemporaryDirectory(prefix='clas12-preflight-') as scratch:
            checkpoint = Path(scratch)/'weights.pt'
            torch.save(model.state_dict(), checkpoint)
            clone = build_backbone(params).cuda()
            clone.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True), strict=True)
        torch.cuda.synchronize()
        report['backbones'][version] = {'status': 'passed', 'loss': float(loss.detach()),
                                        'parameters': params,
                                        'forward_backward_optimizer_step': True,
                                        'strict_checkpoint_reload': True}
        print(f'PASS: {version} CUDA forward/backward, optimizer step, strict reload', flush=True)
        del clone, optimizer, model, prediction, loss, gradients, points
        torch.cuda.empty_cache()
    report['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--backbones', nargs='+', choices=['mamba1', 'mamba2', 'mamba3', 'transformer'],
                        default=['mamba1', 'mamba2', 'mamba3', 'transformer'])
    parser.add_argument('--imports-only', action='store_true', help='Explicitly omit GPU execution; not a training validation')
    args = parser.parse_args()
    report = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'machine': platform.machine(),
              'platform': platform.platform(), 'status': 'failed', 'backbones': {},
              'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
    build = Path(sys.prefix)/'clas12-kernel-build.json'
    if build.is_file():
        report['kernel_build'] = json.loads(build.read_text())
    try:
        verify(args, report)
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(f'Environment receipt: {args.output}', flush=True)


if __name__ == '__main__':
    main()
