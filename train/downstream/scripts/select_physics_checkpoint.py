#!/usr/bin/env python3
"""Reselect saved validation checkpoints with new guardrails, without inference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from train.downstream.physics_checkpoints import CheckpointSummary, resolve_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for the revised selection')
    for name in ('B_macro', 'B_worst', 'T_macro'):
        parser.add_argument('--' + name.replace('_', '-').lower(), type=float)
    parser.add_argument('--width-tie-atol', type=float)
    parser.add_argument('--width-tie-rtol', type=float)
    parser.add_argument('--require-configured-guardrails', action='store_true')
    args = parser.parse_args()
    original = json.loads(args.summary.read_text())
    config = original['config']
    for key in ('B_macro', 'B_worst', 'T_macro'):
        value = getattr(args, key.lower())
        if value is not None:
            config['guardrails'][key] = value
    for key in ('width_tie_atol', 'width_tie_rtol'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.require_configured_guardrails:
        config['require_configured_guardrails'] = True
    config = resolve_config('mom', config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    history = CheckpointSummary(config, args.output_dir, original['checkpoints'])
    payload = history.write()
    print((args.output_dir / 'checkpoint_summary.md').read_text())
    if history.selected is None:
        raise SystemExit('No checkpoint passed; summaries and best candidates were saved.')
    import torch
    checkpoint = torch.load(history.selected['checkpoint'], map_location='cpu', weights_only=False)
    checkpoint['physics_checkpoint_config'] = config
    checkpoint['physics_checkpoint_summary_path'] = str((args.output_dir / 'checkpoint_summary.json').resolve())
    checkpoint['best_step'] = history.selected['step']
    checkpoint['best_epoch'] = history.selected['epoch']
    torch.save(checkpoint, args.output_dir / 'selected_checkpoint.pth')
    print(f"Selected step {payload['selected_step']}: {args.output_dir / 'selected_checkpoint.pth'}")


if __name__ == '__main__':
    main()
