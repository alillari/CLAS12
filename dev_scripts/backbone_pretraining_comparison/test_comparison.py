from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from dev_scripts.backbone_pretraining_comparison.tokens import replay
from dev_scripts.backbone_pretraining_comparison.compare import curve_from_log
from dev_scripts.backbone_pretraining_comparison.flops import describe


class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, lengths): self.lengths = lengths
    def __len__(self): return len(self.lengths)
    def __getitem__(self, index): return int(self.lengths[index])


def length_batch(values):
    return max(values) * len(values), sum(values)


class ComparisonTests(unittest.TestCase):
    def test_replay_matches_persistent_worker_loader_across_epochs(self):
        lengths = np.arange(7, 31)
        loader = torch.utils.data.DataLoader(LengthDataset(lengths), batch_size=4, shuffle=True,
                 drop_last=True, num_workers=2, persistent_workers=True,
                 generator=torch.Generator().manual_seed(42), prefetch_factor=2, collate_fn=length_batch)
        batches = []
        for _ in range(3):
            for value in loader:
                batches.append(value)
                if len(batches) == 15: break
            if len(batches) == 15: break
        padded, valid = replay(lengths, 4, 15)
        np.testing.assert_array_equal(padded, [x[0] for x in batches])
        np.testing.assert_array_equal(valid, [x[1] for x in batches])
        del loader

    def test_curve_parser_preserves_raw_validation_not_running_average(self):
        with TemporaryDirectory() as root:
            p = Path(root) / 'train.log'
            p.write_text('Step = 1000, Val loss = 0.03\nStep = 2000, Val loss = 0.08\nStep = 3000, Val loss = 0.01\n')
            values = curve_from_log(p)
            self.assertEqual([v['validation_mse'] for v in values], [.03, .08, .01])

    def test_dense_count_includes_frozen_projection_forward_only(self):
        model = torch.nn.Module()
        model.embedder = torch.nn.Module(); model.embedder.embed = torch.nn.Module()
        model.embedder.embed.projection = torch.nn.Parameter(torch.ones(3, 4), requires_grad=False)
        model.linear = torch.nn.Linear(4, 2)
        result = describe(model, 'mamba1')
        self.assertEqual(result['dense_training_flops_per_padded_token'], 3 * 2 * 4 * 2 + 2 * 3 * 4)
        self.assertIn('not measured', result['status'])
        self.assertIn('backward recomputation', result['excluded'])


if __name__ == '__main__': unittest.main()
