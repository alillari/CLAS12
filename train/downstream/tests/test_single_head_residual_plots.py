import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from train.downstream.campaign.plot_single_head_residuals import (
    align_reference, binned_statistics, curve, load_evaluation, HDF5_IDENTITY, residual_maps,
)


class SingleHeadResidualPlotsTest(unittest.TestCase):
    def test_empty_display_ranges_still_produce_residual_figures(self):
        data = {task:dict(residual=np.array([50., 60.]), core_limit=2., label=task, ylabel='error')
                for task in ('p','theta')}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'empty_range'
            residual_maps(data,np.array([.5,1.]),np.array([.25,3.]),'True p',
                          {'p':[],'theta':[]},output)
            for suffix in ('.png','.pdf'):
                self.assertGreater(output.with_suffix(suffix).stat().st_size,1000)

    def test_hdf5_identity_joins_across_row_order_and_rejects_other_plans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'summary.json').write_text(json.dumps(dict(evaluation_contract='clas12_momentum_entrance_v1',
                training_target_task='p', swingback_enabled=False, n_records=2)))
            (root/'physics_checkpoint_summary.json').write_text('{}')
            rows = [dict(data_plan_id='plan_a', data_split='test', source_id=17, original_ordinal=i,
                         source_event_index=i, segment_label=3, truth_segment_label=3,
                         adapter_sample_mode='event_segment', n_hits=12, true_p_gev=1., adapter_p_gev=1.)
                    for i in (7, 9)]
            def write(values):
                with gzip.open(root/'predictions.csv.gz', 'wt') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                    writer.writeheader(); writer.writerows(values)
            write(rows)
            primary = load_evaluation(root, 'p')
            self.assertEqual(primary['identity_columns'], HDF5_IDENTITY)
            rows.reverse()
            rows[0]['source_event_index'] = 400
            write(rows)
            np.testing.assert_array_equal(align_reference(primary, load_evaluation(root, 'p')), [1, 0])
            rows[0]['data_plan_id'] = 'other_plan'
            write(rows)
            with self.assertRaisesRegex(ValueError, 'different cohorts'):
                align_reference(primary, load_evaluation(root, 'p'))
            for row in rows:
                del row['data_plan_id']
            write(rows)
            with self.assertRaisesRegex(ValueError, 'missing columns'):
                load_evaluation(root, 'p')

    def test_truth_join_uses_identity_not_row_order(self):
        primary = dict(keys=[('event1', 'segment0'), ('event1', 'segment1')], hits=np.array([12, 15]))
        reference = dict(keys=primary['keys'][::-1], hits=np.array([15, 12]))
        np.testing.assert_array_equal(align_reference(primary, reference), [1, 0])

    def test_truth_join_rejects_missing_duplicate_and_inconsistent_tracks(self):
        primary = dict(keys=[('event1', 'segment0')], hits=np.array([12]))
        for reference in [dict(keys=[('event2', 'segment0')], hits=np.array([12])),
                          dict(keys=primary['keys']*2, hits=np.array([12, 12])),
                          dict(keys=primary['keys'], hits=np.array([13]))]:
            with self.assertRaises(ValueError):
                align_reference(primary, reference)

    def test_quantiles_tail_boundary_and_last_bin_edge(self):
        rows = binned_statistics(np.array([.25, .5, .75, 1.]), np.array([-10., 10., 20., 30.]),
                                 [.25, .75, 1.], threshold=10., min_entries=2)
        self.assertEqual([r['n'] for r in rows], [2, 2])
        self.assertEqual([r['median'] for r in rows], [0., 25.])
        self.assertEqual([r['tail_fraction'] for r in rows], [0., 1.])
        self.assertAlmostEqual(rows[0]['w68'], 6.8)
        self.assertTrue(all(r['valid'] for r in rows))

    def test_sparse_or_nonfinite_bins_do_not_draw_valid_curves(self):
        rows = binned_statistics(np.array([.3, .4, .8]), np.array([1., np.nan, 3.]),
                                 [.25, .75, 1.], threshold=10., min_entries=2)
        self.assertFalse(any(r['valid'] for r in rows))
        self.assertTrue(np.isnan(curve(rows, 'median')[1]).all())


if __name__ == '__main__':
    unittest.main()
