import unittest

import numpy as np

from train.downstream.campaign.plot_single_head_residuals import (
    align_reference, binned_statistics, curve,
)


class SingleHeadResidualPlotsTest(unittest.TestCase):
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
