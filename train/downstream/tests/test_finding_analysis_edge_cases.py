"""Review regressions: archive numeric contracts, pairing and output isolation."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from train.downstream.eval.finding_prediction_archive import load_archive
from train.downstream.eval import calibrate_track_finding as calibration
from train.downstream.eval import compare_track_finding_events as comparison
from train.downstream.tests.test_finding_analysis_tools import fixture, write_fixture


class ArchiveNumericContractTest(unittest.TestCase):
    def test_unsigned_labels_keep_the_signed_rejection_sentinel(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = write_fixture(root, 'unsigned.npz', truth=fixture()['truth'],
                                 pred=np.full(12, 7, dtype=np.uint64))
            archive = load_archive(path, require_scores=True)
            final = archive.final_predictions(.7)
            self.assertEqual(final.dtype.kind, 'i')
            self.assertEqual(final[9], -1)
            self.assertEqual(final[11], -1)
            full = calibration.score_full(archive, .7, comparison.metrics.MatchConfig(min_purity=.9, min_efficiency=.9))
            self.assertAlmostEqual(full['mc_track_dice_mean'], .9)
            self.assertEqual(full['n_qualified_tracks'], 1)

    def test_unrepresentable_labels_are_rejected_before_signed_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for key in ('truth', 'pred', 'point_ids'):
                values = np.arange(12, dtype=np.uint64)
                values[-1] = np.iinfo(np.uint64).max
                path = write_fixture(root, key + '.npz', **{key: values})
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'int64'):
                    load_archive(path)

    def test_negative_original_cluster_indices_are_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_fixture(Path(d), 'negative.npz', point_ids=np.arange(12)-1)
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                load_archive(path)

    def test_every_raw_provenance_field_must_align_to_all_points(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for field in ('background_only', 'ownership', 'origin_noise', 'purity'):
                raw = fixture(); raw.pop('provenance')
                raw.update(background_only=np.array([False]*10 + [True]*2),
                           ownership=np.array([1]*10 + [0]*2),
                           origin_noise=np.array([False]*10 + [True]*2),
                           purity=np.array([1.]*10 + [0.]*2))
                raw[field] = raw[field][:1]
                path = root / (field + '.npz'); np.savez(path, **raw)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    load_archive(path)

    def test_valid_raw_provenance_matches_explicit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = fixture(); raw.pop('provenance')
            raw.update(background_only=np.array([False]*10 + [True]*2),
                       ownership=np.array([1]*10 + [0]*2),
                       origin_noise=np.array([False]*10 + [True]*2),
                       purity=np.array([1.]*10 + [0.]*2))
            path = root/'raw.npz'; np.savez(path, **raw)
            np.testing.assert_array_equal(load_archive(path).arrays['provenance'], fixture()['provenance'])

    def test_complex_geometry_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for key, width in (('xyz', 3), ('endpoints', 6)):
                path = write_fixture(Path(d), key+'.npz', **{key: np.ones((12, width), dtype=complex)})
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    load_archive(path)


class BootstrapNumericContractTest(unittest.TestCase):
    def test_unsigned_counts_cannot_underflow(self):
        result = comparison.paired_efficiency_interval(np.array([1],dtype=np.uint64),
            np.array([2],dtype=np.uint64),np.array([2],dtype=np.uint64),resamples=10)
        self.assertEqual(result['reference_minus_model'], -.5)
        self.assertEqual(result['ci95'], [-.5,-.5])

    def test_counts_and_resample_count_are_integral(self):
        for args, kwargs in [(([.5],[0],[1]),{}), (([1],[0],[1]),dict(resamples=1.5)),
                             (([1],[0],[1]),dict(resamples=0)), (([1],[0],[1]),dict(seed=-1))]:
            with self.subTest(args=args,kwargs=kwargs),self.assertRaises(ValueError):
                comparison.paired_efficiency_interval(*args,**kwargs)

    def test_sparse_truth_bootstrap_reports_omitted_empty_replicates(self):
        result = comparison.paired_efficiency_interval([0,0,1],[0,0,0],[0,0,1],resamples=100,seed=4)
        self.assertEqual(result['reference_minus_model'],1.)
        self.assertEqual(result['ci95'],[1.,1.])
        self.assertGreater(result['valid_resamples'],0)
        self.assertLess(result['valid_resamples'],result['resamples'])


class WorkflowIsolationTest(unittest.TestCase):
    def test_comparison_integer_controls_are_validated_before_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            for controls in [dict(max_examples=1.5),dict(min_truth_clusters=.5),
                             dict(bootstrap_resamples=1.5),dict(seed=-1)]:
                with self.subTest(controls=controls),self.assertRaises(ValueError):
                    comparison.compare_events('missing.json',Path(d)/'out',focus_model='Model',**controls)

    def test_existing_calibration_output_fails_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(calibration,'load_archive') as read:
                with self.assertRaises(FileExistsError):
                    calibration.calibrate('missing.npz', d)
                read.assert_not_called()

    def test_existing_comparison_output_fails_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(comparison.Path,'read_text') as read:
                with self.assertRaises(FileExistsError):
                    comparison.compare_events('missing.json',d,focus_model='Model')
                read.assert_not_called()

    def test_no_truth_and_empty_event_comparison_has_defined_nulls(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); common=dict(truth=np.full(12,-1),pred=np.full(12,-1))
            write_fixture(root,'baseline.npz',**common);write_fixture(root,'model.npz',**common)
            plan=dict(baseline=dict(name='Reference',kind='final',path='baseline.npz'),
                      models=[dict(name='Model',kind='final',path='model.npz')])
            p=root/'manifest.json';p.write_text(json.dumps(plan))
            result=comparison.compare_events(p,root/'out',focus_model='Model',plots=False)
            self.assertEqual(result['selected'],[])
            self.assertIsNone(result['paired_quality_efficiency']['Model'])
            self.assertIsNone(result['quality_summaries']['Model']['quality_efficiency'])

    def test_quality_summary_keeps_configured_cuts_distinct_from_legacy_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_fixture(root,'baseline.npz',pred=np.array([7]*10+[-1]*2))
            write_fixture(root,'model.npz')
            plan=dict(baseline=dict(name='Reference',kind='final',path='baseline.npz'),
                      models=[dict(name='Model',kind='raw',path='model.npz',assignment_threshold=.2)])
            p=root/'manifest.json';p.write_text(json.dumps(plan))
            result=comparison.compare_events(p,root/'out',focus_model='Model',plots=False,bootstrap_resamples=10)
            self.assertEqual(result['summaries']['Model']['track_efficiency_global'],1.)
            self.assertEqual(result['quality_summaries']['Model']['n_qualified_tracks'],0)
            self.assertEqual(result['quality_summaries']['Model']['quality_efficiency'],0.)
            self.assertEqual(result['quality_summaries']['Reference']['quality_efficiency'],1.)
            self.assertEqual(result['selection_config']['min_truth_clusters'],4)

    def test_model_names_are_literal_text_in_geometry_and_geometry_free_plots(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for geometry in (False, True):
                extra = dict(xyz=np.column_stack((np.arange(12),np.zeros(12),np.arange(12)))) if geometry else {}
                write_fixture(root,'baseline.npz',pred=np.array([7]*10+[-1]*2),**extra)
                write_fixture(root,'model.npz',**extra)
                name = 'Adapter $\\not_a_math_macro$'
                plan=dict(baseline=dict(name='Reference',kind='final',path='baseline.npz'),
                          models=[dict(name=name,kind='raw',path='model.npz',assignment_threshold=.2)])
                p=root/'manifest.json';p.write_text(json.dumps(plan))
                out=root/str(geometry)
                import matplotlib
                with matplotlib.rc_context({'text.usetex': True}):
                    comparison.compare_events(p,out,focus_model=name,bootstrap_resamples=10)
                self.assertGreater((out/'example_01_overlay.png').stat().st_size,100)
                self.assertIn('not_a_math_macro',(out/'example_01_overlay.svg').read_text())

    def test_manifest_guards_prevent_ungated_or_unpaired_comparisons(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_fixture(root,'baseline.npz',pred=np.array([7]*10+[-1]*2))
            write_fixture(root,'model.npz')
            base=dict(name='Reference',kind='final',path='baseline.npz')
            model=dict(name='Model',kind='raw',path='model.npz',assignment_threshold=.2)
            cases=[({k:v for k,v in model.items() if k!='assignment_threshold'},'Raw predictions'),
                   ({**model,'kind':'final'},'final predictions'),
                   ({**model,'kind':'unknown'},'kind')]
            no_ids=fixture();no_ids.pop('point_ids');np.savez(root/'no_ids.npz',**no_ids)
            cases.append(({**model,'path':'no_ids.npz'},'original point_ids'))
            write_fixture(root,'other_events.npz',ordinal=2)
            cases.append(({**model,'path':'other_events.npz'},'physical event cohort'))
            for i,(spec,message) in enumerate(cases):
                manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(baseline=base,models=[spec])))
                out=root/f'case_{i}'
                with self.subTest(case=i),self.assertRaisesRegex(ValueError,message):
                    comparison.compare_events(manifest,out,focus_model='Model',plots=False)
                self.assertFalse(out.exists())

    def test_whole_comparison_is_invariant_to_variable_event_and_point_orders(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Three differently sized events, including an empty event.
            truth=np.array([0]*4+[-1]+[0]*6+[-1]*2)
            point_ids=np.r_[np.arange(5),np.arange(8)]
            common=dict(truth=truth,offsets=np.array([0,5,5,13]),point_ids=point_ids,
                        source_id=np.array([3,3,3]),original_ordinal=np.array([1,2,3]))
            pred=np.where(truth>=0,7,-1)
            np.savez(root/'baseline.npz',**common,pred=pred)
            np.savez(root/'model.npz',**common,pred=pred)
            order=np.r_[np.arange(5,13)[::-1],np.arange(5)[::-1]]
            reversed_event=dict(truth=truth[order],pred=pred[order],offsets=np.array([0,8,8,13]),
                point_ids=point_ids[order],source_id=np.array([3,3,3]),original_ordinal=np.array([3,2,1]))
            np.savez(root/'reordered.npz',**reversed_event)
            baseline=dict(name='Reference',kind='final',path='baseline.npz')
            summaries=[]
            for i,filename in enumerate(['model.npz','reordered.npz']):
                manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(baseline=baseline,
                    models=[dict(name='Model',kind='final',path=filename)])))
                result=comparison.compare_events(manifest,root/f'out_{i}',focus_model='Model',plots=False,bootstrap_resamples=20)
                summaries.append((result['summaries'],result['quality_summaries'],result['paired_quality_efficiency']))
            self.assertEqual(summaries[0],summaries[1])


if __name__=='__main__':
    unittest.main()
