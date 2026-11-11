"""CPU regression tests for validation calibration and paired event diagnostics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from train.downstream import track_finding_metrics as metrics
from train.downstream.eval import calibrate_track_finding as calibration
from train.downstream.eval import compare_track_finding_events as comparison
from train.downstream.eval.finding_prediction_archive import load_archive, align_to
from train.downstream.eval.plot_finding_events import measurement_support


def fixture(ordinal=1):
    return dict(truth=np.array([0]*10 + [-1]*2), pred=np.array([7]*12),
        offsets=np.array([0,12]), scores=np.array([.9]*9 + [.4,.9,.6]),
        signal=np.ones(12, dtype=bool), point_ids=np.arange(12),
        source_id=np.array([21]), original_ordinal=np.array([ordinal]),
        provenance=np.array([1]*10 + [2]*2))


def write_fixture(root, name, ordinal=1, **updates):
    data = {**fixture(ordinal), **updates}; path = root / name
    np.savez(path, **data)
    return path


class FindingArchiveTest(unittest.TestCase):
    def test_native_gate_negative_inactive_and_strict_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); data = fixture(); data['signal'][-1] = False; data['scores'][-1] = -.1
            path = write_fixture(root, 'a.npz', **data)
            archive = load_archive(path, require_scores=True)
            self.assertEqual(archive.final_predictions(.9).tolist(), [-1]*12)
            self.assertEqual(archive.final_predictions(.2)[-1], -1)
            data['signal'][-1] = True; np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, 'assignment scores'):
                load_archive(path, require_scores=True)

    def test_invalid_shapes_offsets_ids_provenance_and_sidecars(self):
        cases = [dict(offsets=np.array([1,12])), dict(offsets=np.array([0,13])),
                 dict(truth=np.zeros((12,1), dtype=int)), dict(scores=np.full(12,np.nan)),
                 dict(signal=np.ones(12,dtype=int)), dict(point_ids=np.zeros(12,dtype=int)),
                 dict(provenance=np.full(12,99)), dict(provenance=np.full(12,2))]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i, updates in enumerate(cases):
                path = write_fixture(root, f'{i}.npz', **updates)
                with self.subTest(case=i), self.assertRaises(ValueError):
                    load_archive(path, require_scores=True)
            path = write_fixture(root, 'raw.npz')
            meta = write_fixture(root, 'meta.npz', truth=np.array([1]*10+[-1]*2))
            with self.assertRaisesRegex(ValueError, 'alignment mismatch'):
                load_archive(path, metadata=meta)
            meta = write_fixture(root, 'meta2.npz', point_ids=np.arange(12)[::-1])
            with self.assertRaisesRegex(ValueError, 'field mismatch'):
                load_archive(path, metadata=meta)

    def test_explicit_truth_policy_and_full_event_membership(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path = write_fixture(root,'a.npz',truth=np.array([0]*3+[1]*7+[-1]*2))
            self.assertEqual(load_archive(path).arrays['truth'][0], 0)
            self.assertEqual(load_archive(path,truth_policy='primary_mc_min4_v1').arrays['truth'][:3].tolist(),[-1]*3)
            path = write_fixture(root,'crop.npz',point_ids=np.arange(12)+1)
            with self.assertRaisesRegex(ValueError, 'full-event'):
                load_archive(path,truth_policy='primary_mc_min4_v1')

    def test_ambiguous_provenance_and_geometry_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i,updates in enumerate([dict(background_only=np.zeros(12,dtype=bool)),
                                       dict(xyz=np.full((12,3),np.nan)),dict(endpoints=np.zeros((12,2,3)))]):
                path=write_fixture(root,f'{i}.npz',**updates)
                with self.subTest(i=i),self.assertRaises(ValueError):load_archive(path)

    def test_pair_alignment_reorders_physical_events_and_points(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); data=fixture()
            two={k:np.concatenate([v,v]) for k,v in data.items() if k not in ('offsets','source_id','original_ordinal')}
            two.update(offsets=np.array([0,12,24]),source_id=np.array([21,21]),original_ordinal=np.array([1,2]))
            a=write_fixture(root,'a.npz',**two); ix=np.r_[np.arange(12,24)[::-1],np.arange(12)[::-1]]
            reverse={k:v[ix] if len(v)==24 else v[::-1] if k in ('source_id','original_ordinal') else v for k,v in two.items()}
            b=write_fixture(root,'b.npz',**reverse)
            first,second=load_archive(a),load_archive(b)
            order=align_to(first,second)
            np.testing.assert_array_equal(second.arrays['point_ids'][order],first.arrays['point_ids'])
            second.arrays['truth'][0]=99
            with self.assertRaisesRegex(ValueError,'Truth alignment'):
                align_to(first,second)
            second=load_archive(b);second.arrays['original_ordinal'][0]=3
            with self.assertRaisesRegex(ValueError,'physical event cohort'):
                align_to(first,second)


class CalibrationTest(unittest.TestCase):
    def test_cached_matches_public_evaluators_with_sparse_ids_and_ties(self):
        rng=np.random.default_rng(92)
        configs=[metrics.MatchConfig(min_purity=p,min_efficiency=c,iou_threshold=i)
                 for p,c,i in ((.9,.9,.5),(.5,.5,.5),(.8,.95,.2),(0.,0.,0.))]
        for case in range(50):
            n=int(rng.integers(0,70)); truth=rng.choice([-1,3,11,52],n)
            pred=np.where(truth>=0,truth+100,800); changed=rng.random(n)<.2
            pred[changed]=rng.choice([-1,7,80],int(changed.sum()))
            event=dict(truth=truth,pred=pred,scores=rng.choice(np.array([0,.2,.9,1],np.float32),n),
                       signal=rng.random(n)>.1,point_ids=rng.permutation(n))
            cache=calibration.cache_event(event)
            for config in configs:
                for t in (0.,.2,.9,1.):
                    final=np.where(event['signal']&(event['scores']>t),pred,-1)
                    row=metrics.event_mc_reconstruction_metrics(truth,final,point_ids=event['point_ids'])
                    quality=metrics.event_track_metrics(truth,final,config=config)
                    actual=calibration.score_cached(cache,t,config)
                    self.assertAlmostEqual(actual[0],row['mc_track_dice_sum'])
                    self.assertEqual(actual[1:],(row['n_mc_correct_points'],row['n_mc_associated_pred_points'],quality['n_matched_tracks']))

    def test_dice_and_quality_objectives_select_different_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=write_fixture(root,'validation.npz')
            for objective,expected in [('dice',.2),('quality-efficiency',.7)]:
                result=calibration.calibrate(path,root/objective,objective=objective,thresholds=[.2,.5,.7,1],refine_steps=())
                self.assertEqual(result['selection']['selected']['threshold'],expected)
                self.assertEqual(result['selection']['validation_metrics']['n_true_tracks'],1)
                self.assertIsNone(result['heldout'])
            with self.assertRaises(FileExistsError):
                calibration.calibrate(path,root/'dice',thresholds=[.2],refine_steps=())

    def test_inclusive_quality_boundary_split_merge_and_miss(self):
        for truth,pred,count in [([0]*10+[-1],[7]*9+[-1,7],1),([0]*10,[7]*5+[8]*5,0),
                                  ([0]*10+[1]*10,[7]*20,0),([0]*10,[-1]*10,0)]:
            event=dict(truth=np.array(truth),pred=np.array(pred),signal=np.array(pred)>=0,
                       scores=np.ones(len(truth)),point_ids=np.arange(len(truth)))
            actual=calibration.score_cached(calibration.cache_event(event),.2,
                metrics.MatchConfig(min_purity=.9,min_efficiency=.9))
            self.assertEqual(actual[3],count)

    def test_selection_persisted_before_test_access_and_test_does_not_select(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); val=write_fixture(root,'val.npz',ordinal=1)
            test=write_fixture(root,'test.npz',ordinal=2,scores=np.zeros(12));out=root/'out'
            original=calibration.load_archive
            def checked(path,**kwargs):
                if path==test:
                    self.assertTrue((out/'selection.json').exists())
                return original(path,**kwargs)
            with patch.object(calibration,'load_archive',side_effect=checked):
                result=calibration.calibrate(val,out,test=test,thresholds=[.2,.5,.7],refine_steps=())
            self.assertEqual(result['selection']['selected']['threshold'],.2)
            self.assertEqual(result['heldout']['metrics']['mc_track_dice_mean'],0.)
            with self.assertRaisesRegex(ValueError,'overlap'):
                calibration.calibrate(val,root/'overlap',test=val,thresholds=[.2],refine_steps=())
            self.assertFalse((root/'overlap/summary.json').exists())

    def test_invalid_thresholds_no_truth_and_unknown_objective(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); val=write_fixture(root,'val.npz')
            for kw in [dict(thresholds=[]),dict(thresholds=[np.nan]),dict(refine_steps=[0]),dict(objective='purity'),dict(min_purity=1.1)]:
                with self.subTest(kw=kw),self.assertRaises(ValueError):
                    calibration.calibrate(val,root/'out',**kw)
            val=write_fixture(root,'empty.npz',truth=np.full(12,-1))
            with self.assertRaisesRegex(ValueError,'eligible truth'):
                calibration.calibrate(val,root/'out')


class PairedDiagnosticsTest(unittest.TestCase):
    def test_selection_advances_past_duplicates_to_preserve_category_diversity(self):
        a=dict(event_index=0,true_id=1);b=dict(event_index=1,true_id=1)
        c=dict(event_index=2,true_id=1);d=dict(event_index=3,true_id=1)
        chosen=comparison.select_examples({'overlay':[a,b],'fragmentation':[a,c],'mixing':[d]},3)
        self.assertEqual([x['category'] for x in chosen],['overlay','fragmentation','mixing'])
        self.assertEqual([x['event_index'] for x in chosen],[0,2,3])

    def test_bmt_arc_and_bst_line_preserve_measurement_supports(self):
        event=dict(endpoints=np.array([[2.,0.,3.,0.,2.,3.],[1.,2.,-5.,1.,2.,7.]]),detector=np.array([1,2]))
        arc=measurement_support(event,0)
        np.testing.assert_allclose(np.hypot(arc[:,0],arc[:,1]),2.)
        np.testing.assert_allclose(arc[:,2],3.)
        line=measurement_support(event,1)
        np.testing.assert_allclose(line[:,0],1.);np.testing.assert_allclose(line[:,1],2.)
        self.assertEqual(line[0,2],-5.);self.assertEqual(line[-1,2],7.)

    def test_failure_diagnostics_and_undefined_purity(self):
        truth=np.array([0]*10+[-1]*2);baseline=np.array([7]*10+[-1]*2)
        config=metrics.MatchConfig(min_purity=.9,min_efficiency=.9)
        reference=comparison.track_diagnostics(truth,baseline,config=config)[0]['tracks'][0]
        contaminated=comparison.track_diagnostics(truth,np.array([8]*12),provenance=np.array([1]*10+[2]*2),config=config)[0]['tracks'][0]
        self.assertEqual(comparison.failure_categories(reference,contaminated,config),['overlay'])
        unknown=comparison.track_diagnostics(truth,np.array([8]*12),config=config)[0]['tracks'][0]
        self.assertIsNone(unknown['overlay'])
        self.assertEqual(comparison.failure_categories(reference,unknown,config),[])
        missed=comparison.track_diagnostics(truth,np.full(12,-1),config=config)[0]['tracks'][0]
        self.assertIsNone(missed['purity']);self.assertIn('miss',comparison.failure_categories(reference,missed,config))

    def test_quality_matching_is_independent_of_dice_association(self):
        y=np.array([0]*4+[1]*2+[0]*2);pred=np.array([10]*6+[11]*2)
        detail,_=comparison.track_diagnostics(y,pred,config=metrics.MatchConfig())
        by_id={t['true_id']:t for t in detail['tracks']}
        self.assertEqual(by_id[0]['qualified_pred_id'],10)
        self.assertEqual(by_id[0]['pred_id'],11)

    def test_bootstrap_preserves_pooled_weighting_and_pairing(self):
        result=comparison.paired_efficiency_interval([0,8],[0,4],[1,9],resamples=100,seed=8)
        self.assertAlmostEqual(result['reference_minus_model'],.4)
        self.assertEqual(result,comparison.paired_efficiency_interval([0,8],[0,4],[1,9],resamples=100,seed=8))
        zero=comparison.paired_efficiency_interval([1,9],[1,9],[1,9],resamples=20)
        self.assertEqual(zero['ci95'],[0.,0.])

    def test_cli_roundtrip_arbitrary_names_and_geometry_free_plots(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_fixture(root,'raw.npz');write_fixture(root,'baseline.npz',pred=np.array([7]*10+[-1]*2))
            plan=dict(baseline=dict(name='Reference reconstruction',kind='final',path='baseline.npz'),
                      models=[dict(name='New backbone',kind='raw',path='raw.npz',assignment_threshold=.2)])
            manifest=root/'manifest.json';manifest.write_text(json.dumps(plan))
            run=subprocess.run([sys.executable,'-m','train.downstream.eval.compare_track_finding_events',
                '--manifest',str(manifest),'--focus-model','New backbone','--output-dir',str(root/'plots'),
                '--bootstrap-resamples','20'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            result=json.loads((root/'plots/comparison.json').read_text())
            self.assertEqual(result['selected'][0]['category'],'overlay')
            for ext in ('png','svg'):
                self.assertGreater((root/f'plots/example_01_overlay.{ext}').stat().st_size,100)
            run=subprocess.run([sys.executable,'-m','train.downstream.eval.calibrate_track_finding',
                '--validation',str(root/'raw.npz'),'--objective','quality-efficiency','--output-dir',str(root/'tune'),
                '--thresholds','.2','.5','.7','--refine-steps'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(json.loads((root/'tune/selection.json').read_text())['selected']['threshold'],.7)


if __name__=='__main__':
    unittest.main()
