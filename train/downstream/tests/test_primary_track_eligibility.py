"""Four-cluster eligibility shared by finding, noise, and offline rescoring."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from mmap_ninja import RaggedMmap

from fm4npp.datasets.track_eligibility import (
    PRIMARY_TRACK_POLICY, eligible_primary_labels, EligibleTrackLabelView,
    EligibleNoiseLabelView, configure_primary_track_cut, primary_track_eligibility_contract,
)
from fm4npp.datasets.cvt_downstream import CVTDownstreamSource, MIN4_NOISE_POLICY
from fm4npp.datasets.dataset import TPCBatchDataset, MyCollator, get_data_loader, get_val_loader
from fm4npp.utils import YParams
from train.downstream.tests.test_cvt_downstream import event, publish_fixture
from train.downstream.noise_metrics import training_class_counts
from train.downstream.track_finding_targets import build_track_instance_targets
from train.downstream.eval.rescore_track_finding import rescore_archive
from train.downstream.track_finding_contract import finding_evaluation_metadata


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "scripts/configs/clas12_adapter_recipes.yaml"


def boundary_event(ordinal=0):
    value = event(ordinal)
    for name, data in list(value.items()):
        if name.startswith("clusters/"):
            value[name] = data[np.arange(9) % 8]
    value["clusters/primary_segment"] = np.array([12] * 3 + [35] * 4 + [-1] * 2)
    value["clusters/primary_tid"] = np.array([42] * 3 + [7] * 4 + [-1] * 2)
    value["clusters/ownership"] = np.array([1] * 7 + [0, 3])
    value["clusters/background_only"] = np.array([False] * 7 + [True, False])
    value["clusters/origin_noise"] = np.array([False] * 7 + [True, False])
    value["clusters/purity"] = np.array([1.] * 7 + [0., 0.])
    value["clusters/cluster_id"] = np.arange(9)
    value["clusters/xyz"] = np.column_stack((np.arange(9) + 7., np.zeros(9), np.arange(9))).astype('f4')
    return value


class PrimaryTrackEligibilityTest(unittest.TestCase):
    def test_boundary_sparse_ids_negatives_idempotence_and_no_mutation(self):
        original = np.array([0] + [7] * 2 + [101] * 3 + [900] * 4 + [5] * 5 + [-1, -100])
        saved = original.copy()
        expected = np.array([-1] * 6 + [900] * 4 + [5] * 5 + [-1, -100])
        actual = eligible_primary_labels(original)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(eligible_primary_labels(actual), expected)
        np.testing.assert_array_equal(original, saved)
        self.assertEqual(eligible_primary_labels(np.array([], dtype=int)).size, 0)
        for bad in (np.zeros((1, 3), dtype=int), np.array([0., 1.])):
            with self.assertRaises(ValueError):
                eligible_primary_labels(bad)

    def test_event_local_counting_and_legacy_noise_preservation(self):
        labels = [np.array([5, 5, -1]), np.array([5, 5, 5, 5, -1])]
        seg = EligibleTrackLabelView(labels)
        np.testing.assert_array_equal(seg[0], [-1, -1, -1])
        np.testing.assert_array_equal(seg[1], labels[1])
        noise = [np.array([0, 0, 1]), np.array([0, 0, 0, 0, 1])]
        view = EligibleNoiseLabelView(labels, noise)
        np.testing.assert_array_equal(view[0], [1, 1, 1])
        np.testing.assert_array_equal(view[1], noise[1])
        np.testing.assert_array_equal(noise[0], [0, 0, 1])
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(view))[0], view[0])

    def test_adapter_entrypoints_enable_cut_and_leave_pid_alone(self):
        from train.downstream.track_finding_experiment import TrackFindingExperimentConfig, resolve_params
        from train.downstream.track_finding_trainer import DownstreamTrainer as Finding
        from train.downstream.point_classification_trainer import DownstreamTrainer as Noise
        params = resolve_params(TrackFindingExperimentConfig(yaml_config=str(RECIPE), config='hdf5_finding_5k'))
        self.assertTrue(params.params['apply_primary_track_cut'])
        with tempfile.TemporaryDirectory() as tmp, patch('torch.cuda.is_available', return_value=False):
            args = SimpleNamespace(root_dir=tmp, global_log_dir='logs', config='test', run_num='0')
            for cls, profile in ((Finding, 'hdf5_finding_5k'), (Noise, 'hdf5_noise_50k_balanced')):
                params = YParams(str(RECIPE), profile)
                trainer = cls(params, args)
                self.assertTrue(trainer.params.apply_primary_track_cut)
                self.assertEqual(params.params['track_eligibility_contract']['policy'], PRIMARY_TRACK_POLICY)
            params = YParams(str(RECIPE), 'hdf5_noise_50k_balanced')
            params['task'] = 'pid'
            Noise(params, args)
            self.assertFalse(getattr(params, 'apply_primary_track_cut', False))
        original_tpc = SimpleNamespace(input_layout='exyz')
        configure_primary_track_cut(original_tpc)
        self.assertFalse(original_tpc.apply_primary_track_cut)

    def test_hdf5_cut_matches_finding_noise_and_leaves_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = boundary_event()
            publish_fixture(root, {'train': [[raw]]})
            original = CVTDownstreamSource(root, 'train')
            source = CVTDownstreamSource(root, 'train', apply_primary_track_cut=True)
            expected = np.array([-1] * 3 + [35] * 4 + [-1] * 2)
            np.testing.assert_array_equal(source.point_field(0, 'seg_target'), expected)
            np.testing.assert_array_equal(source.point_field(0, 'noise_target'), expected < 0)
            np.testing.assert_array_equal(original.point_field(0, 'seg_target'), raw['clusters/primary_segment'])
            np.testing.assert_array_equal(source.read(0, ['clusters.primary_segment'])[0], raw['clusters/primary_segment'])
            np.testing.assert_array_equal(source.read(0, ['clusters.ownership'])[0], raw['clusters/ownership'])
            for name in ('features', 'reg_target', 'pid_target', 'coatjava_seg_pred'):
                np.testing.assert_array_equal(source.point_field(0, name), original.point_field(0, name))
            self.assertEqual(source.contract['noise_policy'], MIN4_NOISE_POLICY)
            self.assertEqual(source.contract['primary_track_min_clusters'], 4)
            labels = torch.as_tensor(expected).unsqueeze(0)
            targets, _ = build_track_instance_targets(labels, torch.ones_like(labels, dtype=torch.bool), mode='signal_only')
            self.assertEqual(tuple(targets[0]['masks'].shape), (1, 9))
            self.assertEqual(targets[0]['masks'].sum().item(), 4)
            source.close(); original.close()

    def test_both_storage_formats_sort_chunk_and_pad_after_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); hdf = root/'hdf'; hdf.mkdir(); ragged = root/'ragged'; ragged.mkdir()
            raw = boundary_event(); publish_fixture(hdf, {'train': [[raw]]})
            source = CVTDownstreamSource(hdf, 'train')
            for stem in source.fields:
                RaggedMmap.from_lists(ragged / f'{stem}_train', [source.point_field(0, stem)])
            samples = []
            for storage, reader in ((hdf, 'hdf5'), (ragged, 'ragged_mmap')):
                kwargs = dict(split='train', reader_type=reader, normalize=False, voxelize=False,
                              return_dict=True, compute_neighbors=False, apply_primary_track_cut=True,
                              input_representation='clas12_pos_plus_aux_v1', return_coatjava_seg_pred=True)
                ds = TPCBatchDataset(storage, **kwargs)
                self.assertEqual(finding_evaluation_metadata(ds)['track_eligibility_contract'],
                                 primary_track_eligibility_contract())
                raw_ds = TPCBatchDataset(storage, **{**kwargs, 'apply_primary_track_cut': False})
                self.assertIsNone(raw_ds.track_eligibility_contract)
                with self.assertRaisesRegex(ValueError, 'track_eligibility_contract'):
                    finding_evaluation_metadata(raw_ds)
                item = ds[0]; samples.append(item)
                np.testing.assert_array_equal(training_class_counts(ds), [4, 5])
                self.assertEqual(int((item['target'] >= 0).sum()), 4)
                torch.testing.assert_close(item['noise_target'], (item['target'] < 0).long())
                # The full four-cluster track stays eligible in two-cluster chunks.
                chunked = TPCBatchDataset(storage, train=False, chunk_training=True, len_chunk=2, **kwargs)
                saw_long_track = False
                for i in range(len(chunked)):
                    part = chunked[i]; ids = part['token_context'][:, 3].numpy()
                    expected = np.where((ids >= 3) & (ids < 7), 35, -1)
                    np.testing.assert_array_equal(part['target'].numpy(), expected)
                    np.testing.assert_array_equal(part['noise_target'].numpy(), expected < 0)
                    saw_long_track |= bool(np.any(expected == 35))
                self.assertTrue(saw_long_track)
                batch = MyCollator()([item, chunked[0]])
                self.assertTrue(torch.all(batch['target'][1, 2:] == -100))
                self.assertTrue(torch.all(batch['noise_target'][1, 2:] == -100))
            for name in ('target', 'noise_target', 'points', 'coatjava_seg_pred'):
                torch.testing.assert_close(samples[0][name], samples[1][name])
            source.close()

    def test_all_three_factory_splits_use_cut_and_preserve_event_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            publish_fixture(root, {s: [[boundary_event(i)]] for i, s in enumerate(('train','validation','test'))})
            params = YParams(str(RECIPE), 'hdf5_finding_5k')
            params.update_params(dict(data_root=str(root), data_root_test=str(root), num_data_workers=0,
                local_batch_size=1, local_valid_batch_size=1, limit_size=1, limit_test_size=1,
                drop_last_test=False, compute_neighbors=False, pretrained_ckpt=None))
            configure_primary_track_cut(params)
            train, _, validation, _ = get_data_loader(params, False)
            test = get_val_loader(params, False)
            for loader, split in ((train,'train'), (validation,'validation'), (test,'test')):
                self.assertEqual(loader.dataset.split, split)
                y = loader.dataset.memmap_seg_target[0]
                self.assertEqual(len(y), 9)
                np.testing.assert_array_equal(np.unique(y), [-1, 35])
                self.assertEqual(len(loader.dataset), 1)
            self.assertEqual(params.data_contract['noise_policy'], MIN4_NOISE_POLICY)

    def test_rescoring_policy_is_explicit_and_keeps_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'predictions.npz'
            truth = np.array([12]*3 + [35]*4 + [-1])
            pred = np.array([6]*3 + [7]*4 + [-1])
            np.savez(path, offsets=[0, 8], truth=truth, pred=pred)
            old = rescore_archive(path)
            new = rescore_archive(path, truth_policy=PRIMARY_TRACK_POLICY)
            self.assertEqual(old['metrics']['pred']['n_true_tracks'], 2)
            self.assertEqual(new['metrics']['pred']['n_true_tracks'], 1)
            self.assertEqual(new['metrics']['pred']['n_pred_tracks'], 2)
            self.assertEqual(new['metrics']['pred']['mc_track_dice_mean'], 1)
            self.assertEqual(new['metrics']['pred']['n_non_target_only_pred_tracks'], 1)
            self.assertEqual(new['receipt']['n_relabelled_truth_points'], 3)
            self.assertEqual(new['receipt']['truth_policy'], PRIMARY_TRACK_POLICY)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['truth'], truth)

    def test_hdf5_rescoring_accepts_raw_or_projected_labels_only_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = boundary_event()
            publish_fixture(root, {'test': [[raw]]})
            path = root/'pred.npz'; perm = np.array([3, 8, 1, 4, 0, 7, 5, 2, 6])
            values = dict(offsets=[0, 9], pred=np.array([7]*7 + [9, -1])[perm], point_ids=perm,
                          source_event_index=[0], source_id=[17], original_ordinal=[0])
            kwargs = dict(hdf5_root=root, split='test', plan_id='fixture', truth_policy=PRIMARY_TRACK_POLICY)
            results = []
            for truth in (raw['clusters/primary_segment'], eligible_primary_labels(raw['clusters/primary_segment'])):
                np.savez(path, truth=truth[perm], **values)
                results.append(rescore_archive(path, **kwargs)['metrics']['pred'])
            self.assertEqual(results[0], results[1])
            s = results[0]
            self.assertEqual(s['n_truth_tracks_below_policy_min_clusters'], 0)
            self.assertEqual(s['n_mc_nontarget_mc_only_points'], 3)
            self.assertEqual(s['mc_bg_contamination_global'], 0)
            self.assertAlmostEqual(s['mc_hit_impurity_global'], 3/7)
            with self.assertRaisesRegex(ValueError, 'truth label mismatch'):
                rescore_archive(path, **{**kwargs, 'truth_policy':'as_saved'})

    def test_all_short_tracks_keep_event_and_become_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); publish_fixture(root, {'train': [[event()]]})
            ds = TPCBatchDataset(root, split='train', reader_type='hdf5', apply_primary_track_cut=True,
                                 normalize=False, voxelize=False, return_dict=True, compute_neighbors=False)
            item = ds[0]
            self.assertEqual(len(ds), 1)
            self.assertEqual(len(item['target']), 8)
            self.assertTrue(torch.all(item['target'] == -1))
            self.assertTrue(torch.all(item['noise_target'] == 1))
            labels = item['target'].unsqueeze(0)
            targets, _ = build_track_instance_targets(labels, torch.ones_like(labels, dtype=torch.bool), mode='signal_only')
            self.assertEqual(tuple(targets[0]['masks'].shape), (0, 8))
            from train.downstream.loss import PointHungarianMatcher, compute_point_loss
            logits = torch.zeros(1, 2, 2, requires_grad=True)
            outputs = {'pred_probs': logits.softmax(-1), 'pred_masks': torch.full((1, 2, 8), .5)}
            losses = compute_point_loss(outputs, targets, torch.ones_like(labels, dtype=torch.bool), PointHungarianMatcher())
            self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
            sum(losses.values()).backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(logits.grad.abs().sum().item(), 0)
            noise_logits = torch.zeros(8, 2, requires_grad=True)
            noise_loss = torch.nn.functional.cross_entropy(noise_logits, item['noise_target'])
            noise_loss.backward()
            self.assertTrue(torch.isfinite(noise_logits.grad).all())


if __name__ == '__main__':
    unittest.main()
