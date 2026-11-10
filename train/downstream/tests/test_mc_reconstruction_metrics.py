"""CPU-only scientific contract tests; no trainer, Torch or detector dependency."""

import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from train.downstream.track_finding_metrics import (
    ClusterProvenance as P, EVENT_MEAN_METRICS, MatchConfig,
    MCReconstructionAccumulator, cluster_provenance_from_hdf5,
    event_mc_reconstruction_metrics as score, event_track_metrics,
    summarize_event_metrics, summarize_mc_reconstruction_metrics as summarize,
)
from train.downstream.eval.rescore_track_finding import rescore_archive


def row(y, p, **kwargs):
    return score(np.asarray(y, dtype=np.int64), np.asarray(p, dtype=np.int64), **kwargs)


def summary(y, p, **kwargs):
    return summarize([row(y, p, **kwargs)])


class MCReconstructionTest(unittest.TestCase):
    def test_perfect_and_independent_non_target_predictions(self):
        y = [0] * 4 + [1] * 4 + [-1] * 4
        outputs = [summary(y, [7] * 4 + [9] * 4 + tail, provenance=[P.MC_ONLY] * 8 + [P.OVERLAY_ONLY] * 4)
                   for tail in ([-1] * 4, [2] * 4, [2, 3, 4, 5])]
        for s in outputs:
            for key in ("mc_track_dice_mean", "mc_hit_completeness_global", "mc_hit_purity_global", "track_efficiency_global"):
                self.assertEqual(s[key], 1)
            self.assertEqual(s["mc_bg_contamination_global"], 0)
            self.assertEqual(s["bg_leakage_to_mc_global"], 0)
        self.assertEqual(outputs[1]["n_non_target_only_pred_tracks"], 1)
        self.assertEqual(outputs[2]["n_non_target_only_pred_tracks"], 4)
        self.assertGreater(outputs[1]["fake_rate"], outputs[0]["fake_rate"])

    def test_split_merge_and_miss(self):
        split = summary([0] * 4, [7, 7, 8, 8])
        self.assertAlmostEqual(split["mc_track_dice_mean"], 2/3)
        self.assertEqual(split["mc_hit_completeness_global"], .5)
        self.assertEqual(split["mc_hit_purity_global"], 1)
        self.assertEqual(split["split_rate_global"], 1)
        self.assertEqual(split["n_mc_overlap_unassociated_pred_tracks"], 1)
        self.assertEqual(split["track_efficiency_global"], 1)  # inclusive >= .5
        merged = summary([0] * 4 + [1] * 4, [7] * 8)
        self.assertAlmostEqual(merged["mc_track_dice_mean"], 1/3)
        self.assertEqual(merged["mc_hit_completeness_global"], .5)
        self.assertEqual(merged["mc_hit_purity_global"], .5)
        self.assertEqual(merged["mc_other_track_contamination_global"], .5)
        self.assertEqual(merged["merge_rate_global"], 1)
        missed = summary([0] * 4, [-1] * 4)
        self.assertEqual(missed["mc_track_dice_mean"], 0)
        self.assertEqual(missed["mc_hit_completeness_global"], 0)
        self.assertIsNone(missed["mc_hit_purity_global"])
        self.assertEqual(missed["ari_signal"], 1)  # legacy degeneracy, deliberately retained

    def test_independent_matching_and_unchanged_iou_cut(self):
        y, p = [0] * 4 + [1] * 2 + [0] * 2, [10] * 6 + [11] * 2
        r = row(y, p)
        self.assertEqual([(m["pred_id"], m["true_id"]) for m in r["matches"]], [(10, 0)])
        self.assertEqual([(m["pred_id"], m["true_id"]) for m in r["dice_matches"]], [(10, 1), (11, 0)])
        self.assertEqual(r["mc_track_dice_sum"], 1)
        # Three of four truth hits in a five-hit prediction gives IoU=.5.
        boundary = summary([0] * 4 + [-1] * 2, [8] * 3 + [-1] + [8] * 2)
        self.assertEqual(boundary["track_efficiency_global"], 1)
        below = summary([0] * 5 + [-1] * 2, [8] * 3 + [-1] * 2 + [8] * 2)
        self.assertEqual(below["track_efficiency_global"], 0)  # purity=.6, completeness=.6, IoU=3/7
        self.assertAlmostEqual(below["mc_track_dice_mean"], .6)

    def test_provenance_decomposition(self):
        y = [0] * 4 + [1] * 4 + [-1] * 5
        p = [7] * 4 + [8] * 4 + [7, 7, 7, 7, 9]
        provenance = [P.MC_ONLY] * 8 + [P.MC_ONLY, P.OVERLAY_ONLY, P.MIXED, P.MC_PRESENT_UNRESOLVED, P.OVERLAY_ONLY]
        s = summary(y, p, provenance=provenance)
        self.assertAlmostEqual(s["mc_track_dice_mean"], 5/6)
        self.assertEqual(s["mc_hit_completeness_global"], 1)
        self.assertAlmostEqual(s["mc_hit_purity_global"], 8/12)
        self.assertAlmostEqual(s["mc_hit_impurity_global"], 4/12)
        self.assertAlmostEqual(s["mc_bg_contamination_global"], 1/12)
        self.assertEqual(s["bg_leakage_to_mc_global"], .5)
        self.assertEqual(s["n_mc_nontarget_mc_only_points"], 1)
        self.assertEqual(s["n_mc_nontarget_mixed_points"], 1)
        self.assertEqual(s["n_mc_nontarget_mc_present_unresolved_points"], 1)
        self.assertEqual(s["n_mc_correct_points"] + s["n_mc_other_track_points"] + s["n_mc_nontarget_points"], s["n_mc_associated_pred_points"])
        self.assertEqual(sum(s[f"n_mc_associated_{k.name.lower()}_points"] for k in P), s["n_mc_associated_pred_points"])

    def test_unknown_is_not_overlay_and_missing_metrics_are_null(self):
        s = summary([0] * 4 + [-1], [2] * 5)
        self.assertAlmostEqual(s["mc_track_dice_mean"], 8/9)
        self.assertIsNone(s["mc_bg_contamination_global"])
        self.assertIsNone(s["bg_leakage_to_mc_global"])
        self.assertEqual(s["mc_bg_contamination_lower_bound_global"], 0)
        self.assertEqual(s["mc_provenance_coverage_global"], 0)
        s = summary([0] * 4 + [-1], [2] * 4 + [3], provenance=[P.MC_ONLY] * 4 + [P.UNKNOWN])
        self.assertEqual(s["mc_bg_contamination_global"], 0)
        self.assertIsNone(s["bg_leakage_to_mc_global"])

    def test_zero_denominators_and_empty_events(self):
        for y, p in (([], []), ([-1] * 4, [8] * 4)):
            s = summary(y, p)
            for key in ("mc_track_dice_mean", "mc_hit_completeness_global", "mc_hit_purity_global", "track_efficiency_global"):
                self.assertIsNone(s[key])
            self.assertEqual(s["n_mc_dice_matches"], 0)
        self.assertIsNone(summarize([])["mc_track_dice_mean"])
        s = summary([0] * 4, [8] * 4, provenance=[P.MC_ONLY] * 4)
        self.assertEqual(s["mc_bg_contamination_global"], 0)
        self.assertIsNone(s["bg_leakage_to_mc_global"])

    def test_truth_track_weighting_and_hit_weighting_are_distinct(self):
        # One perfect 4-cluster track, three missed 8-cluster tracks.
        s = summarize([row([0] * 4, [1] * 4), row([0] * 8 + [1] * 8 + [2] * 8, [-1] * 24)])
        self.assertEqual(s["mc_track_dice_mean"], .25)
        self.assertEqual(s["mc_track_dice_event_macro"], .5)
        self.assertAlmostEqual(s["mc_hit_completeness_global"], 4/28)
        self.assertEqual(s["mc_hit_completeness_event_macro"], .5)

    def test_masks_labels_input_immutability_and_policy_audit(self):
        y = np.array([0, 0, -1, -100]); p = np.array([2, 2, 3, 100]); saved = y.copy(), p.copy()
        r = score(y, p, valid_mask=[1, 1, 1, 0], pred_signal_mask=[1, 0, 1, 1])
        self.assertEqual(r["n_true_tracks"], 1)
        self.assertEqual(r["n_truth_tracks_below_policy_min_clusters"], 1)
        self.assertEqual(r["n_mc_correct_points"], 1)
        np.testing.assert_array_equal(y, saved[0]); np.testing.assert_array_equal(p, saved[1])
        with self.assertRaisesRegex(ValueError, "negative label"):
            score(y, p)

    def test_original_point_ids_make_ties_stable_under_relabel_and_permutation(self):
        y = np.array([0] * 4 + [1] * 4); p = np.array([7] * 8); points = np.arange(8)
        first = score(y, p, point_ids=points)
        perm = np.array([6, 1, 4, 0, 7, 3, 2, 5])
        second = score(np.where(y[perm] == 0, 55, 3), np.full(8, 900), point_ids=points[perm])
        self.assertEqual(first["dice_matches"][0]["true_id"], 0)
        self.assertEqual(second["dice_matches"][0]["true_id"], 55)
        self.assertEqual(first["mc_event_metrics"], second["mc_event_metrics"])

    def test_hungarian_maximizes_sum_against_exhaustive_oracle(self):
        rng = np.random.default_rng(712)
        for _ in range(80):
            y, p = rng.integers(-1, 4, (2, 18))
            r = score(y, p)
            tids, pids = np.unique(y[y >= 0]), np.unique(p[p >= 0])
            weights = np.array([[2 * np.count_nonzero((y == t) & (p == j)) /
                                 (np.count_nonzero(y == t) + np.count_nonzero(p == j))
                                 for t in tids] for j in pids])
            best = 0
            if weights.size:
                a = weights if len(pids) <= len(tids) else weights.T
                best = max(sum(a[i, j] for i, j in enumerate(cols))
                           for cols in itertools.permutations(range(a.shape[1]), a.shape[0]))
            self.assertAlmostEqual(r["mc_track_dice_sum"], best)
            matches = r["dice_matches"]
            self.assertEqual(len({m["true_id"] for m in matches}), len(matches))
            self.assertEqual(len({m["pred_id"] for m in matches}), len(matches))
            self.assertTrue(all(m["intersection"] > 0 for m in matches))

    def test_legacy_outputs_are_unchanged(self):
        rng = np.random.default_rng(131)
        new, old = [], []
        for _ in range(40):
            y, p = rng.integers(-1, 5, (2, 30))
            valid, gate = rng.random((2, 30)) > .2
            old.append(event_track_metrics(y, p, valid, gate))
            new.append(score(y, p, valid, gate))
            self.assertEqual(old[-1], {k: new[-1][k] for k in old[-1]})
        expected, actual = summarize_event_metrics(old), summarize(new)
        for k, v in expected.items():
            if v is None:
                self.assertIsNone(actual[k])
            else:
                self.assertAlmostEqual(actual[k], v)
        for k in EVENT_MEAN_METRICS:
            self.assertAlmostEqual(actual[k + "_event_macro"], actual[k])

    def test_global_metrics_invariant_to_batching_and_merge_order(self):
        rng = np.random.default_rng(912)
        rows = []
        for _ in range(73):
            n = int(rng.integers(0, 60))
            y, p = rng.integers(-1, 4, (2, n))
            provenance = np.where(y >= 0, P.MC_ONLY, rng.choice(list(P), n))
            rows.append(score(y, p, provenance=provenance))
        expected = summarize(rows)
        for size in (1, 2, 7, 16, 64, 100):
            total = MCReconstructionAccumulator()
            batches = []
            for start in range(0, len(rows), size):
                batch = MCReconstructionAccumulator()
                for r in rows[start:start+size]:
                    batch.add(r)
                batches.append(batch)
            for batch in reversed(batches):
                total.merge(batch)
            for k, v in expected.items():
                actual = total.summary()[k]
                if isinstance(v, float):
                    self.assertAlmostEqual(actual, v, places=12, msg=f"{size}: {k}")
                else:
                    self.assertEqual(actual, v, f"{size}: {k}")

    def test_invalid_inputs_and_incompatible_merges_fail(self):
        for kwargs in ({"valid_mask": [1]}, {"pred_signal_mask": [1, 3]},
                       {"point_ids": [0, 0]}, {"provenance": [99, 0]},
                       {"provenance": [P.OVERLAY_ONLY, P.MC_ONLY]}):
            with self.assertRaises(ValueError):
                row([0, 0], [1, 1], **kwargs)
        with self.assertRaises(ValueError):
            score(np.array([0.0]), np.array([1]))
        a, b = MCReconstructionAccumulator(), MCReconstructionAccumulator()
        a.add(row([0] * 4, [1] * 4))
        b.add(row([0] * 4, [1] * 4, config=MatchConfig(iou_threshold=.9)))
        with self.assertRaises(ValueError):
            a.merge(b)
        with self.assertRaises(ValueError):
            a.add(event_track_metrics(np.array([0]), np.array([1])))


class ProvenanceAndArchiveTest(unittest.TestCase):
    def test_hdf5_join_checks_identity_full_event_and_serialization(self):
        import h5py
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test").mkdir()
            manifest = {"format": "cvt-clusters-h5-v1", "status": "complete", "plan_id": "fixture",
                        "splits": {"test": [{"path": "test/part.h5", "events": 1, "split": "test"}]}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            y = np.array([0] * 4 + [-1] * 3)
            with h5py.File(root / "test/part.h5", "w") as f:
                f.attrs.update(format=manifest["format"], complete=1, plan_id="fixture", events=1, part_key="test/part.h5")
                f["events/source_id"] = [17]; f["events/original_ordinal"] = [25]
                f["clusters/offsets"] = [0, 7]; f["clusters/primary_segment"] = y
                f["clusters/background_only"] = [0, 0, 0, 0, 1, 0, 0]
                f["clusters/ownership"] = [1, 1, 1, 1, 0, 3, 2]
                f["clusters/origin_noise"] = [0, 0, 0, 0, 1, 0, 0]
                f["clusters/purity"] = [1., 1., 1., 1., 0., 0., .5]
            permutation = np.array([6, 2, 0, 4, 1, 3, 5])
            archive = dict(offsets=[0, 7], truth=y[permutation], pred=np.full(7, 9),
                           point_ids=permutation, source_event_index=[0], source_id=[17], original_ordinal=[25])
            path = root / "pred.npz"
            np.savez(path, **archive)
            kwargs = dict(hdf5_root=root, split="test", plan_id="fixture")
            s = rescore_archive(path, **kwargs)
            self.assertTrue(s["receipt"]["full_event_membership_verified"])
            self.assertAlmostEqual(s["metrics"]["pred"]["mc_bg_contamination_global"], 1/7)
            self.assertAlmostEqual(s["metrics"]["pred"]["mc_hit_impurity_global"], 3/7)
            self.assertEqual(s["metrics"]["pred"]["n_mc_nontarget_mc_present_unresolved_points"], 2)
            for replacement, error in (({"source_id": [18]}, "identity"),
                                       ({"point_ids": [0] * 7}, "permutation"),
                                       ({"truth": y}, "truth label")):
                np.savez(path, **{**archive, **replacement})
                with self.assertRaisesRegex(ValueError, error):
                    rescore_archive(path, **kwargs)
            with self.assertRaisesRegex(ValueError, "plan_id"):
                rescore_archive(path, **{**kwargs, "plan_id": "wrong"})

    def test_existing_hdf5_provenance_is_conservative(self):
        actual = cluster_provenance_from_hdf5(
            background_only=[0, 0, 1, 0, 0, 0], ownership=[1, 1, 0, 2, 3, 3],
            origin_noise=[0, 0, 1, 0, 0, 1], purity=[1., .75, 0, .5, 0, 0])
        np.testing.assert_array_equal(actual, [P.MC_ONLY, P.MC_PRESENT_UNRESOLVED,
            P.OVERLAY_ONLY, P.MC_PRESENT_UNRESOLVED, P.MC_PRESENT_UNRESOLVED, P.UNKNOWN])
        # No simulated contributor is insufficient to prove measured overlay.
        self.assertEqual(cluster_provenance_from_hdf5(background_only=[0], origin_noise=[1])[0], P.UNKNOWN)
        for kwargs in ({"background_only": [1], "origin_noise": [0]},
                       {"background_only": [0], "ownership": [0]},
                       {"background_only": [0], "ownership": [1], "purity": [.5]}):
            with self.assertRaises(ValueError):
                cluster_provenance_from_hdf5(**kwargs)

    def test_archive_roundtrip_and_batch_invariance(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.npz"
            np.savez(path, offsets=[0, 4, 12], truth=[0] * 4 + [0] * 4 + [-1] * 4,
                     pred=[7] * 4 + [8] * 8, baseline=[7] * 4 + [-1] * 8,
                     background_only=[0] * 8 + [1] * 4, ownership=[1] * 8 + [0] * 4,
                     purity=[1.] * 8 + [0.] * 4, origin_noise=[0] * 8 + [1] * 4)
            events = []
            a = rescore_archive(path, ["pred", "baseline"], events_per_batch=1, event_sink=events.append)
            b = rescore_archive(path, ["pred", "baseline"], events_per_batch=17)
            self.assertEqual(a["metrics"], b["metrics"])
            self.assertEqual(len(events), 4)
            self.assertEqual(a["receipt"]["provenance_source"], "archive")
            self.assertEqual(a["receipt"]["input_sha256"], b["receipt"]["input_sha256"])
            json.dumps(a, allow_nan=False)
            self.assertFalse(a["checkpoint_selection_modified"])
            self.assertAlmostEqual(a["metrics"]["pred"]["mc_track_dice_mean"], 5/6)

    def test_archive_rejects_bad_offsets_and_duplicate_events(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.npz"
            for offsets in ([1, 2], [0, 3], [0, 2, 1, 2]):
                np.savez(path, offsets=offsets, truth=[0, 0], pred=[1, 1])
                with self.assertRaises(ValueError):
                    rescore_archive(path)
            np.savez(path, offsets=[0, 1, 2], truth=[0, 0], pred=[1, 1],
                     source_event_index=[0, 0])
            with self.assertRaisesRegex(ValueError, "Repeated"):
                rescore_archive(path)


if __name__ == "__main__":
    unittest.main()
