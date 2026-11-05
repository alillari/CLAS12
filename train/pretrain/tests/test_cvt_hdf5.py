"""Pretraining data-contract tests. Set CVT_HDF5_PILOT for the real-event test."""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from mmap_ninja import RaggedMmap
from torch.utils.data import DataLoader

from fm4npp.datasets.cvt_hdf5 import CVTClusterEvents
from fm4npp.datasets.dataset_pretrain import MyCollator, TPCBatchDataset, get_data_loader, get_val_loader
from fm4npp.models.transformer import TransformerGPT
from fm4npp.utils import YParams


RECIPE = Path(__file__).resolve().parents[3] / "scripts/configs/mamba_chinchilla_sweep.yaml"


def publish_fixture(root, splits):
    """Write small v1 parts. Each event maps HDF5 column paths to their rows."""
    manifest = dict(format="cvt-clusters-h5-v1", status="complete", plan_id="fixture", splits={})
    for split, shards in splits.items():
        parts = manifest["splits"][split] = []
        (root / split).mkdir()
        for i, events in enumerate(shards):
            path = f"{split}/part-{i:04d}.h5"
            parts.append(dict(path=path, events=len(events), split=split))
            with h5py.File(root / path, "w") as f:
                f.attrs.update(
                    format=manifest["format"],
                    complete=1,
                    plan_id="fixture",
                    events=len(events),
                    part_key=path,
                )
                for name in events[0]:
                    domain = name.rsplit("/", 1)[0]
                    rows = [e[name] for e in events]
                    if domain == "events":
                        f[name] = rows
                    else:
                        if domain + "/offsets" not in f:
                            f[domain + "/offsets"] = np.cumsum([0, *map(len, rows)], dtype=np.int64)
                        f[name] = np.concatenate(rows)
    (root / "manifest.json").write_text(json.dumps(manifest))


def event(n, ordinal):
    r = np.linspace(6.8, 20, n, dtype=np.float32)
    return {
        "events/original_ordinal": ordinal,
        "clusters/xyz": np.column_stack((r, r * 0.1, r * 0.2)),
        "clusters/primary_segment": np.full(n, ordinal, dtype=np.int32),
        "particles/tid": np.arange(ordinal % 3, dtype=np.int32),
    }


def loaders_params(root, workers=0):
    p = YParams(RECIPE, "scale_w64_d12_bgmerged_hdf5")
    p.data_root = root
    p.num_data_workers = workers
    p.local_batch_size = p.local_valid_batch_size = 2
    return p


class HDF5PretrainingTest(unittest.TestCase):
    def test_grid_choice_preserves_full_event_targets_and_auxiliary_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            radii = np.repeat([6.8, 9.3, 12.2, 16.], 24)
            phi = np.tile(np.linspace(-2.8, 2.9, 24), 4)
            row = event(96, 0)
            row['clusters/xyz'] = np.column_stack((radii * np.cos(phi), radii * np.sin(phi), np.arange(96) / 10)).astype('f4')
            row['clusters/primary_segment'] = np.arange(96, dtype=np.int32)
            row['clusters/direction'] = np.tile(np.array([1., 0., 0.], dtype='f4'), (96, 1))
            row['clusters/length'] = np.arange(96, dtype='f4') / 3
            row['clusters/pitch'] = np.full(96, .1, dtype='f4')
            publish_fixture(root, {'train': [[row]], 'validation': [[row]]})
            results = []
            for grid in ('legacy_v1', 'unit_v2'):
                p = loaders_params(root)
                p.hilbert_angular_grid = grid
                p.use_aux_features = True
                train, _, valid, _ = get_data_loader(p, False)
                results.append(train.dataset[0])
                self.assertEqual(len(results[-1][0]), 96)
                for a, b in zip(results[-1], valid.dataset[0]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                for a, b in zip(results[-1], get_val_loader(p, False).dataset[0]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertFalse(torch.equal(results[0][1], results[1][1]))
            for a, b in zip(results[0], results[1]):
                torch.testing.assert_close(a[results[0][1].argsort()], b[results[1][1].argsort()], rtol=0, atol=0)

    def test_legacy_readers_keep_their_splits_and_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            for reader_type in ("ragged_mmap", "ragged_npy"):
                root = Path(tmp) / reader_type
                root.mkdir()
                for split, label in (("pretrain", 2), ("test", 4)):
                    rows = [event(n, label) for n in (7, 60)]
                    for prefix, key in (
                        ("features", "clusters/xyz"),
                        ("seg_target", "clusters/primary_segment"),
                    ):
                        path = root / f"{prefix}_{split}"
                        values = [row[key] for row in rows]
                        if reader_type == "ragged_mmap":
                            RaggedMmap.from_lists(path, values)
                        else:
                            path.mkdir()
                            np.save(path / "offsets.npy", [0, 7, 67])
                            np.save(path / "values.npy", np.concatenate(values))
                p = loaders_params(root)
                p.reader_type = reader_type
                p.local_batch_size = p.local_valid_batch_size = 1
                train, _, valid, _ = get_data_loader(p, distributed=False)
                for loader, label in ((train, 2), (valid, 4)):
                    self.assertEqual(len(loader.dataset), 1)
                    _, segments, _ = next(iter(loader))
                    self.assertEqual(segments.tolist(), [[label] * 7])
                self.assertFalse((root / "reg_target_pretrain").exists())

    def test_singleton_and_chunked_train_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            long = event(9, 2)
            long["clusters/primary_segment"] = np.arange(9, dtype=np.int32)
            publish_fixture(root, {"train": [[event(1, 3), long]], "validation": [[long]]})
            p = loaders_params(root)
            p.len_chunk = 4
            p.chunk_training = True
            train, _, valid, _ = get_data_loader(p, distributed=False)
            x, segment, neighbors = train.dataset[0]
            self.assertEqual((x.shape, segment.shape, neighbors.shape), ((1, 3), (1,), (1, 3)))
            self.assertTrue((neighbors == -100).all())
            for _ in range(8):
                x, labels, _ = train.dataset[1]
                self.assertEqual(len(x), 4)
                self.assertEqual(sorted(labels.tolist()), list(range(int(labels.min()), int(labels.min()) + 4)))
            self.assertEqual([sorted(valid.dataset[i][1].tolist()) for i in range(len(valid.dataset))],
                             [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_reader_and_loader_across_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            publish_fixture(
                root,
                {"train": [[event(1, 3)], [event(65, 4), event(4, 5)]], "validation": [[event(6, 8)]]},
            )
            with CVTClusterEvents(root, "train", max_open=1) as reader:
                for i in (0, 1, -1, 0):
                    row = reader.read(i, ["events.original_ordinal", "clusters.xyz", "particles.tid"])
                    self.assertEqual(len(row["particles.tid"]), row["events.original_ordinal"] % 3)
                    self.assertEqual(len(row["clusters.xyz"]), [1, 65, 4][i])
                with self.assertRaises(IndexError):
                    reader.read(3, ["clusters.xyz"])
                with pickle.loads(pickle.dumps(reader)) as reopened:
                    self.assertEqual(
                        reopened.read(1, ["events.original_ordinal"])["events.original_ordinal"], 4
                    )

            p = loaders_params(root)
            train, _, valid, _ = get_data_loader(p, distributed=False)
            self.assertEqual(len(train.dataset), 3)
            self.assertEqual(sum(len(x) for x, _, _ in train), 3)
            for loader in (valid, get_val_loader(p, distributed=False)):
                x, segments, _ = next(iter(loader))
                self.assertEqual(x.shape, (1, 6, 3))
                self.assertTrue((segments == 8).all())

            # Open a file in the parent before either fork or spawn.
            train.dataset[0]
            for mode in ("fork", "spawn"):
                loader = DataLoader(
                    train.dataset,
                    batch_size=2,
                    num_workers=2,
                    multiprocessing_context=mode,
                    collate_fn=MyCollator(),
                )
                self.assertEqual([t[:, 0].tolist() for _, t, _ in loader], [[3, 4], [5]])
            subset = TPCBatchDataset(
                root, reader_type="hdf5", limit_data=True, limit_size=2, data_fraction=0.5
            )
            self.assertEqual(len(subset), 1)

    def test_refuses_missing_split_unpublished_or_foreign_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            publish_fixture(root, {"train": [[event(3, 2)]]})
            with self.assertRaisesRegex(ValueError, "validation"):
                get_data_loader(loaders_params(root), distributed=False)
            with h5py.File(root / "train/part-0000.h5", "r+") as f:
                f.attrs["plan_id"] = "other"
            with CVTClusterEvents(root, "train") as reader:
                with self.assertRaisesRegex(ValueError, "manifest"):
                    reader.read(0, ["clusters.xyz"])
            path = root / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["status"] = "partial"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "published"):
                CVTClusterEvents(root, "train")

    @unittest.skipUnless(os.getenv("CVT_HDF5_PILOT"), "set CVT_HDF5_PILOT to a published pilot root")
    def test_real_events_match_mmap_and_train(self):
        pilot_root = Path(os.environ["CVT_HDF5_PILOT"])
        paths = [
            "clusters/" + key
            for key in ("xyz", "primary_segment", "endpoints", "direction", "length", "pitch")
        ]
        with CVTClusterEvents(pilot_root, "pilot") as pilot:
            lengths = pilot.event_lengths("clusters")
            ids = [0, int(lengths.argmin()), int(lengths.argmax()), len(pilot) - 1]
            # Independent direct slices from the single qualification part.
            with h5py.File(pilot_root / pilot.parts[0][0], "r") as f:
                offsets = f["clusters/offsets"][:]
                rows = [{p: f[p][offsets[i] : offsets[i + 1]] for p in paths} for i in ids]
                for i, row in zip(ids, rows):
                    got = pilot.read(i, [p.replace("/", ".") for p in paths])
                    for path in paths:
                        np.testing.assert_array_equal(got[path.replace("/", ".")], row[path])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            publish_fixture(root, {"train": [[rows[0]], [rows[1], rows[2]]], "validation": [[rows[3]]]})
            for key, values in {
                "features": [r[paths[0]] for r in rows],
                "seg_target": [r[paths[1]].astype(np.int64) for r in rows],
                "reg_target": [np.full((len(r[paths[0]]), 3), np.nan, dtype=np.float32) for r in rows],
                "cluster_geometry_context_target": [
                    np.column_stack([r[p] for p in paths[2:]]) for r in rows
                ],
            }.items():
                RaggedMmap.from_lists(root / (key + "_pretrain"), values)

            for extra in (None, "length", "pitch", "both"):
                opts = dict(use_aux_features=extra is not None, aux_extra_feature=extra or "length")
                legacy = TPCBatchDataset(root, num_pred_points=1, **opts)
                self.assertEqual(len(legacy), 0)  # Historical 50-cluster ceiling.
                legacy.filter_data(1, np.inf)  # Compare the same events in the legacy transforms.
                reference = MyCollator()([legacy[i] for i in range(len(rows))])
                p = loaders_params(root, workers=2)
                p.use_aux_features = extra is not None
                p.aux_extra_feature = extra or "length"
                train, _, valid, _ = get_data_loader(p, distributed=False)
                batch = MyCollator()([train.dataset[i] for i in range(3)] + [valid.dataset[0]])
                for actual, expected in zip(batch, reference):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                x, _, target = next(iter(train))
                model = TransformerGPT(
                    embed_dim=32,
                    num_layers=1,
                    num_heads=4,
                    klen=1,
                    embed_method="pos_plus_aux" if extra else "pos_only",
                    aux_dim=x.shape[-1] - 3,
                    dropout=0,
                )
                mask = target != -100
                loss = (model(x)[mask] - target[mask]).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(
                    all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                )
