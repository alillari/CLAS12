"""Test the native-hit and backbone contracts for the strip geometry encoder."""

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from fm4npp.datasets.cvt_hdf5 import CVTStripEvents
from fm4npp.models.mamba_config import build_backbone
from fm4npp.models.tracking_geometry import TrackingGeometryEmbedder
from fm4npp.tracking_geometry import (
    ARC, CENTER, ENDPOINT_OFFSETS, LINE, PITCH,
    TRACKING_GEOMETRY_V1, pack_tracking_geometry,
)


def measurement_fields():
    return dict(
        center_cm=[[-100, 0, 0], [0, 12, 4], [-100, 0, 0]],
        endpoints_cm=[[[-100, 0, -4], [-100, 0, 4]],
                      [[-3, 11, 4], [3, 11, 4]],
                      [[-100, 0, -4], [-100, 0, 4]]],
        length_cm=[8, 6.4, 8],
        pitch_cm=[.04, .08, .04],
        sweep_rad=[0, 1.2, 0],
        kind=[LINE, ARC, LINE],
        detector=[2, 1, 2],
        layer=[1, 4, 1],
    )


class TrackingGeometryTests(unittest.TestCase):
    def test_native_reader_preserves_each_hit(self):
        fields = measurement_fields()
        expected = pack_tracking_geometry(**fields)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            part = "train/part-0000.h5"
            for format_name in ("cvt-strips-h5-v1", "cvt-strips-h5-v2"):
                with self.subTest(format=format_name):
                    manifest = dict(format=format_name, status="complete", plan_id="plan",
                                    splits={"train": [dict(path=part, split="train", events=2)]})
                    (root / "manifest.json").write_text(json.dumps(manifest))
                    with h5py.File(root / part, "w") as file:
                        file.attrs.update(format=format_name, complete=1, plan_id="plan",
                                          part_key=part, events=2)
                        strips = file.create_group("strips")
                        strips["offsets"] = [0, 3, 3]
                        strips["xyz"] = fields["center_cm"]
                        strips["endpoints"] = np.reshape(fields["endpoints_cm"], (-1, 6))
                        for name, key in (("length", "length_cm"), ("pitch", "pitch_cm"),
                                          ("arc_angle", "sweep_rad"), ("geometry_kind", "kind"),
                                          ("detector", "detector"), ("layer", "layer")):
                            strips[name] = fields[key]
                    with CVTStripEvents(root, "train") as events:
                        np.testing.assert_array_equal(events.geometry_tokens(0), expected)
                        self.assertEqual(events.geometry_tokens(1).shape, (0, expected.shape[-1]))

    def test_endpoint_symmetry_and_pitch(self):
        fields = measurement_fields()
        tokens = torch.from_numpy(pack_tracking_geometry(**fields))[None]
        embedder = TrackingGeometryEmbedder("nerf", 32).eval()
        original, position = embedder(tokens)
        swapped = tokens.clone()
        swapped[..., ENDPOINT_OFFSETS] = swapped[..., ENDPOINT_OFFSETS].reshape(
            1, 3, 2, 3).flip(-2).flatten(-2)
        torch.testing.assert_close(embedder(swapped)[0], original)
        changed_pitch = tokens.clone()
        changed_pitch[..., PITCH] *= 2
        different, unchanged_position = embedder(changed_pitch)
        torch.testing.assert_close(unchanged_position, position)
        self.assertFalse(torch.allclose(different, original))
        torch.testing.assert_close(original[:, 0], original[:, 2])
        original.square().mean().backward()
        for branch in (embedder.endpoint, embedder.line, embedder.arc):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in branch.parameters()))

    def test_transformer_preserves_valid_negative_coordinates(self):
        tokens = torch.from_numpy(pack_tracking_geometry(**measurement_fields()))[None]
        self.assertEqual(tokens[0, 0, CENTER][0].item(), -100)
        padded = torch.nn.functional.pad(tokens, (0, 0, 0, 1), value=-100)
        model = build_backbone(dict(backbone_version="transformer",
                                   embed_dim=32, num_layers_backbone=2,
                                   embed_method=TRACKING_GEOMETRY_V1, dropout=0.)).eval()
        self.assertEqual(model._backbone_config["embed_method"], TRACKING_GEOMETRY_V1)
        with torch.no_grad():
            _, features, positions = model(padded, return_z=True)
            prediction = model(padded)
            torch.testing.assert_close(prediction[:, :3], model(tokens))
        self.assertTrue(torch.count_nonzero(features[-1][0, 0]))
        self.assertTrue(torch.count_nonzero(positions[0, 0]))
        self.assertEqual(torch.count_nonzero(prediction[:, 3:]).item(), 0)

    def test_rejects_invalid_geometry(self):
        for invalid in (dict(pitch_cm=[.04, 0, .04]), dict(sweep_rad=[1, 1.2, 0]),
                        dict(kind=[LINE, 7, LINE])):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pack_tracking_geometry(**{**measurement_fields(), **invalid})


if __name__ == "__main__":
    unittest.main()
