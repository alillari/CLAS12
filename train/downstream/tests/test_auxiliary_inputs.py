"""Auxiliary backbones through real task losses, validation and checkpoint reuse."""
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from fm4npp.datasets.dataset import TPCBatchDataset, MyCollator
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset as PretrainingDataset
from fm4npp.models.mamba_config import architecture_metadata, build_backbone
from fm4npp.pretrained import export_backbone_config
from fm4npp.utils import YParams
from train.downstream.input_representation import (
    input_config, representation_input, validate_checkpoint_inputs, validate_input_representation,
)
# Also installs the historical downstream import paths used by the trainers.
from train.downstream.track_finding_experiment import DownstreamTrainer as Finder
from train.downstream.point_classification_trainer import DownstreamTrainer as Classifier
from train.downstream.model import AttentionHead, MambaHead
from train.downstream.trackinghead import MambaAttentionHead
from train.downstream.loss import PointHungarianMatcher
from train.downstream.noise_checkpoint import training_contract
from train.downstream.tests.test_cvt_downstream import event
from train.pretrain.tests.test_cvt_hdf5 import publish_fixture


ROOT = Path(__file__).resolve().parents[3]
try:
    from fm4npp.models.mamba3 import Mamba3SISO
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False


def params(extra="length", **overrides):
    p = YParams(ROOT / "scripts/configs/clas12_adapter_recipes.yaml", "hdf5_noise_50k_balanced")
    p.update_params(dict(input_representation="clas12_pos_plus_aux_v1", embed_method="pos_plus_aux",
        pe_method="nerf", pos_dim=3, aux_dim=5 if extra == "both" else 4, aux_extra_feature=extra,
        backbone_version="transformer", embed_dim=64, num_layers_backbone=2, num_heads_backbone=4,
        d_state=128, dropout=0., klen=5, use_sincos_target=True, hilbert_angular_grid="unit_v2",
        adapter_version="mamba2", num_embedder_layers=0, reader_type="hdf5",
        validation_ari_mode="signal", max_validation_events=1, max_val_batches=None))
    p.update_params(overrides)
    return p


def fixture(root, p):
    rows = []
    for ordinal, n in enumerate((8, 5)):
        row = event(ordinal)
        for key, value in list(row.items()):
            if key.startswith("clusters/"):
                row[key] = value[:n]
        phi = np.linspace(-2.8, 2.8, n)
        r = np.arange(n) + 7.
        row["clusters/xyz"] = np.column_stack((r*np.cos(phi), r*np.sin(phi), np.arange(n))).astype("f4")
        row["clusters/direction"] = np.tile(np.array([.6, 0., .8], dtype="f4"), (n, 1))
        row["clusters/length"] = np.arange(1, n+1, dtype="f4")
        row["clusters/pitch"] = np.arange(1, n+1, dtype="f4") / 100
        rows.append(row)
    publish_fixture(root, {"train": [rows]})
    ds = TPCBatchDataset(root, split="train", reader_type="hdf5", normalize=True,
        voxelize=False, return_dict=True, compute_neighbors=False, return_coatjava_seg_pred=True,
        input_representation=p.input_representation, hilbert_angular_grid=p.hilbert_angular_grid)
    return MyCollator()([ds[0], ds[1]])


class AuxiliaryInputTest(unittest.TestCase):
    def test_packing_matches_pretraining_with_aligned_hdf5_rows_and_padding(self):
        for extra in ("length", "pitch", "both"):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                p = params(extra)
                batch = fixture(Path(tmp), p)
                original = batch["points"].clone()
                packed = representation_input(batch["points"], batch, p)
                pretrain = PretrainingDataset(tmp, reader_type="hdf5", split="train", normalize=True,
                    use_aux_features=True, aux_extra_feature=extra, hilbert_angular_grid="unit_v2")
                for i, n in enumerate((8, 5)):
                    torch.testing.assert_close(packed[i, :n], pretrain[i][0], rtol=0, atol=2e-6)
                self.assertTrue((packed[1, 5:] == -100).all())
                torch.testing.assert_close(batch["points"], original, rtol=0, atol=0)
                self.assertIs(representation_input(original, {}, {}), original)

    def test_invalid_settings_and_missing_or_misaligned_geometry_fail(self):
        for update in (dict(aux_dim=4, aux_extra_feature="both"), dict(pos_dim=4),
                       dict(embed_method="pos_only"), dict(aux_extra_feature="invalid"),
                       dict(input_representation="center_only")):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_input_representation(params(**update))
        p = params()
        with self.assertRaisesRegex(KeyError, "geometry_context"):
            representation_input(torch.zeros(1, 2, 3), {}, p)
        with self.assertRaisesRegex(ValueError, "row-aligned"):
            representation_input(torch.zeros(1, 2, 3), {"geometry_context": torch.zeros(1, 1, 11)}, p)

    def test_checkpoint_rejects_length_pitch_swap_and_preserves_legacy_centers(self):
        validate_checkpoint_inputs({"params": {"params": {"input_representation": "center_only"}}}, {})
        for saved in ({"input_config": input_config(params("length"))},
                      {"params": {"params": dict(aux_extra_feature="length"),
                                  "input_representation": "clas12_pos_plus_aux_v1"}}):
            validate_checkpoint_inputs(saved, params("length"))
            with self.assertRaisesRegex(ValueError, "input representation differs"):
                validate_checkpoint_inputs(saved, params("pitch"))

    def test_classification_head_builder_uses_the_auxiliary_checkpoint_width(self):
        for attention in (True, False):
            with self.subTest(attention=attention):
                trainer = Classifier.__new__(Classifier)
                trainer.device = "cpu"
                trainer.params = params("both", use_attention_head=attention, num_output_classes=2)
                head = trainer.build_head()
                self.assertEqual(head.embedder.aux_proj.in_features, 5)


class AuxiliaryTrainerTest(unittest.TestCase):
    device = "cpu"
    backbone = "transformer"

    def trainer(self, root, extra, task, head="mamba"):
        p = params(extra, checkpoint_dir=str(root), task=task, backbone_version=self.backbone,
                   num_output_classes=5 if task == "pid" else 2)
        trainer = (Finder if task == "finding" else Classifier).__new__(Finder if task == "finding" else Classifier)
        trainer.params, trainer.device = p, self.device
        trainer.log_to_screen, trainer.world_rank, trainer.use_lora, trainer.use_amp = False, 1, False, False
        trainer.iters = trainer.global_step = 1
        trainer.best_loss, trainer.class_weights = math.inf, None
        p["data_contract"] = {"format": "cvt-clusters-h5-v1", "plan_id": "fixture"}
        trainer.training_cohort_digest = "fixture"
        trainer.model = build_backbone(p).to(self.device).eval()
        path = root / "backbone.tar"
        mike_fixture = os.environ.get("MIKE_AUX_TEST_CHECKPOINT")
        if self.backbone == "mamba3" and extra == "both" and mike_fixture:
            path = Path(mike_fixture)
        else:
            torch.save(dict(model_state=trainer.model.state_dict(), hilbert_angular_grid="unit_v2",
                backbone_architecture=architecture_metadata(trainer.model),
                backbone_config=export_backbone_config(p, trainer.model)), path)
        p["pretrained_ckpt"] = str(path)
        trainer.restore_checkpoint(path, load_optimizer_state=False)
        trainer.model.requires_grad_(False)
        common = dict(input_dim=64, embed_dim=64, num_feature_layers=2, num_embedder_layers=0,
                      embed_method="pos_plus_aux", pe_method="nerf", aux_dim=p.aux_dim, dropout=0.)
        if task == "finding":
            trainer.down_model = MambaAttentionHead(**common, num_layers=0, num_prototypes=3,
                num_output_dim=64, num_self_attn_layers=1, num_heads=4, ffn_dim=64).to(self.device)
            trainer.matcher = PointHungarianMatcher(cost_class=1., cost_dice=1., cost_focal=1.)
            for name in ("loss_matched_ce_weight", "loss_unmatched_ce_weight", "loss_dice_weight", "loss_focal_weight"):
                setattr(trainer, name, 1.)
        else:
            cls = AttentionHead if head == "attention" else MambaHead
            trainer.down_model = cls(**common, num_layers=1 if head == "attention" else 0,
                                     num_output_dim=p.num_output_classes).to(self.device)
            trainer.noise_training_contract = training_contract(p)
        trainer.down_optimizer = torch.optim.AdamW(trainer.down_model.parameters(), lr=.001)
        trainer.down_scheduler = torch.optim.lr_scheduler.StepLR(trainer.down_optimizer, 1)
        trainer.down_results = {k: [] for k in ("val", "precision", "recall", "accuracy")}
        return trainer

    def test_task_training_validation_and_checkpoint_reload(self):
        for extra in ("length", "pitch", "both"):
            for task, head in (("finding", "mamba"), ("nid", "mamba"), ("pid", "attention")):
                with self.subTest(extra=extra, task=task), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    trainer = self.trainer(root, extra, task, head)
                    data = root / "data"
                    data.mkdir()
                    batch = fixture(data, trainer.params)
                    trainer.val_data_loader = [batch]
                    expected_width = 8 if extra == "both" else 7
                    widths = []
                    hook = trainer.model.register_forward_pre_hook(lambda module, args: widths.append(args[0].shape[-1]))
                    trainer.use_pretrained_backbone = True
                    if task == "finding":
                        loss, _, _ = trainer._compute_batch_loss(batch, pretrain=True, train=True)
                        loss.backward()
                        trainer.down_optimizer.step()
                        loss = float(loss)
                    else:
                        loss = trainer._train_one_batch(batch, pretrain=True)
                    self.assertTrue(math.isfinite(loss))
                    self.assertTrue(any(p.grad is not None for p in trainer.down_model.parameters()))
                    self.assertTrue(all(p.grad is None for p in trainer.model.parameters()))
                    validation = trainer.validate_end_to_end_one_epoch(pretrain=True)
                    self.assertTrue(math.isfinite(validation))
                    self.assertEqual(widths, [expected_width, expected_width])
                    hook.remove()
                    trainer._save_checkpoint("head.pth", 1, True, validation)
                    trainer.load_checkpoint(root / "head.pth", inference=True)
                    if task == "nid":
                        from train.downstream.noise_evaluation import evaluate
                        class Loader(list):
                            dataset = SimpleNamespace(split="test")
                        summary = evaluate(trainer, Loader([batch]), root)
                        self.assertTrue(math.isfinite(summary["adapter"]["accuracy"]))
                    # Same tensor width, different physical scalar: reject before loading weights.
                    if extra != "both":
                        trainer.params["aux_extra_feature"] = "pitch" if extra == "length" else "length"
                        with self.assertRaisesRegex(ValueError, "input representation differs"):
                            trainer.load_checkpoint(root / "head.pth", inference=True)
                        trainer.params["aux_extra_feature"] = extra
                    # The adapter-only path must receive packed tokens too.
                    if task == "finding":
                        loss, _, _ = trainer._compute_batch_loss(batch, pretrain=False, train=True)
                        self.assertTrue(torch.isfinite(loss))
                    else:
                        self.assertTrue(math.isfinite(trainer._train_one_batch(batch, pretrain=False)))


@unittest.skipUnless(torch.cuda.is_available() and HAS_MAMBA3, "Mamba3 feature extraction requires CUDA and Mamba3")
class AuxiliaryMamba3CudaTest(AuxiliaryTrainerTest):
    device = "cuda"
    backbone = "mamba3"


if __name__ == "__main__":
    unittest.main()
