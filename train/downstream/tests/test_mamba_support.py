"""Version selection, legacy checkpoint contracts, and optional CUDA execution."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from fm4npp.models.mamba_config import (
    adapter_options, architecture_metadata, backbone_version, build_backbone,
    build_mamba_block, legacy_initialization_parameters, optimizer_parameters,
    validate_checkpoint_architecture,
)
from fm4npp.models.mambagpt import Mamba1GPT, MambaGPT, Mamba3GPT
from train.downstream.model import MambaHead, MambaTrackRegressionHead, AttentionHead
from train.downstream.trackinghead import MambaAttentionHead


try:
    from fm4npp.models.mamba3 import Mamba3SISO
    MAMBA3_AVAILABLE = True
except ImportError:
    MAMBA3_AVAILABLE = False

VERSIONS = ("mamba1", "mamba2", "mamba3") if MAMBA3_AVAILABLE else ("mamba1", "mamba2")


ROOT = Path(__file__).resolve().parents[3]


def backbone_params(version):
    return dict(backbone_version=version, embed_dim=64, num_layers_backbone=2,
                d_state=16, embed_method="pos_only", pe_method="nerf", klen=2,
                dropout=0.0)


def regression_head(version="mamba2", **kwargs):
    options = dict(input_dim=64, embed_dim=64, num_layers=1,
                   num_embedder_layers=1, num_feature_layers=2,
                   d_state=16, embed_method="pos_only", dropout=0.0,
                   adapter_version=version)
    options.update(kwargs)
    return MambaTrackRegressionHead(**options)


class MambaSelectionTest(unittest.TestCase):
    def test_legacy_defaults_and_explicit_backbone_override(self):
        self.assertEqual(backbone_version({}), "mamba2")
        self.assertEqual(backbone_version({}, default="mamba1"), "mamba1")
        self.assertEqual(adapter_options({})["adapter_version"], "mamba2")
        for version in VERSIONS:
            cls = {"mamba1": Mamba1GPT, "mamba2": MambaGPT, "mamba3": Mamba3GPT}[version]
            params = {**backbone_params(version), "mambaversion": "mamba1"}
            self.assertIsInstance(build_backbone(params), cls)
            self.assertEqual(build_backbone(params).backbone_version, version)

    @unittest.skipUnless(MAMBA3_AVAILABLE, "Mamba3 dependencies are optional")
    def test_selections_are_independent(self):
        params = {**backbone_params("mamba1"), "adapter_version": "mamba3"}
        self.assertEqual(build_backbone(params).backbone_version, "mamba1")
        head = regression_head(**adapter_options(params))
        self.assertEqual(head.mamba_layers[0][1]._mamba_config["version"], "mamba3")
        self.assertEqual(head.mamba_embedder_layers[0][1]._mamba_config["version"], "mamba3")
        # Backbone Mamba3 also works with the default Mamba2 adapter.
        self.assertEqual(build_backbone(backbone_params("mamba3")).backbone_version, "mamba3")
        self.assertEqual(regression_head().adapter_version, "mamba2")

    def test_options_reach_backbone_and_both_adapter_paths(self):
        params = {**backbone_params("mamba2"), "headdim": 32, "ngroups": 2,
                  "chunk_size": 32, "use_mem_eff_path": False}
        block = build_backbone(params).mamba_layers[0][1]
        self.assertEqual((block.headdim, block.ngroups, block.chunk_size), (32, 2, 32))
        self.assertFalse(block.use_mem_eff_path)
        if not MAMBA3_AVAILABLE:
            return
        options = adapter_options(dict(adapter_version="mamba3", adapter_headdim=32,
                                       adapter_d_state=32, adapter_chunk_size=32))
        head = regression_head(**options)
        for layer in (head.mamba_layers[0], head.mamba_embedder_layers[0]):
            self.assertEqual(layer[1].headdim, 32)
            self.assertEqual(layer[1].d_state, 32)
            self.assertEqual(layer[1].chunk_size, 32)

    @unittest.skipUnless(MAMBA3_AVAILABLE, "Mamba3 dependencies are optional")
    def test_all_heads_accept_mamba3(self):
        for cls in (MambaHead, AttentionHead, MambaAttentionHead, MambaTrackRegressionHead):
            with self.subTest(head=cls.__name__):
                head = cls(input_dim=64, embed_dim=64, num_layers=1,
                           num_embedder_layers=1, adapter_version="mamba3")
                self.assertTrue(architecture_metadata(head)["blocks"])
                self.assertTrue(all(c["version"] == "mamba3"
                                    for c in architecture_metadata(head)["blocks"].values()))

    def test_invalid_versions_and_mimo_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown backbone"):
            build_backbone({"backbone_version": "mamba4"})
        with self.assertRaisesRegex(ValueError, "Unknown adapter"):
            adapter_options({"adapter_version": "mamba4"})
        if not MAMBA3_AVAILABLE:
            return
        with self.assertRaisesRegex(ValueError, "SISO"):
            build_mamba_block("mamba3", d_model=64, block_kwargs={"is_mimo": True})
        with self.assertRaisesRegex(ValueError, "divisible"):
            build_mamba_block("mamba3", d_model=65)
        with self.assertRaisesRegex(ValueError, "Unsupported mamba3"):
            build_mamba_block("mamba3", d_model=64, block_kwargs={"use_mem_eff_path": True})

    def test_pretraining_yaml_selects_the_named_versions(self):
        configs = yaml.safe_load((ROOT / "scripts/configs/mamba_pretrain.yaml").read_text())
        for config, version in (("mamba_5m", "mamba1"), ("mamba2_5m", "mamba2"),
                                ("mamba3_siso", "mamba3")):
            self.assertEqual(backbone_version(configs[config], default="mamba1"), version)

    def test_mamba3_remains_an_optional_dependency(self):
        with patch.dict("sys.modules", {"fm4npp.models.mamba3": None}):
            self.assertEqual(build_backbone(backbone_params("mamba2")).backbone_version, "mamba2")
            with self.assertRaisesRegex(ImportError, "source installation"):
                build_backbone(backbone_params("mamba3"))

    @unittest.skipUnless(MAMBA3_AVAILABLE, "Mamba3 dependencies are optional")
    def test_mamba3_cpu_fails_with_an_actionable_message(self):
        model = build_backbone(backbone_params("mamba3"))
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            model(torch.rand(2, 5, 3))

    @unittest.skipUnless(MAMBA3_AVAILABLE, "Mamba3 dependencies are optional")
    def test_mamba3_initialization_and_no_decay_are_preserved(self):
        model = build_backbone(backbone_params("mamba3"))
        blocks = [layer[1] for layer in model.mamba_layers]
        before = [block.dt_bias.detach().clone() for block in blocks]
        for name, param in legacy_initialization_parameters(model):
            if "bias" in name:
                torch.nn.init.zeros_(param)
        for block, expected in zip(blocks, before):
            torch.testing.assert_close(block.dt_bias, expected, rtol=0, atol=0)
            self.assertTrue(torch.all(block.B_bias == 1))
            self.assertTrue(torch.all(block.C_bias == 1))
        optimizer = torch.optim.AdamW(optimizer_parameters(model), weight_decay=0.1)
        covered = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(covered), len(set(covered)))
        self.assertEqual(set(covered), {id(p) for p in model.parameters()})
        for group in optimizer.param_groups:
            if any(getattr(p, "_no_weight_decay", False) for p in group["params"]):
                self.assertEqual(group["weight_decay"], 0.0)
        # Preserve the single legacy optimizer group for old checkpoints.
        old = build_backbone(backbone_params("mamba1"))
        self.assertEqual(len(torch.optim.AdamW(optimizer_parameters(old)).param_groups), 1)


class MambaCheckpointTest(unittest.TestCase):
    def payload(self, model, kind, metadata=True):
        state_key = "model_state" if kind == "backbone" else "model_state_dict"
        payload = {state_key: model.state_dict()}
        if metadata:
            payload[f"{kind}_architecture"] = architecture_metadata(model)
        return payload

    def test_all_versions_roundtrip_new_and_legacy_checkpoints(self):
        for version in VERSIONS:
            for kind, model in (("backbone", build_backbone(backbone_params(version))),
                                ("adapter", regression_head(version))):
                for metadata in (False, True):
                    with self.subTest(version=version, kind=kind, metadata=metadata):
                        payload = self.payload(model, kind, metadata)
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / "checkpoint.pt"
                            torch.save(payload, path)
                            loaded = torch.load(path, weights_only=True)
                        validate_checkpoint_architecture(loaded, model, kind)
                        state = loaded["model_state" if kind == "backbone" else "model_state_dict"]
                        clone = copy.deepcopy(model)
                        clone.load_state_dict(state, strict=True)
                        # DDP prefixes remain accepted during validation.
                        loaded["model_state" if kind == "backbone" else "model_state_dict"] = {
                            "module." + k: v for k, v in state.items()}
                        validate_checkpoint_architecture(loaded, model, kind)

    def test_legacy_cross_version_loads_are_rejected_before_non_strict_loading(self):
        for source in VERSIONS:
            payload = self.payload(regression_head(source), "adapter", metadata=False)
            for target in VERSIONS:
                if source != target:
                    with self.subTest(source=source, target=target), self.assertRaisesRegex(ValueError, "configured as"):
                        validate_checkpoint_architecture(payload, regression_head(target), "adapter")
            empty = regression_head(source, num_layers=0, num_embedder_layers=0)
            with self.assertRaisesRegex(ValueError, "absent"):
                validate_checkpoint_architecture(payload, empty, "adapter")

    @unittest.skipUnless(MAMBA3_AVAILABLE, "Mamba3 dependencies are optional")
    def test_same_shape_semantic_changes_are_rejected(self):
        params = backbone_params("mamba3")
        source = build_backbone(params)
        target = build_backbone({**params, "backbone_rope_fraction": 1.0})
        with self.assertRaisesRegex(ValueError, "architecture does not match"):
            validate_checkpoint_architecture(self.payload(source, "backbone"), target, "backbone")

    @unittest.skipUnless(MAMBA3_AVAILABLE, "Mamba3 dependencies are optional")
    def test_frozen_backbone_metadata_is_checked_when_loading_an_adapter(self):
        head = regression_head()
        payload = self.payload(head, "adapter")
        source = build_backbone(backbone_params("mamba3"))
        payload["backbone_architecture"] = architecture_metadata(source)
        validate_checkpoint_architecture(payload, head, "adapter", backbone=source)
        with self.assertRaisesRegex(ValueError, "different backbone"):
            validate_checkpoint_architecture(payload, head, "adapter",
                                             backbone=build_backbone(backbone_params("mamba1")))

    @unittest.skipUnless(MAMBA3_AVAILABLE, "Mamba3 dependencies are optional")
    def test_regression_evaluator_uses_adapter_selection(self):
        from train.downstream.eval.evaluate_track_regression import build_head
        trainer = SimpleNamespace(
            params=SimpleNamespace(embed_dim=64, num_output_classes=3, num_layers_backbone=2,
                                   num_embedder_layers=1, embed_method="pos_only", pe_method="nerf",
                                   adapter_version="mamba3", adapter_headdim=32),
            regression_target_stats={"mean": [0.] * 3, "std": [1.] * 3}, device="cpu")
        head = build_head(trainer)
        self.assertEqual(head.adapter_version, "mamba3")
        self.assertEqual(head.mamba_layers[0][1].headdim, 32)

    def test_mamba2_cpu_fallback_optimizer_step(self):
        # Explicitly test the historical portable path, without treating it as
        # numerical validation of the official fused Mamba2 algorithm.
        with patch("fm4npp.models.mamba2.RMSNormGated", None):
            model = build_backbone({**backbone_params("mamba2"), "use_mem_eff_path": False})
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        prediction = model(torch.rand(2, 5, 3))
        self.assertEqual(prediction.shape, (2, 5, 6))
        prediction.square().mean().backward()
        optimizer.step()
        _, features, _ = model(torch.rand(2, 5, 3), return_z=True)
        self.assertEqual(len(features), 2)
        self.assertTrue(all(torch.isfinite(feature).all() for feature in features))

    def test_pretraining_trainer_step_validation_and_resume(self):
        from train.pretrain.nppmamba import train_multi_gpu_mamba1 as pretrain
        params = SimpleNamespace(
            **backbone_params("mamba2"), use_mem_eff_path=False, stat_dir="/unused",
            min_lr=1e-3, max_lr=2e-3, total_steps=4, warmup_steps=0,
            rep_aaai=False, loss_reweight=False, ablate_loss_scale=False,
            grad_clip_value=1.0, n_eval_steps=1, save_checkpoint=True)
        batch = (torch.rand(2, 5, 3), None, torch.rand(2, 5, 6))
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(root_dir=directory, config="mamba2", run_num="test")
            with patch.object(pretrain, "get_data_loader", return_value=([batch], None, [batch], None)), \
                    patch.object(pretrain, "pickle_load", return_value=[0., 1.]), \
                    patch("fm4npp.models.mamba2.RMSNormGated", None):
                trainer = pretrain.Trainer(params, args)
                trainer.train_one_epoch()
                self.assertEqual(trainer.iters, 1)
                self.assertTrue(torch.isfinite(trainer.logs["val_loss"]).all())
                resumed = pretrain.Trainer(params, args)
                self.assertEqual(resumed.iters, trainer.iters)
                self.assertEqual(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())
                for name, state in trainer.model.state_dict().items():
                    torch.testing.assert_close(resumed.model.state_dict()[name], state, rtol=0, atol=0)
                resumed.train_one_epoch()
                self.assertEqual(resumed.iters, 2)

    @unittest.skipUnless(MAMBA3_AVAILABLE, "Mamba3 dependencies are optional")
    def test_mamba3_pretraining_trainer_preserves_ssm_initialization(self):
        from train.pretrain.nppmamba import train_multi_gpu_mamba1 as pretrain
        params = SimpleNamespace(**backbone_params("mamba3"), stat_dir="/unused",
                                 min_lr=1e-3, max_lr=2e-3, total_steps=4, warmup_steps=0)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(root_dir=directory, config="mamba3", run_num="test")
            with patch.object(pretrain, "get_data_loader", return_value=([], None, [], None)), \
                    patch.object(pretrain, "pickle_load", return_value=[]):
                trainer = pretrain.Trainer(params, args)
                for layer in trainer.model.mamba_layers:
                    self.assertTrue(torch.all(layer[1].dt_bias < 0))
                    self.assertTrue(torch.all(layer[1].B_bias == 1))
                trainer.save_checkpoint(trainer.checkpoint_path)
                resumed = pretrain.Trainer(params, args)
                self.assertEqual(architecture_metadata(resumed.model), architecture_metadata(trainer.model))


@unittest.skipUnless(torch.cuda.is_available() and MAMBA3_AVAILABLE,
                     "CUDA and Mamba3 dependencies are required for real kernel execution")
class MambaCUDATest(unittest.TestCase):
    def test_pretraining_forward_backward_and_reload(self):
        for version, width, state in (("mamba2", 64, 16), ("mamba3", 64, 16),
                                      ("mamba3", 256, 128)):
            with self.subTest(version=version, width=width, state=state):
                params = {**backbone_params(version), "embed_dim": width, "d_state": state}
                model = build_backbone(params).cuda()
                if version == "mamba2":
                    self.assertTrue(all(layer[1].use_mem_eff_path for layer in model.mamba_layers),
                                    "CUDA validation must execute fused Mamba2, not its simplified fallback")
                optimizer = torch.optim.AdamW(optimizer_parameters(model), lr=1e-3)
                # Include a partial chunk and right padding.
                points = torch.rand(2, 65, 3, device="cuda")
                points[1, -7:] = -100
                pred = model(points)
                loss = pred[:, :-7].float().square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertTrue(any(p.grad is not None for p in model.mamba_layers.parameters()))
                optimizer.step()
                clone = build_backbone(params).cuda().eval()
                clone.load_state_dict(model.state_dict(), strict=True)
                model.eval()
                torch.testing.assert_close(clone(points), model(points))

    def test_mixed_backbones_and_adapters_at_representative_widths(self):
        points = torch.rand(2, 65, 3, device="cuda")
        mask = torch.ones(2, 65, dtype=torch.bool, device="cuda")
        mask[1, -7:] = False
        points[~mask] = -100
        cases = (("mamba1", "mamba3", 256), ("mamba2", "mamba3", 256),
                 ("mamba3", "mamba2", 256), ("mamba3", "mamba3", 256),
                 ("mamba3", "mamba2", 512))
        for version, adapter, width in cases:
            with self.subTest(backbone=version, adapter=adapter, width=width):
                params = {**backbone_params(version), "embed_dim": width,
                          "d_state": 16 if version == "mamba1" else 128}
                backbone = build_backbone(params).cuda().eval()
                head = regression_head(adapter, input_dim=width, embed_dim=256,
                                       d_state=64, d_state_embedder=64).cuda()
                optimizer = torch.optim.AdamW(optimizer_parameters(head), lr=1e-3)
                with torch.no_grad():
                    _, features, _ = backbone(points, return_z=True)
                pred = head(points, torch.stack(features), padding_mask=mask,
                            pretrain=True)["pred_regression"]
                pred.square().mean().backward()
                self.assertTrue(torch.isfinite(pred).all())
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                    for p in head.parameters()))
                self.assertTrue(any(p.grad is not None for p in head.mamba_layers.parameters()))
                self.assertTrue(all(p.grad is None for p in backbone.parameters()))
                optimizer.step()

    def test_mamba3_adapteronly_and_frozen_features(self):
        backbone = build_backbone(backbone_params("mamba3")).cuda().eval()
        points = torch.rand(2, 65, 3, device="cuda")
        mask = torch.ones(2, 65, dtype=torch.bool, device="cuda")
        mask[1, -7:] = False
        points[~mask] = -100
        for pretrained in (False, True):
            with self.subTest(pretrained=pretrained):
                head = regression_head("mamba3").cuda()
                optimizer = torch.optim.AdamW(optimizer_parameters(head), lr=1e-3)
                with torch.no_grad():
                    _, features, _ = backbone(points, return_z=True)
                result = head(points, torch.stack(features) if pretrained else None,
                              padding_mask=mask, pretrain=pretrained)["pred_regression"]
                result.square().mean().backward()
                self.assertTrue(torch.isfinite(result).all())
                self.assertTrue(any(p.grad is not None for p in head.mamba_layers.parameters()))
                self.assertTrue(all(p.grad is None for p in backbone.parameters()))
                optimizer.step()

    def test_mamba3_classification_and_tracking_heads(self):
        points = torch.rand(2, 65, 3, device="cuda")
        mask = torch.ones(2, 65, dtype=torch.bool, device="cuda")
        mask[1, -7:] = False
        points[~mask] = -100
        for cls, key, extra in ((MambaHead, "pred_logits", {}),
                                (AttentionHead, "pred", {}),
                                (MambaAttentionHead, "mask_probs", {"num_output_dim": 64, "num_prototypes": 3})):
            for pretrained in (False, True):
                with self.subTest(head=cls.__name__, pretrained=pretrained):
                    head = cls(input_dim=64, embed_dim=64, num_layers=1,
                               num_embedder_layers=1, num_feature_layers=2,
                               embed_method="pos_only", adapter_version="mamba3", **extra).cuda()
                    features = torch.rand(2, 2, 65, 64, device="cuda") if pretrained else None
                    pred = head(points, features, padding_mask=mask, pretrain=pretrained)[key]
                    pred.square().mean().backward()
                    self.assertTrue(torch.isfinite(pred).all())
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))
                    self.assertTrue(any(p.grad is not None for p in head.parameters()
                                        if getattr(p, "_preserve_mamba_init", False))
                                    or (pretrained and cls is AttentionHead))


if __name__ == "__main__":
    unittest.main()
