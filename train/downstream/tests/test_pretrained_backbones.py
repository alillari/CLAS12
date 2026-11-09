"""Portable selection, native input contracts, and strict checkpoint reuse."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from ruamel.yaml import YAML

from fm4npp.models.mamba_config import build_backbone, architecture_metadata
from fm4npp.pretrained import (
    catalog_selections, configure_pretrained_backbone,
    export_backbone_config, read_catalog, resolve_backbone,
)
from fm4npp.utils import YParams

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "train/downstream/campaign"
sys.path.insert(0, str(CAMPAIGN))
from train.downstream.campaign import campaign_util as campaign
from train.downstream.campaign import build_track_finding_manifest as finding_builder
from train.downstream.campaign import build_track_regression_manifest as regression_builder
from train.downstream.campaign import run_track_finding_campaign as finding_runner
from train.downstream.track_finding_experiment import TrackFindingExperimentConfig, resolve_params as finding_params
from train.downstream.track_regression_experiment import TrackRegressionExperimentConfig, resolve_params as regression_params
from train.downstream.train_point_classification import parse_args as classification_args, configure_training

try:
    from fm4npp.models.mamba3 import Mamba3SISO
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False


@unittest.skipUnless(HAS_MAMBA3, "Mamba3 checkpoint fixtures require its optional dependencies")
class PretrainedBackboneTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Real Mamba3 topology with Mike's target width; no CUDA execution needed.
        self.config = dict(backbone_version="mamba3", embed_dim=64, num_layers_backbone=2,
                           d_state=128, headdim=64, expand=2, klen=5, use_sincos_target=True,
                           embed_method="pos_only", pe_method="nerf", dropout=0., hilbert_angular_grid="unit_v2")
        self.model = build_backbone(self.config)
        self.checkpoint = self.root / "weights/arbitrary/nested/training_checkpoints/ckpt_best.tar"
        self.checkpoint.parent.mkdir(parents=True)
        self.payload = {"model_state": self.model.state_dict(),
                        "backbone_architecture": architecture_metadata(self.model),
                        "hilbert_angular_grid": "unit_v2"}
        torch.save(self.payload, self.checkpoint)
        self.catalog = self.root / "catalog.yaml"
        self.catalog_data = dict(version=1, defaults={key: self.config[key] for key in
                                 ("embed_method", "pe_method", "klen", "use_sincos_target")},
                                 backbones={"small": {"checkpoint": str(self.checkpoint.relative_to(self.root)),
                                                       "sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()}})
        self.write_catalog()

    def write_catalog(self):
        with self.catalog.open("w") as stream:
            YAML().dump(self.catalog_data, stream)

    def resolve(self):
        return resolve_backbone("small", catalog=self.catalog)

    def test_catalog_needs_no_directory_tokens_and_resolves_native_dimensions(self):
        resolved = self.resolve()
        self.assertEqual(resolved["checkpoint"], str(self.checkpoint))
        self.assertEqual(resolved["config"]["backbone_version"], "mamba3")
        self.assertEqual(resolved["config"]["num_layers_backbone"], 2)
        self.assertEqual(resolved["config"]["d_state"], 128)
        self.assertEqual(resolved["config"]["backbone_output_dim"], 20)
        self.assertEqual(len(catalog_selections(self.catalog, names=["small", "small"])), 1)

    def test_new_checkpoint_is_self_describing_without_catalog_or_recipe(self):
        self.payload["backbone_config"] = export_backbone_config(SimpleNamespace(**self.config), self.model)
        torch.save(self.payload, self.checkpoint)
        resolved = resolve_backbone({"checkpoint": str(self.checkpoint)})
        restored = build_backbone(resolved["config"])
        restored.load_state_dict(self.payload["model_state"], strict=True)
        self.assertEqual(restored.output_layer.out_features, 20)
        self.assertNotIn("task", resolved["config"])

    def test_catalog_rejects_changed_weights_and_conflicting_metadata(self):
        self.catalog_data["backbones"]["small"]["sha256"] = "0" * 64
        self.write_catalog()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.resolve()
        self.catalog_data["backbones"]["small"].pop("sha256")
        for key, value in (("d_state", 16), ("hilbert_angular_grid", "legacy_v1"), ("backbone_version", "mamba1")):
            with self.subTest(field=key):
                self.catalog_data["backbones"]["small"]["config"] = {key: value}
                self.write_catalog()
                with self.assertRaisesRegex(ValueError, "Conflicting backbone settings"):
                    self.resolve()

    def test_unknown_ids_and_unsafe_ids_are_rejected(self):
        with self.assertRaisesRegex(KeyError, "Unknown backbone"):
            catalog_selections(self.catalog, names=["missing"])
        self.catalog_data["backbones"]["../escape"] = {}
        self.write_catalog()
        with self.assertRaisesRegex(ValueError, "Unsafe backbone"):
            read_catalog(self.catalog)

    def test_missing_input_metadata_is_not_guessed_from_weights(self):
        with self.assertRaisesRegex(ValueError, "missing backbone settings"):
            resolve_backbone({"checkpoint": str(self.checkpoint)})
        self.catalog_data["defaults"]["use_sincos_target"] = False
        self.write_catalog()
        with self.assertRaisesRegex(ValueError, "output width 20"):
            self.resolve()

    def test_root_override_and_environment_do_not_require_editing_catalog(self):
        self.catalog_data["root"] = "${UNSET_BACKBONE_TEST_ROOT}"
        self.write_catalog()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unresolved environment"):
                catalog_selections(self.catalog)
            self.assertEqual(catalog_selections(self.catalog, root=self.root)[0]["checkpoint"], str(self.checkpoint))
        with patch.dict(os.environ, {"CLAS12_PRETRAINED_ROOT": str(self.root)}):
            self.assertEqual(self.resolve()["checkpoint"], str(self.checkpoint))

    def test_resolution_replaces_stale_recipe_aliases_and_keeps_adapter_recipe(self):
        params = dict(backbone_version="mamba1", mambaversion="mamba1", embed_dim=512,
                      base_dim=512, num_layers_backbone=12, backbone_d_state=16,
                      klen=1, d_state=16, adapter_version="mamba2", adapter_embed_dim=256,
                      task="nid", max_lr=.001, use_mem_eff_path=True)
        configure_pretrained_backbone(params, backbone="small", catalog=self.catalog)
        self.assertEqual((params["backbone_version"], params["mambaversion"]), ("mamba3", "mamba3"))
        self.assertEqual((params["embed_dim"], params["base_dim"], params["backbone_d_state"]), (64, 64, 128))
        self.assertEqual((params["adapter_version"], params["adapter_embed_dim"], params["task"], params["max_lr"]),
                         ("mamba2", 256, "nid", .001))
        restored = build_backbone(params)
        restored.load_state_dict(self.payload["model_state"], strict=True)
        self.assertEqual(params["pretrained_backbone"]["sha256"], self.resolve()["sha256"])

    def test_adapter_only_clears_selection_even_if_missing_on_this_machine(self):
        params = {"pretrained_ckpt": "/missing", "pretrained_backbone": "small"}
        configure_pretrained_backbone(params, enabled=False)
        self.assertIsNone(params["pretrained_ckpt"])
        self.assertIsNone(params["pretrained_backbone"])
        configure_pretrained_backbone(params)

    def test_explicit_path_and_selector_must_identify_the_same_weights(self):
        with self.assertRaisesRegex(ValueError, "different checkpoints"):
            configure_pretrained_backbone({}, backbone="small", catalog=self.catalog, checkpoint="/other")

    def test_explicit_selection_relocates_a_previously_resolved_recipe(self):
        params = {}
        configure_pretrained_backbone(params, backbone="small", catalog=self.catalog)
        relocated = self.root / "relocated" / self.checkpoint.relative_to(self.root)
        relocated.parent.mkdir(parents=True)
        relocated.write_bytes(self.checkpoint.read_bytes())
        configure_pretrained_backbone(params, backbone="small", catalog=self.catalog, root=self.root / "relocated")
        self.assertEqual(params["pretrained_ckpt"], str(relocated))

    def test_legacy_path_without_metadata_retains_matching_recipe(self):
        torch.save({"model_state": self.model.state_dict()}, self.checkpoint)
        params = dict(self.config)
        configure_pretrained_backbone(params, checkpoint=str(self.checkpoint))
        build_backbone(params).load_state_dict(self.model.state_dict(), strict=True)

    def test_all_three_real_trainers_load_sincos_head_strictly(self):
        from train.downstream.track_finding_trainer import DownstreamTrainer as Finder
        from train.downstream.track_regression_trainer import DownstreamTrainer as Regressor
        from train.downstream.point_classification_trainer import DownstreamTrainer as Classifier
        resolved = self.resolve()
        original = self.checkpoint.read_bytes()
        for cls in (Finder, Regressor, Classifier):
            with self.subTest(trainer=cls.__module__):
                trainer = cls.__new__(cls)
                trainer.params = SimpleNamespace(**resolved["config"], pretrained_backbone=resolved)
                trainer.device, trainer.world_rank = "cpu", 1
                trainer.model = build_backbone(resolved["config"])
                trainer.restore_checkpoint(self.checkpoint, load_optimizer_state=False)
                torch.testing.assert_close(trainer.model.output_layer.weight, self.model.output_layer.weight)
                with self.checkpoint.open("ab") as stream:
                    stream.write(b"changed after resolution")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    trainer.restore_checkpoint(self.checkpoint, load_optimizer_state=False)
                self.checkpoint.write_bytes(original)

    def test_single_run_entrypoints_accept_catalog_selection_without_architecture_overrides(self):
        cases = [(finding_params, TrackFindingExperimentConfig, "track_finding_adapteronly"),
                 (regression_params, TrackRegressionExperimentConfig, "track_regression_pretrained")]
        for resolve, config_type, profile in cases:
            params = resolve(config_type(yaml_config=str(ROOT / f"scripts/configs/mamba_clas12_{profile}.yaml"),
                                         config=f"clas12_{profile}", backbone="small", backbone_catalog=str(self.catalog)))
            self.assertEqual(params.embed_dim, 64)
            self.assertEqual(params.params["backbone_version"], "mamba3")
            self.assertEqual(params.params["pretrained_ckpt"], str(self.checkpoint))
        args = classification_args(["--yaml_config", "unused", "--config", "noise", "--backbone", "small",
                                    "--backbone-catalog", str(self.catalog)])
        params = YParams(str(ROOT / "scripts/configs/clas12_adapter_recipes.yaml"), "hdf5_noise_50k_balanced")
        configure_training(params, args)
        self.assertEqual((params.embed_dim, params.params["embed_dim"], params.klen), (64, 64, 5))

    def test_both_campaigns_render_resolved_models_and_reject_conflicting_overrides(self):
        for task, builder, render in [("finding", finding_builder, finding_runner.render_model_yaml),
                                       ("regression", regression_builder, campaign.render_model_yaml)]:
            with self.subTest(task=task):
                manifest_path = self.root / f"{task}.yaml"
                with patch.object(sys, "argv", ["build", "--backbone-catalog", str(self.catalog),
                                                "--backbone", "small", "--artifact-root", str(self.root),
                                                "--manifest", str(manifest_path), "--no-adapter-only"]):
                    builder.main()
                manifest = campaign.read_yaml(manifest_path)
                self.assertEqual(len(manifest["runs"]), 1)
                row = manifest["runs"][0]
                self.assertEqual((row["model_family"], row["embed_dim"]), ("mamba3", 64))
                render(manifest, row)
                params = campaign.read_yaml(Path(row["model_yaml"]))[row["model_config"]]
                self.assertEqual((params["klen"], params["backbone_output_dim"], params["d_state"]), (5, 20, 128))
                # Evaluation and subprocess training need no access to the catalog.
                self.assertEqual(params["pretrained_backbone"]["checkpoint"], str(self.checkpoint))
                configure_pretrained_backbone(params)
                build_backbone(params).load_state_dict(self.payload["model_state"], strict=True)
                manifest["training_overrides"]["d_state"] = 16
                with self.assertRaisesRegex(ValueError, "Conflicting backbone"):
                    render(manifest, row)

    def test_discovery_handles_nested_best_and_ambiguous_runs(self):
        source = self.root / "weights/arbitrary"
        (self.checkpoint.parent / "ckpt.tar").touch()
        self.assertEqual(campaign.discover_checkpoint(source), self.checkpoint)
        other = source / "another/ckpt_best.tar"
        other.parent.mkdir()
        other.touch()
        with self.assertRaisesRegex(ValueError, "Multiple best"):
            campaign.discover_checkpoint(source)
        (source / "metadata.json").write_text(json.dumps({"checkpoint": str(self.checkpoint.relative_to(source))}))
        self.assertEqual(campaign.discover_checkpoint(source), self.checkpoint)

    def test_sidecar_can_describe_an_arbitrary_directory(self):
        source = self.root / "weights/arbitrary"
        metadata = {"checkpoint": str(self.checkpoint.relative_to(source)), "backbone_config": self.catalog_data["defaults"]}
        (source / "metadata.json").write_text(json.dumps(metadata))
        row = campaign.build_run_row(source, campaign.discover_checkpoint(source), self.root, 100, 4, 100)
        self.assertEqual((row["model_family"], row["embed_dim"], row["num_layers_backbone"]), ("mamba3", 64, 2))


if __name__ == "__main__":
    unittest.main()
