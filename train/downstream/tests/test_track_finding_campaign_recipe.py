"""A single-run profile remains the recipe used by campaign rendering/preflight."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

CAMPAIGN = Path(__file__).resolve().parents[1] / "campaign"
with patch.object(sys, "path", [str(CAMPAIGN), *sys.path]):
    from train.downstream.campaign import build_track_finding_manifest as builder
    from train.downstream.campaign import run_track_finding_campaign as runner


class TrackFindingCampaignRecipeTest(unittest.TestCase):
    def test_named_single_run_profile_reaches_rows_rendering_and_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = root / "backbones"
            checkpoint = checkpoints / "scale_w512_d12" / "ckpt_best.tar"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()  # Metadata/dry-run fixture; never loaded or trained.
            manifest_path = root / "manifest.yaml"
            with patch.dict(os.environ, {"CLAS12_DATA_ROOT": str(root / "events")}):
                with patch.object(sys, "argv", [
                    "build", "--checkpoint-root", str(checkpoints),
                    "--base-model-yaml", "scripts/configs/clas12_adapter_recipes.yaml",
                    "--base-model-config", "clas12_track_finding_50k",
                    "--artifact-root", str(root), "--campaign-name", "recipe_test",
                    "--manifest", str(manifest_path), "--eventnumber", "50000",
                ]):
                    builder.main()
                manifest = runner.read_yaml(manifest_path)
                self.assertEqual(len(manifest["runs"]), 2)
                for row in manifest["runs"]:
                    self.assertEqual(row["base_model_config"], "clas12_track_finding_50k")
                    runner.render_model_yaml(manifest, row)
                    params = runner.read_yaml(Path(row["model_yaml"]))[row["model_config"]]
                    self.assertEqual(params["num_prototypes"], 16)
                    self.assertEqual(params["max_optimizer_steps"], 30000)
                with patch.object(sys, "argv", ["run", "--manifest", str(manifest_path), "--preflight-only"]), \
                     patch.object(runner, "preflight_dataset", return_value={"passed": True}) as preflight:
                    runner.main()
                self.assertEqual(preflight.call_args.args[0], root / "events")

    def test_explicit_unknown_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(sys, "argv", [
                "build", "--base-model-yaml", "scripts/configs/clas12_adapter_recipes.yaml",
                "--base-model-config", "misspelled", "--artifact-root", temporary,
            ]), self.assertRaisesRegex(KeyError, "misspelled"):
                builder.main()


if __name__ == "__main__":
    unittest.main()
