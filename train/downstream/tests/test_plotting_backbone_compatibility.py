"""Plot coordinates and model sizes must follow the supplied campaign metadata."""
import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from ruamel.yaml import YAML

CAMPAIGN = Path(__file__).resolve().parents[1] / "campaign"
with patch.object(sys, "path", [str(CAMPAIGN), *sys.path]):
    import plot_track_finding_campaign as finding
    import plot_track_regression_campaign as regression
    from momentum_presentation import model_catalog, select_runs


def finding_rows(count, budgets=(32, 5000, 250000)):
    rows = []
    for index in range(count + 1):
        name = "adapteronly" if index == 0 else f"backbone_{index}"
        for budget in budgets:
            row = dict(run_id=f"{name}_label{budget}", backbone_run_id=name,
                       model_family="adapteronly" if index == 0 else "transformer",
                       embed_dim=32 * max(index, 1), num_layers_backbone=2,
                       labeled_events=budget)
            row.update({key: .7 for key, _ in finding.METRICS})
            row.update({key.replace("native_", "native_coatjava_"): .6
                        for key, _ in finding.METRICS})
            rows.append(row)
    return rows


class FindingPlotCompatibilityTest(unittest.TestCase):
    def render_paper(self, rows):
        captures = []
        def capture(figure, path, **kwargs):
            captures.append((Path(path).name, [ax.get_xlim() for ax in figure.axes]))
        with tempfile.TemporaryDirectory() as temporary, patch.object(Figure, "savefig", capture):
            finding.paper_suite(rows, Path(temporary), finding.model_labels(rows))
        return captures

    def test_presentation_axes_include_actual_training_budgets(self):
        captures = self.render_paper(finding_rows(3))
        self.assertTrue(captures)
        for filename, limits in captures:
            for low, high in limits:
                with self.subTest(filename=filename):
                    self.assertLess(low, 32)
                    self.assertGreater(high, 250000)

    def test_focus_layout_does_not_require_exactly_six_backbones(self):
        for count in (1, 4, 6, 7):
            with self.subTest(count=count):
                names = {name for name, _ in self.render_paper(finding_rows(count))}
                self.assertIn(f"track_finding_adapter_m{count}_coatjava_1x3.png", names)

    def test_negative_signal_ari_is_visible(self):
        rows = finding_rows(1)
        for row in rows:
            row["native_ari_signal"] = -.2
        limits = []
        def capture(figure, path, **kwargs):
            limits.append(figure.axes[0].get_ylim())
        with tempfile.TemporaryDirectory() as temporary, patch.object(Figure, "savefig", capture):
            finding.paper_suite(rows, Path(temporary), finding.model_labels(rows))
        self.assertTrue(limits)
        self.assertTrue(all(low < -.2 for low, high in limits))


class BackbonePlotMetadataTest(unittest.TestCase):
    def test_same_width_new_backbones_remain_separate_plot_series(self):
        metadata = {name: dict(backbone_run_id=name, model_family="transformer",
                              embed_dim=32, num_layers_backbone=2, labeled_events=32)
                    for name in ("nope", "rope")}
        metrics = [dict(run_name=name, method="adapter", record_type="ml_error",
                        space="component", variable="px_gev", mae=.1, rmse=.2, r2=.8)
                   for name in metadata]
        rows = regression.build_plot_rows(metrics, metadata)
        self.assertEqual(len({row["backbone_width_label"] for row in rows}), 2)

    def test_parameter_counter_uses_full_transformer_config_and_cache_identity(self):
        from fm4npp.models.mamba_config import build_backbone
        from fm4npp.utils import count_parameters
        count = regression.parameter_counter()
        self.assertIsNotNone(count)
        values = []
        for ratio in (1., 3.):
            config = dict(model_family="transformer", backbone_version="transformer",
                          embed_dim=32, num_layers_backbone=2, klen=1,
                          embed_method="pos_only", pe_method="nerf", dropout=0.,
                          num_heads_backbone=4, backbone_sequence_encoding="none",
                          backbone_mlp_ratio=ratio)
            with contextlib.redirect_stdout(io.StringIO()):
                expected = count_parameters(build_backbone(config))
            values.append(count(config))
            self.assertEqual(values[-1], expected)
        self.assertNotEqual(*values)

    def test_csv_architecture_values_are_typed_before_construction(self):
        config = dict(model_family="transformer", backbone_version="transformer",
                      embed_dim=32, num_layers_backbone=2, klen=1, embed_method="pos_only",
                      pe_method="nerf", dropout=0., num_heads_backbone=4,
                      backbone_mlp_ratio=1., backbone_sequence_encoding="none")
        count = regression.parameter_counter()
        self.assertEqual(count({key: str(value) for key,value in config.items()}), count(config))

    def test_rendered_model_overrides_stale_manifest_family(self):
        from fm4npp.models.mamba_config import build_backbone
        from fm4npp.utils import count_parameters
        config = dict(backbone_version="transformer", embed_dim=32, num_layers_backbone=2,
                      klen=1, embed_method="pos_only", pe_method="nerf", dropout=0.,
                      num_heads_backbone=4, backbone_sequence_encoding="rope",
                      backbone_mlp_ratio=1.)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.yaml"
            with path.open("w") as stream:
                YAML().dump({"run": config}, stream)
            row = dict(config, backbone_version="mamba1", model_family="mamba1",
                       model_yaml=str(path), model_config="run")
            with contextlib.redirect_stdout(io.StringIO()):
                expected = count_parameters(build_backbone(config))
            self.assertEqual(regression.parameter_counter()(row), expected)

    def test_portable_descriptor_preserves_native_output_width_without_checkpoint_access(self):
        from fm4npp.models.mamba_config import build_backbone
        from fm4npp.utils import count_parameters
        config = dict(backbone_version="transformer", embed_dim=32, num_layers_backbone=2,
                      klen=5, use_sincos_target=True, backbone_output_dim=20,
                      embed_method="pos_only", pe_method="nerf", dropout=0.,
                      num_heads_backbone=4, backbone_sequence_encoding="rope")
        row = dict(model_family="mamba1", embed_dim=64,
                   model_yaml="/unavailable/training-machine/model.yaml", model_config="run",
                   pretrained_backbone={"checkpoint": "/unavailable/weights.tar", "config": config})
        with contextlib.redirect_stdout(io.StringIO()):
            expected = count_parameters(build_backbone(config))
        self.assertEqual(regression.parameter_counter()(row), expected)
        self.assertEqual(finding.nparams(row), expected)

    def test_finding_csv_recovers_portable_descriptor_from_manifest(self):
        config = dict(backbone_version="transformer", embed_dim=32, num_layers_backbone=2,
                      klen=5, backbone_output_dim=20, embed_method="pos_only", pe_method="nerf")
        descriptor = {"checkpoint": "/other-machine/weights.tar", "config": config}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary"
            summary.mkdir()
            with (root / "manifest.yaml").open("w") as stream:
                YAML().dump({"runs": [{"run_id": "catalog_model", "pretrained_backbone": descriptor}]}, stream)
            with (summary / "run_table.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["run_id", "summary_found", "native_ari_signal"])
                writer.writeheader()
                writer.writerow(dict(run_id="catalog_model", summary_found="true", native_ari_signal=.75))
            row, = finding.campaign_rows(root, summary)
            self.assertEqual(row["pretrained_backbone"], descriptor)
            self.assertEqual(float(row["native_ari_signal"]), .75)
            external = root / "external.yaml"
            other = dict(descriptor, checkpoint="/another-machine/weights.tar")
            with external.open("w") as stream:
                YAML().dump({"runs": [{"run_id": "catalog_model", "pretrained_backbone": other}]}, stream)
            external_row, = finding.campaign_rows(root, summary, external)
            self.assertEqual(external_row["pretrained_backbone"], other)


    def test_momentum_budget_selection_accepts_new_families_without_50k(self):
        rows = {}
        for family in ("mamba3", "nope", "rope"):
            for budget in (32, 5000, 250000):
                rows[f"{family}_{budget}"] = dict(backbone_run_id=family,
                    model_family="mamba3" if family == "mamba3" else "transformer",
                    embed_dim=64, num_layers_backbone=2, labeled_events=budget)
        catalog, runs, ordered = model_catalog(rows, rows)
        budget, selected = select_runs(runs)
        self.assertEqual(budget, 250000)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(ordered), 3)
        self.assertEqual(len({catalog[name]["plot_label"] for name in ordered}), 3)


if __name__ == "__main__":
    unittest.main()
