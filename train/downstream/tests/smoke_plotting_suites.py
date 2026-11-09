"""CPU artifact smoke for maintained plot CLIs; inputs are explicitly synthetic.

Run from the repository root with --output-dir in an isolated writable directory.
An optional real finding campaign also checks stored evaluation/history products.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "train/downstream"), str(ROOT / "train/downstream/campaign")]
import numpy as np
from ruamel.yaml import YAML
from train.downstream.evaluation_contract import EVALUATION_CONTRACT
from train.downstream.tests.test_plotting_backbone_compatibility import finding_rows
from train.downstream.tests.test_noise_campaign import write_run, manifest as noise_manifest
from train.downstream.tests.test_regression_campaign_physics import CampaignPhysicsTest
from train.downstream.campaign.campaign_util import collate_summary

CONTRACT = dict(evaluation_contract=EVALUATION_CONTRACT,
                comparison_truth="mctrue_inner_hit", swingback_enabled=False)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def yaml_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        YAML().dump(value, stream)


def relocated_history(receipt_path):
    receipt = json.loads(receipt_path.read_text())
    history = Path(receipt["validation_history_dir"])
    if not history.is_absolute():
        return receipt_path.parent / history
    if history.is_dir():
        return history
    source_run = Path(receipt["checkpoint_dir"]).parent
    return receipt_path.parent.parent / history.relative_to(source_run)


def make_joint_campaign(directory):
    rows, headlines, fits = [], [], []
    for model in ("adapteronly", "mamba3", "transformer_nope", "transformer_rope"):
        for budget in (32, 250000):
            name = f"{model}_label{budget}"
            config = dict(backbone_version="mamba3" if model == "mamba3" else "transformer",
                          embed_dim=64, num_layers_backbone=2, klen=5,
                          backbone_output_dim=20, use_sincos_target=True,
                          embed_method="pos_only", pe_method="nerf",
                          d_state=128, headdim=64, num_heads_backbone=4,
                          backbone_mlp_ratio=1., backbone_sequence_encoding="none" if model.endswith("nope") else "rope")
            path = directory / "configs" / f"{name}.yaml"
            yaml_file(path, {name: config})
            row = dict(run_id=name, backbone_run_id=model,
                       model_family="adapteronly" if model == "adapteronly" else config["backbone_version"],
                       use_pretrained_backbone=model != "adapteronly", embed_dim=64,
                       num_layers_backbone=2, pretrain_events=10000, labeled_events=budget,
                       eventnumber=budget, model_yaml=str(path), model_config=name)
            rows.append(row)
            for method in ("adapter", "cvt"):
                value = .1 if method == "cvt" else .08 if budget == 250000 else .12
                for space, variables in (("component", ("px_gev", "py_gev", "pz_gev")),
                                         ("kinematic", ("p_gev", "theta_deg", "phi_deg"))):
                    for variable in variables:
                        headlines.append(dict(CONTRACT, run_name=name, run_num=name, method=method,
                            record_type="ml_error", space=space, variable=variable,
                            training_target_task="mom", unit="deg" if variable.endswith("deg") else "GeV",
                            mae=value, rmse=value * 1.3, r2=.7, n=64))
                for low, high in ((.25, .5), (.5, 1.0)):
                    fits.append(dict(evaluation_contract=EVALUATION_CONTRACT, comparison_truth="mctrue_inner_hit",
                        run_id=name, method=method, model_family=row["model_family"],
                        use_pretrained_backbone=row["use_pretrained_backbone"], embed_dim=64,
                        pretrain_events=10000, labeled_events=budget, bin_low_gev=low, bin_high_gev=high,
                        bin_center_gev=(low + high) / 2, fit_mean=.001, fit_sigma=.02,
                        fit_mean_error=.0001, fit_sigma_error=.001, n=32, fit_status="ok"))
    yaml_file(directory / "manifest.yaml", {"runs": rows})
    summary = directory / "summary"
    summary.mkdir()
    (summary / "campaign_headline_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in headlines))
    for metric in ("delta_p_over_p", "delta_theta"):
        write_csv(summary / f"{metric}_fits.csv", fits)
    return rows


def make_scalar_campaigns(directory):
    fixture = CampaignPhysicsTest()
    evaluations = {}
    for task in ("p", "theta"):
        root = directory / task
        root.mkdir(parents=True)
        run = fixture.create_run(directory, 32, task=task)
        campaign = fixture.manifest(root)
        campaign["runs"] = [run]
        collate_summary(campaign)
        evaluation = Path(run["evaluation_dir"])
        count = 64
        values = np.linspace(.3, 2.8, count) if task == "p" else np.linspace(.7, 2.3, count)
        variable = "p_gev" if task == "p" else "theta_rad"
        rows = [dict(source_file="synthetic", event=i, source_event_index=i, segment_label=0,
                     truth_segment_label=0, adapter_sample_mode="event_segment", n_hits=12,
                     **{f"true_{variable}": float(value),
                        f"adapter_{variable}": float(value + .005 * np.sin(i))})
                for i, value in enumerate(values)]
        with gzip.open(evaluation / "predictions.csv.gz", "wt") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = json.loads((evaluation / "summary.json").read_text())
        summary["n_records"] = count
        write_json(evaluation / "summary.json", summary)
        evaluations[task] = evaluation
    return evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-finding-campaign", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixtures = output / "fixtures"
    fixtures.mkdir()
    results = []
    def run(name, command, destination):
        full = [sys.executable, *command]
        log = output / f"{name}.log"
        with log.open("w") as stream:
            process = subprocess.run(full, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        files = sorted(path for path in destination.rglob("*") if path.suffix in (".png", ".pdf"))
        result = dict(suite=name, command=full, returncode=process.returncode,
                      figure_count=len(files), files=[str(path.relative_to(output)) for path in files],
                      input_kind="synthetic" if not name.startswith("real_") else "stored_real_evaluation")
        results.append(result)
        write_json(output / "plot_checks.json", {"suites": results})
        if process.returncode:
            raise RuntimeError(f"{name} failed; inspect {log}")
        assert files, f"{name} produced no figures"
        for path in files:
            assert path.stat().st_size > 1000, path
            if path.suffix == ".png":
                from PIL import Image
                with Image.open(path) as image:
                    image.verify()
            else:
                assert path.read_bytes().startswith(b"%PDF-"), path
        print(f"{name}: {len(files)} figures", flush=True)

    finding = fixtures / "finding"
    rows = finding_rows(3, (32, 250000))
    configurations = make_joint_campaign(fixtures / "momentum")
    configs = {row["run_id"].split("_label")[0]: row for row in configurations}
    names = ("adapteronly", "mamba3", "transformer_nope", "transformer_rope")
    for row in rows:
        index = 0 if row["backbone_run_id"] == "adapteronly" else int(row["backbone_run_id"].split("_")[-1])
        source = configs[names[index]]
        row.update(summary_found=True, model_yaml=source["model_yaml"], model_config=source["model_config"],
                   embed_dim=source["embed_dim"], model_family=source["model_family"])
    write_csv(finding / "summary/run_table.csv", rows)
    yaml_file(finding / "manifest.yaml", {"runs": rows})
    run("finding", ["train/downstream/campaign/plot_track_finding_campaign.py", "--campaign-dir", str(finding),
                    "--output-dir", str(output / "finding")], output / "finding")
    run("momentum_all", ["train/downstream/campaign/plot_track_regression_campaign.py", "--campaign-dir",
                         str(fixtures / "momentum"), "--plot-suite", "all",
                         "--output-dir", str(output / "momentum")], output / "momentum")
    scalars = make_scalar_campaigns(fixtures / "scalars")
    run("scalar_residuals", ["train/downstream/campaign/plot_single_head_residuals.py",
                            "--p-evaluation", str(scalars["p"]), "--theta-evaluation", str(scalars["theta"]),
                            "--min-bin-entries", "2", "--output-dir", str(output / "residuals")], output / "residuals")
    run("scalar_physics", ["train/downstream/campaign/plot_track_regression_campaign.py",
                          "--campaign-dir", str(fixtures / "scalars/p"), "--plot-suite", "physics",
                          "--output-dir", str(output / "physics")], output / "physics")
    noise = fixtures / "noise"
    noise.mkdir()
    noise_runs = [write_run(noise, "small"), write_run(noise, "large", labels=4)]
    run("noise", ["train/downstream/campaign/plot_noise_tagging_campaign.py",
                  "--manifest", str(noise_manifest(noise, noise_runs)),
                  "--output-dir", str(output / "noise")], output / "noise")
    history = fixtures / "pretraining.csv"
    history.write_text("split,step,loss,lr\ntrain,1,0.5,0.001\nval,1,0.4,0.001\ntrain,2,0.4,0.001\nval,2,0.3,0.001\n")
    run("pretraining", ["train/pretrain/plot_pretraining_history.py", "--log", str(history),
                        "--output-dir", str(output / "pretraining")], output / "pretraining")
    if args.real_finding_campaign:
        real = args.real_finding_campaign
        run("real_finding", ["train/downstream/campaign/plot_track_finding_campaign.py", "--campaign-dir", str(real),
                             "--output-dir", str(output / "real_finding")], output / "real_finding")
        receipt_path = next((real / "runs").glob("*/train/artifacts.json"))
        history = relocated_history(receipt_path)
        run("real_validation", ["train/downstream/campaign/plot_track_finding_validation.py",
                                "--history", str(history), "--output-dir", str(output / "validation")],
                               output / "validation")
    report = dict(suites=results, synthetic_budgets=[32,250000], synthetic_backbones=list(names),
                  total_figures=sum(row["figure_count"] for row in results), all_passed=True,
                  caveat="Synthetic fixtures test plot execution and metadata; they are not measured physics results.")
    write_json(output / "plot_checks.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
