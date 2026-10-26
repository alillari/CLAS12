"""Validate, pilot, and run matched AdapterOnly/frozen-M6/last-two-M6 arms."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--campaign-root", type=Path, required=True)
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--backbone", type=Path, required=True)
args = parser.parse_args()
ROOT = args.campaign_root.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
SOURCE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "train/downstream/campaign"))
sys.path.insert(0, str(SOURCE / "train/downstream"))

import numpy as np
import torch
from fm4npp.utils import YParams
from fm4npp.datasets.dataset import get_data_loader
from fm4npp.models.mambagpt import Mamba1GPT
from train.downstream.scripts.compute_regression_target_stats import compute_stats
from train.downstream.backbone_finetuning import configure_backbone, backbone_features
from campaign_util import read_yaml, write_yaml, run_paths, render_model_yaml, render_analysis_yaml

BACKBONE = args.backbone.resolve()
ARMS = (("adapteronly", False, 0), ("frozen_m6", True, 0), ("last_two_m6", True, 2))


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, log):
    print("RUN", json.dumps([str(x) for x in command]), flush=True)
    with (ROOT / log).open("w") as stream:
        subprocess.run(command, cwd=SOURCE, stdout=stream, stderr=subprocess.STDOUT, check=True)


def prepare():
    original = read_yaml(Path(__file__).with_name("recipe.yaml"))["recipe"]
    recipe = dict(original)
    recipe.update(input_representation="center_only",
                  data_root=str(args.data_root.resolve()), data_root_train=str(args.data_root.resolve()),
                  data_root_test=str(args.data_root.resolve()),
                  artifact_root=str(ROOT), stat_dir=str(ROOT / "stats"),
                  regression_target_stats=str(ROOT / "stats/mom_train50k.json"))
    write_yaml(ROOT / "base_model.yaml", {"recipe": recipe})
    analysis = read_yaml(SOURCE / "train/downstream/eval/track_regression_analysis_pretrained.yaml")["analysis"]
    analysis.pop("pretrain_root", None)
    write_yaml(ROOT / "base_analysis.yaml", {"analysis": analysis})
    manifests = []
    for pilot in (True, False):
        destination = ROOT / "pilot" if pilot else ROOT
        destination.mkdir(exist_ok=True)
        overrides = {}
        if pilot:
            overrides = dict(max_optimizer_steps=10, val_interval_steps=5,
                             early_stopping_min_steps=10, limit_test_size=256,
                             num_data_workers=0)
        manifest = dict(campaign_name=ROOT.name + ("_pilot" if pilot else ""),
                        artifact_root=str(ROOT.parent.parent), campaign_dir=str(destination),
                        base_model_yaml=str(ROOT / "base_model.yaml"),
                        base_analysis_yaml=str(ROOT / "base_analysis.yaml"),
                        recipe_source="conventional_v602_cartesian_mae_50k",
                        training_overrides=overrides, runs=[])
        for name, pretrained, count in ARMS:
            run_id = name + "_seed42"
            row = dict(run_id=run_id, backbone_run_id="campaign4_m6" if pretrained else "adapteronly",
                       model_family="mamba1" if pretrained else "adapteronly",
                       source_dir=str(BACKBONE.parent),
                       pretrained_checkpoint=str(BACKBONE) if pretrained else None,
                       use_pretrained_backbone=pretrained, backbone_unfreeze_last_n=count,
                       base_model_yaml=str(ROOT / "base_model.yaml"), base_model_config="recipe",
                       model_config=run_id, base_dim=1536, embed_dim=1536,
                       num_layers_backbone=12, pretrain_events=39553933 if pretrained else 0,
                       eventnumber=256 if pilot else 50000, labeled_events=256 if pilot else 50000,
                       train_batch_size=128, max_samples=256 if pilot else 50000,
                       seed=42, status="pending", training_overrides=dict(backbone_unfreeze_last_n=count))
            row.update({k: str(v) for k, v in run_paths(destination, run_id).items()})
            manifest["runs"].append(row)
            render_model_yaml(manifest, row)
            render_analysis_yaml(manifest, row)
        write_yaml(destination / "manifest.yaml", manifest)
        manifests.append(manifest)
    full = manifests[1]
    rendered = [read_yaml(Path(row["model_yaml"]))[row["model_config"]] for row in full["runs"]]
    allowed = {"backbone_unfreeze_last_n", "model_version", "checkpoint_dir", "downstream_dir"}
    for candidate in rendered[1:]:
        differences = {k for k in set(rendered[0]) | set(candidate)
                       if rendered[0].get(k) != candidate.get(k)}
        assert differences <= allowed, differences
    write_json(ROOT / "recipe_audit.json", dict(
        source_recipe_sha256=sha(Path(__file__).with_name("recipe.yaml")),
        production_recipe=rendered[0], allowed_arm_differences=sorted(allowed),
        note="The original test split is reused for validation and evaluation, as in the source recipe; evaluation is not an independent holdout."))
    return full, manifests[0]


def preflight(full):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE), "Commit source before launch"
    source_files = subprocess.check_output(["git", "ls-files"], cwd=SOURCE, text=True).splitlines()
    write_json(ROOT / "source_provenance.json", dict(commit=commit,
        source_dir=str(SOURCE), source_files={name: sha(SOURCE / name) for name in source_files}))
    assert torch.cuda.is_available(), "CUDA unavailable"
    audit = dict(torch_version=torch.__version__, device=torch.cuda.get_device_name(0),
                 capability=list(torch.cuda.get_device_capability(0)),
                 backbone_path=str(BACKBONE), backbone_sha256=sha(BACKBONE),
                 source_snapshot_sha256=sha(ROOT / "source_provenance.json"))
    state = torch.load(BACKBONE, map_location="cpu", weights_only=False)
    weights = {key.removeprefix("module."): value for key, value in state["model_state"].items()}
    assert tuple(weights["output_layer.weight"].shape) == (3, 1536)
    model = Mamba1GPT(embed_dim=1536, num_layers=12, d_state=16, d_conv=4, expand=2,
                      klen=1, dropout=.05019431407230154, embed_method="pos_only", pe_method="nerf")
    model.load_state_dict(weights, strict=True)
    audit["backbone_checkpoint_step"] = int(state["iters"])
    audit["strict_backbone_reload"] = True
    del model, state, weights

    (ROOT / "stats").mkdir(exist_ok=True)
    params = YParams(full["runs"][0]["model_yaml"], full["runs"][0]["model_config"])
    computed = compute_stats(Path(params.data_root), "pretrain", 1, 100, 50000, 10000,
                             task="mom", segment_min_clusters=12)
    assert computed["selected_events"] == 50000
    assert computed["count"] == [50000] * 3
    write_json(ROOT / "stats/mom_train50k.json", computed)
    audit["target_stats_regenerated_from_training_cohort"] = True
    audit["target_stats_sha256"] = sha(ROOT / "stats/mom_train50k.json")

    params["num_data_workers"] = 0
    training, _, validation, _ = get_data_loader(params, False)
    cohorts = {}
    for name, loader, expected in (("train", training, 50000), ("validation", validation, 50000)):
        keys = [tuple(map(int, key)) for key in loader.dataset.idxlist]
        assert len(keys) == expected and len(set(keys)) == expected
        cohorts[name] = dict(selected_tracks=len(keys),
                             used_tracks=len(loader) * loader.batch_size if loader.drop_last else len(keys),
                             ordered_identity_sha256=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
                             split=loader.dataset.split)
    batch = next(iter(training))
    assert batch["points"].shape[-1] == 3
    assert torch.isfinite(batch["points"]).all()
    audit["cohorts"] = cohorts
    audit["first_batch_shape"] = list(batch["points"].shape)
    # Real M6 CUDA backward: exact frozen prefix and last-two-block updates.
    model = Mamba1GPT(embed_dim=1536, num_layers=12, d_state=16, d_conv=4, expand=2,
                      klen=1, dropout=.05019431407230154, embed_method="pos_only", pe_method="nerf").cuda()
    state = torch.load(BACKBONE, map_location="cpu", weights_only=False)
    weights = {k.removeprefix("module."): v for k, v in state["model_state"].items()}
    model.load_state_dict(weights, strict=True)
    from train.downstream.model import MambaTrackRegressionHead
    head = MambaTrackRegressionHead(input_dim=1536, num_layers=1, num_embedder_layers=0,
                                    num_feature_layers=12, embed_method="pos_only",
                                    pooling="attention").cuda()
    trainable = configure_backbone(model, 2)
    optimizer = torch.optim.AdamW(trainable, lr=3e-5)
    points = batch["points"].cuda()
    feature = backbone_features(model, points, tune=True)
    output = head(points, feature, padding_mask=points[..., 0] != -100, pretrain=True)["pred_regression"]
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    for name, p in model.named_parameters():
        expected = name.startswith(("mamba_layers.10.", "mamba_layers.11."))
        assert (p.grad is not None) == expected, name
        if expected:
            assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    optimizer.step()
    changed = []
    for name, value in model.state_dict().items():
        if not torch.equal(value.cpu(), weights[name]):
            changed.append(name)
    assert changed and all(name.startswith(("mamba_layers.10.", "mamba_layers.11.")) for name in changed)
    audit["real_batch_cuda_backward"] = True
    audit["changed_backbone_tensors"] = changed
    audit["backbone_trainable_parameters"] = sum(p.numel() for p in trainable)
    audit["frozen_prefix_unchanged"] = True
    write_json(ROOT / "preflight_pass.json", audit)
    print("PREFLIGHT_PASSED", flush=True)


def assert_outputs(manifest, steps):
    identities = []
    results = {}
    for row in manifest["runs"]:
        artifact = json.loads(Path(row["artifact_summary"]).read_text())
        assert artifact["final_step"] == steps, artifact
        count = row["backbone_unfreeze_last_n"]
        assert artifact["backbone_unfreeze_last_n"] == count
        assert math.isfinite(artifact["best_loss"])
        checkpoint = torch.load(row["adapter_checkpoint"], map_location="cpu", weights_only=False)
        assert checkpoint["backbone_unfreeze_last_n"] == count
        assert ("backbone_state_dict" in checkpoint) == bool(count)
        if count:
            groups = checkpoint["optimizer_state_dict"]["param_groups"]
            assert len(groups) == 2
            assert math.isclose(groups[1]["lr"], groups[0]["lr"] * .1, rel_tol=1e-10)
            assert artifact["backbone_trainable_parameters"] > 0
            original = torch.load(BACKBONE, map_location="cpu", weights_only=False)["model_state"]
            original = {k.removeprefix("module."): v for k, v in original.items()}
            changes = [k for k, v in checkpoint["backbone_state_dict"].items() if not torch.equal(v, original[k])]
            assert changes and all(k.startswith(("mamba_layers.10.", "mamba_layers.11.")) for k in changes)
            del original
        del checkpoint
        with gzip.open(Path(row["evaluation_dir"]) / "predictions.csv.gz", "rt") as stream:
            records = list(csv.DictReader(stream))
        assert len(records) == row["max_samples"], len(records)
        # Compare serialized identity plus truth columns across arms.
        fields = [k for k in records[0] if k in {"source_event_index", "event_index", "segment_label", "sample_index"}
                  or k.startswith("true_")]
        assert fields, list(records[0])
        identity = [{k: record[k] for k in fields} for record in records]
        identities.append(identity)
        results[row["run_id"]] = dict(final_step=artifact["final_step"], best_loss=artifact["best_loss"],
                                       evaluated_tracks=len(records), matched_fields=fields)
    assert all(identity == identities[0] for identity in identities[1:]), "Evaluation cohorts/truth differ"
    return results


def main():
    full, pilot = prepare()
    preflight(full)
    runner = [sys.executable, "train/downstream/campaign/run_track_regression_campaign.py", "--manifest"]
    run(runner + [str(ROOT / "pilot/manifest.yaml")], "pilot_launcher.log")
    results = assert_outputs(pilot, 10)
    write_json(ROOT / "pilot_pass.json", dict(finished_unix=time.time(), arms=results))
    print("ALL_THREE_PILOTS_PASSED", flush=True)
    run(runner + [str(ROOT / "manifest.yaml")], "campaign_launcher.log")
    results = assert_outputs(full, 30000)
    write_json(ROOT / "completed.json", dict(finished_unix=time.time(), arms=results))
    print("CAMPAIGN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
