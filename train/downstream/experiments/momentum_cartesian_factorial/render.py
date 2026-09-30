"""Render the six matched Cartesian MAE arms from the active a02c recipe."""
import argparse
import json
from pathlib import Path
from ruamel.yaml import YAML

MODES = {
    "track": "event_segment",
    "gather": "event_segment_context",
    "membership": "event_segment_membership",
}
DATA_ROOT = Path("/home/alessio/ML-work/HIPO_processing/multi_particle_storage/big-data/mmap_canonical_loose_truthseg_event_v6_02")
BACKBONE = Path("/home/alessio/ML-work/pretrained-FMs/campaign_4/scale_w1536_d12_n39553933/ckpt_best.tar")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracks", type=int, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    configs = {}
    source = Path(__file__).resolve().parent
    for kind in ("pretrained", "adapteronly"):
        template = next(iter(yaml.load((source / f"{kind}_template.yaml").read_text()).values()))
        for label, mode in MODES.items():
            name = f"{kind}_{label}"
            config = dict(template)
            # The active sweep's newer physics diagnostics include per-bin
            # guardrails. This branch selects by validation loss, and its
            # older diagnostic parser does not accept that unused option.
            config["physics_checkpoint"] = dict(config["physics_checkpoint"])
            config["physics_checkpoint"].pop("per_bin_guardrails", None)
            config.update(
                adapter_sample_mode=mode,
                data_root=str(DATA_ROOT), data_root_train=str(DATA_ROOT),
                data_root_test=str(DATA_ROOT),
                regression_target_stats=str(root / "target_stats.json"),
                checkpoint_dir=str(root / name / "checkpoints"),
                downstream_dir=str(root / name),
                model_version=name,
                limit_size=args.tracks,
            )
            configs[name] = config
    with (root / "experiment.yaml").open("w") as stream:
        yaml.dump(configs, stream)
    (root / "experiment_design.json").write_text(json.dumps(dict(
        source="Active a02c Cartesian MAE 70k rendered configs; data root, cohort limit, paths, adapter mode, and unused per-bin physics guardrails changed",
        data_root=str(DATA_ROOT), labeled_events=100000,
        labeled_tracks=args.tracks, backbone=str(BACKBONE),
        modes=MODES, seed=11, batch_size=128, max_optimizer_steps=30000,
        validation_tracks=50000, final_evaluation_tracks=500000,
        evaluation_overlaps_checkpoint_validation=True,
        adapteronly_gather_equals_track_input=True,
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
