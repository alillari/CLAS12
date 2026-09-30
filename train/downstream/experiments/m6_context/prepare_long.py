"""Render a matched 30k-step experiment using complete labeled events."""
import argparse
import json
from pathlib import Path
import numpy as np
from ruamel.yaml import YAML
from fm4npp.datasets.dataset import (RaggedMmap, EventSegmentTPCBatchDataset,
                                      EventContextSegmentTPCBatchDataset)


def select_cohort(segments, event_budget):
    """Match the mctrue/min12/high100 dataset filters; never split an event."""
    identities = []
    events = []
    for event_index in range(len(segments)):
        labels, counts = np.unique(np.asarray(segments[event_index]), return_counts=True)
        accepted = [int(label) for label, count in zip(labels, counts)
                    if int(label) != -1 and 12 <= count <= 100]
        if not accepted:
            continue
        events.append(event_index)
        identities.extend((event_index, label) for label in accepted)
        if len(events) == event_budget:
            return np.asarray(identities, dtype=np.int64), events
    raise ValueError(f"Only {len(events)} labeled events available; need {event_budget}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ids, events = select_cohort(RaggedMmap(str(args.data_root / "seg_target_pretrain")), 100000)
    for cls in (EventSegmentTPCBatchDataset, EventContextSegmentTPCBatchDataset):
        dataset = cls(data_root=str(args.data_root), split="pretrain", return_dict=True,
                      require_reg_target=True, num_pred_points=1, segment_min_clusters=12,
                      high_thr=100, voxelize=False, limit_data=True, limit_size=len(ids))
        np.testing.assert_array_equal(np.asarray(dataset.idxlist), ids)
        del dataset
    np.save(args.output / "training_track_identities.npy", ids)
    report = dict(training_distinct_labeled_events=len(events), training_labeled_tracks=len(ids),
                  first_event_index=events[0], last_event_index=events[-1],
                  selection="First 100000 events with eligible tracks; all eligible tracks per event",
                  validation_tracks=50000, evaluation_tracks=500000,
                  evaluation_split="test", evaluation_overlaps_checkpoint_validation=True)
    # Fail early if the existing test product cannot supply the requested evaluation.
    total = 0
    test = RaggedMmap(str(args.data_root / "seg_target_test"))
    for i in range(len(test)):
        labels, counts = np.unique(np.asarray(test[i]), return_counts=True)
        total += int(np.sum((labels != -1) & (counts >= 12) & (counts <= 100)))
        if total >= 500000:
            break
    if total < 500000:
        raise ValueError(f"Only {total} evaluation tracks available")
    yaml = YAML()
    configs = yaml.load((Path(__file__).parent / "experiment.yaml").read_text())
    for config in configs.values():
        config.update(limit_size=len(ids), max_optimizer_steps=30000,
                      early_stopping_min_steps=30000, first_cycle_steps=30000,
                      val_interval_steps=1000)
    with (args.output / "experiment.yaml").open("w") as f:
        yaml.dump(configs, f)
    (args.output / "training_track_count.txt").write_text(str(len(ids)) + "\n")
    (args.output / "cohort.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
