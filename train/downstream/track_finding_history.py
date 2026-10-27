"""Append-only validation histories with one immutable checkpoint per record."""

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import tempfile


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ValidationHistory:
    """Keep fresh training invocations separate, including repeated run names."""

    def __init__(self, training_log, metadata):
        log = Path(training_log).resolve()
        root = log.parent / f"{log.stem}_validation"
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="run_", dir=root))
        self.jsonl_path = self.directory / "metrics.jsonl"
        self.csv_path = self.directory / "metrics.csv"
        self.metadata = json_safe({"created_at": datetime.now(timezone.utc).isoformat(), **metadata})
        self.index = 0
        self.fields = None
        (self.directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n"
        )

    def append(self, record, checkpoint_writer=None):
        self.index += 1
        record = json_safe({"schema_version": 1, "history_id": self.directory.name, "validation_index": self.index, **record})
        record["checkpoint"] = None
        fields = list(record)
        if self.fields is not None and fields != self.fields:
            raise ValueError("Validation history field layout changed within a training run")
        if checkpoint_writer is not None:
            folder = self.directory / "checkpoints"
            folder.mkdir(exist_ok=True)
            path = folder / (
                f"validation_{self.index:06d}_step_{record['step']:09d}_epoch_{record['epoch']:06d}.pth"
            )
            record["checkpoint"] = str(path)
            checkpoint_writer(path, record)
        with self.jsonl_path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        with self.csv_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if self.fields is None:
                writer.writeheader()
            writer.writerow(record)
        self.fields = fields
        return record
