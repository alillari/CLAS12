"""Event access for the published cvt-clusters-h5-v1 skim.

Follows the skim's standalone reader contract. Fields are selected by name and
each ragged collection uses its own event offsets. No training targets or
cluster-to-particle joins are inferred here.
"""

import json
import operator
import os
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np


class CVTClusterEvents:
    """Index one split in manifest order, with bounded, process-local file handles."""

    def __init__(self, root, split, *, max_open=4):
        self.root = Path(root).resolve()
        self.split = split
        self.max_open = operator.index(max_open)
        if self.max_open < 1:
            raise ValueError("max_open must be positive")
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest["format"] != "cvt-clusters-h5-v1" or manifest["status"] != "complete":
            raise ValueError(f"Expected a published cvt-clusters-h5-v1 manifest at {self.root}")
        self.plan_id = manifest["plan_id"]
        if split not in manifest["splits"]:
            raise ValueError(f"Split {split!r} is absent from {self.root}/manifest.json")

        self.parts = []
        self.ends = []
        seen = set()
        total = 0
        for part in manifest["splits"][split]:
            path = Path(part["path"])
            count = operator.index(part["events"])
            if (
                part["split"] != split
                or path.parent != Path(split)
                or path.suffix != ".h5"
                or path in seen
                or count < 0
            ):
                raise ValueError(f"Invalid {split} part in manifest: {path}")
            seen.add(path)
            self.parts.append((str(path), count))
            total += count
            self.ends.append(total)
        self._files = OrderedDict()
        self._pid = os.getpid()

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def _file(self, part_index):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if part_index not in self._files:
            path, count = self.parts[part_index]
            file = h5py.File(self.root / path, "r")
            attrs = file.attrs
            if (
                attrs.get("format") != "cvt-clusters-h5-v1"
                or attrs.get("complete") != 1
                or attrs.get("plan_id") != self.plan_id
                or attrs.get("part_key") != path
                or attrs.get("events") != count
            ):
                file.close()
                raise ValueError(f"Part {path} does not match the published manifest")
            self._files[part_index] = file
            if len(self._files) > self.max_open:
                self._files.popitem(last=False)[1].close()
        self._files.move_to_end(part_index)
        return self._files[part_index]

    def read(self, index, fields):
        """Return named scalar event fields or ragged slices, e.g. clusters.xyz."""
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = bisect_right(self.ends, index)
        local = index - (self.ends[part - 1] if part else 0)
        file = self._file(part)
        ranges = {}
        result = {}
        for field in fields:
            path = field.replace(".", "/")
            domain, key = path.rsplit("/", 1)
            if key == "offsets":
                raise ValueError("Offsets are part-level, not event fields")
            if domain == "events":
                result[field] = file[path][local]
                continue
            if domain not in ranges:
                start, stop = file[domain + "/offsets"][local : local + 2]
                ranges[domain] = slice(start, stop)
            selection = ranges[domain]
            column = file[path]
            if not 0 <= selection.start <= selection.stop <= len(column):
                raise ValueError(f"Invalid {domain} offsets in {file.filename} at event {local}")
            result[field] = column[selection]
        return result

    def event_lengths(self, domain):
        """Read only offsets, e.g. when constructing validation chunk indices."""
        lengths = np.empty(len(self), dtype=np.int64)
        start = 0
        for part, stop in enumerate(self.ends):
            lengths[start:stop] = np.diff(self._file(part)[domain + "/offsets"][:])
            start = stop
        return lengths

    def close(self):
        for file in self._files.values():
            file.close()
        self._files.clear()

    def __getstate__(self):
        return {**self.__dict__, "_files": OrderedDict()}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
