"""Project named HDF5 fields onto the existing downstream tensor contract.

All joins are event-local. Compact segment labels, particle IDs, native track
IDs and bank row numbers remain distinct. Unassigned primary clusters (including
contested ownership) are background for finding and noise for binary tagging.
"""

import numpy as np
from .track_eligibility import eligible_primary_labels, PRIMARY_TRACK_MIN_CLUSTERS, PRIMARY_TRACK_POLICY

NOISE_POLICY = "primary_unassigned_is_noise_v1"
ENTRANCE_COLUMNS = ("cvt_benchmark_p", "cvt_benchmark_theta", "cvttraj_entrance_phi")
MIN4_NOISE_POLICY = "primary_unassigned_or_lt4_is_noise_v2"


def downstream_split(params, role):
    hdf5 = getattr(params, "reader_type", "ragged_mmap") == "hdf5"
    defaults = {"train": "train" if hdf5 else "pretrain",
                "validation": "validation" if hdf5 else "test", "test": "test"}
    return str(getattr(params, role + "_split", defaults[role]))


class PointFieldView:
    """Small indexed view; data stays in the HDF5 shards."""

    def __init__(self, source, name):
        self.source, self.name = source, name

    def __len__(self):
        return len(self.source.reader)

    def __getitem__(self, index):
        return self.source.point_field(index, self.name)


class CVTDownstreamSource:
    fields = {"features", "seg_target", "reg_target", "pid_target", "noise_target",
              "coatjava_seg_pred", "cluster_token_context_target",
              "cluster_geometry_context_target"}

    def __init__(self, root, split, *, apply_primary_track_cut=False):
        from .cvt_hdf5 import CVTClusterEvents
        self.reader = CVTClusterEvents(root, split)
        self.apply_primary_track_cut = bool(apply_primary_track_cut)
        self._index = None
        self._rows = {}

    def view(self, name):
        if name not in self.fields:
            raise KeyError(name)
        return PointFieldView(self, name)

    def read(self, index, fields):
        if index != self._index:
            self._index, self._rows = index, {}
        missing = [field for field in fields if field not in self._rows]
        if missing:
            self._rows.update(self.reader.read(index, missing))
        return [self._rows[field] for field in fields]

    def identity(self, index):
        source, ordinal = self.read(index, ["events.source_id", "events.original_ordinal"])
        return {"source_id": int(source), "original_ordinal": int(ordinal)}

    @property
    def contract(self):
        result = {"format": "cvt-clusters-h5-v1", "plan_id": self.reader.plan_id,
                  "split": self.reader.split, "noise_policy": NOISE_POLICY,
                  "momentum_unit": "GeV/c", "target": "observed_direct_primary_momentum"}
        if self.apply_primary_track_cut:
            result.update(noise_policy=MIN4_NOISE_POLICY,
                          track_eligibility_policy=PRIMARY_TRACK_POLICY,
                          primary_track_min_clusters=PRIMARY_TRACK_MIN_CLUSTERS)
        return result

    def point_field(self, index, name):
        if name == "seg_target":
            labels = self.read(index, ["clusters.primary_segment"])[0]
            return eligible_primary_labels(labels) if self.apply_primary_track_cut else labels
        direct = {"features": "clusters.xyz",
                  "coatjava_seg_pred": "clusters.reco_segment"}
        if name in direct:
            return self.read(index, [direct[name]])[0]
        if name == "noise_target":
            return (self.point_field(index, "seg_target") < 0).astype(np.int64)
        if name == "pid_target":
            pid, valid = self.read(index, ["clusters.primary_pid", "clusters.primary_pid_valid"])
            return np.where(valid, pid, -100).astype(np.int64)
        if name == "cluster_token_context_target":
            return np.column_stack(self.read(index, ["clusters." + key for key in
                ("detector", "layer", "sector", "cluster_id", "seed_strip")])).astype(np.int64)
        if name == "cluster_geometry_context_target":
            return np.column_stack(self.read(index, ["clusters." + key for key in
                ("endpoints", "direction", "length", "pitch")])).astype(np.float32)
        if name == "reg_target":
            owner, tids, momentum, valid, vertex, energy = self.read(index, [
                "clusters.primary_tid", "particles.tid", "particles.observed_momentum",
                "particles.observed_valid", "particles.vertex", "particles.generated_energy"])
            if len(np.unique(tids)) != len(tids) or np.any(tids <= 0):
                raise ValueError(f"Invalid particle IDs in event {index}")
            rows = {int(tid): i for i, tid in enumerate(tids)}
            result = np.full((len(owner), 7), np.nan, dtype=np.float32)
            for tid in np.unique(owner[owner > 0]):
                if int(tid) not in rows:
                    raise ValueError(f"Cluster owner {tid} absent from particles in event {index}")
                row = rows[int(tid)]
                value = np.concatenate((np.where(valid[row], momentum[row], np.nan),
                                        vertex[row], np.atleast_1d(energy[row])))
                result[owner == tid] = value
            return result
        raise KeyError(name)

    def primary_tid(self, index, segment_label):
        labels, tids = self.read(index, ["clusters.primary_segment", "clusters.primary_tid"])
        selected = np.unique(tids[labels == segment_label])
        if len(selected) != 1 or selected[0] <= 0:
            raise ValueError(f"Segment {segment_label} has no unique primary ID in event {index}")
        return int(selected[0])

    def benchmark(self, index, truth_segment, candidate_segment=None):
        """First-pass CVT entrance reference; never choose arbitrarily among clones.

        Truth samples require exactly one candidate with a valid unique owner.
        COATJAVA-segment samples use their own native track ID. Readout purity is
        recorded as evidence, independently of the dataset's cluster-count cut.
        """
        tid = self.primary_tid(index, truth_segment)
        native, owners, valid, unique, purity, p, theta, phi = self.read(index, [
            "reco.cvt_tracks." + key for key in ("native", "purity_tid", "purity_valid",
            "unique_owner", "purity", "benchmark_p", "benchmark_theta", "entrance_phi")])
        eligible = (owners == tid) & valid & unique
        if candidate_segment is not None:
            labels, ids = self.read(index, ["clusters.reco_segment", "clusters.native_track_id"])
            ids = np.unique(ids[labels == candidate_segment])
            if len(ids) != 1 or ids[0] <= 0:
                raise ValueError("Candidate segment has no unique native track ID")
            eligible &= native["ID"] == ids[0]
        matches = np.flatnonzero(eligible)
        result = {"truth_tid": tid, "pass": "CVT", "candidate_count": len(matches),
                  "track_row": -1, "status": "missing" if not len(matches) else "ambiguous",
                  "values": np.full(3, np.nan), "purity": np.nan}
        if len(matches) == 1:
            row = int(matches[0])
            values = np.array([p[row], theta[row], phi[row]], dtype=float)
            good = np.isfinite(values).all() and p[row] > 0 and 0 <= theta[row] <= np.pi
            result.update(track_row=row, status="matched" if good else "invalid_entrance",
                          values=values if good else np.full(3, np.nan), purity=float(purity[row]))
        return result

    def close(self):
        self.reader.close()

    def __getstate__(self):
        return {**self.__dict__, "_index": None, "_rows": {}}
