"""Versioned tensor layout for individual tracking measurements."""

import numpy as np


TRACKING_GEOMETRY_V1 = "tracking_geometry_v1"
LINE, ARC = 0, 1
CENTER = slice(0, 3)
ENDPOINT_OFFSETS = slice(3, 9)
LENGTH, PITCH, SWEEP, KIND, DETECTOR, LAYER = range(9, 15)
TOKEN_WIDTH = 15

# Positive category IDs are independent of the number and order of CVT layers.
# Zero is reserved for padding in both categorical embeddings.
NUM_DETECTOR_IDS = 32
NUM_LAYER_IDS = 128


def valid_tracking_hits(tokens):
    """Identify real rows independently of their Cartesian coordinates."""
    if tokens.shape[-1] != TOKEN_WIDTH:
        raise ValueError(f"Expected {TOKEN_WIDTH} columns for {TRACKING_GEOMETRY_V1}")
    return tokens[..., DETECTOR] > 0


def pack_tracking_geometry(
    *, center_cm, endpoints_cm, length_cm, pitch_cm, sweep_rad, kind, detector, layer
):
    """Pack one line or circular arc per hit in the given order.

    Endpoints are global positions with shape (hits, 2, 3). Centers are the
    geometry anchors, including the arc midpoint for an arc. Length and pitch
    are in cm. Sweep is an unsigned angle in radians, zero for lines.
    """
    centers = np.asarray(center_cm, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Tracking centers must have shape (hits, 3)")
    n = len(centers)
    ends = np.asarray(endpoints_cm, dtype=np.float32)
    if ends.shape != (n, 2, 3):
        raise ValueError("Tracking endpoints must have shape (hits, 2, 3)")
    length, pitch, sweep = (np.asarray(field, dtype=np.float32)
                            for field in (length_cm, pitch_cm, sweep_rad))
    types, detectors, layers = (np.asarray(field) for field in (kind, detector, layer))
    if any(field.shape != (n,) for field in (length, pitch, sweep, types, detectors, layers)):
        raise ValueError("Tracking metadata must have one entry per hit")
    if any(field.dtype.kind not in "iu" for field in (types, detectors, layers)):
        raise ValueError("Geometry type, detector and layer IDs must be integers")
    if not np.isin(types, (LINE, ARC)).all():
        raise ValueError("Expected line or arc geometry")
    if not ((detectors > 0) & (detectors < NUM_DETECTOR_IDS)).all():
        raise ValueError("Detector IDs are outside the tracking geometry vocabulary")
    if not ((layers > 0) & (layers < NUM_LAYER_IDS)).all():
        raise ValueError("Layer IDs are outside the tracking geometry vocabulary")
    if (length <= 0).any() or (pitch <= 0).any() or (ends[:, 0] == ends[:, 1]).all(axis=1).any():
        raise ValueError("Tracking channels need distinct endpoints and positive length and pitch")
    if (sweep[types == LINE] != 0).any() or not ((sweep[types == ARC] > 0) &
                                               (sweep[types == ARC] < 2 * np.pi)).all():
        raise ValueError("Expected zero line sweep and a positive arc sweep below 2π")

    tokens = np.empty((n, TOKEN_WIDTH), dtype=np.float32)
    tokens[:, CENTER] = centers
    tokens[:, ENDPOINT_OFFSETS] = (ends - centers[:, None, :]).reshape(n, 6)
    for index, field in ((LENGTH, length), (PITCH, pitch), (SWEEP, sweep),
                         (KIND, types), (DETECTOR, detectors), (LAYER, layers)):
        tokens[:, index] = field
    if not np.isfinite(tokens).all():
        raise ValueError("Tracking geometry contains nonfinite values")
    return tokens
