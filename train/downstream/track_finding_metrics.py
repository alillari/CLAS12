"""Metrics for event-level track-finding segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True)
class MatchConfig:
    background_label: int = -1
    iou_threshold: float = 0.5
    min_purity: float = 0.5
    min_efficiency: float = 0.5


def _safe_div(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def finite_or_none(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def event_track_metrics(
    truth_labels: np.ndarray,
    pred_labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
    pred_signal_mask: np.ndarray | None = None,
    config: MatchConfig | None = None,
) -> dict[str, Any]:
    """Compute one event's clustering and matched-track metrics.

    ``truth_labels`` uses ``config.background_label`` for background/no segment.
    ``pred_labels`` uses arbitrary predicted instance ids; values < 0 are
    treated as predicted background.
    """

    config = config or MatchConfig()
    truth = np.asarray(truth_labels).reshape(-1)
    pred = np.asarray(pred_labels).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError(f"truth/pred shape mismatch: {truth.shape} vs {pred.shape}")

    if valid_mask is None:
        valid = np.ones(truth.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask).reshape(-1).astype(bool)
    if pred_signal_mask is None:
        pred_signal = pred >= 0
    else:
        pred_signal = np.asarray(pred_signal_mask).reshape(-1).astype(bool) & (pred >= 0)

    truth_background = valid & (truth == config.background_label)
    truth_signal = valid & ~truth_background
    pred_signal = pred_signal & valid

    true_ids = sorted(np.unique(truth[truth_signal]).tolist())
    pred_ids = sorted(np.unique(pred[pred_signal]).tolist())

    if truth_signal.sum() > 1 and len(true_ids) > 1:
        ari_signal = float(adjusted_rand_score(truth[truth_signal], pred[truth_signal]))
    elif truth_signal.sum() > 0:
        ari_signal = 1.0 if len(pred_ids) <= 1 else 0.0
    else:
        ari_signal = None

    if valid.sum() > 1 and len(np.unique(truth[valid])) > 1:
        pred_with_bg = pred.copy()
        pred_with_bg[~pred_signal] = -1
        ari_with_background = float(adjusted_rand_score(truth[valid], pred_with_bg[valid]))
    else:
        ari_with_background = None

    n_true = len(true_ids)
    n_pred = len(pred_ids)
    iou = np.zeros((n_pred, n_true), dtype=np.float64)
    purity = np.zeros_like(iou)
    efficiency = np.zeros_like(iou)

    for pidx, pred_id in enumerate(pred_ids):
        pred_points = pred_signal & (pred == pred_id)
        pred_count = int(pred_points.sum())
        for tidx, true_id in enumerate(true_ids):
            true_points = truth_signal & (truth == true_id)
            true_count = int(true_points.sum())
            intersection = int((pred_points & true_points).sum())
            union = int((pred_points | true_points).sum())
            iou[pidx, tidx] = _safe_div(intersection, union) or 0.0
            purity[pidx, tidx] = _safe_div(intersection, pred_count) or 0.0
            efficiency[pidx, tidx] = _safe_div(intersection, true_count) or 0.0

    matches = []
    if n_pred and n_true:
        row_ind, col_ind = linear_sum_assignment(-iou)
        for pidx, tidx in zip(row_ind, col_ind):
            is_match = (
                iou[pidx, tidx] >= config.iou_threshold
                and purity[pidx, tidx] >= config.min_purity
                and efficiency[pidx, tidx] >= config.min_efficiency
            )
            if is_match:
                matches.append({
                    "pred_id": int(pred_ids[pidx]),
                    "true_id": int(true_ids[tidx]),
                    "iou": float(iou[pidx, tidx]),
                    "purity": float(purity[pidx, tidx]),
                    "efficiency": float(efficiency[pidx, tidx]),
                })

    matched_true = {match["true_id"] for match in matches}
    matched_pred = {match["pred_id"] for match in matches}
    split_truth = 0
    for true_id in true_ids:
        overlaps = [
            pred_id for pred_id in pred_ids
            if (pred_signal & (pred == pred_id) & truth_signal & (truth == true_id)).sum() > 0
        ]
        if len(overlaps) > 1:
            split_truth += 1
    merged_pred = 0
    for pred_id in pred_ids:
        overlaps = np.unique(truth[pred_signal & (pred == pred_id) & truth_signal])
        if len(overlaps) > 1:
            merged_pred += 1

    background_pred_background = int((truth_background & ~pred_signal).sum())
    background_as_signal = int((truth_background & pred_signal).sum())
    signal_as_background = int((truth_signal & ~pred_signal).sum())

    return {
        "n_points": int(valid.sum()),
        "n_signal_points": int(truth_signal.sum()),
        "n_background_points": int(truth_background.sum()),
        "n_true_tracks": int(n_true),
        "n_pred_tracks": int(n_pred),
        "n_matched_tracks": int(len(matches)),
        "ari_signal": finite_or_none(ari_signal),
        "ari_with_background": finite_or_none(ari_with_background),
        "track_efficiency": _safe_div(len(matched_true), n_true),
        "track_purity": _safe_div(len(matched_pred), n_pred),
        "matched_iou_mean": _mean_or_none([m["iou"] for m in matches]),
        "matched_purity_mean": _mean_or_none([m["purity"] for m in matches]),
        "matched_efficiency_mean": _mean_or_none([m["efficiency"] for m in matches]),
        "fake_rate": _safe_div(n_pred - len(matched_pred), n_pred),
        "miss_rate": _safe_div(n_true - len(matched_true), n_true),
        "split_rate": _safe_div(split_truth, n_true),
        "merge_rate": _safe_div(merged_pred, n_pred),
        "background_rejection": _safe_div(background_pred_background, int(truth_background.sum())),
        "background_contamination": _safe_div(background_as_signal, int(pred_signal.sum())),
        "signal_loss_to_background": _safe_div(signal_as_background, int(truth_signal.sum())),
        "matches": matches,
    }


def summarize_event_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    summary: dict[str, Any] = {
        "n_events": len(rows),
        "n_points": int(sum(row.get("n_points", 0) for row in rows)),
        "n_signal_points": int(sum(row.get("n_signal_points", 0) for row in rows)),
        "n_background_points": int(sum(row.get("n_background_points", 0) for row in rows)),
        "n_true_tracks": int(sum(row.get("n_true_tracks", 0) for row in rows)),
        "n_pred_tracks": int(sum(row.get("n_pred_tracks", 0) for row in rows)),
        "n_matched_tracks": int(sum(row.get("n_matched_tracks", 0) for row in rows)),
    }
    for key in (
        "ari_signal",
        "ari_with_background",
        "track_efficiency",
        "track_purity",
        "matched_iou_mean",
        "matched_purity_mean",
        "matched_efficiency_mean",
        "fake_rate",
        "miss_rate",
        "split_rate",
        "merge_rate",
        "background_rejection",
        "background_contamination",
        "signal_loss_to_background",
    ):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary[key] = finite_or_none(float(np.mean(values))) if values else None
    summary["track_efficiency_global"] = _safe_div(summary["n_matched_tracks"], summary["n_true_tracks"])
    summary["track_purity_global"] = _safe_div(summary["n_matched_tracks"], summary["n_pred_tracks"])
    return summary


def track_momentum_by_label(
    truth_labels: np.ndarray,
    reg_target: np.ndarray,
    background_label: int = -1,
    momentum_scale: float = 0.001,
) -> dict[int, dict[str, float]]:
    labels = np.asarray(truth_labels).reshape(-1)
    reg = np.asarray(reg_target)
    if reg.ndim != 2 or reg.shape[0] != labels.shape[0] or reg.shape[1] < 3:
        return {}
    out: dict[int, dict[str, float]] = {}
    for label in sorted(np.unique(labels).tolist()):
        if int(label) == int(background_label):
            continue
        rows = reg[labels == label, :3].astype(np.float64) * float(momentum_scale)
        rows = rows[np.isfinite(rows).all(axis=1)]
        if rows.size == 0:
            continue
        vec = rows.mean(axis=0)
        pt = float(np.hypot(vec[0], vec[1]))
        p = float(np.sqrt(np.dot(vec, vec)))
        out[int(label)] = {"p_gev": p, "pt_gev": pt}
    return out
