"""Replay binary option-2 assignments for a configurable query eligibility cut."""
import numpy as np


def query_winners(confidence, candidate_scores, threshold):
    """Scores are indexed by cluster, then query. Ties favor the first query."""
    eligible_scores = np.where(confidence[..., None, :] > threshold, candidate_scores, -1)
    pred = np.argmax(eligible_scores, axis=-1)
    scores = np.take_along_axis(eligible_scores, pred[..., None], axis=-1)[..., 0]
    signal = scores > 0
    return np.where(signal, pred, -1), scores, signal
