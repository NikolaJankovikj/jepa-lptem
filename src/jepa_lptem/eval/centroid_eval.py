"""Scoring decoded centroids: detection counts, threshold choice, pooled MSD.

Shared by Gate 1 (probes on encoded latents) and Gate 2 (the same probes on
predicted latents), so both grade positions the same way. A *window* here is
``(occ_prob (T,G,G), offsets (T,G,G,2), GT centroids per frame, meta)``, with
``meta`` carrying the Step-6 cache fields (``crop``, ``D``, ``drift``,
``frames_per_latent``, ``msd_max_lag``).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from jepa_lptem.eval.observables import compute_msd
from jepa_lptem.eval.probes import decode_centroids
from jepa_lptem.eval.tracking import link_tracks

GRID = 16
THRESHOLDS = np.round(np.arange(0.30, 1.00, 0.05), 2)  # searched on val
MIN_FULL_TRACKS = 2  # an ensemble MSD needs more than one track


def match_distances(pred: np.ndarray, gt: np.ndarray, tol: float) -> np.ndarray:
    """Distances of the one-to-one detection/GT pairs closer than ``tol``."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros(0)
    dist = cdist(pred, gt)
    rows, cols = linear_sum_assignment(dist)
    paired = dist[rows, cols]
    return paired[paired < tol]


def match_count(pred: np.ndarray, gt: np.ndarray, tol: float) -> int:
    """True positives: one-to-one pairs of detections and GT closer than ``tol``."""
    return len(match_distances(pred, gt, tol))


def decode_and_match(window, threshold: float):
    """Decode a window's frames -> ([tp, n_pred, n_gt], per-frame centroids).

    A detection counts as a hit within half a cell of an unclaimed GT centre."""
    occ, off, gt, meta = window
    cell = meta["crop"] / GRID
    pred = [
        decode_centroids(o, f, threshold=threshold, cell_size=cell)
        for o, f in zip(occ, off, strict=True)
    ]
    tp = sum(match_count(p, g, cell / 2) for p, g in zip(pred, gt, strict=True))
    return np.array([tp, sum(map(len, pred)), sum(map(len, gt))]), pred


def pick_threshold(windows, thresholds=THRESHOLDS) -> float:
    """Occupancy threshold maximising detection F1 -- chosen on val, applied to
    test. The BCE pos-weight shifts every probability up by design, so 0.5 is
    not a calibrated cut."""
    counts = np.zeros((len(thresholds), 3))
    for window in windows:
        for k, th in enumerate(thresholds):
            counts[k] += decode_and_match(window, th)[0]
    tp, n_pred, n_gt = counts.T
    f1 = 2 * tp / np.maximum(n_pred + n_gt, 1)
    return float(thresholds[int(f1.argmax())])


def window_msd(frames, meta, memory: int = 0):
    """Link one window's per-frame centroids -> (MSD/4Dt ratio or None, LinkResult).

    Drift is removed exactly and lags are capped by the clip's ``msd_max_lag``.
    ``memory`` > 0 closes gaps of up to that many frames (missed frames stay NaN).
    Raises trackpy's ``SubnetOversizeException`` when even adaptive linking gives up.
    """
    dt = meta["frames_per_latent"]
    step = 1.772 * np.sqrt(meta["D"] * dt) + dt * np.hypot(*meta["drift"])
    link = link_tracks(frames, max_dist=max(8.0, 3.0 * step), memory=memory)
    if link.n_full < MIN_FULL_TRACKS:
        return None, link
    tracks = link.tracks.copy()
    drift = np.asarray(meta["drift"]) * dt
    tracks -= drift[None, None, :] * np.arange(tracks.shape[1])[None, :, None]
    msd = compute_msd(tracks)
    max_lag = max(2, min(len(msd), int(meta["msd_max_lag"] / dt)))
    lags = np.arange(1, max_lag + 1)
    return float(np.mean(msd[:max_lag] / (4 * meta["D"] * dt * lags))), link


def pool_msd(results, suffix: str = "") -> dict:
    """``(ratio or None, LinkResult)`` per window -> n_full-weighted pooled ratio.

    Per-window MSD at these particle counts is noise (Step 4e); only the pool is
    a number."""
    ratios = [r for r, _ in results if r is not None]
    weights = [link.n_full for r, link in results if r is not None]
    coverages = [link.coverage for _, link in results]
    return {
        f"msd_ratio_pooled{suffix}": (
            float(np.average(ratios, weights=weights)) if ratios else float("nan")
        ),
        f"msd_windows_used{suffix}": len(ratios),
        f"track_coverage_mean{suffix}": float(np.mean(coverages)) if coverages else 0.0,
    }


def eval_centroid_msd(windows, threshold: float, memory: int = 0) -> dict:
    """Decode -> link -> pooled MSD/4Dt ratio, exactly as Gate 2 will consume it.

    Reported next to the same pipeline run on the ground-truth centroids (the
    ``_gt`` keys): the ceiling a perfect probe reaches. It is not 1 -- only
    particles that stay inside the crop for the whole window form full tracks,
    which biases MSD low (0.93-0.98 simulated) and caps coverage near 0.3-0.5 --
    so the probe is judged against this ceiling, not against 1. ``memory`` applies
    the same gap closing to both.
    """
    from trackpy.linking.utils import SubnetOversizeException

    probe, truth = [], []
    counts = np.zeros(3)
    n_unlinkable = 0
    for window in windows:
        c, pred = decode_and_match(window, threshold)
        counts += c
        truth.append(window_msd(window[2], window[3], memory))
        try:
            probe.append(window_msd(pred, window[3], memory))
        except SubnetOversizeException:
            # the probe emitted a wall of false positives on this window --
            # skip it, but never silently
            n_unlinkable += 1
    tp, n_pred, n_gt = counts
    return {
        "threshold": threshold,
        "det_precision": tp / max(n_pred, 1),
        "det_recall": tp / max(n_gt, 1),
        "msd_windows_unlinkable": n_unlinkable,
        **pool_msd(probe),
        **pool_msd(truth, "_gt"),
    }
