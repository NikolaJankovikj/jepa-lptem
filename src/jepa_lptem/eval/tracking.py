"""Centroid extraction and track linking.

Turns model-predicted occupancy heatmaps into particle trajectories, from which
MSD is computed. Linking uses ``trackpy`` (Crocker--Grier), the community-standard
particle tracker, wrapped so that **track coverage is always reported** — we never
silently drop particles the linker lost, because an MSD computed on an unknown
fraction of particles is not a publishable number.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import trackpy as tp
from scipy.ndimage import center_of_mass, label

tp.quiet()  # trackpy logs one INFO line per frame otherwise


def heatmap_to_centroids(hm: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Connected-component centroids of a 2D occupancy heatmap.

    Parameters
    ----------
    hm : np.ndarray
        Shape ``(H, W)`` occupancy map (higher = more likely a particle).
    threshold : float
        Cells with ``hm > threshold`` are foreground.

    Returns
    -------
    np.ndarray
        Shape ``(K, 2)`` of ``(row, col)`` centroids; ``(0, 2)`` if none.
    """
    hm = np.asarray(hm, dtype=float)
    lab, n = label(hm > threshold)
    if n == 0:
        return np.zeros((0, 2))
    return np.asarray(center_of_mass(hm, lab, range(1, n + 1)), dtype=float)


class LinkResult(NamedTuple):
    """Result of :func:`link_tracks`.

    Attributes
    ----------
    tracks : np.ndarray
        Shape ``(n_full, T, 2)`` — only tracks spanning *every* frame, sorted by
        first-frame position for determinism. With ``memory > 0`` a spanning track
        may skip up to ``memory`` frames at a time; those entries are NaN. Feed
        these to ``compute_msd``, which averages over observed pairs only.
    coverage : float
        ``n_full / n_particles`` — fraction of linked particles that survived the
        full clip. Report this alongside any MSD derived from ``tracks``.
    n_particles : int
        Total distinct particles the linker identified.
    n_full : int
        Number of full-length tracks (``== tracks.shape[0]``).
    """

    tracks: np.ndarray
    coverage: float
    n_particles: int
    n_full: int


def link_tracks(
    frames: list[np.ndarray], max_dist: float, memory: int = 0
) -> LinkResult:
    """Link per-frame ``(row, col)`` centroid lists into trajectories via trackpy.

    Parameters
    ----------
    frames : list[np.ndarray]
        One ``(K_i, 2)`` array of ``(row, col)`` centroids per frame.
    max_dist : float
        Maximum inter-frame displacement to link (trackpy ``search_range``).
    memory : int
        Frames a particle may vanish for and still be re-linked (gap-closing);
        the skipped frames come back as NaN in ``tracks``.

    Returns
    -------
    LinkResult
        Full-length tracks plus coverage bookkeeping.

    Notes
    -----
    Linking is adaptive: when a dense cluster of detections makes the assignment
    subnet combinatorially intractable at ``max_dist``, trackpy halves the search
    range for that subnet (down to 2 px) instead of raising
    ``SubnetOversizeException``. A half-trained centroid probe emits exactly such
    clusters of false positives, and one bad window must not kill a batch
    evaluation. If even 2 px cannot resolve it, the exception still propagates --
    callers grading many windows should catch it and report the skip.
    """
    n_frames = len(frames)
    rows = []
    for f, cents in enumerate(frames):
        cents = np.asarray(cents, dtype=float).reshape(-1, 2)
        for row, col in cents:
            rows.append({"frame": f, "y": row, "x": col})

    if not rows:
        return LinkResult(np.zeros((0, n_frames, 2)), 0.0, 0, 0)

    df = pd.DataFrame(rows)
    linked = tp.link(
        df, search_range=max_dist, memory=memory, adaptive_stop=2.0, adaptive_step=0.5
    )
    n_particles = int(linked["particle"].nunique())
    tracks = _spanning_tracks(linked, n_frames)
    coverage = len(tracks) / n_particles if n_particles else 0.0
    return LinkResult(tracks, coverage, n_particles, len(tracks))


def _spanning_tracks(linked: pd.DataFrame, n_frames: int) -> np.ndarray:
    """Tracks observed in the first and the last frame -> ``(n, n_frames, 2)``.

    With memory > 0 a spanning track may skip frames; they stay NaN. Never
    interpolate them: a straight line under-states the random step and biases
    MSD low. With memory = 0 a spanning track has every frame.
    """
    full = []
    for _, grp in linked.groupby("particle"):
        if grp["frame"].min() != 0 or grp["frame"].max() != n_frames - 1:
            continue
        track = np.full((n_frames, 2), np.nan)
        track[grp["frame"].to_numpy()] = grp[["y", "x"]].to_numpy()
        full.append(track)
    if not full:
        return np.zeros((0, n_frames, 2))
    tracks = np.stack(full)
    order = np.lexsort((tracks[:, 0, 1], tracks[:, 0, 0]))  # by first (row, col)
    return tracks[order]
