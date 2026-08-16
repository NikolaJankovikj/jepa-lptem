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
        Shape ``(n_full, T, 2)`` — only tracks present in *every* frame, sorted by
        first-frame position for determinism. Feed these to ``compute_msd``.
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
        Frames a particle may vanish for and still be re-linked (gap-closing).

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
    T = len(frames)
    rows = []
    for f, cents in enumerate(frames):
        cents = np.asarray(cents, dtype=float).reshape(-1, 2)
        for row, col in cents:
            rows.append({"frame": f, "y": row, "x": col})

    if not rows:
        return LinkResult(np.zeros((0, T, 2)), 0.0, 0, 0)

    df = pd.DataFrame(rows)
    linked = tp.link(df, search_range=max_dist, memory=memory,
                     adaptive_stop=2.0, adaptive_step=0.5)

    n_particles = int(linked["particle"].nunique())
    full = []
    for pid, grp in linked.groupby("particle"):
        if grp["frame"].nunique() == T:
            grp = grp.sort_values("frame")
            full.append(grp[["y", "x"]].to_numpy())

    n_full = len(full)
    coverage = n_full / n_particles if n_particles else 0.0
    if n_full == 0:
        return LinkResult(np.zeros((0, T, 2)), coverage, n_particles, 0)

    tracks = np.stack(full)  # (n_full, T, 2)
    order = np.lexsort((tracks[:, 0, 1], tracks[:, 0, 0]))  # by first (row, col)
    return LinkResult(tracks[order], coverage, n_particles, n_full)
