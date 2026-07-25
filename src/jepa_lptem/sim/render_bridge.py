"""Render bridge: turn ground-truth (positions, radii) into realistic LPTEM frames.

This is the bridge between ``dynamics.py`` (which produces the exact numbers we
grade the model against) and V-JEPA (which only ever sees pixels). The physics
renderer is a clean re-implementation of the Jamali lab's ``LPTEMsimulator``
(``github.com/JamaliLab/LPTEMsimulator``), itself a Python port of the MATLAB
simulator by Yao et al. (``github.com/chenlabUIUC/simulatedLPTEM``). Using the
same image-formation model as the real SAM-EM data removes the sim-to-real
appearance gap that would otherwise threaten Tier-2 transfer.

Two separable pieces:

- ``draw_spheres``  our geometry front-end: circles at our particle positions,
  with a physically correct spherical-chord gold-thickness map. (The upstream
  code draws nanorods; spheres are ours, matching the growth physics we test.)
- ``render_frame``  the adapted image-formation physics: Beer--Lambert
  attenuation through a gold/water/SiNx thickness stack, Poisson shot noise plus
  Gaussian read noise, and an optional MTF (microscope transfer function) blur.

``render_video`` stitches per-frame renders; ``normalize_uint8`` is a display
helper for saving previews.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Inelastic mean free paths (nm) and defaults from the LPTEMsimulator notebook.
IMFP_GOLD = 70.0
IMFP_WATER = 185.0
IMFP_SINX = 144.0
BEAM_FLUX = 30.0
THICKNESS_WATER = 150.0
THICKNESS_CELL = 50.0
CONVERSION_FACTOR = 37.46


def draw_spheres(
    positions: np.ndarray,
    radii: np.ndarray,
    image_size: tuple[int, int],
    pixel_size_nm: float = 2.0,
):
    """Rasterize spheres into an occupancy mask and a gold-thickness map.

    Parameters
    ----------
    positions : np.ndarray
        Shape ``(N, 2)`` particle centres in pixel coordinates ``(row, col)``
        (matching ``eval/tracking.py``).
    radii : np.ndarray
        Shape ``(N,)`` radii in pixels.
    image_size : (int, int)
        ``(height, width)``.
    pixel_size_nm : float
        Physical size of one pixel; converts the spherical chord to nanometres.

    Returns
    -------
    (occupancy, gold_nm) : tuple[np.ndarray, np.ndarray]
        ``occupancy`` bool ``(H, W)`` union of disks (ground truth for masks).
        ``gold_nm`` float ``(H, W)`` gold path length in nm: for a sphere the
        chord at distance ``d`` from the centre is ``2 sqrt(r^2 - d^2)``,
        summed over particles (overlaps add, as attenuation would).
    """
    h, w = image_size
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    occupancy = np.zeros((h, w), dtype=bool)
    gold_nm = np.zeros((h, w), dtype=float)
    if positions.shape[0] == 0:
        return occupancy, gold_nm

    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    for (r0, c0), rad in zip(positions, radii):
        if rad <= 0:
            continue
        d2 = (rows - r0) ** 2 + (cols - c0) ** 2
        inside = d2 <= rad ** 2
        occupancy |= inside
        chord_px = 2.0 * np.sqrt(np.maximum(rad ** 2 - d2, 0.0))  # 0 outside disk
        gold_nm += chord_px * pixel_size_nm
    return occupancy, gold_nm


def render_frame(
    gold_nm: np.ndarray,
    *,
    beam_flux: float = BEAM_FLUX,
    thickness_water: float = THICKNESS_WATER,
    thickness_cell: float = THICKNESS_CELL,
    imfp_gold: float = IMFP_GOLD,
    imfp_water: float = IMFP_WATER,
    imfp_sinx: float = IMFP_SINX,
    conversion_factor: float = CONVERSION_FACTOR,
    mtf: np.ndarray | None = None,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Form one LPTEM image from a gold-thickness map (Beer--Lambert + noise + MTF).

    Intensity follows attenuation through the sample stack::

        I = beam_flux * exp(-(t_gold/IMFP_gold + t_water/IMFP_water + t_cell/IMFP_SiNx))

    where water is displaced by the particle (``t_water = max(thickness_water -
    t_gold, 0)``  a particle thicker than the cell leaves no water, never negative)
    and the SiNx window contributes both walls (``2 * thickness_cell``). Gold has the
    shortest IMFP, so particle regions attenuate most and appear darker. Poisson shot
    noise (dominant in LPTEM) and Gaussian read noise are added when ``add_noise``,
    then an optional MTF low-pass is applied in Fourier space.

    Parameters
    ----------
    gold_nm : np.ndarray
        ``(H, W)`` gold path length in nm (from :func:`draw_spheres`).
    mtf : np.ndarray or None
        ``(H, W)`` modulation transfer function (fft-shifted, DC at centre). If
        ``None``, no optical blur is applied.
    rng : np.random.Generator or None
        Seeded generator for reproducible noise; ``None`` uses a fresh default.

    Returns
    -------
    np.ndarray
        ``(H, W)`` float image (physical intensity times ``conversion_factor``).
    """
    gold_nm = np.asarray(gold_nm, dtype=float)
    # Gold displaces water, but only down to an empty channel: a particle thicker
    # than the cell cannot displace *negative* water. Without the clamp a large or
    # overlapping particle silently subtracts optical depth and renders too bright.
    water = np.maximum(thickness_water - gold_nm, 0.0)
    cell = 2.0 * thickness_cell
    optical_depth = gold_nm / imfp_gold + water / imfp_water + cell / imfp_sinx
    img = beam_flux * np.exp(-optical_depth)

    if add_noise:
        if rng is None:
            rng = np.random.default_rng()
        img = rng.poisson(img).astype(float)
        img = rng.normal(img, 1.0)

    if mtf is not None:
        ft = fftshift(fft2(img))
        ft = ft * mtf
        img = np.abs(ifft2(ifftshift(ft)))

    return img * conversion_factor


def render_video(
    positions_t,
    radii_t,
    image_size: tuple[int, int],
    *,
    pixel_size_nm: float = 2.0,
    mtf: np.ndarray | None = None,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
    **render_kwargs,
) -> np.ndarray:
    """Render a full clip from per-frame particle states.

    Parameters
    ----------
    positions_t, radii_t : sequence
        Length ``T`` sequences; ``positions_t[i]`` is ``(N_i, 2)`` and
        ``radii_t[i]`` is ``(N_i,)`` for frame ``i`` (particle counts may vary).
    image_size, pixel_size_nm, mtf, add_noise, rng, **render_kwargs :
        Forwarded to :func:`draw_spheres` / :func:`render_frame`.

    Returns
    -------
    np.ndarray
        ``(T, H, W)`` float video.
    """
    if rng is None and add_noise:
        rng = np.random.default_rng()
    frames = []
    for pos, rad in zip(positions_t, radii_t):
        _, gold = draw_spheres(pos, rad, image_size, pixel_size_nm=pixel_size_nm)
        frames.append(
            render_frame(gold, mtf=mtf, add_noise=add_noise, rng=rng, **render_kwargs)
        )
    return np.stack(frames, axis=0)


def normalize_uint8(x: np.ndarray) -> np.ndarray:
    """Min--max scale to ``[0, 255]`` uint8 for saving/preview (per-array)."""
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    return np.round((x - lo) / (hi - lo) * 255.0).astype(np.uint8)
