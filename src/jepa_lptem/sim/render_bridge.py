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

``render_video`` stitches per-frame renders; ``to_uint8`` quantizes them against
the analytically known background so absolute contrast survives to disk.
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

# Readout sigma in electrons. Upstream uses 1.0, but upstream also low-passed its
# read noise along with the signal, which hid what 1.0 costs: a direct electron
# detector *counts* electrons, and sigma = 1.0 e means it cannot tell zero from one
# -- at the 0.45 e/px background of our lowest dose it clipped a third of every
# frame to black. 0.3 e is the effective read noise of a real DED and keeps low-dose
# clips usable. Raise it to model a scintillator-coupled CCD instead.
READ_NOISE = 0.3

# 1-sigma liquid-thickness variation across the cell, in nm.
#
# Real LPTEM illumination is not flat. Measured on the Au-nanorod video (40-px
# Gaussian lowpass, so particles are suppressed): the large-scale background has
# sd/mean = 0.212 and a 1-99 percentile spread of 0.74 -- it varies by 74 % of the
# mean across the field of view, and carries 26 % of the frame's total variance.
# Ours read 0.006, a 34x gap, and it was the whole of the low-frequency deficit in
# the radial power spectrum.
#
# The amplitude is not fitted to that curve. Intensity goes as exp(-t/IMFP_WATER),
# so sd(ln I) = sd(t)/185, and reproducing 0.212 needs sd(t) ~ 39 nm -- about +-26 %
# on a 150 nm cell, which is what a bulging SiNx window does. This is the scalar
# becoming the field it always should have been.
#
# It matters past appearance: V-JEPA predicts masked spatiotemporal blocks, and
# against a byte-identical uniform background predicting a masked *background*
# block is free. Real background needs spatial context to predict. A flat
# background is a free lunch in the pretraining objective that does not exist in
# real data.
THICKNESS_SD_NM = 39.0


def thickness_field(image_size, sd_nm: float = THICKNESS_SD_NM, n_modes: int = 3,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Smooth, zero-mean liquid-thickness variation across the cell, in nm.

    A handful of low-order Fourier modes with random phase and a ``1/k^2`` (red)
    amplitude, so the large scales dominate the way a window bulge does rather than
    looking like filtered noise. Normalized to exactly ``sd_nm`` so the amplitude is
    a parameter and not an accident of which phases were drawn.

    Zero-mean by construction (each mode spans a whole number of periods), which is
    what keeps :func:`background_intensity` -- and therefore the fixed ``to_uint8``
    scale that stops brightness leaking particle size -- valid unchanged. The one
    residual is Jensen: ``E[exp(-(t0+d)/L)] > exp(-t0/L)``, so the realised mean
    background sits ~2 % above the analytic value at ``sd_nm = 39``. That is far
    inside the 4x headroom and is independent of particle size, so it shifts no
    label.

    ponytail: static within a clip -- it models the beam-profile / window-shape
    component, which really is fixed in detector coordinates. The part of the
    structure that lives in *sample* coordinates should translate with the stage
    drift; real background moves at only 0.0046/frame against a 0.212 sd, so the
    static field carries almost all of the effect. Upgrade path if it matters:
    evaluate the modes at coordinates offset by ``drift * t``, which is exact and
    seamless because the field is periodic.
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = image_size
    yy = np.arange(h)[:, None] / h
    xx = np.arange(w)[None, :] / w
    field = np.zeros((h, w))
    for ky in range(n_modes + 1):
        for kx in range(n_modes + 1):
            if ky == 0 and kx == 0:
                continue
            amp = rng.normal() / (ky ** 2 + kx ** 2)
            field += amp * np.cos(2 * np.pi * (ky * yy + kx * xx)
                                  + rng.uniform(0, 2 * np.pi))
    field -= field.mean()
    return field / field.std() * sd_nm


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
    read_noise: float = READ_NOISE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Form one LPTEM image from a gold-thickness map (Beer--Lambert + noise + MTF).

    Intensity follows attenuation through the sample stack::

        I = beam_flux * exp(-(t_gold/IMFP_gold + t_water/IMFP_water + t_cell/IMFP_SiNx))

    where water is displaced by the particle (``t_water = max(thickness_water -
    t_gold, 0)``  a particle thicker than the cell leaves no water, never negative)
    and the SiNx window contributes both walls (``2 * thickness_cell``). Gold has the
    shortest IMFP, so particle regions attenuate most and appear darker.

    **Order matters.** Poisson shot noise, then the MTF, then Gaussian read noise --
    because charge spreading happens *at* the sensor: it blurs the electron events
    that arrived, and the readout electronics add their noise downstream of it.
    Blurring the read noise too (the order this used to have) strips high-frequency
    power that a real detector keeps, and it does so by a factor independent of dose:
    the ratio of temporal to spatial noise sat at 2.9 at every flux, against 1.67
    measured on real video. In this order it is 1.3-1.5 and moves with flux.

    Parameters
    ----------
    gold_nm : np.ndarray
        ``(H, W)`` gold path length in nm (from :func:`draw_spheres`).
    mtf : np.ndarray or None
        ``(H, W)`` modulation transfer function (fft-shifted, DC at centre). If
        ``None``, no optical blur is applied.
    read_noise : float
        Gaussian readout sigma in electrons (see :data:`READ_NOISE`). The one
        detector parameter a real camera actually varies, so it is a knob.
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

    if mtf is not None:
        ft = fftshift(fft2(img))
        ft = ft * mtf
        # real, not abs: MTF ringing beside a dark core is genuinely negative, and
        # folding it positive puts a bright halo where the optics put a dark one.
        img = np.real(ifft2(ifftshift(ft)))

    if add_noise:
        img = rng.normal(img, read_noise)

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
    thickness_sd_nm: float = 0.0,
    **render_kwargs,
) -> np.ndarray:
    """Render a full clip from per-frame particle states.

    Parameters
    ----------
    positions_t, radii_t : sequence
        Length ``T`` sequences; ``positions_t[i]`` is ``(N_i, 2)`` and
        ``radii_t[i]`` is ``(N_i,)`` for frame ``i`` (particle counts may vary).
    thickness_sd_nm : float
        1-sigma liquid-thickness variation (see :func:`thickness_field`). Drawn once
        per clip -- it is the shape of one cell, so it must not change between
        frames. ``0.0`` = the flat background this used to have; the unit tests keep
        that so they grade the image-formation physics without a random field in the
        way, and the generator passes the real value.
    image_size, pixel_size_nm, mtf, add_noise, rng, **render_kwargs :
        Forwarded to :func:`draw_spheres` / :func:`render_frame`.

    Returns
    -------
    np.ndarray
        ``(T, H, W)`` float video.
    """
    if rng is None and add_noise:
        rng = np.random.default_rng()
    if thickness_sd_nm > 0:
        base = render_kwargs.pop("thickness_water", THICKNESS_WATER)
        render_kwargs["thickness_water"] = base + thickness_field(
            image_size, thickness_sd_nm, rng=rng)
    frames = []
    for pos, rad in zip(positions_t, radii_t):
        _, gold = draw_spheres(pos, rad, image_size, pixel_size_nm=pixel_size_nm)
        frames.append(
            render_frame(gold, mtf=mtf, add_noise=add_noise, rng=rng, **render_kwargs)
        )
    return np.stack(frames, axis=0)


BACKGROUND_LEVEL = 64  # uint8 value a particle-free pixel maps to


def background_intensity(
    *,
    beam_flux: float = BEAM_FLUX,
    thickness_water: float = THICKNESS_WATER,
    thickness_cell: float = THICKNESS_CELL,
    imfp_water: float = IMFP_WATER,
    imfp_sinx: float = IMFP_SINX,
    conversion_factor: float = CONVERSION_FACTOR,
) -> float:
    """Intensity of a particle-free pixel: pure water plus both SiNx walls.

    Known in closed form, which is what makes :func:`to_uint8` a *fixed* scale.
    """
    optical_depth = thickness_water / imfp_water + 2.0 * thickness_cell / imfp_sinx
    return float(beam_flux * np.exp(-optical_depth) * conversion_factor)


def to_uint8(img: np.ndarray, **background_kwargs) -> np.ndarray:
    """Quantize to uint8 against the analytic background, not a min--max stretch.

    A per-frame min--max stretch would rescale every clip to span 0--255, which
    **erases absolute contrast**: a 32 nm particle (25 % attenuation) and a 160 nm
    one (77 %) come out looking identical. Worse, the stretch depends on the
    darkest pixel, so overall brightness ends up correlated with particle size --
    a label leak a probe can exploit without ever locating a particle, and with the
    sign inverted relative to the real physics.

    Dividing by the known background instead gives transmittance (1.0 = clean
    water) on one fixed scale for the whole dataset, independent of dose. Values
    above 1.0 are noise excursions, so ``BACKGROUND_LEVEL`` sits at 64 rather than
    mid-range: 4x headroom keeps clipping under 1 % even at the lowest dose we
    generate (~0.4 electrons per pixel), where the raw frame is nearly binary
    before the MTF averages it. The coarser scale costs nothing -- one quantization
    step stays far below the shot noise at every flux in ``BEAM_FLUX_RANGE``.
    """
    background = background_intensity(**background_kwargs)
    scaled = np.asarray(img, dtype=float) / background * BACKGROUND_LEVEL
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)
