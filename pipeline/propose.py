"""Stage 3 - candidate proposal. This is the COOL-accelerated core pass.

Deliberately built from the three operations COOL names as accelerated on
Graviton: `resize`, `adaptiveThreshold` with ADAPTIVE_THRESH_GAUSSIAN_C, and
`findContours`. Those three carry the per-frame cost of the product, so the
COOL measurement is of the workload that actually matters rather than of a
synthetic side benchmark.

The pass runs downscaled, on every frame. Its job is recall, not precision -
anything it discards is never measured, while a false positive only costs one
refinement. The thresholds are therefore set loose, and the agent tightens them
when a frame returns implausibly many candidates.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .config import ProposeParams
from .instrument import Timings
from .types import Candidate

# Lane paint is separated from asphalt by a robust spread, not a ratio. A
# ratio test on a noisy surface flags a fifth of the road as paint, which then
# dilates over the whole frame; paint is genuinely far brighter (around 230 on
# a 70-level surface), so a median-plus-k-MAD test isolates it cleanly.
_PAINT_MAD_SIGMAS = 6.0
_PAINT_MIN_MARGIN = 30.0

# A candidate brighter than this multiple of the road median is not a defect.
_CANDIDATE_MAX_BRIGHTNESS_RATIO = 1.08


def _candidate_id(frame_id: str, index: int, bbox: tuple[int, int, int, int]) -> str:
    digest = hashlib.sha1(f"{frame_id}:{index}:{bbox}".encode()).hexdigest()[:10]
    return f"c_{digest}"


def median_and_spread(road_pixels: np.ndarray) -> tuple[float, float]:
    """Exact median and scaled MAD of 8-bit pixels, from a 256-bin histogram.

    Both statistics come out of one pass and a 256-element reduction, with no
    sorting. `np.median` was the single largest cost in this pass - 38% of its
    runtime, because the median and the MAD together need three full partitions
    of roughly 300k pixels per frame. For 8-bit data a histogram gives the same
    answers exactly, not approximately: the median is the first bin where the
    cumulative count passes half, and the distribution of |x - median| is
    recoverable from the same counts.

    This matters beyond speed. The three operations COOL accelerates were only
    6% of this pass while incidental statistics dominated it, which would have
    made any COOL measurement a measurement of the overhead around it.
    """
    flat = road_pixels.ravel()
    if flat.dtype != np.uint8:
        median = float(np.median(flat))
        spread = 1.4826 * float(np.median(np.abs(flat.astype(np.float64) - median)))
        return median, spread

    counts = np.bincount(flat, minlength=256).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return 0.0, 0.0
    half = total / 2.0
    median = int(np.searchsorted(np.cumsum(counts), half))

    deviations = np.abs(np.arange(256) - median)
    deviation_counts = np.bincount(deviations, weights=counts, minlength=256)
    mad = int(np.searchsorted(np.cumsum(deviation_counts), half))
    return float(median), 1.4826 * float(mad)


def paint_threshold(road_pixels: np.ndarray) -> float:
    """Brightness above which a pixel is lane paint rather than pavement."""
    median, spread = median_and_spread(road_pixels)
    return median + max(_PAINT_MIN_MARGIN, _PAINT_MAD_SIGMAS * spread)


def paint_halo_mask(gray: np.ndarray, paint_level: float, window_px: int) -> np.ndarray:
    """Lane paint plus the halo that adaptive thresholding puts around it.

    `adaptiveThreshold` compares each pixel to a Gaussian mean of its
    neighbourhood. Beside a bright lane marking that local mean is pulled up,
    so ordinary asphalt within roughly one window radius of the paint is
    reported as darker than its surroundings and survives as a large, spurious
    candidate hugging the marking.

    Masking the paint alone does not help, because the halo is *outside* it.
    The paint is therefore dilated by the adaptive window radius before being
    excluded - the artifact is a known property of the operator, not noise to
    be filtered downstream.
    """
    paint = (gray > paint_level).astype(np.uint8) * 255
    radius = max(3, int(window_px))

    # Dilating a 51 px ellipse over a full bird's-eye view cost about 10 ms a
    # frame. The halo is a deliberately conservative safety margin rather than
    # a precise boundary, so it is computed at quarter resolution and scaled
    # back up - the same margin for roughly a sixteenth of the work.
    shrink = 4 if radius >= 12 else 1
    if shrink > 1:
        small = cv2.resize(
            paint,
            (max(1, paint.shape[1] // shrink), max(1, paint.shape[0] // shrink)),
            interpolation=cv2.INTER_NEAREST,
        )
        small_radius = max(3, radius // shrink) | 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (small_radius, small_radius)
        )
        dilated = cv2.dilate(small, kernel)
        return cv2.resize(
            dilated, (paint.shape[1], paint.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius | 1, radius | 1))
    return cv2.dilate(paint, kernel)


def valid_road_mask(bev: np.ndarray) -> np.ndarray:
    """Pixels inside the warped road quad.

    `warpPerspective` leaves black borders outside the source trapezoid. A
    contour straddling that border would have a meaningless area, so the border
    is excluded explicitly rather than hoped away.
    """
    gray = bev if bev.ndim == 2 else cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    mask = (gray > 0).astype(np.uint8) * 255
    # Erode so candidates must sit clear of the border, not merely touch it.
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def propose(
    bev: np.ndarray,
    frame_id: str,
    params: ProposeParams,
    timings: Timings | None = None,
) -> tuple[list[Candidate], Timings]:
    """Propose defect candidates from a rectified bird's-eye view."""
    params = params.normalised()
    timings = timings or Timings()

    gray = bev if bev.ndim == 2 else cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    mask = valid_road_mask(bev)

    road_pixels = gray[mask > 0]
    if road_pixels.size < 100:
        return [], timings
    # One pass for both statistics, reused by the paint test and the candidate
    # brightness filter below.
    road_median, road_spread = median_and_spread(road_pixels)
    paint_level = road_median + max(_PAINT_MIN_MARGIN, _PAINT_MAD_SIGMAS * road_spread)

    # The adaptive window measured in full bird's-eye pixels, which is the
    # coordinate system the paint halo lives in.
    window_px = params.block_size / max(params.scale, 1e-6)
    halo = paint_halo_mask(gray, paint_level, int(window_px / 2))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(halo))
    if cv2.countNonZero(mask) < 100:
        return [], timings

    # --- COOL op 1: resize -------------------------------------------------
    scale = max(0.05, min(1.0, params.scale))
    small_size = (max(8, round(gray.shape[1] * scale)), max(8, round(gray.shape[0] * scale)))
    with timings.measure("resize"):
        small = cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, small_size, interpolation=cv2.INTER_NEAREST)

    # --- COOL op 2: adaptive Gaussian threshold ----------------------------
    # THRESH_BINARY_INV because pavement defects are darker than the surface
    # around them; the adaptive window makes this robust to the illumination
    # gradients that dominate real dashcam footage.
    with timings.measure("adaptiveThreshold"):
        binary = cv2.adaptiveThreshold(
            small,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            params.block_size,
            params.c_constant,
        )
    binary = cv2.bitwise_and(binary, small_mask)

    with timings.measure("morphology"):
        if params.morph_open_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (params.morph_open_kernel, params.morph_open_kernel)
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        if params.morph_close_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (params.morph_close_kernel, params.morph_close_kernel)
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # --- COOL op 3: contour detection --------------------------------------
    with timings.measure("findContours"):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    inverse_scale = 1.0 / scale
    candidates: list[Candidate] = []
    for index, contour in enumerate(contours):
        # Back to full bird's-eye resolution before any geometry is judged, so
        # the thresholds in config are expressed in one coordinate system
        # regardless of the proposal scale the agent chose.
        full = (contour.astype(np.float64) * inverse_scale).astype(np.int32)
        area_px = float(cv2.contourArea(full))
        if not (params.min_area_px <= area_px <= params.max_area_px):
            continue

        hull_area = float(cv2.contourArea(cv2.convexHull(full)))
        solidity = area_px / hull_area if hull_area > 0 else 0.0
        if solidity < params.min_solidity:
            continue

        x, y, w, h = cv2.boundingRect(full)
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect > params.max_aspect_ratio:
            continue

        # Allocating and filling a full-view mask per candidate made this loop
        # O(candidates x 640 x 640). The contour lives inside its bounding box,
        # so the mask only has to.
        blob = np.zeros((h, w), np.uint8)
        cv2.drawContours(blob, [full], -1, 255, -1, offset=(-x, -y))
        blob = cv2.bitwise_and(blob, mask[y : y + h, x : x + w])
        if cv2.countNonZero(blob) == 0:
            continue
        mean_intensity = float(cv2.mean(gray[y : y + h, x : x + w], mask=blob)[0])

        # Lane paint and other bright features are not pavement defects.
        if mean_intensity > road_median * _CANDIDATE_MAX_BRIGHTNESS_RATIO:
            continue

        bbox = (int(x), int(y), int(w), int(h))
        candidates.append(
            Candidate(
                candidate_id=_candidate_id(frame_id, index, bbox),
                frame_id=frame_id,
                bbox=bbox,
                area_px=area_px,
                solidity=round(solidity, 4),
                aspect_ratio=round(aspect, 4),
                mean_intensity=round(mean_intensity, 2),
                contour=full,
            )
        )

    # Largest first: if the agent caps how many candidates it refines, it
    # should spend that budget on the ones that matter most.
    candidates.sort(key=lambda c: c.area_px, reverse=True)
    return candidates, timings
