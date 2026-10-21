"""Cue extraction, separated from the decision rule.

Splitting these apart was a deliberate correction. The cues are physics and
transfer across datasets; the decision boundary is statistics and must be
*fitted* and cross-validated, not hand-tuned. Hand-tuned trapezoidal bands were
tried first and reached 3/12 on held-out synthetic confusers, because every
adjustment was being fitted by hand to one renderer's noise model. See
docs/negative-results.md.

`CUE_NAMES` is the feature vector contract shared by the extractor, the fitting
script and the classifier.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import Candidate

CUE_NAMES = (
    "darkness_ratio",
    "edge_width_px",
    "texture_ratio",
    "elongation",
    "solidity",
    "rect_fill",
    "boundary_contrast",
    "boundary_sharpness",
)


def boundary_sharpness(gray: np.ndarray, contour: np.ndarray) -> float:
    """Mean gradient magnitude on the contour itself."""
    band = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(band, [contour], -1, 255, thickness=3)
    if cv2.countNonZero(band) == 0:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(cv2.mean(cv2.magnitude(gx, gy), mask=band)[0])


def _masks(gray: np.ndarray, contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inside = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(inside, [contour], -1, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    annulus = cv2.subtract(cv2.dilate(inside, kernel, iterations=2), inside)
    return inside, annulus


def extract(
    bev: np.ndarray, candidate: Candidate, road_median: float | None = None
) -> dict[str, float] | None:
    """Extract the cue vector for one candidate, or None if degenerate.

    The cues and the physics they encode:

    * `darkness_ratio` - region brightness over road brightness. A repair sits
      near 1; a defect, a shadow and standing water are all well below it.
    * `edge_width_px` - width of the intensity transition at the boundary,
      recovered as contrast over gradient. A pothole rim is a depth
      discontinuity and images as a step one to three pixels wide; a cast
      shadow has a penumbra, because the sun is not a point source, and ramps
      over several pixels. This is the cue that separates cavity from shadow.
    * `texture_ratio` - contrast-normalised roughness inside against the
      surrounding annulus. Normalisation matters because shading is
      multiplicative. Note that additive sensor noise raises relative roughness
      in any dark region, so this cue alone cannot separate a smooth wet patch
      from a shadow - it does separate a rough cavity from both.
    * `elongation`, `solidity`, `rect_fill` - geometry. Cracks are elongated
      and poorly filled by their own hull; repairs are near-rectangular.
    * `boundary_contrast`, `boundary_sharpness` - the raw magnitudes behind
      `edge_width_px`, kept because their absolute scale is informative.
    """
    contour = candidate.contour
    if contour is None or len(contour) < 5:
        return None

    gray = bev if bev.ndim == 2 else cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    if road_median is None:
        valid = gray[gray > 0]
        road_median = float(np.median(valid)) if valid.size else 1.0

    inside, annulus = _masks(gray, contour)
    if cv2.countNonZero(inside) < 10 or cv2.countNonZero(annulus) < 10:
        return None

    intensity = gray.astype(np.float64)
    mean_inside = float(intensity[inside > 0].mean())
    mean_outside = float(intensity[annulus > 0].mean())

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    squared = laplacian * laplacian
    rough_inside = float(squared[inside > 0].mean()) / max(mean_inside**2, 1e-9)
    rough_outside = float(squared[annulus > 0].mean()) / max(mean_outside**2, 1e-9)
    texture_ratio = rough_inside / rough_outside if rough_outside > 1e-12 else 1.0

    contrast = abs(mean_outside - mean_inside)
    sharpness = boundary_sharpness(gray, contour)
    # Sobel responds to a unit step with about four times the contrast, so the
    # ratio reads in pixels.
    edge_width = 4.0 * contrast / sharpness if sharpness > 1e-6 else 0.0

    (_, (rect_w, rect_h), _) = cv2.minAreaRect(contour)
    short, long_side = sorted((max(rect_w, 1e-6), max(rect_h, 1e-6)))
    area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))

    return {
        "darkness_ratio": mean_inside / max(road_median, 1e-9),
        "edge_width_px": edge_width,
        "texture_ratio": texture_ratio,
        "elongation": long_side / short,
        "solidity": area / hull_area if hull_area > 0 else 0.0,
        "rect_fill": area / (rect_w * rect_h) if rect_w * rect_h > 0 else 0.0,
        "boundary_contrast": contrast,
        "boundary_sharpness": sharpness,
    }


def vector(cues: dict[str, float]) -> np.ndarray:
    """Cue dict to feature vector, in the one canonical order."""
    return np.array([float(cues[name]) for name in CUE_NAMES], dtype=np.float64)
