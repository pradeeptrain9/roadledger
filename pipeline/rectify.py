"""Stage 2 - road-plane rectification to a metric bird's-eye view.

Measuring a defect in image pixels is meaningless: the same pothole covers four
times the pixels when it is half as far away. So the road surface is rectified
onto a metric plane before anything is measured.

Two independent routes to metric scale:

1. Lane-width prior. The two dominant lane markings converge at the vanishing
   point; their separation at the bottom of the frame corresponds to a known
   lane width. This fixes the *lateral* scale and needs nothing but the image.
2. Camera geometry. Given camera height and vertical field of view, pinhole
   geometry over a flat road fixes *both* scales exactly.

When both are available they are cross-checked, and disagreement is reported
rather than averaged away. When only the prior is available, the longitudinal
scale is an explicit assumption, the quality score is capped, and the agent can
see that it is measuring on softer ground.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import RectifyParams
from .types import Frame, Rectification

# A rectification this poor is not worth measuring from at all.
_UNUSABLE = 0.0
# Quality ceiling when the depth scale is assumed rather than calibrated.
_ASSUMED_SCALE_QUALITY_CAP = 0.75


def _segments(gray: np.ndarray, params: RectifyParams) -> np.ndarray:
    """Lane-marking candidate segments in the near-field road band."""
    height = gray.shape[0]
    band_top = int(height * (1.0 - params.road_band_frac))
    band = gray[band_top:, :]

    # Lane paint is brighter than asphalt; a mild blur suppresses aggregate
    # texture without dissolving the markings.
    blurred = cv2.GaussianBlur(band, (5, 5), 0)
    edges = cv2.Canny(blurred, params.canny_low, params.canny_high)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=params.hough_threshold,
        minLineLength=params.hough_min_line_length,
        maxLineGap=params.hough_max_line_gap,
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.float64)

    segments = lines.reshape(-1, 4).astype(np.float64)
    segments[:, 1] += band_top  # back to full-image coordinates
    segments[:, 3] += band_top

    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.abs(np.where(dx == 0, np.inf, dy / dx))
    keep = (slope >= params.min_abs_slope) & (slope <= params.max_abs_slope)
    return segments[keep]


def _homogeneous_lines(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert segments to normalised line coefficients plus their lengths."""
    x1, y1, x2, y2 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]
    a = y2 - y1
    b = -(x2 - x1)
    c = -(a * x1 + b * y1)
    norm = np.hypot(a, b)
    norm[norm == 0] = 1.0
    lengths = np.hypot(x2 - x1, y2 - y1)
    return np.stack([a / norm, b / norm, c / norm], axis=1), lengths


def _fit_point(lines: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    """Weighted least-squares point closest to all the given lines."""
    if len(lines) < 2:
        return None
    a = lines[:, :2] * weights[:, None]
    d = -lines[:, 2] * weights
    try:
        point, *_ = np.linalg.lstsq(a, d, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return point if np.all(np.isfinite(point)) else None


def estimate_vanishing_point(
    gray: np.ndarray, params: RectifyParams, inlier_tol_px: float = 6.0
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, float]:
    """RANSAC the vanishing point, then refine on inliers.

    Returns the point, the inlier segments, their line coefficients, and the
    mean inlier residual in pixels.
    """
    segments = _segments(gray, params)
    if len(segments) < 2:
        return None, segments, np.empty((0, 3)), math.inf

    lines, lengths = _homogeneous_lines(segments)
    rng = np.random.default_rng(12345)  # fixed seed: an agent trace must replay exactly

    best_inliers = np.zeros(len(lines), dtype=bool)
    n = len(lines)
    iterations = min(200, max(30, n * 4))
    for _ in range(iterations):
        i, j = rng.choice(n, size=2, replace=False)
        pair = _fit_point(lines[[i, j]], lengths[[i, j]])
        if pair is None:
            continue
        residual = np.abs(lines[:, :2] @ pair + lines[:, 2])
        inliers = residual < inlier_tol_px
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers.sum() < 2:
        return None, segments, lines, math.inf

    point = _fit_point(lines[best_inliers], lengths[best_inliers])
    if point is None:
        return None, segments, lines, math.inf

    residual = float(np.mean(np.abs(lines[best_inliers, :2] @ point + lines[best_inliers, 2])))
    return point, segments[best_inliers], lines[best_inliers], residual


def _x_at_row(line: np.ndarray, y: float) -> float | None:
    """Where a line crosses a given image row."""
    a, b, c = line
    if abs(a) < 1e-9:
        return None
    return float(-(b * y + c) / a)


def _lane_pair(
    lines: np.ndarray, lengths: np.ndarray, y_base: float, width: int
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """The dominant left and right lane markings, by bottom-row crossing."""
    centre = width / 2.0
    left: list[tuple[float, float, np.ndarray]] = []
    right: list[tuple[float, float, np.ndarray]] = []

    for line, length in zip(lines, lengths):
        x = _x_at_row(line, y_base)
        if x is None or not (-width <= x <= 2 * width):
            continue
        (left if x < centre else right).append((length, x, line))

    if not left or not right:
        return None

    # Longest segment in each group is the best-supported marking.
    _, x_left, line_left = max(left, key=lambda t: t[0])
    _, x_right, line_right = max(right, key=lambda t: t[0])
    if x_right - x_left < width * 0.08:
        return None  # the two "lanes" are the same marking
    return line_left, line_right, x_left, x_right


def _quality(
    n_inliers: int, residual: float, lane_found: bool, vp_plausible: bool, calibrated: bool
) -> float:
    """A single honest 0-1 score the agent can threshold on.

    Deliberately conservative: anything missing the lane pair or a plausible
    vanishing point lands below the default decision threshold.
    """
    if not lane_found or not vp_plausible or not math.isfinite(residual):
        return _UNUSABLE

    support = min(1.0, n_inliers / 12.0)
    fit = 1.0 / (1.0 + residual / 3.0)
    score = 0.55 * support + 0.45 * fit
    if not calibrated:
        score = min(score, _ASSUMED_SCALE_QUALITY_CAP)
    return round(float(score), 4)


def rectify_calibrated(
    frame: Frame, params: RectifyParams
) -> tuple[Rectification, np.ndarray | None]:
    """Rectify from camera calibration alone, with no lane markings required.

    This exists because the lane-marking route does not survive contact with
    real roads. Measured on 175 RDD2022 India frames it produced a usable
    rectification on **4** of them - 2.3% - because unmarked and worn roads
    simply have no converging lane lines to find, and that failure takes the
    entire measurement chain with it.

    Nothing about the road plane actually requires markings. For a camera of
    known height, field of view and downward tilt, the ground-plane homography
    follows from geometry: the horizon is fixed by the pitch, and a ground
    point at lateral offset u and forward distance v images at

        x = cx + f*u/v,    y = y_horizon + f*h/v

    which is linear in homogeneous coordinates and therefore a homography. The
    markings were only ever a way to *estimate* those parameters from an
    uncalibrated camera, and on a camera we control - a mounted dashcam, a
    phone on a bracket - they are known exactly and better.

    The trade: this requires calibration, and it trusts it. A wrong camera
    height scales every area by the square of the error, which is precisely why
    `rectify` cross-checks it against the lane-width prior when markings happen
    to be present.
    """
    if frame.image is None:
        raise ValueError(f"frame {frame.frame_id} carries no image")
    camera = params.camera
    if camera is None:
        return _static_fallback(frame, params, "no camera calibration supplied"), None

    image = frame.image
    height, width = image.shape[:2]
    focal = camera.focal_px(height)
    horizon = camera.horizon_px(height)
    centre_x = width / 2.0

    # The measured band runs from the bottom of the frame to a row kept clear
    # of the horizon, where a flat-road assumption breaks down and one pixel
    # spans many metres.
    y_base = float(height - 1)
    y_top = max(horizon + 0.08 * height, height * (1.0 - params.road_band_frac))
    if y_top >= y_base:
        return _static_fallback(frame, params, "degenerate band: horizon below the frame"), None

    def ground_distance(y: float) -> float:
        delta = y - horizon
        return camera.height_m * focal / delta if delta > 1e-6 else math.inf

    d_base = ground_distance(y_base)
    d_top = ground_distance(y_top)
    if not (math.isfinite(d_base) and math.isfinite(d_top)) or d_top <= d_base:
        return _static_fallback(frame, params, "calibration gives a degenerate horizon"), None

    half = params.bev_metres_across / 2.0

    def project(lateral_m: float, distance_m: float) -> tuple[float, float]:
        return (
            centre_x + lateral_m * focal / distance_m,
            horizon + camera.height_m * focal / distance_m,
        )

    source = np.array(
        [
            project(-half, d_base), project(half, d_base),
            project(half, d_top), project(-half, d_top),
        ],
        dtype=np.float32,
    )
    destination = np.array(
        [
            [0.0, params.bev_height_px - 1],
            [params.bev_width_px - 1, params.bev_height_px - 1],
            [params.bev_width_px - 1, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(source, destination)
    bev = cv2.warpPerspective(
        image, homography, (params.bev_width_px, params.bev_height_px),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    metres_ahead = d_top - d_base
    return (
        Rectification(
            frame_id=frame.frame_id,
            homography=homography,
            # Calibrated geometry is trustworthy, but it is trusted rather than
            # verified: with no markings in view there is nothing to cross-check
            # it against, so it does not score as highly as a corroborated one.
            quality=0.80,
            method="calibrated",
            vanishing_point=(centre_x, horizon),
            lane_px_width=None,
            m_per_px_x=params.bev_metres_across / params.bev_width_px,
            m_per_px_y=metres_ahead / params.bev_height_px,
            near_distance_m=round(d_base, 4),
            notes=(
                f"from calibration alone (h={camera.height_m} m, vfov={camera.vfov_deg} deg, "
                f"pitch={camera.pitch_deg} deg); covers {d_base:.1f}-{d_top:.1f} m ahead. "
                "No lane markings were needed, and none were used to check it."
            ),
        ),
        bev,
    )


def rectify(frame: Frame, params: RectifyParams) -> tuple[Rectification, np.ndarray | None]:
    """Rectify one frame's road surface to a metric bird's-eye view.

    Prefers the lane-marking route, because when markings are present they
    corroborate the scale independently. Falls back to calibration when they
    are not - which on real unmarked roads is most of the time.
    """
    if frame.image is None:
        raise ValueError(f"frame {frame.frame_id} carries no image")

    image = frame.image
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    point, inlier_segments, inlier_lines, residual = estimate_vanishing_point(gray, params)
    if point is None:
        if params.camera is not None:
            return rectify_calibrated(frame, params)
        return _static_fallback(frame, params, "no vanishing point: too few lane-like segments"), None

    vp_x, vp_y = float(point[0]), float(point[1])
    # A vanishing point below the road band or far outside the frame means the
    # geometry estimate has collapsed.
    vp_plausible = (-width <= vp_x <= 2 * width) and (0.0 <= vp_y <= height * (1.0 - params.road_band_frac))

    _, lengths = _homogeneous_lines(inlier_segments) if len(inlier_segments) else (None, np.array([]))
    y_base = float(height - 1)
    pair = _lane_pair(inlier_lines, lengths, y_base, width) if len(inlier_lines) else None

    if pair is None or not vp_plausible:
        reason = "no lane pair at frame base" if pair is None else "implausible vanishing point"
        if params.camera is not None:
            return rectify_calibrated(frame, params)
        return _static_fallback(frame, params, reason), None

    line_left, line_right, x_left, x_right = pair
    lane_px = x_right - x_left

    # Top of the measured band, kept clear of the horizon where a flat-road
    # assumption breaks down and one pixel spans many metres.
    y_top = max(vp_y + 0.08 * height, height * (1.0 - params.road_band_frac))
    x_left_top = _x_at_row(line_left, y_top)
    x_right_top = _x_at_row(line_right, y_top)
    if x_left_top is None or x_right_top is None or x_right_top <= x_left_top:
        if params.camera is not None:
            return rectify_calibrated(frame, params)
        return _static_fallback(frame, params, "lane lines do not bracket the band top"), None

    calibrated = params.camera is not None
    notes: list[str] = []

    # Lateral scale from the lane-width prior.
    m_per_px_lateral_prior = params.lane_width_m / max(lane_px, 1e-6)

    metres_ahead = params.bev_metres_ahead
    near_distance_m = 0.0
    scale_disagreement = None
    if calibrated:
        camera = params.camera
        assert camera is not None
        focal = camera.focal_px(height)
        # Flat-road pinhole: a row y below the horizon images ground at
        # distance h*f/(y - y_vp).
        def ground_distance(y: float) -> float:
            delta = y - vp_y
            return camera.height_m * focal / delta if delta > 1e-6 else math.inf

        d_base = ground_distance(y_base)
        d_top = ground_distance(y_top)
        if math.isfinite(d_base) and math.isfinite(d_top) and d_top > d_base:
            metres_ahead = d_top - d_base
            near_distance_m = d_base
            # At the base row the lane spans lane_px pixels; geometry predicts
            # a lateral scale of d/f metres per pixel there.
            m_per_px_lateral_geom = d_base / focal
            scale_disagreement = abs(m_per_px_lateral_geom - m_per_px_lateral_prior) / max(
                m_per_px_lateral_prior, 1e-9
            )
            if scale_disagreement > params.max_scale_disagreement:
                notes.append(
                    f"lane-prior and camera-geometry lateral scales disagree by "
                    f"{scale_disagreement:.0%}; depth scale may be wrong"
                )
            else:
                notes.append(f"scale cross-check agrees within {scale_disagreement:.0%}")
        else:
            calibrated = False
            notes.append("camera geometry supplied but degenerate for this horizon; depth scale assumed")
    else:
        notes.append(
            f"depth scale assumed at {params.bev_metres_ahead} m ahead - supply CameraGeometry to measure it"
        )

    # Source quad: the lane trapezoid. Destination: that same lane as an
    # upright rectangle whose width is the lane's share of the BEV.
    lane_dst_px = (params.lane_width_m / params.bev_metres_across) * params.bev_width_px
    dst_x_left = (params.bev_width_px - lane_dst_px) / 2.0
    dst_x_right = dst_x_left + lane_dst_px

    source = np.array(
        [[x_left, y_base], [x_right, y_base], [x_right_top, y_top], [x_left_top, y_top]],
        dtype=np.float32,
    )
    destination = np.array(
        [
            [dst_x_left, params.bev_height_px - 1],
            [dst_x_right, params.bev_height_px - 1],
            [dst_x_right, 0.0],
            [dst_x_left, 0.0],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(source, destination)
    bev = cv2.warpPerspective(
        image,
        homography,
        (params.bev_width_px, params.bev_height_px),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    quality = _quality(
        n_inliers=len(inlier_lines),
        residual=residual,
        lane_found=True,
        vp_plausible=True,
        calibrated=calibrated,
    )

    rect = Rectification(
        frame_id=frame.frame_id,
        homography=homography,
        quality=quality,
        method="vanishing_point" + ("+camera" if calibrated else "+lane_prior"),
        vanishing_point=(vp_x, vp_y),
        lane_px_width=float(lane_px),
        m_per_px_x=params.bev_metres_across / params.bev_width_px,
        m_per_px_y=metres_ahead / params.bev_height_px,
        near_distance_m=round(near_distance_m, 4),
        notes="; ".join(notes),
    )
    return rect, bev


def _static_fallback(frame: Frame, params: RectifyParams, reason: str) -> Rectification:
    """Fixed-trapezoid rectification for frames with no usable lane geometry.

    Reported at low quality on purpose. The agent's job is to notice that and
    either re-rectify with different parameters or skip the frame - not to
    measure from it as though the geometry were known.
    """
    return Rectification(
        frame_id=frame.frame_id,
        homography=None,
        quality=_UNUSABLE,
        method="static_fallback",
        vanishing_point=None,
        lane_px_width=None,
        m_per_px_x=params.bev_metres_across / params.bev_width_px,
        m_per_px_y=params.bev_metres_ahead / params.bev_height_px,
        notes=f"fallback: {reason}",
    )
