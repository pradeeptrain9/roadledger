"""Stage 5 - multi-frame association. This stage produces the agent's signal.

One measurement of a pothole is a number. Several independent measurements of
the same pothole, from frames taken at different distances, are a number plus a
spread - and the spread is what tells the system whether to trust itself.

`iqr_ratio` (interquartile range over median) is therefore the most important
output of the whole pipeline. It is what makes the agent loop genuinely agentic:
a high spread is a measured fact that changes what the system does next, by
pulling more frames or escalating to a human, rather than a confidence score
invented by a language model.

Association exploits the rectification. Once two frames are rectified to a
metric bird's-eye view of the same road plane, forward driving separates them
by a near-pure translation, so the ego-motion is recoverable in one step with
phase correlation over the two bird's-eye views. The defect's next position is
then simply its current position plus that shift.

Lucas-Kanade on the defect's own centroid was tried first and failed: a pothole
interior is a smooth dark region with no gradient structure, so LK either
refuses to converge or drifts, and the forward-backward check then discards it.
Measured on an eight-frame synthetic drive-past, LK followed the pothole into
*one* frame - leaving the spread undefined and the whole multi-frame argument
inert. Phase correlation over the bird's-eye views is both more robust here and
better posed, because it uses the road surface texture, which is everywhere,
rather than the defect, which is featureless. LK is retained for the
textured-point case and as a fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import TrackParams
from .types import Aggregate, Candidate, Measurement


@dataclass
class TrackedFrame:
    """One frame prepared for tracking: its image, rectification and candidates."""

    frame_id: str
    image: np.ndarray
    homography: np.ndarray | None
    bev: np.ndarray | None
    m_per_px_x: float
    m_per_px_y: float
    candidates: list[Candidate]


def _bev_to_source(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float] | None:
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None
    mapped = cv2.perspectiveTransform(
        np.array([[[point[0], point[1]]]], dtype=np.float64), inverse
    )[0][0]
    return float(mapped[0]), float(mapped[1])


def _source_to_bev(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    mapped = cv2.perspectiveTransform(
        np.array([[[point[0], point[1]]]], dtype=np.float64), homography
    )[0][0]
    return float(mapped[0]), float(mapped[1])


def centroid(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        point = contour.reshape(-1, 2).mean(axis=0)
        return float(point[0]), float(point[1])
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def track_point(
    previous_gray: np.ndarray,
    next_gray: np.ndarray,
    point: tuple[float, float],
    params: TrackParams,
) -> tuple[float, float] | None:
    """Track one point between two frames, verified forward and backward.

    A tracker that silently returns a wrong position is worse than one that
    admits failure, because the measurement it enables would be attributed to
    the wrong defect and would corrupt the spread. So the point is tracked
    forward and then back, and discarded unless it returns to where it started.
    """
    start = np.array([[point]], dtype=np.float32)
    lk_params = {
        "winSize": (params.lk_win_size, params.lk_win_size),
        "maxLevel": params.lk_max_level,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, next_gray, start, None, **lk_params
    )
    if forward is None or not status_forward.flatten()[0]:
        return None
    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        next_gray, previous_gray, forward, None, **lk_params
    )
    if backward is None or not status_backward.flatten()[0]:
        return None

    error = float(np.linalg.norm(backward[0][0] - start[0][0]))
    if error > params.max_fb_error_px:
        return None
    return float(forward[0][0][0]), float(forward[0][0][1])


def bev_shift_lk(
    previous_bev: np.ndarray, next_bev: np.ndarray, params: TrackParams
) -> tuple[tuple[float, float], float]:
    """Ego-motion between two bird's-eye views, from road-texture features.

    Returns ((dx, dy), confidence), where confidence is the fraction of tracked
    features that agreed with the median flow.

    This replaced a single global phase correlation, which was too fragile: on
    a fourteen-frame drive it recovered the motion correctly for six frame
    pairs (response 0.35-0.46, forward 0.58 m against a 0.60 m truth) and
    collapsed on the other seven (response 0.003-0.035, forward essentially
    zero). One global peak has to be right about the whole image at once, and
    a bird's-eye view is dominated by lane markings that are translation
    invariant along the direction of travel - driving along a solid line looks
    like standing still.

    Tracking many small road-texture features instead and taking the median
    flow is both more robust and better posed. The features come from the road
    surface, which has persistent aggregate texture everywhere.

    Crucially, lane markings are *excluded* rather than outvoted. Driving along
    a solid line looks exactly like standing still, so the markings are the
    brightest, highest-contrast, most trackable thing in the view and they
    report zero motion. Leaving them in dragged both estimators to nearly zero:
    a correlation scan peaked at 16 px per frame where the true shift, verified
    by projecting a fixed ground point through both homographies, is 28.9 px.
    A median over inliers does not save you when the confuser is the majority
    of the signal. Asphalt aggregate is lower contrast but it is the only thing
    in the frame that actually moves with the road.
    """
    from .propose import paint_halo_mask, paint_threshold, valid_road_mask

    previous_gray = (
        previous_bev if previous_bev.ndim == 2 else cv2.cvtColor(previous_bev, cv2.COLOR_BGR2GRAY)
    )
    next_gray = next_bev if next_bev.ndim == 2 else cv2.cvtColor(next_bev, cv2.COLOR_BGR2GRAY)
    if previous_gray.shape != next_gray.shape:
        return (0.0, 0.0), 0.0

    mask = valid_road_mask(previous_bev)
    road_pixels = previous_gray[mask > 0]
    if road_pixels.size < 100:
        return (0.0, 0.0), 0.0

    # Exclude lane paint and the halo around it: those pixels are static along
    # the direction of travel and would report no motion.
    paint = paint_halo_mask(previous_gray, paint_threshold(road_pixels), 9)
    asphalt = cv2.bitwise_and(mask, cv2.bitwise_not(paint))
    if cv2.countNonZero(asphalt) < 500:
        return (0.0, 0.0), 0.0

    features = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=params.lk_max_corners,
        qualityLevel=params.lk_quality_level,
        minDistance=params.lk_min_distance,
        mask=asphalt,
    )
    if features is None or len(features) < 8:
        return (0.0, 0.0), 0.0

    lk_params = {
        "winSize": (params.lk_win_size, params.lk_win_size),
        "maxLevel": params.lk_max_level,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, next_gray, features, None, **lk_params
    )
    if forward is None:
        return (0.0, 0.0), 0.0
    backward, status_back, _ = cv2.calcOpticalFlowPyrLK(
        next_gray, previous_gray, forward, None, **lk_params
    )
    if backward is None:
        return (0.0, 0.0), 0.0

    good = (status.flatten() == 1) & (status_back.flatten() == 1)
    if good.sum() < 8:
        return (0.0, 0.0), 0.0

    start = features[good].reshape(-1, 2)
    end = forward[good].reshape(-1, 2)
    returned = backward[good].reshape(-1, 2)

    # Forward-backward error: a feature that does not come home was not tracked.
    fb_error = np.linalg.norm(returned - start, axis=1)
    kept = fb_error <= params.max_fb_error_px
    if kept.sum() < 8:
        return (0.0, 0.0), 0.0

    flow = end[kept] - start[kept]
    median_flow = np.median(flow, axis=0)

    # Confidence is agreement: the share of tracked features whose flow sits
    # within two pixels of the median. Lane markings and the defect itself are
    # expected to dissent, so this is a share rather than a requirement.
    residual = np.linalg.norm(flow - median_flow, axis=1)
    agreement = float((residual <= 2.0).mean())
    return (float(median_flow[0]), float(median_flow[1])), agreement


def bev_shift(
    previous_bev: np.ndarray, next_bev: np.ndarray
) -> tuple[tuple[float, float], float]:
    """Global phase-correlation ego-motion. Kept as a cross-check only.

    Retained because it is a useful independent estimate when it is confident,
    and because the comparison between the two estimators is reported in the
    evaluation. It is no longer the primary path - see `bev_shift_lk`.
    """
    previous_gray = (
        previous_bev if previous_bev.ndim == 2 else cv2.cvtColor(previous_bev, cv2.COLOR_BGR2GRAY)
    )
    next_gray = next_bev if next_bev.ndim == 2 else cv2.cvtColor(next_bev, cv2.COLOR_BGR2GRAY)
    if previous_gray.shape != next_gray.shape:
        return (0.0, 0.0), 0.0

    previous_float = previous_gray.astype(np.float32)
    next_float = next_gray.astype(np.float32)
    # A Hanning window suppresses the edge discontinuity that the warp border
    # introduces, which would otherwise dominate the correlation peak.
    window = cv2.createHanningWindow(
        (previous_gray.shape[1], previous_gray.shape[0]), cv2.CV_32F
    )
    (dx, dy), response = cv2.phaseCorrelate(previous_float, next_float, window)
    return (float(dx), float(dy)), float(response)


# Below this phase-correlation response the recovered shift is not trusted and
# association falls back to tracking a point with Lucas-Kanade. Calibrated on
# the ground-plane rig: correct shifts came back at 0.37-0.40 while the two
# frame pairs whose rectifications disagreed scored 0.034 and 0.060 and
# produced shifts an order of magnitude too small. A threshold of 0.03 let
# those through as though they were measurements.
MIN_SHIFT_RESPONSE = 0.15

# Minimum share of tracked road features that must agree with the median flow.
MIN_SHIFT_AGREEMENT = 0.35

# The shift must also be physically possible for a vehicle between two sampled
# frames. A high correlation response on a wrong peak is still a wrong peak,
# and these bounds are what a road vehicle can actually do.
MIN_FORWARD_M_PER_FRAME = 0.02
MAX_FORWARD_M_PER_FRAME = 5.0


def associate(
    frames: list[TrackedFrame],
    anchor_index: int,
    anchor: Candidate,
    params: TrackParams,
    max_frames: int | None = None,
) -> list[tuple[int, Candidate]]:
    """Follow one candidate from its anchor frame into subsequent frames.

    Returns (frame_index, candidate) pairs, the anchor included. Tracking stops
    at the first frame where the point is lost or no candidate is nearby, since
    beyond that the chain of identity is broken.
    """
    anchor_frame = frames[anchor_index]
    if anchor_frame.homography is None or anchor.contour is None:
        return []

    matched: list[tuple[int, Candidate]] = [(anchor_index, anchor)]
    limit = max_frames if max_frames is not None else params.k_frames

    previous_frame = anchor_frame
    current_bev_point = centroid(anchor.contour)

    for offset in range(1, limit + 1):
        index = anchor_index + offset
        if index >= len(frames):
            break
        frame = frames[index]
        if frame.homography is None or not frame.candidates:
            break

        predicted: tuple[float, float] | None = None
        if previous_frame.bev is not None and frame.bev is not None:
            (dx, dy), agreement = bev_shift_lk(previous_frame.bev, frame.bev, params)
            # Forward motion moves ground features toward the camera, which is
            # +y in the bird's-eye view. A shift that is backwards, stationary
            # or impossibly fast is wrong however confident the estimator.
            forward_m = dy * frame.m_per_px_y
            plausible = MIN_FORWARD_M_PER_FRAME <= forward_m <= MAX_FORWARD_M_PER_FRAME
            if agreement >= MIN_SHIFT_AGREEMENT and plausible:
                predicted = (current_bev_point[0] + dx, current_bev_point[1] + dy)

        if predicted is None:
            # Fall back to tracking the point itself through the source frames.
            source_point = _bev_to_source(current_bev_point, previous_frame.homography)
            if source_point is None:
                break
            tracked = track_point(
                cv2.cvtColor(previous_frame.image, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY),
                source_point,
                params,
            )
            if tracked is None:
                break
            predicted = _source_to_bev(tracked, frame.homography)

        best, best_distance = None, math.inf
        for candidate in frame.candidates:
            if candidate.contour is None:
                continue
            point = centroid(candidate.contour)
            dx = (point[0] - predicted[0]) * frame.m_per_px_x
            dy = (point[1] - predicted[1]) * frame.m_per_px_y
            distance = math.hypot(dx, dy)
            if distance < best_distance:
                best, best_distance = candidate, distance

        # Half a metre is generous for a tracked centroid and tight enough that
        # a different defect nearby is not absorbed into this one's spread.
        if best is None or best_distance > 0.5:
            break

        matched.append((index, best))
        previous_frame = frame
        # Re-anchor on the matched candidate rather than the prediction, so
        # small shift errors do not accumulate over the chain.
        current_bev_point = centroid(best.contour) if best.contour is not None else predicted

    return matched


# A measurement taken from a vantage this much worse than the best one is used
# for nothing: not for the value, and not for the spread.
#
# Footprint falls as the square of distance, so this fraction is really a
# distance window: 0.25 admits observations out to twice the closest approach,
# which over a drive-past at 0.6 m steps yields four or five comparable views.
# A 0.5 window admitted only two - below the three-measurement minimum, so
# every defect escalated for want of evidence that was there to be had.
# Only views whose source footprint is within this fraction of the best view's
# contribute to the aggregate.
#
# Re-calibrated from 0.25 against the live path, where it turned out to be the
# binding constraint on whether a defect could be confirmed at all. Projected
# ground area falls as roughly the cube of distance, so a fraction v admits
# distances up to v^(-1/3) times the closest approach: 0.25 is a 1.59x band,
# which at the near edge of the bird's-eye view is about a metre and a half of
# travel. A 128-drive sweep over four defect sizes and four frame spacings
# (eval/live_eval.py) measured what that costs:
#
#   vantage  band    bias at 1.2 m/frame   confirmed
#     0.10   2.15x        -18.7%             8/8
#     0.18   1.77x        -17.9%             8/8
#     0.25   1.59x        -20.0%             7/8
#     0.40   1.36x        -20.0%             1/8
#
# The accuracy is flat across the whole range - within 3 points - while the
# confirmation rate collapses. The original 0.25 was calibrated against the
# first synthetic rig, which regenerated surface noise every frame and so made
# distant views look catastrophic (99% spread, 50% low); on the persistent
# ground plane that replaced it, distant views are only mildly worse. So the
# tight band was buying almost nothing and costing most of the confirmations.
VANTAGE_FOOTPRINT_FRACTION = 0.18


def aggregate(measurements: list[Measurement], params: TrackParams) -> Aggregate:
    """Reduce several measurements of one defect to a value plus a spread.

    Median and interquartile range rather than mean and standard deviation: a
    single badly segmented frame should move the reported area hardly at all,
    and with four to ten measurements a robust statistic is the honest choice.

    Measurements are first restricted to *comparable vantage points*. Tracking
    a defect over a drive-past yields observations from very different
    distances, and one taken at the far edge of the bird's-eye view rests on a
    few dozen source pixels while the closest rests on thousands. Pooling them
    is not averaging, it is diluting: measured on the ground-plane rig, pooling
    every vantage gave a spread of 99% of the median and an area 50% low, which
    looks like a disagreeing measurement when it is really one good measurement
    and three bad ones.

    So the best vantage - the largest source footprint, which is the closest
    approach - sets the standard, and only observations within
    VANTAGE_FOOTPRINT_FRACTION of it contribute. This mirrors what a surveyor
    does: measure from where you can see it, and use the other views to
    confirm. Discarded observations are counted, not hidden, because "three of
    seven views were too distant to use" is information the reviewer wants.
    """
    candidates = [m for m in measurements if m.measurable and m.area_m2 > 0]
    discarded = 0
    best_footprint = max((m.source_footprint_px for m in candidates), default=0.0)
    if best_footprint > 0:
        threshold = VANTAGE_FOOTPRINT_FRACTION * best_footprint
        comparable = [m for m in candidates if m.source_footprint_px >= threshold]
        discarded = len(candidates) - len(comparable)
        usable = comparable
    else:
        usable = candidates
    if not usable:
        return Aggregate(
            candidate_id=measurements[0].candidate_id if measurements else "",
            n_measurements=0,
            area_m2_median=0.0,
            area_m2_iqr=0.0,
            # No measurements means maximum distrust, not zero spread. Reporting
            # 0.0 here would read as perfect agreement and let an unmeasured
            # defect pass straight through the agent's confidence gate.
            iqr_ratio=math.inf,
            area_m2_min=0.0,
            area_m2_max=0.0,
            frame_ids=[m.frame_id for m in measurements],
            n_discarded_vantage=discarded,
            best_footprint_px=round(best_footprint, 1),
        )

    areas = np.array([m.area_m2 for m in usable], dtype=np.float64)
    median = float(np.median(areas))
    q1, q3 = np.percentile(areas, [25, 75])
    iqr = float(q3 - q1)

    # With fewer than the required minimum, the spread is not yet meaningful;
    # it is reported as infinite so the agent gathers more rather than
    # concluding from two frames that happened to agree.
    ratio = iqr / median if median > 1e-9 else math.inf
    if len(usable) < params.min_measurements:
        ratio = math.inf

    return Aggregate(
        candidate_id=usable[0].candidate_id,
        n_measurements=len(usable),
        area_m2_median=round(median, 6),
        area_m2_iqr=round(iqr, 6),
        iqr_ratio=round(ratio, 4) if math.isfinite(ratio) else math.inf,
        area_m2_min=round(float(areas.min()), 6),
        area_m2_max=round(float(areas.max()), 6),
        frame_ids=[m.frame_id for m in usable],
        n_discarded_vantage=discarded,
        best_footprint_px=round(best_footprint, 1),
    )
