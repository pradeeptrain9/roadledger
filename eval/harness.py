"""Evaluation harness: measured claims, not eyeballed ones.

Every defect in a synthetic scene has an exactly known centroid and area, so
candidates can be matched to ground truth by projecting the truth centroid
through the same homography the pipeline used. That gives honest precision and
recall alongside the area error, and it stops a missed defect from being hidden
by zipping two sorted lists of different lengths - which is how an earlier
version of this check quietly dropped a false negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline.config import PipelineConfig
from pipeline.propose import propose
from pipeline.rectify import rectify
from pipeline.refine import refine
from pipeline.types import Frame, Measurement

# A candidate counts as the same defect as a truth item if their bird's-eye
# centroids are within this many metres.
MATCH_TOLERANCE_M = 0.35


@dataclass
class Match:
    truth_area_m2: float
    truth_distance_m: float
    cheap_area_m2: float
    measurement: Measurement
    centroid_error_m: float

    @property
    def refined_error_pct(self) -> float:
        return 100.0 * (self.measurement.area_m2 - self.truth_area_m2) / self.truth_area_m2

    @property
    def cheap_error_pct(self) -> float:
        return 100.0 * (self.cheap_area_m2 - self.truth_area_m2) / self.truth_area_m2


@dataclass
class SceneResult:
    matches: list[Match] = field(default_factory=list)
    false_positives: int = 0
    false_negatives: int = 0
    missed_truth: list[dict] = field(default_factory=list)
    rectification_quality: float = 0.0
    rectification_method: str = ""


def _bev_centroid(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        point = contour.reshape(-1, 2).mean(axis=0)
        return float(point[0]), float(point[1])
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def evaluate_scene(image: np.ndarray, truth: dict, config: PipelineConfig) -> SceneResult:
    """Run the pipeline on one scene and match its output to ground truth."""
    frame = Frame(
        frame_id="eval",
        index=0,
        timestamp_s=0.0,
        sharpness=999.0,
        image=image,
        source="synthetic",
    )
    rect, bev = rectify(frame, config.rectify)
    result = SceneResult(
        rectification_quality=rect.quality, rectification_method=rect.method
    )
    truth_defects = truth.get("defects", [])
    if bev is None or rect.homography is None:
        result.false_negatives = len(truth_defects)
        result.missed_truth = [dict(d, reason="rectification failed") for d in truth_defects]
        return result

    candidates, _ = propose(bev, frame.frame_id, config.propose)
    m2_per_px = rect.m_per_px_x * rect.m_per_px_y

    # Project each truth centroid into the bird's-eye view the pipeline built.
    truth_points: list[tuple[dict, tuple[float, float]]] = []
    for defect in truth_defects:
        source_point = np.array([[list(defect["centre_px"])]], dtype=np.float64)
        bev_point = cv2.perspectiveTransform(source_point, rect.homography)[0][0]
        truth_points.append((defect, (float(bev_point[0]), float(bev_point[1]))))

    candidate_points = [(c, _bev_centroid(c.contour)) for c in candidates if c.contour is not None]

    # Greedy nearest matching, closest pair first, each item used once.
    pairs: list[tuple[float, int, int]] = []
    for ti, (_, tp) in enumerate(truth_points):
        for ci, (_, cp) in enumerate(candidate_points):
            dx = (tp[0] - cp[0]) * rect.m_per_px_x
            dy = (tp[1] - cp[1]) * rect.m_per_px_y
            pairs.append((math.hypot(dx, dy), ti, ci))
    pairs.sort()

    used_truth: set[int] = set()
    used_candidate: set[int] = set()
    for distance_m, ti, ci in pairs:
        if distance_m > MATCH_TOLERANCE_M or ti in used_truth or ci in used_candidate:
            continue
        used_truth.add(ti)
        used_candidate.add(ci)
        defect, _ = truth_points[ti]
        candidate, _ = candidate_points[ci]
        measurement = refine(
            bev,
            candidate,
            config.refine,
            rect.m_per_px_x,
            rect.m_per_px_y,
            rect.homography,
        )
        result.matches.append(
            Match(
                truth_area_m2=float(defect["area_m2"]),
                truth_distance_m=float(defect["distance_m"]),
                cheap_area_m2=candidate.area_px * m2_per_px,
                measurement=measurement,
                centroid_error_m=distance_m,
            )
        )

    result.false_positives = len(candidate_points) - len(used_candidate)
    result.false_negatives = len(truth_points) - len(used_truth)
    result.missed_truth = [
        dict(defect, reason="not proposed")
        for index, (defect, _) in enumerate(truth_points)
        if index not in used_truth
    ]
    return result


@dataclass
class Summary:
    n_truth: int
    n_matched: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    cheap_bias_pct: float
    cheap_mae_pct: float
    refined_bias_pct: float
    refined_mae_pct: float
    refined_max_abs_pct: float
    edge_offset_px: float
    corrected_bias_pct: float
    corrected_mae_pct: float

    def report(self) -> str:
        return "\n".join(
            [
                f"defects: {self.n_truth} truth, {self.n_matched} matched, "
                f"{self.false_negatives} missed, {self.false_positives} false positive",
                f"precision: {self.precision:.3f}   recall: {self.recall:.3f}",
                f"cheap pass    bias {self.cheap_bias_pct:+6.1f}%   MAE {self.cheap_mae_pct:5.1f}%",
                f"refined       bias {self.refined_bias_pct:+6.1f}%   MAE {self.refined_mae_pct:5.1f}%"
                f"   max |err| {self.refined_max_abs_pct:5.1f}%",
                *(
                    [
                        f"edge-correction diagnostic (REJECTED, not applied): "
                        f"offset {self.edge_offset_px:.3f} px would give bias "
                        f"{self.corrected_bias_pct:+.1f}% MAE {self.corrected_mae_pct:.1f}%"
                    ]
                    if self.edge_offset_px
                    else []
                ),
            ]
        )


def fit_edge_offset(matches: list[Match], m_per_px_x: float, m_per_px_y: float) -> float:
    """Estimate the boundary-localisation offset, in bird's-eye pixels.

    The refined area is biased consistently low, which is the signature of a
    boundary placed just inside the true edge rather than of random noise. For
    an offset of d pixels, the shortfall is approximately d times the perimeter,
    so one least-squares fit over (shortfall, perimeter) recovers d.

    Retained as a diagnostic only. The hypothesis was tested on a train/holdout
    split and REJECTED: the fitted offset made the holdout worse (bias -5.2% to
    -8.5%, MAE 6.5% to 9.1%), so the bias is not perimeter-proportional. See
    docs/negative-results.md. Callers pass `edge_offset_px=0.0` to summarise().
    """
    mean_m_per_px = (m_per_px_x + m_per_px_y) / 2.0
    shortfalls, perimeters = [], []
    for match in matches:
        if not match.measurement.measurable or match.measurement.perimeter_m <= 0:
            continue
        shortfall_m2 = match.truth_area_m2 - match.measurement.area_m2
        shortfalls.append(shortfall_m2 / (m_per_px_x * m_per_px_y))  # to px^2
        perimeters.append(match.measurement.perimeter_m / mean_m_per_px)  # to px
    if not perimeters:
        return 0.0
    perimeter_array = np.array(perimeters)
    shortfall_array = np.array(shortfalls)
    # Least squares through the origin: shortfall_px2 = d * perimeter_px.
    denominator = float(perimeter_array @ perimeter_array)
    if denominator <= 0:
        return 0.0
    return float((perimeter_array @ shortfall_array) / denominator)


def summarise(
    results: list[SceneResult],
    m_per_px_x: float,
    m_per_px_y: float,
    edge_offset_px: float | None = None,
) -> Summary:
    matches = [m for r in results for m in r.matches]
    n_truth = sum(len(r.matches) + r.false_negatives for r in results)
    false_positives = sum(r.false_positives for r in results)
    false_negatives = sum(r.false_negatives for r in results)
    matched = len(matches)

    precision = matched / (matched + false_positives) if matched + false_positives else 0.0
    recall = matched / n_truth if n_truth else 0.0

    cheap = np.array([m.cheap_error_pct for m in matches]) if matches else np.zeros(1)
    refined = np.array([m.refined_error_pct for m in matches]) if matches else np.zeros(1)

    offset = (
        edge_offset_px
        if edge_offset_px is not None
        else fit_edge_offset(matches, m_per_px_x, m_per_px_y)
    )
    mean_m_per_px = (m_per_px_x + m_per_px_y) / 2.0
    corrected = []
    for match in matches:
        perimeter_px = match.measurement.perimeter_m / mean_m_per_px
        corrected_px2 = (
            match.measurement.area_m2 / (m_per_px_x * m_per_px_y) + offset * perimeter_px
        )
        corrected_m2 = corrected_px2 * m_per_px_x * m_per_px_y
        corrected.append(100.0 * (corrected_m2 - match.truth_area_m2) / match.truth_area_m2)
    corrected_array = np.array(corrected) if corrected else np.zeros(1)

    return Summary(
        n_truth=n_truth,
        n_matched=matched,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=round(precision, 4),
        recall=round(recall, 4),
        cheap_bias_pct=round(float(cheap.mean()), 2),
        cheap_mae_pct=round(float(np.abs(cheap).mean()), 2),
        refined_bias_pct=round(float(refined.mean()), 2),
        refined_mae_pct=round(float(np.abs(refined).mean()), 2),
        refined_max_abs_pct=round(float(np.abs(refined).max()), 2),
        edge_offset_px=round(offset, 4),
        corrected_bias_pct=round(float(corrected_array.mean()), 2),
        corrected_mae_pct=round(float(np.abs(corrected_array).mean()), 2),
    )
