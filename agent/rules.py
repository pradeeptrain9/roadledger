"""The decision policy: measurements in, next action out.

Every rule is a pure function of observed values and declared thresholds, which
makes the whole policy testable without AWS, without a model, and without
re-running the vision pipeline. It also makes the trace meaningful: each
decision records the rule that fired and the measured values that fired it.

This follows the deterministic-engine pattern used elsewhere in this codebase.
The policy decides *which measurement to take next*; the pipeline computes every
number; a language model, where one is used, selects among these same actions
and never invents a value. The engine fails closed - where no rule applies and
no measurement is trustworthy, the outcome is escalation, not a quiet default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from pipeline.config import AgentThresholds, ProposeParams, RectifyParams, RefineParams


@dataclass
class Action:
    """What to do next, and why."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    rule: str = ""
    because: dict[str, Any] = field(default_factory=dict)
    decision: str = ""


# -- perception quality ------------------------------------------------------


def rectification_unusable(
    quality: float, thresholds: AgentThresholds
) -> tuple[bool, dict[str, Any]]:
    return quality < thresholds.homography_min_quality, {
        "rectification_quality": round(quality, 4),
        "minimum": thresholds.homography_min_quality,
    }


def widen_road_band(params: RectifyParams) -> RectifyParams:
    """Retry rectification over more of the frame, with looser edge detection.

    A frame whose lane markings are faint, worn or partly occluded yields no
    vanishing point from the default near-field band. Looking further up the
    frame and lowering the Canny thresholds recovers some of those, at the cost
    of a flat-road assumption over a longer stretch.
    """
    return replace(
        params,
        road_band_frac=min(0.75, params.road_band_frac * 1.5),
        canny_low=max(20, int(params.canny_low * 0.6)),
        canny_high=max(60, int(params.canny_high * 0.6)),
        hough_threshold=max(25, int(params.hough_threshold * 0.7)),
    )


# -- proposal density --------------------------------------------------------


def too_many_candidates(
    n_candidates: int, thresholds: AgentThresholds
) -> tuple[bool, dict[str, Any]]:
    """An implausible candidate count means the threshold is too loose.

    Refining fifty candidates on one frame of road is not a measurement
    problem, it is a segmentation problem, and the right response is to tighten
    the proposal rather than spend the expensive pass on noise.
    """
    return n_candidates > thresholds.max_candidates, {
        "n_candidates": n_candidates,
        "maximum": thresholds.max_candidates,
    }


def no_candidates_on_textured_frame(
    n_candidates: int, thresholds: AgentThresholds
) -> tuple[bool, dict[str, Any]]:
    return n_candidates < thresholds.min_candidates_for_textured_frame, {
        "n_candidates": n_candidates,
        "minimum": thresholds.min_candidates_for_textured_frame,
    }


def tighten_proposal(params: ProposeParams) -> ProposeParams:
    """Raise the adaptive window and offset so fewer, stronger blobs survive."""
    return replace(
        params,
        block_size=int(params.block_size * 1.4) | 1,
        c_constant=params.c_constant * 1.5,
    )


def loosen_proposal(params: ProposeParams) -> ProposeParams:
    return replace(
        params,
        block_size=max(11, int(params.block_size * 0.7) | 1),
        c_constant=max(2.0, params.c_constant * 0.7),
    )


# -- classification confidence ----------------------------------------------


def class_ambiguous(
    confidence: float, label: str, thresholds: AgentThresholds
) -> tuple[bool, dict[str, Any]]:
    return (
        confidence < thresholds.min_class_confidence or label == "unknown",
        {
            "label": label,
            "class_confidence": round(confidence, 4),
            "minimum": thresholds.min_class_confidence,
        },
    )


def sharpen_refinement(params: RefineParams) -> RefineParams:
    """Re-examine an ambiguous region with more local contrast and more context.

    More CLAHE separates a shallow defect from its surround; more ROI padding
    gives the threshold a larger surround to be bimodal against. Both change
    the cues the classifier sees, which is the point - this is a second look,
    not a retry of the same look.
    """
    return replace(
        params,
        clahe_clip_limit=min(6.0, params.clahe_clip_limit * 1.8),
        roi_pad_frac=min(0.8, params.roi_pad_frac * 1.8),
    )


# -- measurement agreement ---------------------------------------------------


def spread_too_wide(
    iqr_ratio: float, thresholds: AgentThresholds
) -> tuple[bool, dict[str, Any]]:
    """The central rule. Disagreement between views is a reason to gather more.

    This is what makes the loop agentic rather than a fixed pipeline: the
    quantity being thresholded is produced by the vision stages themselves, and
    exceeding it causes a different, more expensive tool call.
    """
    exceeded = not math.isfinite(iqr_ratio) or iqr_ratio > thresholds.max_area_iqr_ratio
    return exceeded, {
        "iqr_ratio": "inf" if not math.isfinite(iqr_ratio) else round(iqr_ratio, 4),
        "threshold": thresholds.max_area_iqr_ratio,
    }


def spread_needs_human(
    iqr_ratio: float, n_measurements: int, thresholds: AgentThresholds
) -> tuple[bool, dict[str, Any]]:
    """After escalated tracking, persistent disagreement goes to a person."""
    unresolved = (
        not math.isfinite(iqr_ratio)
        or iqr_ratio > thresholds.escalate_to_human_iqr_ratio
        or n_measurements < 3
    )
    return unresolved, {
        "iqr_ratio": "inf" if not math.isfinite(iqr_ratio) else round(iqr_ratio, 4),
        "threshold": thresholds.escalate_to_human_iqr_ratio,
        "n_measurements": n_measurements,
    }


def unmeasurable(
    measurable: bool, footprint_px: float
) -> tuple[bool, dict[str, Any]]:
    return not measurable, {
        "source_footprint_px": footprint_px,
        "measurable": measurable,
    }


# -- action -------------------------------------------------------------------


def requires_approval(
    priority: int, thresholds: AgentThresholds
) -> tuple[bool, dict[str, Any]]:
    return priority <= thresholds.always_approve_priority, {
        "priority": priority,
        "always_approve_at_or_below": thresholds.always_approve_priority,
    }


def budget_exhausted(
    tool_calls: int, thresholds: AgentThresholds
) -> tuple[bool, dict[str, Any]]:
    """A confused loop must not be able to spend the compute budget."""
    return tool_calls >= thresholds.max_tool_calls, {
        "tool_calls": tool_calls,
        "maximum": thresholds.max_tool_calls,
    }
