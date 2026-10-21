"""Data carried between pipeline stages.

These are the objects the MCP tool surface serialises, so they stay plain and
JSON-friendly. Every measurement records how it was obtained - which frame,
which parameters, which stage - because the agent reasons over provenance as
much as over values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class Frame:
    """One sampled frame, already privacy-scrubbed."""

    frame_id: str
    index: int
    timestamp_s: float
    sharpness: float
    image: np.ndarray | None = None  # not serialised
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("image", None)
        return d


@dataclass
class Rectification:
    """Bird's-eye homography plus an honest quality score for it."""

    frame_id: str
    homography: np.ndarray | None = None  # not serialised
    quality: float = 0.0
    method: str = "vanishing_point"
    vanishing_point: tuple[float, float] | None = None
    lane_px_width: float | None = None
    m_per_px_x: float = 0.0
    m_per_px_y: float = 0.0
    # Ground distance ahead of the camera at the bird's-eye view's near (bottom)
    # edge. Bird's-eye y is relative to that edge, so this offset is what turns
    # a bird's-eye position into a distance from the camera.
    near_distance_m: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("homography", None)
        return d


@dataclass
class Candidate:
    """A proposal from the cheap pass. Not yet a measurement."""

    candidate_id: str
    frame_id: str
    # Bounding box in bird's-eye pixel coordinates: x, y, w, h.
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    area_px: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 0.0
    mean_intensity: float = 0.0
    contour: np.ndarray | None = None  # not serialised

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("contour", None)
        return d


@dataclass
class Measurement:
    """One independent metric measurement of one candidate in one frame.

    `measurable` is false when the defect's footprint in the *source* image is
    too small for its boundary to mean anything. Such a candidate is reported,
    never silently measured: a confident number derived from four pixels is
    worse than an admission that the defect is too far away to size.
    """

    candidate_id: str
    frame_id: str
    area_m2: float
    perimeter_m: float
    convexity: float
    max_width_m: float
    texture_energy: float
    stage: str = "refine"
    source_footprint_px: float = 0.0
    measurable: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Classification:
    """Surface class with calibrated-ish probabilities and the cues behind them."""

    candidate_id: str
    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    cues: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Aggregate:
    """Several measurements of one defect, reduced to a value plus a spread.

    `iqr_ratio` is the number the agent acts on: high spread means the
    measurement is not yet trustworthy and more evidence is needed.
    """

    candidate_id: str
    n_measurements: int
    area_m2_median: float
    area_m2_iqr: float
    iqr_ratio: float
    area_m2_min: float
    area_m2_max: float
    frame_ids: list[str] = field(default_factory=list)
    n_discarded_vantage: int = 0
    best_footprint_px: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Defect:
    """A scored, ledger-ready defect."""

    defect_id: str
    label: str
    class_confidence: float
    area_m2: float
    area_iqr_ratio: float
    n_measurements: int
    severity: float
    priority: int
    cost_estimate: float
    currency: str
    billable: bool
    needs_human_approval: bool
    reasons: list[str] = field(default_factory=list)
    frame_ids: list[str] = field(default_factory=list)
    geo: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
