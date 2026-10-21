"""The stateful session the agent drives.

Every tool the agent can call maps onto one method here, and each method is
small, side-effect-light and individually replayable. Deliberately so: the
agent's job is to decide *which* measurement to take next, and that is only
auditable if each measurement is a discrete, named, logged operation rather
than a step buried inside a monolithic `process_video()`.

The session holds the expensive intermediate state - decoded frames,
rectifications, bird's-eye views, candidates, measurements - keyed by id, so an
agent can revisit a frame at a different scale or with different parameters
without re-decoding anything.

All numbers come from the pipeline modules. The session computes nothing itself;
it only stores, dispatches and records.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classify import classify as classify_candidate
from .config import PipelineConfig, ProposeParams, RefineParams
from .instrument import Timings
from .privacy import scrub
from .propose import propose
from .rectify import rectify
from .refine import refine
from .score import score_defect
from .track import TrackedFrame, aggregate, associate
from .types import (
    Aggregate,
    Candidate,
    Classification,
    Defect,
    Frame,
    Measurement,
    Rectification,
)


@dataclass
class Session:
    """One analysis run over one input source."""

    config: PipelineConfig = field(default_factory=PipelineConfig)
    timings: Timings = field(default_factory=Timings)
    # How candidates are proposed. The classical pass is the default and is the
    # workload the COOL benchmark measures; a detector backend can be supplied
    # instead without changing anything downstream of this stage.
    proposer: Any = None

    frames: list[Frame] = field(default_factory=list)
    rectifications: dict[str, Rectification] = field(default_factory=dict)
    bevs: dict[str, np.ndarray] = field(default_factory=dict)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    frame_candidates: dict[str, list[str]] = field(default_factory=dict)
    measurements: dict[str, list[Measurement]] = field(default_factory=dict)
    classifications: dict[str, Classification] = field(default_factory=dict)
    aggregates: dict[str, Aggregate] = field(default_factory=dict)
    defects: dict[str, Defect] = field(default_factory=dict)
    # defect_id -> candidate_id, so a scored defect can be traced back to the
    # measurements it came from.
    defect_candidate: dict[str, str] = field(default_factory=dict)
    privacy_counts: dict[str, int] = field(default_factory=dict)
    source: str = ""
    # Cumulative forward distance travelled at each frame, in metres, built by
    # chaining frame-to-frame ego-motion. This is what puts defects in a single
    # shared coordinate frame.
    odometry_m: dict[str, float] = field(default_factory=dict)

    # -- loading ----------------------------------------------------------

    def load(self, source: str | Path) -> dict[str, Any]:
        """Sample frames and scrub them before anything else touches them."""
        from .frames import sample

        self.source = str(source)
        with self.timings.measure("decode_and_sample"):
            sampled = sample(source, self.config.frames)

        faces = plates = 0
        with self.timings.measure("privacy_scrub"):
            for frame in sampled:
                if frame.image is None:
                    continue
                # Scrubbing happens here, on load, so no later stage can ever
                # see or persist an unscrubbed frame.
                frame.image, counts = scrub(frame.image, self.config.privacy)
                faces += counts["faces"]
                plates += counts["plates"]

        self.frames = sampled
        self.privacy_counts = {"faces_blurred": faces, "plates_blurred": plates}
        return {
            "source": self.source,
            "frames_sampled": len(sampled),
            "privacy": self.privacy_counts,
            "frame_ids": [f.frame_id for f in sampled],
            "mean_sharpness": round(
                float(np.mean([f.sharpness for f in sampled])) if sampled else 0.0, 1
            ),
        }

    def load_images(
        self, images: list[np.ndarray], source: str = "memory", fps: float = 2.0
    ) -> dict[str, Any]:
        """Load an in-memory frame sequence, scrubbed and sharpness-gated.

        The same contract as `load`, without touching the filesystem. Used by
        the test rig and by callers that already hold decoded frames, so a long
        evaluation sweep does not have to write hundreds of megabytes of
        intermediate PNGs to disk.
        """
        import hashlib

        from .frames import sharpness

        self.source = source
        frames: list[Frame] = []
        faces = plates = 0
        with self.timings.measure("privacy_scrub"):
            for index, image in enumerate(images):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                sharp = sharpness(gray)
                if sharp < self.config.frames.min_sharpness:
                    continue
                scrubbed, counts = scrub(image, self.config.privacy)
                faces += counts["faces"]
                plates += counts["plates"]
                digest = hashlib.sha1(f"{source}:{index}".encode()).hexdigest()[:10]
                frames.append(
                    Frame(
                        frame_id=f"f_{digest}",
                        index=index,
                        timestamp_s=index / max(fps, 1e-6),
                        sharpness=sharp,
                        image=scrubbed,
                        source=source,
                    )
                )

        self.frames = frames
        self.privacy_counts = {"faces_blurred": faces, "plates_blurred": plates}
        return {
            "source": source,
            "frames_sampled": len(frames),
            "privacy": self.privacy_counts,
            "frame_ids": [f.frame_id for f in frames],
        }

    def _frame(self, frame_id: str) -> Frame:
        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame
        raise KeyError(f"unknown frame: {frame_id}")

    # -- perception -------------------------------------------------------

    def rectify_frame(self, frame_id: str, params=None) -> Rectification:
        """Rectify one frame. Low quality is reported, never hidden."""
        frame = self._frame(frame_id)
        with self.timings.measure("rectify"):
            rectification, bev = rectify(frame, params or self.config.rectify)
        self.rectifications[frame_id] = rectification
        if bev is not None:
            self.bevs[frame_id] = bev
        return rectification

    def propose_candidates(
        self, frame_id: str, params: ProposeParams | None = None
    ) -> list[Candidate]:
        """Run the cheap COOL-accelerated pass on one rectified frame."""
        if frame_id not in self.bevs:
            rectification = self.rectify_frame(frame_id)
            if frame_id not in self.bevs:
                raise ValueError(
                    f"frame {frame_id} could not be rectified "
                    f"(quality {rectification.quality}): {rectification.notes}"
                )
        if self.proposer is None:
            found, _ = propose(
                self.bevs[frame_id], frame_id, params or self.config.propose, self.timings
            )
        else:
            frame = self._frame(frame_id)
            rectification = self.rectifications.get(frame_id)
            found = self.proposer.propose(
                self.bevs[frame_id],
                frame_id,
                params or self.config.propose,
                source_image=frame.image,
                homography=rectification.homography if rectification else None,
                timings=self.timings,
            )
        for candidate in found:
            self.candidates[candidate.candidate_id] = candidate
        self.frame_candidates[frame_id] = [c.candidate_id for c in found]
        return found

    def refine_candidate(
        self, candidate_id: str, params: RefineParams | None = None
    ) -> Measurement:
        """Measure one candidate at full resolution in its own frame."""
        candidate = self.candidates[candidate_id]
        rectification = self.rectifications[candidate.frame_id]
        with self.timings.measure("refine"):
            measurement = refine(
                self.bevs[candidate.frame_id],
                candidate,
                params or self.config.refine,
                rectification.m_per_px_x,
                rectification.m_per_px_y,
                rectification.homography,
            )
        self.measurements.setdefault(candidate_id, []).append(measurement)
        return measurement

    def classify(self, candidate_id: str) -> Classification:
        candidate = self.candidates[candidate_id]
        with self.timings.measure("classify"):
            classification = classify_candidate(self.bevs[candidate.frame_id], candidate)
        self.classifications[candidate_id] = classification
        return classification

    # -- multi-frame evidence --------------------------------------------

    def _tracked_frames(self) -> list[TrackedFrame]:
        tracked = []
        for frame in self.frames:
            rectification = self.rectifications.get(frame.frame_id)
            tracked.append(
                TrackedFrame(
                    frame_id=frame.frame_id,
                    image=frame.image,
                    homography=rectification.homography if rectification else None,
                    bev=self.bevs.get(frame.frame_id),
                    m_per_px_x=rectification.m_per_px_x if rectification else 0.0,
                    m_per_px_y=rectification.m_per_px_y if rectification else 0.0,
                    candidates=[
                        self.candidates[cid]
                        for cid in self.frame_candidates.get(frame.frame_id, [])
                    ],
                )
            )
        return tracked

    def track_across_frames(self, candidate_id: str, k: int | None = None) -> Aggregate:
        """Gather independent measurements of one defect from k further frames.

        This is the tool the agent reaches for when a measurement's spread is
        too wide to trust. It prepares each subsequent frame as needed, follows
        the defect into it, measures it there, and returns the aggregate whose
        `iqr_ratio` drives the next decision.
        """
        candidate = self.candidates[candidate_id]
        anchor_index = next(
            i for i, f in enumerate(self.frames) if f.frame_id == candidate.frame_id
        )
        limit = k if k is not None else self.config.track.k_frames

        # Subsequent frames must be rectified and proposed before the defect
        # can be followed into them.
        for offset in range(1, limit + 1):
            index = anchor_index + offset
            if index >= len(self.frames):
                break
            frame_id = self.frames[index].frame_id
            if frame_id not in self.rectifications:
                self.rectify_frame(frame_id)
            if frame_id in self.bevs and frame_id not in self.frame_candidates:
                self.propose_candidates(frame_id)

        with self.timings.measure("track"):
            matched = associate(
                self._tracked_frames(),
                anchor_index,
                candidate,
                self.config.track,
                max_frames=limit,
            )

        measurements: list[Measurement] = []
        for index, tracked_candidate in matched:
            self.candidates[tracked_candidate.candidate_id] = tracked_candidate
            measurement = self.refine_candidate(tracked_candidate.candidate_id)
            # Every measurement is attributed to the anchor, since they are all
            # measurements of the same physical defect.
            measurements.append(
                Measurement(
                    candidate_id=candidate_id,
                    frame_id=self.frames[index].frame_id,
                    area_m2=measurement.area_m2,
                    perimeter_m=measurement.perimeter_m,
                    convexity=measurement.convexity,
                    max_width_m=measurement.max_width_m,
                    texture_energy=measurement.texture_energy,
                    stage=measurement.stage,
                    source_footprint_px=measurement.source_footprint_px,
                    measurable=measurement.measurable,
                    note=measurement.note,
                )
            )

        self.measurements[candidate_id] = measurements
        result = aggregate(measurements, self.config.track)
        self.aggregates[candidate_id] = result
        return result

    # -- global position --------------------------------------------------

    def build_odometry(self) -> dict[str, float]:
        """Chain ego-motion across frames into a travelled-distance estimate.

        Each bird's-eye view is camera-relative, so the same pothole seen from
        six frames has six different bird's-eye positions and looks like six
        different potholes. Accumulating forward motion gives every frame a
        station along the route, and a defect's station plus its lateral offset
        put it in one shared frame - which is what lets duplicates be merged
        and what the map and the work-order ledger are keyed on.

        Where ego-motion between two frames cannot be trusted, the gap is
        filled with the running median step rather than with zero, so one bad
        pair does not collapse the route.
        """
        from .track import (
            MAX_FORWARD_M_PER_FRAME,
            MIN_FORWARD_M_PER_FRAME,
            MIN_SHIFT_AGREEMENT,
            bev_shift_lk,
        )

        steps: list[float] = []
        odometry: dict[str, float] = {}
        travelled = 0.0
        previous_id: str | None = None

        for frame in self.frames:
            if frame.frame_id not in self.rectifications:
                self.rectify_frame(frame.frame_id)
            if previous_id is not None:
                step = None
                if previous_id in self.bevs and frame.frame_id in self.bevs:
                    with self.timings.measure("odometry"):
                        (_, dy), agreement = bev_shift_lk(
                            self.bevs[previous_id], self.bevs[frame.frame_id], self.config.track
                        )
                    forward = dy * self.rectifications[frame.frame_id].m_per_px_y
                    if (
                        agreement >= MIN_SHIFT_AGREEMENT
                        and MIN_FORWARD_M_PER_FRAME <= forward <= MAX_FORWARD_M_PER_FRAME
                    ):
                        step = forward
                        steps.append(forward)
                if step is None:
                    step = float(np.median(steps)) if steps else 0.0
                travelled += step
            odometry[frame.frame_id] = round(travelled, 4)
            previous_id = frame.frame_id

        self.odometry_m = odometry
        return odometry

    def ground_position(self, candidate_id: str) -> tuple[float, float] | None:
        """A candidate's (lateral_m, station_m) in the route's own frame."""
        candidate = self.candidates.get(candidate_id)
        if candidate is None or candidate.contour is None:
            return None
        rectification = self.rectifications.get(candidate.frame_id)
        if rectification is None:
            return None
        if not self.odometry_m:
            self.build_odometry()

        from .track import centroid

        x, y = centroid(candidate.contour)
        # Bird's-eye x is measured across the lane, y downward from the far
        # edge, so a larger y is nearer the camera.
        lateral_m = (x - self.config.rectify.bev_width_px / 2.0) * rectification.m_per_px_x
        ahead_m = (
            rectification.near_distance_m
            + (self.config.rectify.bev_height_px - y) * rectification.m_per_px_y
        )
        station_m = self.odometry_m.get(candidate.frame_id, 0.0) + ahead_m
        return round(lateral_m, 3), round(station_m, 3)

    # -- decision ---------------------------------------------------------

    def score(self, candidate_id: str) -> Defect:
        """Score a defect from its aggregate and class. Deterministic."""
        aggregated = self.aggregates.get(candidate_id)
        if aggregated is None:
            # Fail closed: scoring without multi-frame evidence would silently
            # price a defect measured once.
            measurements = self.measurements.get(candidate_id, [])
            aggregated = aggregate(measurements, self.config.track)
            self.aggregates[candidate_id] = aggregated

        classification = self.classifications.get(candidate_id)
        if classification is None:
            classification = self.classify(candidate_id)

        defect = score_defect(
            aggregated, classification, self.config.score, self.config.agent
        )
        self.defects[defect.defect_id] = defect
        self.defect_candidate[defect.defect_id] = candidate_id
        return defect

    # -- reporting --------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "frames": len(self.frames),
            "privacy": self.privacy_counts,
            "rectified": len(self.rectifications),
            "rectification_quality": {
                frame_id: rectification.quality
                for frame_id, rectification in self.rectifications.items()
            },
            "candidates": len(self.candidates),
            "defects": [defect.to_dict() for defect in self.defects.values()],
            "needs_approval": [
                defect.defect_id
                for defect in self.defects.values()
                if defect.needs_human_approval
            ],
            "total_cost": round(
                sum(d.cost_estimate for d in self.defects.values() if d.billable), 2
            ),
            "currency": self.config.score.currency,
            "timings": self.timings.to_dict(),
        }
