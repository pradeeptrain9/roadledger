"""Streaming capture: measure the road while the car is still driving on it.

The batch session in `runner.py` is *pull*-based. Tracking a defect means
reaching forward into frames that have already been decoded, which is exactly
what an agent re-examining a recording wants and exactly what a live stream
cannot offer: the frames that would confirm a measurement have not been
recorded yet.

So this module inverts it. State is carried forward instead of fetched
backward. Each arriving frame is rectified and proposed once, open tracks are
predicted into it using the vehicle's own motion, candidates are matched to
those predictions, and each match appends one more independent measurement to
a track that is already open. Nothing is ever re-decoded, and nothing waits
for the end of the video, because there is no end.

Three properties this buys, which the batch path cannot have:

* **A number that visibly firms up.** A defect is reported the moment it is
  seen - as provisional, with one view behind it - and re-reported with a
  tighter spread on every subsequent view. The driver watches the confidence
  arrive. In the batch path that same process is invisible because it is over
  before anything is displayed.
* **Position for free, and honestly bounded.** The phone's GNSS fix is attached
  to the frame at capture time, so a defect gets a dispatchable location. Its
  accuracy is GPS accuracy, not measurement accuracy, and the two are kept
  apart in `geo.py`.
* **Ego-motion from the speedometer.** Frame-to-frame forward distance comes
  from the fix's own speed, which is Doppler-derived and far more robust than
  recovering it from asphalt texture. The visual estimate is retained as a
  cross-check, which is the same two-independent-routes discipline the metric
  scale uses.

The hard constraint is that this must keep up. A frame that takes longer to
measure than the interval between frames does not produce a late answer, it
produces a growing queue and then a useless one. So the per-frame work is
bounded: at most `MAX_REFINES_PER_FRAME` candidates are measured, nearest
first, and the count of what was skipped is reported rather than hidden.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .classify import classify as classify_candidate
from .config import PipelineConfig
from .geo import GeoFix, GeoLedger, GeoPosition, Sighting, locate, track_heading
from .instrument import Timings
from .privacy import scrub
from .propose import propose
from .rectify import rectify
from .refine import refine
from .score import score_defect
from .track import aggregate, bev_shift_lk, centroid
from .types import Candidate, Frame, Measurement

# A defect is matched to an open track if it lands within this distance of
# where that track was predicted to be. Set from the geometry rather than
# guessed: at 2 fps and 25 km/h the vehicle covers 3.5 m between frames, and
# the forward prediction carries roughly 10% error, so the residual is tens of
# centimetres. Wider than this and two adjacent potholes become one track.
MATCH_RADIUS_M = 0.45

# Measuring every candidate in a noisy frame is how a live pipeline falls
# behind and stops being live. Nearest candidates are measured first because
# they have the largest footprint in the source image, so they carry the most
# evidence - and because they are the ones the driver is about to hit.
MAX_REFINES_PER_FRAME = 6

# Consecutive frames a track may go unmatched before it is closed. One missed
# frame is an occlusion or a threshold flicker; three means it is gone.
MAX_TRACK_MISSES = 3

# The forward step from GPS speed and the one from bird's-eye image shift
# should agree. Disagreement beyond this is reported on the frame: it usually
# means the mount moved, which invalidates the calibration and therefore every
# area measured after it.
MAX_ODOMETRY_DISAGREEMENT = 0.35

# Physical bounds on the forward step between two sampled frames, in metres.
# Outside these the step is not believed from either source.
MIN_STEP_M = 0.05
MAX_STEP_M = 12.0

# How often the expensive image-shift cross-check runs when GPS speed is
# already supplying the step. 1 means every frame.
CROSS_CHECK_EVERY = 5

# Largest course change between two sampled frames that is believed. A phone's
# course-over-ground occasionally jumps by a large angle when the fix is poor,
# and a bogus yaw would throw every open track's prediction sideways and break
# the matching that the whole multi-view argument rests on.
MAX_YAW_PER_FRAME_DEG = 30.0


def _jpeg(image: np.ndarray, quality: int = 72) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return bytes(buffer) if ok else b""


@dataclass
class LiveTrack:
    """One physical defect, accumulating evidence as the vehicle approaches it."""

    track_id: str
    bev_xy: tuple[float, float]
    measurements: list[Measurement] = field(default_factory=list)
    # (weight, probabilities) per view. Weighted by source footprint, because a
    # near view's opinion about whether this is a pothole or a shadow is worth
    # more than a distant one's.
    class_votes: list[tuple[float, dict[str, float]]] = field(default_factory=list)
    fixes: list[GeoFix] = field(default_factory=list)
    best_footprint_px: float = 0.0
    best_fix: GeoFix | None = None
    best_ahead_m: float = 0.0
    best_lateral_m: float = 0.0
    best_heading_deg: float | None = None
    crop_jpeg: bytes = b""
    first_seq: int = 0
    last_seq: int = 0
    misses: int = 0
    defect_id: str = ""
    confirmed: bool = False
    closed: bool = False
    entry_id: str = ""

    @property
    def n_views(self) -> int:
        return sum(1 for m in self.measurements if m.measurable and m.area_m2 > 0)


@dataclass
class LiveFrameResult:
    """What one ingested frame produced. Serialised straight to the phone."""

    seq: int
    accepted: bool
    reason: str = ""
    frame_id: str = ""
    sharpness: float = 0.0
    rectification_quality: float = 0.0
    step_m: float = 0.0
    step_source: str = ""
    step_disagreement: float | None = None
    yaw_deg: float = 0.0
    candidates: int = 0
    refined: int = 0
    candidates_skipped: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    processing_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "frame",
            "seq": self.seq,
            "accepted": self.accepted,
            "reason": self.reason,
            "frame_id": self.frame_id,
            "sharpness": round(self.sharpness, 1),
            "rectification_quality": round(self.rectification_quality, 3),
            "step_m": round(self.step_m, 3),
            "step_source": self.step_source,
            "step_disagreement": (
                None if self.step_disagreement is None else round(self.step_disagreement, 3)
            ),
            "yaw_deg": round(self.yaw_deg, 2),
            "candidates": self.candidates,
            "refined": self.refined,
            "candidates_skipped": self.candidates_skipped,
            "events": self.events,
            "processing_ms": round(self.processing_ms, 1),
            "warnings": self.warnings,
        }


@dataclass
class LiveSession:
    """A capture stream in progress. One per connected phone."""

    config: PipelineConfig = field(default_factory=PipelineConfig)
    run_id: str = ""
    ledger: GeoLedger = field(default_factory=GeoLedger)
    timings: Timings = field(default_factory=Timings)

    tracks: dict[str, LiveTrack] = field(default_factory=dict)
    fixes: list[GeoFix] = field(default_factory=list)
    defects: dict[str, dict[str, Any]] = field(default_factory=dict)
    crops: dict[str, bytes] = field(default_factory=dict)

    # Running totals, all reported to the phone so the operator sees the real
    # throughput rather than the nominal one.
    frames_received: int = 0
    frames_measured: int = 0
    frames_skipped_blurry: int = 0
    frames_skipped_unrectifiable: int = 0
    frames_dropped_behind: int = 0
    proposals_without_measurement: int = 0
    distance_m: float = 0.0
    started_epoch_s: float = field(default_factory=time.time)

    _previous_bev: np.ndarray | None = None
    _previous_timestamp_s: float | None = None
    _previous_heading_deg: float | None = None
    _next_track: int = 0

    # -- helpers ----------------------------------------------------------

    def _track_id(self) -> str:
        self._next_track += 1
        digest = hashlib.sha1(f"{self.run_id}:{self._next_track}".encode()).hexdigest()[:10]
        return f"t_{digest}"

    def _forward_step(
        self, bev: np.ndarray, fix: GeoFix, m_per_px_y: float
    ) -> tuple[float, str, float | None, list[str]]:
        """Forward distance since the previous measured frame, two ways.

        GPS speed is preferred: it is Doppler-derived, it does not care what the
        road surface looks like, and it is the one source that remains correct
        when the asphalt is featureless. The image shift is computed anyway and
        compared, because a disagreement between them is the earliest available
        signal that the mount has moved - and a moved mount invalidates the
        calibration that every area depends on.
        """
        warnings: list[str] = []
        dt = None
        if self._previous_timestamp_s is not None:
            dt = fix.timestamp_s - self._previous_timestamp_s

        gps_step: float | None = None
        if fix.speed_mps is not None and dt is not None and dt > 0:
            candidate = fix.speed_mps * dt
            if MIN_STEP_M <= candidate <= MAX_STEP_M:
                gps_step = candidate

        # The visual estimate is the most expensive thing in the frame - nearly
        # twelve milliseconds against three for rectification - so when GPS
        # speed is available it is run periodically rather than every frame.
        # It is a drift and mount-movement check, and a check that runs every
        # second catches a moved mount just as well as one that runs five times
        # a second, at a fifth of the cost.
        needed = gps_step is None or (self.frames_measured % CROSS_CHECK_EVERY == 0)
        visual_step: float | None = None
        if needed and self._previous_bev is not None and m_per_px_y > 0:
            from .track import MIN_SHIFT_AGREEMENT

            with self.timings.measure("live_odometry"):
                (_, dy), agreement = bev_shift_lk(self._previous_bev, bev, self.config.track)
            candidate = dy * m_per_px_y
            if agreement >= MIN_SHIFT_AGREEMENT and MIN_STEP_M <= candidate <= MAX_STEP_M:
                visual_step = candidate

        disagreement: float | None = None
        if gps_step is not None and visual_step is not None:
            disagreement = abs(gps_step - visual_step) / max(gps_step, 1e-6)
            if disagreement > MAX_ODOMETRY_DISAGREEMENT:
                warnings.append(
                    f"speed-derived step {gps_step:.2f} m and image-derived step "
                    f"{visual_step:.2f} m disagree by {disagreement:.0%}; if this persists the "
                    "mount has moved and the calibration - and every area from it - is stale"
                )

        if gps_step is not None:
            return gps_step, "gps_speed", disagreement, warnings
        if visual_step is not None:
            return visual_step, "image_shift", disagreement, warnings
        return 0.0, "unknown", disagreement, warnings

    def _yaw_since_last_frame(self, fix: GeoFix) -> tuple[float, float | None]:
        """Course change since the previous measured frame, in radians.

        A defect ahead of a turning vehicle sweeps sideways across the
        bird's-eye view, by its forward distance times the course change. On a
        straight road this is zero and costs nothing; on a bend it is the
        difference between following a defect across four frames and losing it
        after one, because the match radius is only 45 cm and a 10 degree turn
        moves a defect 8 m ahead by 1.4 m.
        """
        heading = fix.heading_deg if fix.has_usable_heading else track_heading(self.fixes)
        if heading is None or not math.isfinite(heading):
            return 0.0, self._previous_heading_deg
        previous = self._previous_heading_deg
        if previous is None:
            return 0.0, heading
        delta = (heading - previous + 180.0) % 360.0 - 180.0
        if abs(delta) > MAX_YAW_PER_FRAME_DEG:
            # Not believed: a fix this erratic would wreck the predictions.
            return 0.0, heading
        return math.radians(delta), heading

    def _ahead_and_lateral(self, xy: tuple[float, float], rectification) -> tuple[float, float]:
        """Bird's-eye position to metres ahead of, and right of, the camera."""
        x, y = xy
        lateral = (x - self.config.rectify.bev_width_px / 2.0) * rectification.m_per_px_x
        ahead = (
            rectification.near_distance_m
            + (self.config.rectify.bev_height_px - y) * rectification.m_per_px_y
        )
        return ahead, lateral

    # -- the one method that matters --------------------------------------

    def ingest(
        self, image: np.ndarray, fix: GeoFix, seq: int, scrub_frame: bool = True
    ) -> LiveFrameResult:
        """Measure one captured frame and advance every open track.

        Returns immediately with a refusal rather than a bad measurement when
        the frame cannot carry one: too blurry, or unrectifiable. A refused
        frame still advances nothing, so the next usable frame's step is
        measured against the last one that was actually used.
        """
        started = time.perf_counter()
        self.frames_received += 1
        self.fixes.append(fix)
        frame_id = f"f_{hashlib.sha1(f'{self.run_id}:{seq}'.encode()).hexdigest()[:10]}"
        result = LiveFrameResult(seq=seq, accepted=False, frame_id=frame_id)

        from .frames import sharpness as sharpness_of

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result.sharpness = sharpness_of(gray)
        if result.sharpness < self.config.frames.min_sharpness:
            self.frames_skipped_blurry += 1
            result.reason = (
                f"too blurry to measure from (sharpness {result.sharpness:.0f} below "
                f"{self.config.frames.min_sharpness:.0f})"
            )
            result.processing_ms = (time.perf_counter() - started) * 1000
            return result

        if scrub_frame and self.config.privacy.enabled:
            with self.timings.measure("privacy_scrub"):
                image, _ = scrub(image, self.config.privacy)

        frame = Frame(
            frame_id=frame_id,
            index=seq,
            timestamp_s=fix.timestamp_s,
            sharpness=result.sharpness,
            image=image,
            source=f"live:{self.run_id}",
        )

        with self.timings.measure("rectify"):
            rectification, bev = rectify(frame, self.config.rectify)
        result.rectification_quality = rectification.quality
        if bev is None or rectification.quality < self.config.agent.homography_min_quality:
            self.frames_skipped_unrectifiable += 1
            result.reason = (
                f"road plane not recoverable (quality {rectification.quality:.2f}): "
                f"{rectification.notes}"
            )
            result.processing_ms = (time.perf_counter() - started) * 1000
            return result

        step_m, step_source, disagreement, warnings = self._forward_step(
            bev, fix, rectification.m_per_px_y
        )
        result.step_m = step_m
        result.step_source = step_source
        result.step_disagreement = disagreement
        result.warnings.extend(warnings)
        self.distance_m += step_m

        with self.timings.measure("propose"):
            found, _ = propose(bev, frame_id, self.config.propose, self.timings)
        result.candidates = len(found)

        if len(found) > self.config.agent.max_candidates:
            # The same decision the batch agent makes, made inline because
            # there is no time to re-propose: trust the frame less rather than
            # measure noise.
            result.warnings.append(
                f"{len(found)} candidates in one frame, above the "
                f"{self.config.agent.max_candidates} ceiling; the threshold is too loose for "
                "this surface and the extra candidates were not measured"
            )

        yaw_rad, heading_now = self._yaw_since_last_frame(fix)
        matched = self._advance_tracks(
            bev, found, step_m, yaw_rad, rectification, fix, seq, result
        )
        result.refined = matched
        result.yaw_deg = math.degrees(yaw_rad)
        self.frames_measured += 1
        self._previous_bev = bev
        self._previous_timestamp_s = fix.timestamp_s
        self._previous_heading_deg = heading_now
        result.accepted = True
        result.processing_ms = (time.perf_counter() - started) * 1000
        return result

    # -- tracking ---------------------------------------------------------

    def _advance_tracks(
        self,
        bev: np.ndarray,
        found: list[Candidate],
        step_m: float,
        yaw_rad: float,
        rectification,
        fix: GeoFix,
        seq: int,
        result: LiveFrameResult,
    ) -> int:
        """Predict open tracks into this frame, match, measure, report, close."""
        m_per_px_x = rectification.m_per_px_x or 1e-6
        m_per_px_y = rectification.m_per_px_y or 1e-6
        step_px = step_m / m_per_px_y

        open_tracks = [t for t in self.tracks.values() if not t.closed]
        # Bird's-eye y grows toward the camera, so driving forward moves a
        # defect *down* the view until it leaves the bottom edge and passes
        # under the vehicle. A course change on top of that sweeps it sideways
        # by its forward distance times the turn, away from the direction of
        # the turn.
        predicted: dict[str, tuple[float, float]] = {}
        for track in open_tracks:
            x, y = track.bev_xy
            ahead, _ = self._ahead_and_lateral((x, y), rectification)
            dx = -(ahead * math.sin(yaw_rad)) / m_per_px_x
            predicted[track.track_id] = (x + dx, y + step_px)

        centroids = {c.candidate_id: centroid(c.contour) for c in found if c.contour is not None}

        pairs: list[tuple[float, str, str]] = []
        for track in open_tracks:
            px, py = predicted[track.track_id]
            for candidate_id, (cx, cy) in centroids.items():
                distance = math.hypot((cx - px) * m_per_px_x, (cy - py) * m_per_px_y)
                if distance <= MATCH_RADIUS_M:
                    pairs.append((distance, track.track_id, candidate_id))
        pairs.sort()

        taken_tracks: set[str] = set()
        taken_candidates: set[str] = set()
        assignment: list[tuple[str, Candidate]] = []
        by_id = {c.candidate_id: c for c in found}
        for _distance, track_id, candidate_id in pairs:
            if track_id in taken_tracks or candidate_id in taken_candidates:
                continue
            taken_tracks.add(track_id)
            taken_candidates.add(candidate_id)
            assignment.append((track_id, by_id[candidate_id]))

        # Continuing an open track is worth more than opening a new one: three
        # views of one defect produce a trustworthy number, one view each of
        # three defects produces none. So the frame's measurement budget is
        # spent on matches first, then on the nearest new candidates.
        budget = MAX_REFINES_PER_FRAME
        refined = 0

        for track_id, candidate in assignment:
            if budget <= 0:
                break
            self._measure_into(
                self.tracks[track_id], bev, candidate, rectification, fix, seq, result
            )
            budget -= 1
            refined += 1

        fresh = [c for c in found if c.candidate_id not in taken_candidates and c.contour is not None]
        # Nearest first: largest source footprint, most evidence, most urgent.
        fresh.sort(key=lambda c: c.bbox[1] + c.bbox[3], reverse=True)
        for candidate in fresh:
            if budget <= 0:
                result.candidates_skipped += 1
                continue
            track = LiveTrack(
                track_id=self._track_id(),
                bev_xy=centroid(candidate.contour),
                first_seq=seq,
            )
            self.tracks[track.track_id] = track
            self._measure_into(track, bev, candidate, rectification, fix, seq, result)
            budget -= 1
            refined += 1

        # Anything not matched this frame either drifted out of view or was
        # never real.
        for track in open_tracks:
            if track.track_id in taken_tracks:
                continue
            track.misses += 1
            px, py = predicted[track.track_id]
            track.bev_xy = (px, py)
            if py >= self.config.rectify.bev_height_px or track.misses >= MAX_TRACK_MISSES:
                self._close_track(track, result)

        return refined

    def _measure_into(
        self,
        track: LiveTrack,
        bev: np.ndarray,
        candidate: Candidate,
        rectification,
        fix: GeoFix,
        seq: int,
        result: LiveFrameResult,
    ) -> None:
        """One more independent measurement of one track, then re-report it."""
        with self.timings.measure("refine"):
            measurement = refine(
                bev,
                candidate,
                self.config.refine,
                rectification.m_per_px_x,
                rectification.m_per_px_y,
                rectification.homography,
            )

        # Every measurement is attributed to the track, not to the per-frame
        # candidate, because they are all measurements of one physical defect.
        track.measurements.append(
            Measurement(
                candidate_id=track.track_id,
                frame_id=result.frame_id,
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
        track.bev_xy = centroid(candidate.contour) if candidate.contour is not None else track.bev_xy
        track.last_seq = seq
        track.misses = 0
        track.fixes.append(fix)

        with self.timings.measure("classify"):
            classification = classify_candidate(bev, candidate)
        if classification.probabilities:
            track.class_votes.append(
                (max(measurement.source_footprint_px, 1.0), dict(classification.probabilities))
            )

        # The closest approach is the authoritative vantage: its position is
        # the one with the shortest lever arm from the camera, and its crop is
        # the clearest evidence.
        if measurement.source_footprint_px >= track.best_footprint_px:
            ahead, lateral = self._ahead_and_lateral(track.bev_xy, rectification)
            track.best_footprint_px = measurement.source_footprint_px
            track.best_fix = fix
            track.best_ahead_m = ahead
            track.best_lateral_m = lateral
            track.best_heading_deg = track_heading(self.fixes)
            track.crop_jpeg = self._crop(bev, candidate)

        self._report(track, result)

    def _crop(self, bev: np.ndarray, candidate: Candidate) -> bytes:
        x, y, w, h = candidate.bbox
        pad = max(16, int(0.3 * max(w, h)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(bev.shape[1], x + w + pad), min(bev.shape[0], y + h + pad)
        patch = bev[y0:y1, x0:x1]
        if patch.size == 0:
            return b""
        drawn = patch.copy()
        if candidate.contour is not None:
            shifted = candidate.contour.copy()
            shifted[:, :, 0] -= x0
            shifted[:, :, 1] -= y0
            cv2.drawContours(drawn, [shifted], -1, (40, 220, 255), 2)
        return _jpeg(drawn)

    # -- reporting --------------------------------------------------------

    def _merged_class(self, track: LiveTrack):
        """One class for the defect, from every view, weighted by evidence.

        A single distant view calling a pothole a shadow is the dominant real
        failure mode of the classifier - on RDD2022 it called 35 of 45 real
        potholes shadows. Pooling views weighted by source footprint lets the
        near view, which has the texture and the edge profile to tell them
        apart, outvote the distant ones.
        """
        from .types import Classification

        if not track.class_votes:
            return Classification(
                candidate_id=track.track_id, label="unknown", confidence=0.0
            )
        totals: dict[str, float] = {}
        weight_sum = 0.0
        for weight, probabilities in track.class_votes:
            weight_sum += weight
            for label, probability in probabilities.items():
                totals[label] = totals.get(label, 0.0) + weight * probability
        if weight_sum <= 0:
            return Classification(
                candidate_id=track.track_id, label="unknown", confidence=0.0
            )
        merged = {label: value / weight_sum for label, value in totals.items()}
        label = max(merged, key=merged.get)
        return Classification(
            candidate_id=track.track_id,
            label=label,
            confidence=round(merged[label], 4),
            probabilities={k: round(v, 4) for k, v in merged.items()},
            cues={"views_pooled": float(len(track.class_votes))},
        )

    def _report(self, track: LiveTrack, result: LiveFrameResult) -> None:
        """Emit this track's current state: provisional, then confirmed, then updated.

        The event type is the honest description of how much evidence stands
        behind the number, and it is what the phone colours the overlay by. A
        provisional reading is shown because a driver wants to know *now* that
        something is there; it is labelled provisional because one view cannot
        establish a spread.
        """
        if track.n_views == 0:
            # Proposed, looked at, and not measurable from any view yet: too
            # distant, or clipped by the edge of the rectified band. Reporting
            # it as a 0.000 m2 defect would put a row on the operator's screen
            # that says nothing and scrolls the real findings away.
            return

        aggregated = aggregate(track.measurements, self.config.track)
        classification = self._merged_class(track)
        defect = score_defect(
            aggregated, classification, self.config.score, self.config.agent
        )
        # The id must survive a class changing between views, so it is derived
        # from the track rather than from the label.
        if not track.defect_id:
            track.defect_id = f"d_{track.track_id[2:]}"
        defect.defect_id = track.defect_id

        # Confirmation tracks the *current* aggregate and is allowed to go
        # backwards. It was sticky at first, and that produced a defect
        # labelled confirmed whose own reasons said its spread could not be
        # established: a later, closer view had been discarded by the vantage
        # filter, dropping the comparable count from three to two, and the
        # earlier confirmation stayed on the record contradicting it. Evidence
        # that weakens should be displayed as weaker.
        established = (
            aggregated.n_measurements >= self.config.track.min_measurements
            and math.isfinite(aggregated.iqr_ratio)
        )
        was_confirmed = track.confirmed
        track.confirmed = established

        if established:
            kind = "defect_updated" if was_confirmed else "defect_confirmed"
        elif was_confirmed:
            kind = "defect_weakened"
        else:
            kind = "defect_provisional"

        record = self._record(track, defect, aggregated, classification, kind)
        self.defects[track.defect_id] = record
        result.events.append(record)

    def _record(
        self, track: LiveTrack, defect, aggregated, classification, kind: str
    ) -> dict[str, Any]:
        position: GeoPosition | None = None
        if track.best_fix is not None:
            position = locate(
                track.best_fix,
                track.best_ahead_m,
                track.best_lateral_m,
                track.best_heading_deg,
            )

        if track.crop_jpeg:
            self.crops[track.defect_id] = track.crop_jpeg

        record: dict[str, Any] = {
            "type": kind,
            "defect_id": track.defect_id,
            "track_id": track.track_id,
            "label": defect.label,
            "class_confidence": defect.class_confidence,
            "area_m2": defect.area_m2,
            "area_iqr_ratio": defect.area_iqr_ratio,
            "n_views": track.n_views,
            "n_measurements": aggregated.n_measurements,
            "n_discarded_vantage": aggregated.n_discarded_vantage,
            "best_footprint_px": aggregated.best_footprint_px,
            "severity": defect.severity,
            "priority": defect.priority,
            "cost_estimate": defect.cost_estimate,
            "currency": defect.currency,
            "billable": defect.billable,
            "needs_human_approval": defect.needs_human_approval,
            "reasons": defect.reasons,
            "probabilities": classification.probabilities,
            "confirmed": track.confirmed,
            "closed": track.closed,
            "areas_m2": [
                m.area_m2 for m in track.measurements if m.measurable and m.area_m2 > 0
            ],
            "position": position.to_dict() if position else None,
            "has_crop": bool(track.crop_jpeg),
            "approval": None,
            "revisit_candidates": [],
            "entry_id": track.entry_id,
        }
        return record

    def _close_track(self, track: LiveTrack, result: LiveFrameResult) -> None:
        """Finalise a track whose defect has passed under the vehicle.

        This is where the defect enters the persistent geographic ledger, and
        the only place it does, because until the track closes the closest
        approach - and therefore the best measurement and the tightest
        position - has not happened yet.
        """
        track.closed = True
        if not track.measurements:
            return
        if track.n_views == 0:
            # Never measurable from any view. It is counted so the proposal
            # pass's false-positive rate stays visible, but it is not a defect.
            self.proposals_without_measurement += 1
            return

        aggregated = aggregate(track.measurements, self.config.track)
        classification = self._merged_class(track)
        defect = score_defect(
            aggregated, classification, self.config.score, self.config.agent
        )
        if not track.defect_id:
            track.defect_id = f"d_{track.track_id[2:]}"
        defect.defect_id = track.defect_id

        kind = "defect_final" if track.confirmed else "defect_unconfirmed"
        record = self._record(track, defect, aggregated, classification, kind)
        record["closed"] = True

        if not track.confirmed:
            record["needs_human_approval"] = True
            record["reasons"] = list(record["reasons"]) + [
                f"the defect left the field of view after {track.n_views} usable view(s), "
                f"below the {self.config.track.min_measurements} needed to establish a spread; "
                "recorded for review rather than costed"
            ]
            record["billable"] = False
            record["cost_estimate"] = 0.0

        if track.best_fix is not None and record["position"] is not None:
            position = locate(
                track.best_fix, track.best_ahead_m, track.best_lateral_m, track.best_heading_deg
            )
            record["revisit_candidates"] = self.ledger.revisit_candidates(
                position.lat, position.lon, position.accuracy_m, exclude_run_id=self.run_id
            )
            sighting = Sighting(
                run_id=self.run_id,
                defect_id=track.defect_id,
                at_epoch_s=time.time(),
                area_m2=record["area_m2"],
                label=record["label"],
                priority=record["priority"],
                position=position,
            )
            entry, created = self.ledger.merge_or_add(sighting, same_drive=True)
            track.entry_id = entry.entry_id
            record["entry_id"] = entry.entry_id
            record["merged_into_existing"] = not created

        self.defects[track.defect_id] = record
        result.events.append(record)

    # -- lifecycle --------------------------------------------------------

    def finish(self) -> list[dict[str, Any]]:
        """Close every open track. Called when the phone stops streaming.

        A track still open at the end has not reached its closest approach, so
        its measurement is the weakest it will ever be. Closing it here records
        that honestly rather than discarding it.
        """
        closing = LiveFrameResult(seq=-1, accepted=True, reason="stream ended")
        for track in list(self.tracks.values()):
            if not track.closed:
                self._close_track(track, closing)
        return closing.events

    def stats(self) -> dict[str, Any]:
        elapsed = max(time.time() - self.started_epoch_s, 1e-6)
        confirmed = [d for d in self.defects.values() if d.get("confirmed")]
        return {
            "type": "stats",
            "run_id": self.run_id,
            "elapsed_s": round(elapsed, 1),
            "frames_received": self.frames_received,
            "frames_measured": self.frames_measured,
            "frames_skipped_blurry": self.frames_skipped_blurry,
            "frames_skipped_unrectifiable": self.frames_skipped_unrectifiable,
            "frames_dropped_behind": self.frames_dropped_behind,
            # Candidates the cheap pass proposed that no view could measure.
            # Kept visible because it is the live path's false-positive rate.
            "proposals_without_measurement": self.proposals_without_measurement,
            # The rate frames were actually *measured* at, not the rate the
            # phone sent them. If these differ the phone is sending faster than
            # the server can measure, and the operator should see that.
            "measured_fps": round(self.frames_measured / elapsed, 2),
            "received_fps": round(self.frames_received / elapsed, 2),
            "distance_m": round(self.distance_m, 1),
            "open_tracks": sum(1 for t in self.tracks.values() if not t.closed),
            "defects_total": len(self.defects),
            "defects_confirmed": len(confirmed),
            "defects_needing_approval": sum(
                1 for d in self.defects.values() if d.get("needs_human_approval")
            ),
            "total_cost": round(
                sum(d["cost_estimate"] for d in self.defects.values() if d.get("billable")), 2
            ),
            "currency": self.config.score.currency,
            "ledger_entries": len(self.ledger.entries),
            "timings": self.timings.to_dict(),
        }
