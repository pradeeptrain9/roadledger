"""The perception-decision-action loop.

One pass of the loop, for one candidate defect:

    perceive  rectify the frame, propose candidates
    decide    is the rectification usable? are there implausibly many
              candidates? is the class ambiguous? do the views agree?
    act       re-rectify with a wider band, re-propose with a tighter
              threshold, re-examine the region with more contrast, gather more
              views, escalate to a human, or emit a work order

What makes this a loop rather than a pipeline is that each decision is taken on
a quantity the vision stages produced, and it changes which tool is called
next. A frame returning 40 candidates gets its threshold raised and is
re-proposed. A defect whose views disagree by more than 20% of the median gets
tracked across more frames; if they still disagree by more than 35%, it goes to
a person. None of those branches are reachable without the measurement that
triggers them.

Two drivers, same tool surface and same policy:

* `policy` - the deterministic rules in agent/rules.py choose the action.
  Reproducible, offline, and what the evaluation runs against.
* `llm` - a Bedrock model chooses among the same actions, with the rules'
  findings supplied as observations. The engine still computes every number;
  the model only sequences tool calls.

The deterministic driver is the reference. A run is only as trustworthy as its
numbers, and those come from the pipeline either way.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent import rules
from agent.trace import Trace
from pipeline.config import PipelineConfig
from pipeline.runner import Session
from pipeline.types import Defect


# Two defects whose ground positions are within this distance are the same
# physical defect seen from two frames, not two defects.
DEDUPE_RADIUS_M = 0.6


@dataclass
class Outcome:
    """What one run produced."""

    run_id: str
    trace: Trace
    defects: list[Defect] = field(default_factory=list)
    escalated: list[dict[str, Any]] = field(default_factory=list)
    frames_skipped: list[str] = field(default_factory=list)
    work_orders: list[Defect] = field(default_factory=list)
    duplicates_merged: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "defects": [d.to_dict() for d in self.defects],
            "work_orders": [d.defect_id for d in self.work_orders],
            "escalated": self.escalated,
            "frames_skipped": self.frames_skipped,
            "trace": self.trace.to_dict(),
        }


class AgentLoop:
    """Drives a Session through the policy, recording every decision."""

    def __init__(
        self,
        session: Session,
        config: PipelineConfig | None = None,
        trace: Trace | None = None,
    ) -> None:
        self.session = session
        self.config = config or session.config
        self.thresholds = self.config.agent
        self.run_id = uuid.uuid4().hex[:12]
        self.trace = trace or Trace(run_id=self.run_id, source=session.source)

    # -- helpers ----------------------------------------------------------

    def _perceive(self, tool: str, arguments: dict, observation: dict, elapsed: float):
        return self.trace.add(
            "perceive", tool, arguments, observation, elapsed_ms=elapsed
        )

    def _decide(self, rule: str, because: dict, decision: str):
        return self.trace.add("decide", "", rule=rule, because=because, decision=decision)

    def _over_budget(self) -> bool:
        exhausted, because = rules.budget_exhausted(self.trace.tool_calls, self.thresholds)
        if exhausted:
            self._decide(
                "budget_exhausted", because, "halt: tool-call budget reached"
            )
            self.trace.halted_reason = "tool-call budget reached"
        return exhausted

    # -- the loop ---------------------------------------------------------

    def run(self, max_frames: int | None = None) -> Outcome:
        outcome = Outcome(run_id=self.run_id, trace=self.trace)
        frames = self.session.frames[: max_frames or len(self.session.frames)]

        seen: set[str] = set()
        # Ground positions of defects already recorded, so one pothole seen
        # from six frames becomes one work order rather than six.
        placed: list[tuple[float, float]] = []

        # Odometry first: it puts every frame on a shared station along the
        # route, which is what makes the dedupe possible at all.
        self.session.build_odometry()

        for frame in frames:
            if self._over_budget():
                break

            bev_ready = self._rectify_with_retry(frame.frame_id, outcome)
            if not bev_ready:
                continue

            candidates = self._propose_with_retry(frame.frame_id)
            if candidates is None:
                continue

            for candidate in candidates:
                if self._over_budget():
                    break
                # One physical defect appears in many frames. Measuring it once
                # per frame would produce duplicate work orders for the same
                # pothole, so a candidate already absorbed into another
                # defect's track is skipped.
                if candidate.candidate_id in seen:
                    continue

                position = self.session.ground_position(candidate.candidate_id)
                if position is not None:
                    duplicate = next(
                        (
                            existing
                            for existing in placed
                            if abs(existing[0] - position[0]) < DEDUPE_RADIUS_M
                            and abs(existing[1] - position[1]) < DEDUPE_RADIUS_M
                        ),
                        None,
                    )
                    if duplicate is not None:
                        self._decide(
                            "duplicate_of_recorded_defect",
                            {
                                "station_m": position[1],
                                "lateral_m": position[0],
                                "matches_station_m": duplicate[1],
                                "radius_m": DEDUPE_RADIUS_M,
                            },
                            "skip: already measured from an earlier frame",
                        )
                        outcome.duplicates_merged += 1
                        continue

                defect = self._measure_candidate(candidate.candidate_id, outcome, seen)
                if defect is not None:
                    if position is not None:
                        placed.append(position)
                        defect.geo = {"lateral_m": position[0], "station_m": position[1]}
                    outcome.defects.append(defect)

        self.trace.finished_at = time.time()
        return outcome

    def _rectify_with_retry(self, frame_id: str, outcome: Outcome) -> bool:
        """Perceive, then decide whether the geometry can be measured from."""
        start = time.perf_counter()
        rectification = self.session.rectify_frame(frame_id)
        self._perceive(
            "rectify_frame",
            {"frame_id": frame_id},
            {
                "quality": rectification.quality,
                "method": rectification.method,
                "notes": rectification.notes,
            },
            (time.perf_counter() - start) * 1000,
        )

        unusable, because = rules.rectification_unusable(
            rectification.quality, self.thresholds
        )
        if not unusable:
            return frame_id in self.session.bevs

        # Vision output changes the next tool call: re-rectify over a wider
        # band with looser edge detection.
        self._decide(
            "rectification_unusable",
            because,
            "re-rectify over a wider road band with looser edge thresholds",
        )
        retry_params = rules.widen_road_band(self.config.rectify)
        start = time.perf_counter()
        retried = self.session.rectify_frame(frame_id, retry_params)
        self._perceive(
            "rectify_frame",
            {"frame_id": frame_id, "road_band_frac": retry_params.road_band_frac},
            {"quality": retried.quality, "method": retried.method},
            (time.perf_counter() - start) * 1000,
        )

        still_unusable, because = rules.rectification_unusable(
            retried.quality, self.thresholds
        )
        if still_unusable or frame_id not in self.session.bevs:
            self._decide(
                "rectification_unusable_after_retry",
                because,
                "skip frame: no trustworthy geometry, so nothing is measured from it",
            )
            outcome.frames_skipped.append(frame_id)
            return False
        return True

    def _propose_with_retry(self, frame_id: str):
        """Perceive candidates, then decide whether the threshold was sane."""
        params = self.config.propose
        start = time.perf_counter()
        candidates = self.session.propose_candidates(frame_id, params)
        self._perceive(
            "propose_candidates",
            {"frame_id": frame_id, "block_size": params.block_size},
            {"n_candidates": len(candidates)},
            (time.perf_counter() - start) * 1000,
        )

        excessive, because = rules.too_many_candidates(len(candidates), self.thresholds)
        if excessive:
            self._decide(
                "too_many_candidates",
                because,
                "tighten the adaptive threshold and re-propose rather than "
                "refining noise",
            )
            tightened = rules.tighten_proposal(params)
            start = time.perf_counter()
            candidates = self.session.propose_candidates(frame_id, tightened)
            self._perceive(
                "propose_candidates",
                {"frame_id": frame_id, "block_size": tightened.block_size},
                {"n_candidates": len(candidates)},
                (time.perf_counter() - start) * 1000,
            )

        empty, because = rules.no_candidates_on_textured_frame(
            len(candidates), self.thresholds
        )
        if empty:
            self._decide(
                "no_candidates", because, "nothing to measure on this frame"
            )
            return None
        return candidates

    def _measure_candidate(
        self, candidate_id: str, outcome: Outcome, seen: set[str]
    ) -> Defect | None:
        """Classify, measure, gather more views if they disagree, then decide."""
        start = time.perf_counter()
        classification = self.session.classify(candidate_id)
        self._perceive(
            "classify",
            {"candidate_id": candidate_id},
            {
                "label": classification.label,
                "confidence": classification.confidence,
                "edge_width_px": classification.cues.get("edge_width_px"),
                "texture_ratio": classification.cues.get("texture_ratio"),
            },
            (time.perf_counter() - start) * 1000,
        )

        ambiguous, because = rules.class_ambiguous(
            classification.confidence, classification.label, self.thresholds
        )
        if ambiguous:
            # A second, different look - not a retry of the same one.
            self._decide(
                "class_ambiguous",
                because,
                "re-examine the region with more local contrast and more surround",
            )
            sharper = rules.sharpen_refinement(self.config.refine)
            start = time.perf_counter()
            measurement = self.session.refine_candidate(candidate_id, sharper)
            classification = self.session.classify(candidate_id)
            self._perceive(
                "refine_and_reclassify",
                {"candidate_id": candidate_id, "clahe_clip_limit": sharper.clahe_clip_limit},
                {"label": classification.label, "confidence": classification.confidence},
                (time.perf_counter() - start) * 1000,
            )
            still_ambiguous, because = rules.class_ambiguous(
                classification.confidence, classification.label, self.thresholds
            )
            if still_ambiguous:
                self._decide(
                    "class_ambiguous_after_second_look",
                    because,
                    "escalate to human review: surface class cannot be established",
                )
                self.trace.add(
                    "escalate",
                    "request_human_review",
                    {"candidate_id": candidate_id},
                    {"reason": "class ambiguous after a second look"},
                )
                outcome.escalated.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "class ambiguous after a second look",
                        "label": classification.label,
                        "confidence": classification.confidence,
                    }
                )
                return None

        # First measurement, single frame.
        start = time.perf_counter()
        measurement = self.session.refine_candidate(candidate_id)
        self._perceive(
            "refine_candidate",
            {"candidate_id": candidate_id},
            {
                "area_m2": measurement.area_m2,
                "source_footprint_px": measurement.source_footprint_px,
                "measurable": measurement.measurable,
            },
            (time.perf_counter() - start) * 1000,
        )

        too_small, because = rules.unmeasurable(
            measurement.measurable, measurement.source_footprint_px
        )
        if too_small:
            self._decide(
                "below_resolution_floor",
                because,
                "defer: too distant to size, so no number is reported for it",
            )
            outcome.escalated.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "below the source-footprint floor; too distant to size",
                    "footprint_px": measurement.source_footprint_px,
                }
            )
            return None

        # Multi-frame evidence.
        start = time.perf_counter()
        aggregate = self.session.track_across_frames(
            candidate_id, k=self.config.track.k_frames
        )
        seen.update(aggregate.frame_ids)
        self._perceive(
            "track_across_frames",
            {"candidate_id": candidate_id, "k": self.config.track.k_frames},
            {
                "n_measurements": aggregate.n_measurements,
                "area_m2_median": aggregate.area_m2_median,
                "iqr_ratio": aggregate.iqr_ratio,
                "discarded_vantage": aggregate.n_discarded_vantage,
            },
            (time.perf_counter() - start) * 1000,
        )

        wide, because = rules.spread_too_wide(aggregate.iqr_ratio, self.thresholds)
        if wide:
            # The headline decision: measured disagreement buys more evidence.
            self._decide(
                "measurement_spread_too_wide",
                because,
                f"gather more views: re-track across {self.config.track.k_frames_escalated} frames",
            )
            start = time.perf_counter()
            aggregate = self.session.track_across_frames(
                candidate_id, k=self.config.track.k_frames_escalated
            )
            seen.update(aggregate.frame_ids)
            self._perceive(
                "track_across_frames",
                {"candidate_id": candidate_id, "k": self.config.track.k_frames_escalated},
                {
                    "n_measurements": aggregate.n_measurements,
                    "area_m2_median": aggregate.area_m2_median,
                    "iqr_ratio": aggregate.iqr_ratio,
                },
                (time.perf_counter() - start) * 1000,
            )

            unresolved, because = rules.spread_needs_human(
                aggregate.iqr_ratio, aggregate.n_measurements, self.thresholds
            )
            if unresolved:
                self._decide(
                    "spread_unresolved_after_more_views",
                    because,
                    "escalate to human review: the measurement does not converge",
                )
                self.trace.add(
                    "escalate",
                    "request_human_review",
                    {"candidate_id": candidate_id},
                    {"reason": "measurements do not converge", **because},
                )

        defect = self.session.score(candidate_id)
        self.trace.add(
            "act",
            "score_defect",
            {"candidate_id": candidate_id},
            {
                "label": defect.label,
                "area_m2": defect.area_m2,
                "priority": defect.priority,
                "cost_estimate": defect.cost_estimate,
            },
        )

        approval, because = rules.requires_approval(defect.priority, self.thresholds)
        if defect.needs_human_approval or approval:
            self._decide(
                "work_order_requires_approval",
                {**because, "reasons": defect.reasons},
                "hold work order for human approval before dispatch",
            )
            self.trace.add(
                "escalate",
                "request_human_approval",
                {"defect_id": defect.defect_id},
                {"priority": defect.priority, "reasons": defect.reasons},
            )
        else:
            self.trace.add(
                "act",
                "emit_work_order",
                {"defect_id": defect.defect_id},
                {
                    "label": defect.label,
                    "area_m2": defect.area_m2,
                    "cost_estimate": defect.cost_estimate,
                    "currency": defect.currency,
                },
            )
            outcome.work_orders.append(defect)
        return defect
