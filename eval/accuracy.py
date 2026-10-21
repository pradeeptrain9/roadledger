"""End-to-end measurement accuracy against exact ground truth.

Runs the whole pipeline - rectify, propose, classify, track, aggregate, score -
over simulated drive-pasts where the defect is a true circle of known radius on
the ground plane, so its area is pi*r^2 exactly rather than approximately.

Everything is rendered in memory. An earlier version wrote every frame of every
sweep to disk and filled the volume.

Run:  python -m eval.accuracy
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pipeline.config import (
    CameraGeometry,
    FrameParams,
    PipelineConfig,
    PrivacyParams,
    RectifyParams,
)
from pipeline.runner import Session
from tests.ground import build_ground, render_drive
from tests.synthetic import SceneCamera


@dataclass
class Row:
    truth_r_m: float
    truth_area_m2: float
    detected: bool
    label: str
    views: int
    discarded: int
    best_footprint_px: float
    measured_area_m2: float
    iqr_ratio: float
    priority: int
    needs_approval: bool

    @property
    def error_pct(self) -> float:
        if not self.views:
            return math.nan
        return 100.0 * (self.measured_area_m2 - self.truth_area_m2) / self.truth_area_m2


def default_config(camera: SceneCamera) -> PipelineConfig:
    return PipelineConfig(
        rectify=RectifyParams(camera=CameraGeometry(camera.height_m, camera.vfov_deg)),
        frames=FrameParams(sample_fps=2.0, min_sharpness=10.0, max_frames=40),
        # Scrubbing is exercised in its own test; disabling it here keeps the
        # sweep fast and the measurement the only thing under test.
        privacy=PrivacyParams(enabled=False),
    )


def measure_one(
    truth_r_m: float,
    start_distance_m: float = 11.0,
    n_frames: int = 14,
    step_m: float = 0.6,
    lateral_m: float = -0.4,
    seed: int = 5,
    confusers: list[tuple[float, float, float, str]] | None = None,
) -> Row:
    """One drive-past, one defect, measured end to end."""
    camera = SceneCamera()
    config = default_config(camera)
    ground = build_ground(
        defects=[(lateral_m, start_distance_m, truth_r_m)],
        confusers=confusers,
        seed=seed,
    )
    images, _ = render_drive(ground, camera, n_frames=n_frames, step_m=step_m)

    session = Session(config=config)
    session.load_images(images)

    truth_area = math.pi * truth_r_m**2

    # Anchor on the *earliest* frame where the defect is seen, then track
    # forward. Anchoring on the closest approach instead leaves no frames ahead
    # to track into, which silently reduces the evidence to one view.
    anchor = None
    for frame in session.frames:
        session.rectify_frame(frame.frame_id)
        if frame.frame_id not in session.bevs:
            continue
        for candidate in session.propose_candidates(frame.frame_id):
            if session.classify(candidate.candidate_id).label == "pothole":
                anchor = candidate
                break
        if anchor is not None:
            break

    if anchor is None:
        return Row(
            truth_r_m, truth_area, False, "none", 0, 0, 0.0, 0.0, math.inf, 3, True
        )

    aggregate = session.track_across_frames(anchor.candidate_id, k=n_frames - 1)
    defect = session.score(anchor.candidate_id)
    return Row(
        truth_r_m=truth_r_m,
        truth_area_m2=truth_area,
        detected=True,
        label=defect.label,
        views=aggregate.n_measurements,
        discarded=aggregate.n_discarded_vantage,
        best_footprint_px=aggregate.best_footprint_px,
        measured_area_m2=aggregate.area_m2_median,
        iqr_ratio=aggregate.iqr_ratio,
        priority=defect.priority,
        needs_approval=defect.needs_human_approval,
    )


def sweep(radii: list[float] | None = None) -> list[Row]:
    return [measure_one(r) for r in (radii or [0.45, 0.38, 0.32, 0.26, 0.20, 0.15])]


def report(rows: list[Row]) -> str:
    lines = [
        f"{'truth r':>8} {'truth m2':>9} {'label':>9} {'views':>6} {'disc':>5} "
        f"{'best_fp':>8} {'measured':>9} {'err':>8} {'iqr':>7} {'pri':>4} {'appr':>5}"
    ]
    errors = []
    for row in rows:
        iqr = "inf" if not math.isfinite(row.iqr_ratio) else f"{row.iqr_ratio:.3f}"
        error = "n/a" if math.isnan(row.error_pct) else f"{row.error_pct:+.1f}%"
        if not math.isnan(row.error_pct):
            errors.append(row.error_pct)
        lines.append(
            f"{row.truth_r_m:8.2f} {row.truth_area_m2:9.4f} {row.label:>9} {row.views:6d} "
            f"{row.discarded:5d} {row.best_footprint_px:8.0f} {row.measured_area_m2:9.4f} "
            f"{error:>8} {iqr:>7} {row.priority:4d} {str(row.needs_approval):>5}"
        )
    if errors:
        import numpy as np

        array = np.array(errors)
        lines.append("")
        lines.append(
            f"measured {len(errors)}/{len(rows)}  bias {array.mean():+.1f}%  "
            f"MAE {np.abs(array).mean():.1f}%  max |err| {np.abs(array).max():.1f}%"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sweep()))
