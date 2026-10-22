"""What the pipeline does on real phone footage, and what it refuses to do.

This is deliberately **not** an accuracy evaluation. The clip it was written
for is hand-held and uncalibrated, so no area it produces means anything, and
pretending otherwise would be the exact failure this project is built to avoid.
What it measures instead is the half of the system that does not need
calibration:

* **Applicability** - can the road plane be recovered at all? On real roads the
  lane-marking route managed 2.3%, which is what forced the calibrated route.
* **Proposal behaviour** - how many candidates per frame, and is that a
  plausible number or a threshold flooding on texture?
* **Refusal behaviour** - this is the interesting one. A measurement system is
  only trustworthy if it declines when it cannot measure, so every refusal is
  counted by reason. A run where most candidates are refused for *sensible*
  reasons is a better result than one where everything is confidently sized.
* **Privacy** - what the face and plate scrubbers actually fire on in a real
  street scene, including when they fire too often.

    python -m eval.real_video --video data/footage/drive.mp4 --frames 60
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.config import (
    CameraGeometry,
    FrameParams,
    PipelineConfig,
    PrivacyParams,
    RectifyParams,
)
from pipeline.runner import Session

# Stand-in geometry for an uncalibrated clip. Every area derived from it is
# arithmetic, not measurement, and the report says so on every line that could
# be mistaken for one.
ASSUMED = CameraGeometry(height_m=1.10, vfov_deg=55.0, pitch_deg=15.0)


def analyse(
    video: Path,
    max_frames: int = 60,
    camera: CameraGeometry | None = None,
    sample_fps: float = 4.0,
    artifacts: Path | None = None,
) -> dict[str, Any]:
    calibrated = camera is not None
    config = PipelineConfig(
        rectify=RectifyParams(camera=camera or ASSUMED),
        frames=FrameParams(
            sample_fps=sample_fps, min_sharpness=18.0, max_frames=max_frames,
            resize_long_edge=1280,
        ),
        privacy=PrivacyParams(enabled=True),
    )
    session = Session(config=config)
    loaded = session.load(str(video))

    qualities: list[float] = []
    refusals: Counter[str] = Counter()
    candidates = measurable = 0
    areas: list[float] = []
    footprints: list[float] = []
    saved: list[str] = []

    for index, frame in enumerate(session.frames):
        rectification = session.rectify_frame(frame.frame_id)
        qualities.append(rectification.quality)
        if frame.frame_id not in session.bevs:
            refusals["frame could not be rectified"] += 1
            continue
        found = session.propose_candidates(frame.frame_id)
        candidates += len(found)
        for candidate in found:
            measurement = session.refine_candidate(candidate.candidate_id)
            if measurement.measurable:
                measurable += 1
                areas.append(measurement.area_m2)
                footprints.append(measurement.source_footprint_px)
            else:
                refusals[_category(measurement.note)] += 1

        if artifacts and index in (5, 15, 25) and found:
            saved.extend(_draw(session, frame, found, artifacts, index))

    usable = [q for q in qualities if q >= config.agent.homography_min_quality]
    return {
        "video": str(video),
        "calibrated": calibrated,
        "camera": {
            "height_m": (camera or ASSUMED).height_m,
            "vfov_deg": (camera or ASSUMED).vfov_deg,
            "pitch_deg": (camera or ASSUMED).pitch_deg,
        },
        "frames_sampled": loaded["frames_sampled"],
        "privacy": loaded["privacy"],
        "rectification": {
            "applicability": round(len(usable) / max(len(qualities), 1), 4),
            "quality_min": round(min(qualities), 3) if qualities else 0.0,
            "quality_median": round(float(np.median(qualities)), 3) if qualities else 0.0,
            "method": next(iter(session.rectifications.values())).method
            if session.rectifications else "none",
        },
        "candidates": candidates,
        "candidates_per_frame": round(candidates / max(len(qualities), 1), 2),
        "measurable": measurable,
        "measurable_fraction": round(measurable / max(candidates, 1), 4),
        "refusals": dict(refusals.most_common()),
        "source_footprint_px_median": round(float(np.median(footprints)), 1) if footprints else 0.0,
        # Reported only to show the magnitude the pipeline is working at. With
        # an assumed camera these are not measurements of anything.
        "areas_m2_uncalibrated": {
            "n": len(areas),
            "min": round(min(areas), 4) if areas else 0.0,
            "median": round(float(np.median(areas)), 4) if areas else 0.0,
            "max": round(max(areas), 4) if areas else 0.0,
        } if not calibrated else None,
        "artifacts": saved,
    }


# Each refusal message carries its own numbers ("footprint 104 source px..."),
# so counting the raw strings produces one bucket per candidate and no counts
# at all. Refusals are grouped by kind.
REFUSAL_KINDS = (
    ("below the", "too distant: source footprint under the measurable floor"),
    ("edge of the rectified view", "clipped: boundary cut off by the view edge"),
    ("beyond the proposal", "growth guard: refinement leaked past the proposal"),
    ("did not retain", "refinement did not retain the candidate"),
    ("no contour", "no contour survived refinement"),
    ("ROI degenerate", "ROI too small to refine"),
)


def _category(note: str | None) -> str:
    text = note or ""
    for needle, label in REFUSAL_KINDS:
        if needle in text:
            return label
    return (text.split(";")[0].strip() or "unstated")[:70]


def _draw(session: Session, frame, found, out_dir: Path, index: int) -> list[str]:
    """Source frame and bird's-eye view, both with the candidates outlined."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rectification = session.rectifications[frame.frame_id]
    bev = session.bevs[frame.frame_id].copy()
    for candidate in found:
        if candidate.contour is not None:
            cv2.drawContours(bev, [candidate.contour], -1, (40, 220, 255), 2)
    bev_path = out_dir / f"real_bev_{index:03d}.png"
    cv2.imwrite(str(bev_path), bev)

    source = frame.image.copy()
    inverse = np.linalg.inv(rectification.homography)
    for candidate in found:
        if candidate.contour is None:
            continue
        mapped = cv2.perspectiveTransform(
            candidate.contour.astype(np.float64).reshape(-1, 1, 2), inverse
        ).astype(np.int32)
        cv2.polylines(source, [mapped], True, (40, 220, 255), 2)
    src_path = out_dir / f"real_frame_{index:03d}.png"
    cv2.imwrite(str(src_path), source)
    return [str(bev_path), str(src_path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/footage/drive.mp4")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--artifacts", default="data/artifacts")
    parser.add_argument("--out", default="")
    # Supplying all three promotes every area from arithmetic to measurement.
    parser.add_argument("--camera-height", type=float, default=0.0)
    parser.add_argument("--vfov", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    args = parser.parse_args()

    camera = None
    if args.camera_height and args.vfov:
        camera = CameraGeometry(args.camera_height, args.vfov, args.pitch)

    result = analyse(
        Path(args.video), args.frames, camera, args.fps,
        Path(args.artifacts) if args.artifacts else None,
    )

    print(f"\n{result['video']}")
    print(f"  frames sampled           {result['frames_sampled']}")
    print(f"  faces / plates blurred   {result['privacy']['faces_blurred']} / "
          f"{result['privacy']['plates_blurred']}")
    r = result["rectification"]
    print(f"  rectification            {r['applicability']:.0%} applicable, "
          f"quality {r['quality_median']} ({r['method']})")
    print(f"  candidates               {result['candidates']} "
          f"({result['candidates_per_frame']}/frame)")
    print(f"  measurable               {result['measurable']} "
          f"({result['measurable_fraction']:.0%} of candidates)")
    print(f"  median source footprint  {result['source_footprint_px_median']:.0f} px")
    print("\n  refused, by reason:")
    for reason, count in result["refusals"].items():
        print(f"    {count:4d}  {reason}")
    if not result["calibrated"]:
        a = result["areas_m2_uncalibrated"]
        print(f"\n  areas spanned {a['min']}-{a['max']} m2 (median {a['median']}) from an")
        print("  ASSUMED camera. These are arithmetic, not measurements: the clip is")
        print("  uncalibrated, so no area from it means anything.")
    if result["artifacts"]:
        print(f"\n  wrote {len(result['artifacts'])} annotated images")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
