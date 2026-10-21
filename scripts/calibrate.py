"""Solve camera geometry from two ground markers at known distances.

Rectification needs the camera's height, focal length and downward tilt. Height
is measured with a tape. The other two do not have to be looked up or guessed,
because two markers at known distances pin them down exactly.

For a camera at height h with focal length f and horizon at row y_h, a ground
point at distance d images at

    y = y_h + h*f / d

Two markers give two equations in two unknowns (y_h and the product h*f):

    y1 - y2 = h*f * (1/d1 - 1/d2)      ->   h*f
    y_h     = y1 - h*f/d1

and f follows from the measured h. Field of view and pitch are then derived
rather than assumed:

    vfov  = 2*atan((H/2) / f)
    pitch = atan((H/2 - y_h) / f)

Usage, after recording the calibration clip described in
docs/recording-guide.md:

    # 1. Extract a frame and overlay a row grid to read coordinates from
    python -m scripts.calibrate --video data/footage/calibration.mp4 --grid

    # 2. Solve, giving the image rows of the two markers' contact points
    python -m scripts.calibrate --video data/footage/calibration.mp4 \\
        --camera-height 1.27 --distances 3.0 6.0 --rows 612 486

Read the row where each object *touches the road*, not its top edge: the
geometry is about where the ground is.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def extract_frame(video: Path, index: int = 30) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"cannot read a frame from {video}")
        return frame
    finally:
        capture.release()


def draw_grid(image: np.ndarray, step: int = 25) -> np.ndarray:
    """Overlay labelled horizontal rules, so marker rows can be read off."""
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for y in range(0, height, step):
        heavy = (y % (step * 4) == 0)
        cv2.line(annotated, (0, y), (width, y), (0, 200, 255), 2 if heavy else 1)
        if heavy:
            for x in (8, width - 86):
                cv2.putText(annotated, f"y={y}", (x, max(16, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(annotated, f"y={y}", (x, max(16, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
    cv2.line(annotated, (width // 2, 0), (width // 2, height), (0, 200, 255), 1)
    return annotated


def solve(
    height_m: float, d1: float, d2: float, y1: float, y2: float, image_height: int
) -> dict:
    """Recover horizon, focal length, field of view and pitch."""
    if d1 <= 0 or d2 <= 0 or abs(d1 - d2) < 1e-6:
        raise SystemExit("the two distances must be positive and different")
    if y1 <= y2:
        raise SystemExit(
            "the nearer marker must sit LOWER in the frame (a larger row). "
            f"got d1={d1} m at row {y1}, d2={d2} m at row {y2}. "
            "Check that --rows is given in the same order as --distances."
        )

    hf = (y1 - y2) / (1.0 / d1 - 1.0 / d2)
    horizon = y1 - hf / d1
    focal = hf / height_m

    if focal <= 0:
        raise SystemExit("solved a non-positive focal length; check the inputs")

    vfov = 2.0 * math.degrees(math.atan((image_height / 2.0) / focal))
    pitch = math.degrees(math.atan((image_height / 2.0 - horizon) / focal))

    return {
        "height_m": round(height_m, 4),
        "vfov_deg": round(vfov, 3),
        "pitch_deg": round(pitch, 3),
        "focal_px": round(focal, 2),
        "horizon_px": round(horizon, 2),
        "image_height_px": image_height,
        "solved_from": {
            "distances_m": [d1, d2],
            "rows_px": [y1, y2],
        },
    }


def check(result: dict, height_m: float, d1: float, d2: float, y1: float, y2: float) -> None:
    """Re-project the markers through the solution as a self-check."""
    focal, horizon = result["focal_px"], result["horizon_px"]
    print("\nself-check: re-projecting the markers through the solution")
    for distance, observed in ((d1, y1), (d2, y2)):
        predicted = horizon + height_m * focal / distance
        print(f"  {distance:.2f} m -> row {predicted:7.1f}  (you read {observed:.0f}, "
              f"off by {abs(predicted-observed):.1f} px)")
    # A third distance shows how the solution extrapolates.
    for distance in (2.0, 10.0, 20.0):
        print(f"  {distance:5.2f} m -> row {horizon + height_m*focal/distance:7.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--grid", action="store_true", help="write a grid overlay and exit")
    parser.add_argument("--camera-height", type=float, help="lens height above the road, metres")
    parser.add_argument("--distances", type=float, nargs=2, metavar=("D_NEAR", "D_FAR"))
    parser.add_argument("--rows", type=float, nargs=2, metavar=("Y_NEAR", "Y_FAR"),
                        help="image rows where each marker touches the road")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    video = Path(args.video)
    frame = extract_frame(video, args.frame)
    height, width = frame.shape[:2]
    print(f"{video.name}: frame {args.frame} is {width}x{height}")

    if args.grid or not (args.camera_height and args.distances and args.rows):
        out = video.with_name(f"{video.stem}_grid.png")
        cv2.imwrite(str(out), draw_grid(frame))
        print(f"\nwrote {out}")
        print("Open it, read the row where each marker TOUCHES THE ROAD, then run:")
        print(f"  python -m scripts.calibrate --video {args.video} \\")
        print(f"      --camera-height <metres> --distances 3.0 6.0 --rows <y_near> <y_far>")
        return

    d1, d2 = args.distances
    y1, y2 = args.rows
    result = solve(args.camera_height, d1, d2, y1, y2, height)

    print("\nsolved camera geometry")
    for key in ("height_m", "vfov_deg", "pitch_deg", "focal_px", "horizon_px"):
        print(f"  {key:16s} {result[key]}")

    check(result, args.camera_height, d1, d2, y1, y2)

    if not (30.0 <= result["vfov_deg"] <= 110.0):
        print(f"\n  WARNING: a {result['vfov_deg']:.1f} deg field of view is outside the range "
              "a phone camera plausibly has. Re-check the measured distances and rows.")
    if abs(result["pitch_deg"]) > 30.0:
        print(f"\n  WARNING: {result['pitch_deg']:.1f} deg of tilt is steep for a dashboard mount. "
              "Re-check the inputs.")

    out = Path(args.out) if args.out else video.with_suffix(".json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    print("Pass this to the pipeline with --calibration.")


if __name__ == "__main__":
    main()
