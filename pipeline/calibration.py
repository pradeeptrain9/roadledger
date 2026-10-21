"""Recovering the camera geometry the measurement depends on.

Every area this system reports is scaled by the camera's height, focal length
and downward tilt. Get the height wrong by 5% and every area is wrong by 10%,
because area scales as the square of it. So calibration is not setup overhead,
it is the measurement's foundation, and this module keeps the two ways of
establishing it in one place with the same output format.

**Two markers** (`solve_from_two_markers`). Objects at two known distances give
two equations in the two unknowns. Nothing has to be guessed. This is the route
for a recorded calibration clip and is what `scripts/calibrate.py` drives.

**Horizon plus one marker** (`solve_from_horizon_and_marker`). On a phone the
horizon can simply be pointed at: the operator drags a line onto the horizon in
the live preview, which reads off `y_h` directly, and then taps one object at a
measured distance. One tap instead of two placements, and the horizon - the
quantity rectification is most sensitive to - comes from a long, easily judged
straight line rather than being extrapolated from two nearby points.

The second route is the more accurate one where it is available, and that is
worth stating precisely rather than leaving as a preference. Extrapolating the
horizon from markers at 3 m and 6 m amplifies the reading error on those rows:

    y_h = y1 - (y1 - y2) / (1 - d1/d2)

With d1/d2 = 0.5 the factor on (y1 - y2) is 2, so a 2 px error on each row can
put the horizon out by 4-6 px. Judging a horizon line directly is good to
1-2 px, because it is a boundary hundreds of pixels wide.
"""

from __future__ import annotations

import math
from typing import Any


class CalibrationError(ValueError):
    """The inputs cannot produce a usable camera geometry."""


def _derive(height_m: float, focal_px: float, horizon_px: float, image_height: int) -> dict[str, Any]:
    if focal_px <= 0:
        raise CalibrationError(
            f"solved a non-positive focal length ({focal_px:.1f} px); check the distances and rows"
        )
    vfov = 2.0 * math.degrees(math.atan((image_height / 2.0) / focal_px))
    pitch = math.degrees(math.atan((image_height / 2.0 - horizon_px) / focal_px))
    return {
        "height_m": round(height_m, 4),
        "vfov_deg": round(vfov, 3),
        "pitch_deg": round(pitch, 3),
        "focal_px": round(focal_px, 2),
        "horizon_px": round(horizon_px, 2),
        "image_height_px": int(image_height),
    }


def solve_from_two_markers(
    height_m: float, d1: float, d2: float, y1: float, y2: float, image_height: int
) -> dict[str, Any]:
    """Two ground markers at known distances. `d1` is the nearer one."""
    if height_m <= 0:
        raise CalibrationError("camera height must be positive")
    if d1 <= 0 or d2 <= 0 or abs(d1 - d2) < 1e-6:
        raise CalibrationError("the two distances must be positive and different")
    if y1 <= y2:
        raise CalibrationError(
            "the nearer marker must sit lower in the frame (a larger row); "
            f"got {d1} m at row {y1} and {d2} m at row {y2}, which are the wrong way round"
        )
    hf = (y1 - y2) / (1.0 / d1 - 1.0 / d2)
    horizon = y1 - hf / d1
    result = _derive(height_m, hf / height_m, horizon, image_height)
    result["solved_from"] = {
        "method": "two_markers",
        "distances_m": [d1, d2],
        "rows_px": [y1, y2],
    }
    return result


def solve_from_horizon_and_marker(
    height_m: float, horizon_px: float, distance_m: float, row_px: float, image_height: int
) -> dict[str, Any]:
    """A pointed-at horizon plus one marker at a measured distance.

        row = horizon + height*focal/distance   ->   focal = (row - horizon)*distance/height
    """
    if height_m <= 0:
        raise CalibrationError("camera height must be positive")
    if distance_m <= 0:
        raise CalibrationError("the marker distance must be positive")
    if not (0 <= horizon_px < image_height):
        raise CalibrationError(
            f"the horizon row {horizon_px} is outside a {image_height} px frame"
        )
    if row_px <= horizon_px:
        raise CalibrationError(
            f"the marker at row {row_px} is at or above the horizon at row {horizon_px}; "
            "a point on the road is always below the horizon, so one of the two is misplaced"
        )
    focal = (row_px - horizon_px) * distance_m / height_m
    result = _derive(height_m, focal, horizon_px, image_height)
    result["solved_from"] = {
        "method": "horizon_and_marker",
        "distance_m": distance_m,
        "row_px": row_px,
        "horizon_px": horizon_px,
    }
    return result


# A phone's rear camera has a vertical field of view in roughly this range once
# the sensor is cropped to a 16:9 video frame. A solution outside it means an
# input was mistyped, and since nothing downstream can detect that, it is
# refused here rather than silently scaling every area.
PLAUSIBLE_VFOV_DEG = (30.0, 110.0)
# A windscreen or dashboard mount is aimed slightly down. Steeper than this and
# the flat-road band barely reaches past the bonnet.
PLAUSIBLE_PITCH_DEG = 30.0
# Lens height above the road for a car, a van, or a hand-held phone at head
# height. A number outside this is a units mistake - centimetres for metres is
# the usual one, and it would scale every area by ten thousand.
PLAUSIBLE_HEIGHT_M = (0.3, 4.0)


def warnings_for(result: dict[str, Any]) -> list[str]:
    """Implausible-but-solvable geometries, named rather than rejected."""
    out: list[str] = []
    low, high = PLAUSIBLE_VFOV_DEG
    if not (low <= result["vfov_deg"] <= high):
        out.append(
            f"a {result['vfov_deg']:.1f} deg vertical field of view is outside the "
            f"{low:.0f}-{high:.0f} deg range a phone camera plausibly has; the marker "
            "distance or row is probably wrong, and every area will be wrong with it"
        )
    if abs(result["pitch_deg"]) > PLAUSIBLE_PITCH_DEG:
        out.append(
            f"{result['pitch_deg']:.1f} deg of tilt is steeper than a dashboard mount; "
            "check the horizon row"
        )
    low, high = PLAUSIBLE_HEIGHT_M
    if not (low <= result["height_m"] <= high):
        out.append(
            f"a camera height of {result['height_m']} m is outside {low}-{high} m; "
            "if this was entered in centimetres every area will be out by a factor of 10000"
        )
    return out


def reprojection_check(result: dict[str, Any], distances_m: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)) -> list[dict[str, float]]:
    """Where the solution says the road is, at distances the operator can judge.

    The most useful check available without more measurements: if the solution
    claims 10 m is at a row that is visibly the far kerb, something is wrong
    and it is better to find that out before driving.
    """
    focal, horizon, height = result["focal_px"], result["horizon_px"], result["height_m"]
    return [
        {"distance_m": d, "row_px": round(horizon + height * focal / d, 1)} for d in distances_m
    ]


def scale_to_frame(result: dict[str, Any], frame_height_px: int) -> dict[str, Any]:
    """Re-express a calibration for a frame of a different height.

    The phone previews and calibrates at one resolution and may upload at
    another. Field of view and pitch are properties of the lens and the mount,
    not of the pixel grid, so they carry over unchanged - but focal length and
    horizon are in pixels and do not. Carrying a pixel focal length across a
    resolution change unnoticed would scale every distance, and therefore every
    area, by the ratio of the two heights.
    """
    if frame_height_px <= 0:
        raise CalibrationError("frame height must be positive")
    ratio = frame_height_px / result["image_height_px"]
    return {
        **result,
        "focal_px": round(result["focal_px"] * ratio, 2),
        "horizon_px": round(result["horizon_px"] * ratio, 2),
        "image_height_px": int(frame_height_px),
        "rescaled_from_px": result["image_height_px"],
    }
