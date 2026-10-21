"""A persistent ground plane, rendered per frame through a real camera.

The first synthetic rig regenerated the asphalt noise independently for every
frame. That made multi-frame association impossible to test by any method: no
tracker can follow texture that did not exist in the previous frame, and the
lane markings are continuous lines that look identical as the camera drives
along them. Lucas-Kanade and phase correlation both failed on it, and the
failure was the rig's, not theirs.

So the world is built once, in metric ground coordinates, and each frame is a
projection of that same world from a camera that has moved. Road texture,
markings and defects are all fixed features of the ground, so they move
correctly between frames and a tracker has something real to lock onto. As a
bonus the ground truth gets simpler and stronger: a pothole is a *circle* of
known radius in ground coordinates, so its area is pi*r^2 by construction
rather than by approximation.

The ground-to-image mapping is an exact homography. For lateral offset u and
forward distance v, a pinhole camera at height h with focal length f and
horizon row y0 images the point at x = cx + f*u/v, y = y0 + f*h/v, which in
homogeneous coordinates is linear:

    [x]   [ f   cx    0  ] [u]
    [y] ~ [ 0   y0   f*h ] [v]
    [1]   [ 0    1    0  ] [1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .synthetic import SceneCamera


@dataclass
class GroundPlane:
    """The world, in metres, as an image indexed by (lateral, forward)."""

    lateral_half_m: float = 3.0
    length_m: float = 60.0
    m_per_px: float = 0.01
    lane_width_m: float = 3.5
    image: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 3), np.float64))
    defects: list[dict] = field(default_factory=list)
    confusers: list[dict] = field(default_factory=list)

    @property
    def width_px(self) -> int:
        return int(round(2 * self.lateral_half_m / self.m_per_px))

    @property
    def height_px(self) -> int:
        return int(round(self.length_m / self.m_per_px))

    def to_px(self, lateral_m: float, forward_m: float) -> tuple[float, float]:
        """Ground metres to ground-texture pixels."""
        u = (lateral_m + self.lateral_half_m) / self.m_per_px
        v = forward_m / self.m_per_px
        return u, v


def build_ground(
    lane_width_m: float = 3.5,
    defects: list[tuple[float, float, float]] | None = None,
    confusers: list[tuple[float, float, float, str]] | None = None,
    seed: int = 7,
    m_per_px: float = 0.01,
    length_m: float = 60.0,
) -> GroundPlane:
    """Build the road surface once: texture, markings, defects, confusers.

    `defects` and `confusers` are (lateral_m, forward_m, radius_m[, kind]) in
    ground coordinates, so a defect is a true circle and its area is exact.
    """
    ground = GroundPlane(
        lane_width_m=lane_width_m, m_per_px=m_per_px, length_m=length_m
    )
    height, width = ground.height_px, ground.width_px
    rng = np.random.default_rng(seed)

    # Asphalt: fine aggregate grain plus a coarser mottle, so there is texture
    # at more than one scale for a tracker to use.
    fine = rng.normal(0.0, 7.0, (height, width))
    coarse = cv2.GaussianBlur(rng.normal(0.0, 18.0, (height, width)), (0, 0), 8.0)
    surface = 72.0 + fine + coarse
    image = np.repeat(surface[:, :, None], 3, axis=2)

    radius_px = lambda metres: max(1, int(round(metres / m_per_px)))

    # Lane edge markings: solid lines at +/- half a lane width.
    half = lane_width_m / 2.0
    marking_half_px = radius_px(0.06)
    for lateral in (-half, half):
        u, _ = ground.to_px(lateral, 0.0)
        u = int(round(u))
        image[:, max(0, u - marking_half_px) : u + marking_half_px] = 232.0

    # Dashed centre line: 3 m mark, 6 m gap, the usual highway pattern.
    u_centre = int(round(ground.to_px(0.0, 0.0)[0]))
    dash_half_px = radius_px(0.05)
    position = 2.0
    while position < length_m:
        v0 = int(round(position / m_per_px))
        v1 = int(round((position + 3.0) / m_per_px))
        image[v0:v1, max(0, u_centre - dash_half_px) : u_centre + dash_half_px] = 228.0
        position += 9.0

    def circle_mask(lateral_m: float, forward_m: float, r_m: float) -> np.ndarray:
        mask = np.zeros((height, width), np.uint8)
        u, v = ground.to_px(lateral_m, forward_m)
        cv2.circle(mask, (int(round(u)), int(round(v))), radius_px(r_m), 255, -1, cv2.LINE_AA)
        return mask

    for lateral_m, forward_m, r_m in defects or []:
        mask = circle_mask(lateral_m, forward_m, r_m)
        selected = mask > 0
        # A cavity: much darker, and rougher than the surrounding surface.
        rough = rng.normal(0.0, 16.0, (height, width))
        image[selected] = np.clip(
            28.0 + rough[selected][:, None].repeat(3, axis=1), 0, 255
        )
        ground.defects.append(
            {
                "lateral_m": lateral_m,
                "forward_m": forward_m,
                "radius_m": r_m,
                "area_m2": math.pi * r_m**2,
                "kind": "pothole",
            }
        )

    for lateral_m, forward_m, r_m, kind in confusers or []:
        mask = circle_mask(lateral_m, forward_m, r_m)
        if kind == "shadow":
            # Multiplicative with a penumbra: darkens, preserves texture.
            alpha = cv2.GaussianBlur(mask.astype(np.float64) / 255.0, (0, 0), radius_px(0.08))
            image *= (1.0 - 0.45 * alpha)[:, :, None]
        elif kind == "wet":
            alpha = (cv2.GaussianBlur(mask.astype(np.float64) / 255.0, (0, 0), radius_px(0.03)))[:, :, None]
            smooth = cv2.GaussianBlur(image, (0, 0), radius_px(0.05))
            image = image * (1.0 - alpha) + alpha * (0.6 * (0.2 * image + 0.8 * smooth))
        elif kind == "patch":
            # A rectangular repair: near road brightness, smoother, sharp edge.
            u, v = ground.to_px(lateral_m, forward_m)
            box = radius_px(r_m)
            rect = np.zeros((height, width), np.uint8)
            cv2.rectangle(
                rect,
                (int(round(u)) - box, int(round(v)) - box),
                (int(round(u)) + box, int(round(v)) + box),
                255,
                -1,
            )
            alpha = (rect.astype(np.float64) / 255.0)[:, :, None]
            smooth = cv2.GaussianBlur(image, (0, 0), radius_px(0.04))
            image = image * (1.0 - alpha) + alpha * (0.88 * smooth)
        ground.confusers.append(
            {
                "lateral_m": lateral_m,
                "forward_m": forward_m,
                "radius_m": r_m,
                "kind": kind,
            }
        )

    ground.image = image
    return ground


@dataclass(frozen=True)
class CameraPose:
    """Where the camera is on the ground plane, and which way it points.

    Straight driving needs only `forward_m`, which is why the original rig had
    nothing else. A curve needs all three, and curves are not a detail: the
    live tracker predicts each open defect's next position from the vehicle's
    motion, and on a bend a defect 8 m ahead sweeps sideways by 1.4 m for every
    10 degrees of turn - three times the match radius. Prediction that ignores
    yaw loses the defect after one frame, which silently destroys the
    multi-view agreement the whole measurement rests on. So the rig has to be
    able to turn.
    """

    forward_m: float = 0.0
    lateral_m: float = 0.0
    yaw_deg: float = 0.0


def arc_path(n_frames: int, step_m: float, radius_m: float | None = None) -> list[CameraPose]:
    """A driven path: straight when `radius_m` is None, otherwise a constant arc.

    Positive radius turns right. Integrated as a bicycle model at constant
    speed, so the yaw advances by step/radius per frame.
    """
    poses: list[CameraPose] = []
    forward = lateral = yaw = 0.0
    for _ in range(n_frames):
        poses.append(CameraPose(forward_m=forward, lateral_m=lateral, yaw_deg=math.degrees(yaw)))
        lateral += step_m * math.sin(yaw)
        forward += step_m * math.cos(yaw)
        if radius_m:
            yaw += step_m / radius_m
    return poses


def ground_to_image_homography(
    ground: GroundPlane, camera: SceneCamera, travelled_m: float = 0.0,
    pose: CameraPose | None = None,
) -> np.ndarray:
    """Exact homography from ground-texture pixels to image pixels."""
    focal = camera.focal_px
    cx = camera.width / 2.0
    y0 = camera.horizon_y
    height_m = camera.height_m
    if pose is None:
        pose = CameraPose(forward_m=travelled_m)

    # Ground-texture pixel (u_px, v_px) to metres on the plane:
    #   u_m = u_px * mpp - lateral_half,   v_m = v_px * mpp
    # then into the camera's own frame, which is the plane translated to the
    # camera's position and rotated by its yaw:
    #   right   =  cos(psi) * du - sin(psi) * dv
    #   forward =  sin(psi) * du + cos(psi) * dv
    # with du = u_m - lateral_0 and dv = v_m - forward_0.
    mpp = ground.m_per_px
    psi = math.radians(pose.yaw_deg)
    c, s = math.cos(psi), math.sin(psi)
    du0 = -ground.lateral_half_m - pose.lateral_m
    dv0 = -pose.forward_m
    metres = np.array(
        [
            [c * mpp, -s * mpp, c * du0 - s * dv0],
            [s * mpp, c * mpp, s * du0 + c * dv0],
            [0.0, 0.0, 1.0],
        ]
    )
    # Metres (u_m, v_m) to image, per the derivation in the module docstring.
    project = np.array(
        [
            [focal, cx, 0.0],
            [0.0, y0, focal * height_m],
            [0.0, 1.0, 0.0],
        ]
    )
    return project @ metres


def render_frame(
    ground: GroundPlane,
    camera: SceneCamera,
    travelled_m: float = 0.0,
    noise_sigma: float = 6.0,
    seed: int = 0,
    pose: CameraPose | None = None,
) -> tuple[np.ndarray, dict]:
    """Project the ground plane into one camera frame.

    Sensor noise is added last and independently per frame, which is physically
    right - read noise does not persist between exposures - while the *surface*
    texture does persist, which is what makes tracking possible.
    """
    if pose is None:
        pose = CameraPose(forward_m=travelled_m)
    homography = ground_to_image_homography(ground, camera, pose=pose)
    image = cv2.warpPerspective(
        ground.image,
        homography,
        (camera.width, camera.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(72.0, 72.0, 72.0),
    )
    # Everything above the horizon is sky, not road; fill it so the frame is
    # not dominated by a replicated border.
    horizon = int(camera.horizon_y)
    image[: max(0, horizon + 2)] = 150.0

    rng = np.random.default_rng(seed)
    image = np.clip(
        image + rng.normal(0.0, noise_sigma, image.shape), 0, 255
    ).astype(np.uint8)

    psi = math.radians(pose.yaw_deg)
    visible = []
    for defect in ground.defects + ground.confusers:
        # Forward distance in the camera's own frame. On a straight path this
        # reduces to the old subtraction; on a bend it does not, and using the
        # subtraction there would mislabel every truth distance.
        du = defect["lateral_m"] - pose.lateral_m
        dv = defect["forward_m"] - pose.forward_m
        distance = math.sin(psi) * du + math.cos(psi) * dv
        if distance <= 1.5:
            continue
        u, v = ground.to_px(defect["lateral_m"], defect["forward_m"])
        point = cv2.perspectiveTransform(
            np.array([[[u, v]]], dtype=np.float64), homography
        )[0][0]
        if not (0 <= point[0] < camera.width and horizon < point[1] < camera.height):
            continue
        visible.append(dict(defect, distance_m=distance, centre_px=(float(point[0]), float(point[1]))))

    truth = {
        "travelled_m": travelled_m,
        "defects": [d for d in visible if d["kind"] == "pothole"],
        "confusers": [d for d in visible if d["kind"] != "pothole"],
        "camera": camera,
    }
    return image, truth


def render_drive(
    ground: GroundPlane,
    camera: SceneCamera,
    n_frames: int = 8,
    step_m: float = 0.6,
    seed: int = 21,
    radius_m: float | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """A vehicle driving over the same fixed ground plane.

    Straight by default. `radius_m` bends the path into a constant arc, which
    is what exercises the live tracker's yaw-aware prediction.
    """
    poses = arc_path(n_frames, step_m, radius_m)
    images, truths = [], []
    for index, pose in enumerate(poses):
        image, truth = render_frame(ground, camera, seed=seed + index, pose=pose)
        truth["pose"] = {
            "forward_m": round(pose.forward_m, 4),
            "lateral_m": round(pose.lateral_m, 4),
            "yaw_deg": round(pose.yaw_deg, 4),
        }
        images.append(image)
        truths.append(truth)
    return images, truths
