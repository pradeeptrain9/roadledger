"""Physically consistent synthetic road scenes.

Scenes are rendered by projecting ground-plane points through an actual pinhole
camera model, so the ground truth of every geometric quantity - vanishing
point, lane pixel width, metres per pixel, defect area in m2 - is known exactly.
That makes it possible to measure the pipeline's error rather than merely
eyeballing its output, and it is what the rectification cross-check is tested
against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SceneCamera:
    """Pinhole camera looking straight down a flat road."""

    width: int = 1280
    height: int = 720
    height_m: float = 1.35
    vfov_deg: float = 58.0

    @property
    def focal_px(self) -> float:
        return (self.height / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)

    @property
    def horizon_y(self) -> float:
        """With zero pitch the horizon sits on the optical axis."""
        return self.height / 2.0

    def project(self, lateral_m: float, distance_m: float) -> tuple[float, float]:
        """Ground point (lateral offset, forward distance) to pixel coordinates."""
        if distance_m <= 1e-6:
            raise ValueError("distance must be positive")
        x = self.width / 2.0 + lateral_m * self.focal_px / distance_m
        y = self.horizon_y + self.height_m * self.focal_px / distance_m
        return x, y

    def distance_at_row(self, y: float) -> float:
        delta = y - self.horizon_y
        return math.inf if delta <= 1e-6 else self.height_m * self.focal_px / delta

    def lateral_m_per_px_at_row(self, y: float) -> float:
        return self.distance_at_row(y) / self.focal_px


def render_road(
    camera: SceneCamera = SceneCamera(),
    lane_width_m: float = 3.5,
    near_m: float = 3.0,
    far_m: float = 60.0,
    defects: list[tuple[float, float, float]] | None = None,
    confusers: list[tuple[float, float, float, str]] | None = None,
    seed: int = 7,
    supersample: int = 4,
) -> tuple[np.ndarray, dict]:
    """Render a road with lane markings and optional circular defects.

    `defects` are (lateral_m, distance_m, radius_m) tuples. Returns the image
    and a ground-truth dictionary.

    Rendering is supersampled because it must not become the thing being
    measured. Drawing an ellipse directly at output resolution truncates its
    semi-axes to whole pixels, which under-draws every defect by several
    percent - enough to masquerade as a pipeline bias and send the
    investigation after the wrong cause. Drawing at `supersample` times the
    resolution and area-averaging down keeps the rendered area faithful to the
    intended pi*r^2.
    """
    scale = max(1, int(supersample))
    big = SceneCamera(
        width=camera.width * scale,
        height=camera.height * scale,
        height_m=camera.height_m,
        vfov_deg=camera.vfov_deg,
    )

    rng = np.random.default_rng(seed)
    # Rendered in float and kept that way until the end. Shading is applied
    # before sensor noise, which is the physical order and matters: additive
    # read noise after shading means a shadowed region has *higher* relative
    # roughness than a lit one, so "texture is unchanged under shadow" is not a
    # usable cue on real footage. Quantising mid-pipeline hid that.
    image = np.full((big.height, big.width, 3), 72.0, np.float64)
    camera_render = big

    half = lane_width_m / 2.0
    truth_confusers: list[dict] = []

    def polyline(lateral_m: float, steps: int = 240) -> np.ndarray:
        distances = np.linspace(near_m, far_m, steps)
        points = [camera_render.project(lateral_m, float(d)) for d in distances]
        return np.array(points, dtype=np.int32)

    # Lane edge markings, drawn in perspective so their width tapers correctly.
    for lateral in (-half, half):
        points = polyline(lateral)
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            distance = camera_render.distance_at_row(float(y0))
            thickness = max(1, int(round(0.12 * camera_render.focal_px / max(distance, 1e-6))))
            cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), (232, 232, 232), thickness)

    def ground_ellipse_axes(lateral_m: float, distance_m: float, radius_m: float):
        lateral_px = radius_m * camera_render.focal_px / distance_m
        near = camera_render.project(lateral_m, distance_m - radius_m)
        far = camera_render.project(lateral_m, distance_m + radius_m)
        return lateral_px, abs(near[1] - far[1]) / 2.0

    # Confusers are rendered with the physics that distinguishes them, so the
    # classifier is tested against the real discrimination problem rather than
    # against arbitrarily coloured blobs.
    for lateral_m, distance_m, radius_m, kind in confusers or []:
        centre = camera_render.project(lateral_m, distance_m)
        ax, ay = ground_ellipse_axes(lateral_m, distance_m, radius_m)
        axes = (max(1, int(round(ax))), max(1, int(round(ay))))
        region = np.zeros(image.shape[:2], np.uint8)
        cv2.ellipse(region, (int(round(centre[0])), int(round(centre[1]))), axes,
                    0, 0, 360, 255, -1, lineType=cv2.LINE_AA)
        mask = region > 0
        if kind == "shadow":
            # A cast shadow has a penumbra: the sun is not a point source, so
            # the edge is a ramp a few centimetres wide, not a step. Rendered
            # as a soft-edged multiplicative field - this softness, not the
            # texture, is what separates a shadow from a cavity rim.
            penumbra_px = max(2.0, 0.035 * camera_render.focal_px / distance_m)
            alpha = cv2.GaussianBlur(
                region.astype(np.float64) / 255.0, (0, 0), penumbra_px
            )[:, :, None]
            image *= 1.0 - 0.45 * alpha
        elif kind == "wet":
            # Darker and specular-smooth: texture suppressed, not just dimmed.
            alpha = cv2.GaussianBlur(region.astype(np.float64) / 255.0, (0, 0), 2.0)[:, :, None]
            smooth = cv2.GaussianBlur(image, (0, 0), 3.0)
            image = image * (1.0 - alpha) + alpha * (0.6 * (0.2 * image + 0.8 * smooth))
        elif kind == "patch":
            # A repair: near road brightness, smoother, near-rectangular, with
            # a sharp machined edge.
            box_w = max(2, int(round(ax * 1.6)))
            box_h = max(2, int(round(ay * 1.6)))
            x0 = int(round(centre[0])) - box_w // 2
            y0 = int(round(centre[1])) - box_h // 2
            patch_region = np.zeros(image.shape[:2], np.uint8)
            cv2.rectangle(patch_region, (x0, y0), (x0 + box_w, y0 + box_h), 255, -1)
            alpha = (patch_region.astype(np.float64) / 255.0)[:, :, None]
            smooth = cv2.GaussianBlur(image, (0, 0), 1.5)
            image = image * (1.0 - alpha) + alpha * (0.88 * smooth)
        truth_confusers.append(
            {
                "kind": kind,
                "lateral_m": lateral_m,
                "distance_m": distance_m,
                "radius_m": radius_m,
                "centre_px": (centre[0] / scale, centre[1] / scale),
            }
        )

    truth_defects = []
    for lateral_m, distance_m, radius_m in defects or []:
        centre = camera_render.project(lateral_m, distance_m)
        # A circle on the ground images as an ellipse; its extent follows from
        # the local lateral and longitudinal scales. Rounded, not truncated,
        # and at supersampled resolution so the error is sub-output-pixel.
        lateral_px = radius_m * camera_render.focal_px / distance_m
        near = camera_render.project(lateral_m, distance_m - radius_m)
        far = camera_render.project(lateral_m, distance_m + radius_m)
        longitudinal_px = abs(near[1] - far[1]) / 2.0
        cv2.ellipse(
            image,
            (int(round(centre[0])), int(round(centre[1]))),
            (max(1, int(round(lateral_px))), max(1, int(round(longitudinal_px)))),
            0,
            0,
            360,
            (28.0, 28.0, 30.0),
            -1,
            lineType=cv2.LINE_AA,
        )
        truth_defects.append(
            {
                "lateral_m": lateral_m,
                "distance_m": distance_m,
                "radius_m": radius_m,
                "area_m2": math.pi * radius_m**2,
                "centre_px": (centre[0] / scale, centre[1] / scale),
                "rendered_semi_axes_px": (lateral_px / scale, longitudinal_px / scale),
            }
        )

    if scale > 1:
        image = cv2.resize(
            image, (camera.width, camera.height), interpolation=cv2.INTER_AREA
        )

    # Sensor noise last, at output resolution: additive and signal-independent,
    # so relative roughness rises where the scene is dark.
    image = np.clip(
        image + rng.normal(0.0, 6.0, (camera.height, camera.width, 3)), 0, 255
    ).astype(np.uint8)

    y_base = camera.height - 1
    x_left, _ = camera.project(-half, camera.distance_at_row(y_base))
    x_right, _ = camera.project(half, camera.distance_at_row(y_base))

    truth = {
        "vanishing_point": (camera.width / 2.0, camera.horizon_y),
        "lane_px_width_at_base": x_right - x_left,
        "lane_width_m": lane_width_m,
        "lateral_m_per_px_at_base": camera.lateral_m_per_px_at_row(y_base),
        "distance_at_base_m": camera.distance_at_row(y_base),
        "defects": truth_defects,
        "confusers": truth_confusers,
        "camera": camera,
    }
    return image, truth


def render_sequence(
    camera: SceneCamera = SceneCamera(),
    defects: list[tuple[float, float, float]] | None = None,
    confusers: list[tuple[float, float, float, str]] | None = None,
    n_frames: int = 8,
    forward_step_m: float = 0.6,
    seed: int = 21,
) -> tuple[list[np.ndarray], list[dict]]:
    """A vehicle driving forward past fixed road features.

    Each frame re-renders the same ground-truth features from a camera that has
    advanced `forward_step_m`, so a defect is genuinely seen from several
    distances. That is what gives the multi-frame stage independent
    measurements to disagree about - replaying one frame repeatedly would
    manufacture perfect agreement and make the spread meaningless.
    """
    images: list[np.ndarray] = []
    truths: list[dict] = []
    for index in range(n_frames):
        travelled = index * forward_step_m
        moved_defects = [
            (lateral, distance - travelled, radius)
            for lateral, distance, radius in (defects or [])
            if distance - travelled > 2.5
        ]
        moved_confusers = [
            (lateral, distance - travelled, radius, kind)
            for lateral, distance, radius, kind in (confusers or [])
            if distance - travelled > 2.5
        ]
        image, truth = render_road(
            camera,
            defects=moved_defects,
            confusers=moved_confusers,
            seed=seed + index,
        )
        truth["travelled_m"] = travelled
        images.append(image)
        truths.append(truth)
    return images, truths
