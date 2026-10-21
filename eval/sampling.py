"""Generate labelled cue samples by running the real pipeline on synthetic scenes.

Samples are produced through the actual proposal pass, not from the renderer's
ground truth directly, so every sample is a candidate the pipeline would really
have to classify - including the awkward ones near its detection limit. A
confuser the proposal pass never surfaces contributes no sample, which is
correct: the classifier is never asked about regions it does not see.

Samples come from the **ground-plane** rig (tests/ground.py), which is the one
the pipeline is actually evaluated on. An earlier version drew them from the
single-frame renderer instead, and the resulting train/test mismatch showed up
exactly where it would be expected to: a rendered shadow at 7.5 m was labelled
`wet_stain`. The safety-critical property held - a confuser was never called a
pothole - but the distinction between the two confuser classes was being made
on cues drawn from a different distribution than the one in use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pipeline.config import ProposeParams, RectifyParams
from pipeline.config import CameraGeometry
from pipeline.cues import extract, vector
from pipeline.propose import propose
from pipeline.rectify import rectify
from pipeline.types import Frame
from tests.ground import build_ground, render_frame
from tests.synthetic import SceneCamera

# A candidate is labelled with a rendered item when their bird's-eye centroids
# are within this distance, in metres.
LABEL_TOLERANCE_M = 0.40


@dataclass
class Sample:
    label: str
    features: np.ndarray
    cues: dict[str, float]
    distance_m: float


def _bev_point(source_point, homography) -> tuple[float, float]:
    import cv2

    point = np.array([[list(source_point)]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(point, homography)[0][0]
    return float(mapped[0]), float(mapped[1])


def _centroid(contour) -> tuple[float, float]:
    import cv2

    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        point = contour.reshape(-1, 2).mean(axis=0)
        return float(point[0]), float(point[1])
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def generate_samples(n_scenes: int = 90, seed: int = 0) -> list[Sample]:
    """Render varied scenes and collect labelled cue vectors from them."""
    import cv2

    camera = SceneCamera()
    rectify_params = RectifyParams(
        camera=CameraGeometry(camera.height_m, camera.vfov_deg)
    )
    propose_params = ProposeParams()
    rng = np.random.default_rng(seed)
    samples: list[Sample] = []

    for scene_index in range(n_scenes):
        scene_seed = 1000 + scene_index
        local = np.random.default_rng(scene_seed)

        # Features are placed in the 5.5-10.5 m band, which is where the
        # proposal pass actually surfaces them: closer than about 5.5 m a
        # region grows larger than the adaptive window and stops reading as
        # locally dark, and beyond about 10.5 m it falls under the resolution
        # floor. Sampling outside that band would train the classifier on
        # candidates it will never be shown.
        defects, confusers = [], []
        for _ in range(int(local.integers(0, 3))):
            defects.append(
                (
                    float(local.uniform(-1.3, 1.3)),
                    float(local.uniform(5.5, 10.5)),
                    float(local.uniform(0.15, 0.45)),
                )
            )
        for kind in ("shadow", "wet", "patch"):
            if local.random() < 0.7:
                confusers.append(
                    (
                        float(local.uniform(-1.3, 1.3)),
                        float(local.uniform(5.5, 10.5)),
                        float(local.uniform(0.18, 0.45)),
                        kind,
                    )
                )

        ground = build_ground(defects=defects, confusers=confusers, seed=scene_seed)
        image, truth = render_frame(ground, camera, 0.0, seed=scene_seed + 500)
        frame = Frame(
            frame_id=f"s{scene_index}",
            index=0,
            timestamp_s=0.0,
            sharpness=999.0,
            image=image,
            source="synthetic",
        )
        rectification, bev = rectify(frame, rectify_params)
        if bev is None or rectification.homography is None:
            continue

        candidates, _ = propose(bev, frame.frame_id, propose_params)
        if not candidates:
            continue

        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        valid = gray[gray > 0]
        road_median = float(np.median(valid)) if valid.size else 1.0

        # Every rendered item, with the label the classifier should produce.
        items: list[tuple[str, tuple[float, float], float]] = []
        for defect in truth["defects"]:
            items.append(("pothole", defect["centre_px"], defect["distance_m"]))
        for confuser in truth["confusers"]:
            label = {"shadow": "shadow", "wet": "wet_stain", "patch": "patch"}[
                confuser["kind"]
            ]
            items.append((label, confuser["centre_px"], confuser["distance_m"]))

        projected = [
            (label, _bev_point(point, rectification.homography), distance)
            for label, point, distance in items
        ]

        for candidate in candidates:
            if candidate.contour is None:
                continue
            centroid = _centroid(candidate.contour)
            best_label, best_distance_m, best_distance = None, 0.0, math.inf
            for label, point, distance_m in projected:
                dx = (point[0] - centroid[0]) * rectification.m_per_px_x
                dy = (point[1] - centroid[1]) * rectification.m_per_px_y
                separation = math.hypot(dx, dy)
                if separation < best_distance:
                    best_label, best_distance, best_distance_m = label, separation, distance_m
            if best_label is None or best_distance > LABEL_TOLERANCE_M:
                continue  # unmatched candidate: no reliable label, so not used

            cues = extract(bev, candidate, road_median)
            if cues is None:
                continue
            samples.append(
                Sample(
                    label=best_label,
                    features=vector(cues),
                    cues=cues,
                    distance_m=best_distance_m,
                )
            )

    return samples
