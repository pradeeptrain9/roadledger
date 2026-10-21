"""Labelled cue samples from RDD2022 — real road imagery.

The classifier fitted on synthetic cues does not survive contact with real
roads. Measured on 130 annotated India frames it labelled **35 of 45** detected
potholes as "shadow", and only 2 as pothole. The cue extractor is physics and
transfers; the decision boundary is statistics fitted to one renderer's noise
model, and it does not.

Labels come from the dataset's own annotations by weak supervision:

* a candidate whose centroid falls inside a D40 box is a **pothole**;
* inside a D00/D10/D20 box, a **crack**;
* on an annotated frame but far from every annotation, **not damage**.

The last rule is the one that needs care. RDD2022 annotates damage, not
shadows, water or sound surface, so an unannotated region is not guaranteed to
be undamaged - it is only unlabelled. A margin is therefore enforced: a
candidate near but not inside an annotation is discarded rather than called
negative, so that a slightly misplaced box does not manufacture a false
negative. Frames with no annotations at all are skipped entirely, since on
those an absent label means nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from eval.real_images import CRACK_CLASSES, POTHOLE_CLASSES, RealImage, load_dataset
from pipeline.config import CameraGeometry, ProposeParams, RectifyParams
from pipeline.cues import extract, vector
from pipeline.propose import propose
from pipeline.rectify import rectify
from pipeline.types import Frame

# A candidate within this many pixels of an annotation, but not inside it, is
# too ambiguous to label either way.
AMBIGUITY_MARGIN_PX = 60


@dataclass
class RealSample:
    label: str
    features: np.ndarray
    cues: dict[str, float]
    image_name: str
    source: str


def _expand(box: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return x0 - margin, y0 - margin, x1 + margin, y1 + margin


def generate_real_samples(
    root: str = "data/rdd2022/India",
    limit: int = 300,
    camera: CameraGeometry | None = None,
    propose_params: ProposeParams | None = None,
) -> list[RealSample]:
    camera = camera or CameraGeometry(1.25, 55.0, 5.0)
    rectify_params = RectifyParams(camera=camera)
    propose_params = propose_params or ProposeParams()

    samples: list[RealSample] = []
    for record in load_dataset(root, limit=limit):
        # No annotations means no usable label, positive or negative.
        if not record.annotations:
            continue
        image = record.load()
        if image is None:
            continue

        frame = Frame(record.name, 0, 0.0, 999.0, image, str(record.path))
        rectification, bev = rectify(frame, rectify_params)
        if bev is None or rectification.homography is None:
            continue

        candidates, _ = propose(bev, record.name, propose_params)
        if not candidates:
            continue
        try:
            inverse = np.linalg.inv(rectification.homography)
        except np.linalg.LinAlgError:
            continue

        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        valid = gray[gray > 0]
        road_median = float(np.median(valid)) if valid.size else 1.0

        boxes = [
            (a.label, (a.xmin, a.ymin, a.xmax, a.ymax)) for a in record.annotations
        ]

        for candidate in candidates:
            if candidate.contour is None:
                continue
            moments = cv2.moments(candidate.contour)
            if moments["m00"] <= 0:
                continue
            point = cv2.perspectiveTransform(
                np.array([[[moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]]]),
                inverse,
            )[0][0]

            label = None
            ambiguous = False
            for annotation_label, box in boxes:
                x0, y0, x1, y1 = box
                if x0 <= point[0] <= x1 and y0 <= point[1] <= y1:
                    if annotation_label in POTHOLE_CLASSES:
                        label = "pothole"
                    elif annotation_label in CRACK_CLASSES:
                        label = "crack"
                    else:
                        ambiguous = True  # D43/D44/D50: not damage, not clean either
                    break
                ex0, ey0, ex1, ey1 = _expand(box, AMBIGUITY_MARGIN_PX)
                if ex0 <= point[0] <= ex1 and ey0 <= point[1] <= ey1:
                    ambiguous = True

            if ambiguous:
                continue
            if label is None:
                label = "not_damage"

            cues = extract(bev, candidate, road_median)
            if cues is None:
                continue
            samples.append(
                RealSample(
                    label=label,
                    features=vector(cues),
                    cues=cues,
                    image_name=record.name,
                    source=root,
                )
            )
    return samples
