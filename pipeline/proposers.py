"""Pluggable proposal backends.

Measured on RDD2022, the classical proposal pass recalls 0.429 of annotated
damage in India and 0.092 in Czechia, and essentially none of the fine linear
cracking that dominates European roads. It finds regions that are darker than
their surroundings *over an area*, which suits potholes and alligator cracking
and does not suit a two-pixel crack.

That is a detection problem, and detection is the replaceable part. Everything
downstream - full-resolution refinement, multi-frame agreement, the vantage
filter, the resolution floor, deterministic costing, the approval gate - works
on a `Candidate` and does not care where it came from. So the proposal stage is
an interface with two implementations:

* `ClassicalProposer` - resize, adaptive threshold, contours. The default, and
  the workload the COOL benchmark measures, because those three operations are
  what COOL accelerates on Graviton.
* `DetectorProposer` - a trained detector supplies the regions, and the same
  segmentation turns each into a measurable contour.

Reporting both is the honest arrangement. The classical pass is what COOL
accelerates and what is benchmarked; a detector is what would make the product
deployable on real roads. Swapping the classical pass out for a model would
move the core per-frame workload onto that model, which is a real tension with
the COOL claim rather than something to paper over.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import ProposeParams, RefineParams
from .instrument import Timings
from .propose import propose as classical_propose
from .propose import valid_road_mask
from .types import Candidate


class Proposer(Protocol):
    """Produces measurable candidates from a rectified bird's-eye view."""

    name: str

    def propose(
        self,
        bev: np.ndarray,
        frame_id: str,
        params: ProposeParams,
        source_image: np.ndarray | None = None,
        homography: np.ndarray | None = None,
        timings: Timings | None = None,
    ) -> list[Candidate]: ...


@dataclass
class ClassicalProposer:
    """The COOL-accelerated pass: resize, adaptiveThreshold, findContours."""

    name: str = "classical"

    def propose(
        self,
        bev: np.ndarray,
        frame_id: str,
        params: ProposeParams,
        source_image: np.ndarray | None = None,
        homography: np.ndarray | None = None,
        timings: Timings | None = None,
    ) -> list[Candidate]:
        candidates, _ = classical_propose(bev, frame_id, params, timings)
        return candidates


# RDD2022's own classes, in the order the reference models emit them.
RDD_CLASSES = ("D00", "D10", "D20", "D40")
RDD_TO_SURFACE = {"D00": "crack", "D10": "crack", "D20": "crack", "D40": "pothole"}


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x, y, w, h in source-image pixels
    score: float
    class_id: int

    @property
    def label(self) -> str:
        return RDD_CLASSES[self.class_id] if self.class_id < len(RDD_CLASSES) else "unknown"


class DetectorProposer:
    """A trained detector proposes regions; segmentation makes them measurable.

    The detector runs on the **source image**, because that is what detectors
    are trained on, and its boxes are then mapped into the bird's-eye view
    through the same homography everything else uses. Within each mapped
    region the usual Otsu segmentation recovers a contour, so a detection
    becomes a `Candidate` indistinguishable from a classically proposed one and
    the whole measurement chain is unchanged.

    The model is run through OpenCV's own DNN module, so this adds no runtime
    dependency - the same `cv2` that does the vision does the inference, which
    on the Graviton worker means the COOL build.

    A detection is a region of interest, not a measurement. The box is never
    used as the defect's extent: a bounding box around an irregular pothole
    overstates its area badly, and area is the number this system exists to
    get right.
    """

    name = "detector"

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        refine_params: RefineParams | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"no detector model at {self.model_path}. Export one with:\n"
                "  pip install ultralytics && yolo export "
                "model=YOLOv8_Small_RDD.pt format=onnx imgsz=640"
            )
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.refine_params = refine_params or RefineParams()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run the model and decode YOLO-style output."""
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward()

        # Ultralytics exports (1, 4+nc, N); older exports are (1, N, 4+nc).
        predictions = np.squeeze(outputs)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        scale_x = width / self.input_size
        scale_y = height / self.input_size

        boxes, scores, class_ids = [], [], []
        for row in predictions:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.score_threshold:
                continue
            cx, cy, w, h = row[:4]
            boxes.append([
                int((cx - w / 2) * scale_x), int((cy - h / 2) * scale_y),
                int(w * scale_x), int(h * scale_y),
            ])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(boxes, scores, self.score_threshold, self.nms_threshold)
        if len(keep) == 0:
            return []
        return [
            Detection(tuple(boxes[i]), scores[i], class_ids[i])
            for i in np.array(keep).flatten()
        ]

    def propose(
        self,
        bev: np.ndarray,
        frame_id: str,
        params: ProposeParams,
        source_image: np.ndarray | None = None,
        homography: np.ndarray | None = None,
        timings: Timings | None = None,
    ) -> list[Candidate]:
        if source_image is None or homography is None:
            raise ValueError(
                "DetectorProposer needs the source image and its homography: "
                "the model runs on the source frame, not the bird's-eye view"
            )

        timings = timings or Timings()
        with timings.measure("detector_inference"):
            detections = self.detect(source_image)

        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY) if bev.ndim == 3 else bev
        road = valid_road_mask(bev)
        candidates: list[Candidate] = []

        for index, detection in enumerate(detections):
            x, y, w, h = detection.box
            if w <= 0 or h <= 0:
                continue
            # Map the box into the bird's-eye view. Only the lower part of a
            # box is the defect's footprint on the road - the upper part of a
            # box around a pothole is the road beyond it - so the mapped region
            # is taken from the box's lower portion.
            corners = np.array(
                [[[x, y + 0.35 * h]], [[x + w, y + 0.35 * h]],
                 [[x + w, y + h]], [[x, y + h]]], dtype=np.float64
            )
            mapped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
            bx, by, bw, bh = cv2.boundingRect(mapped.astype(np.int32))
            bx, by = max(0, bx), max(0, by)
            bw = min(bw, gray.shape[1] - bx)
            bh = min(bh, gray.shape[0] - by)
            if bw < 4 or bh < 4:
                continue

            region = gray[by : by + bh, bx : bx + bw]
            region_mask = road[by : by + bh, bx : bx + bw]
            if cv2.countNonZero(region_mask) < 16:
                continue

            # Same segmentation as refinement: an ROI holding one defect plus
            # its surround is bimodal, which is what Otsu is for.
            _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binary = cv2.bitwise_and(binary, region_mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            close = max(3, int(self.refine_params.morph_close_kernel) | 1)
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)),
            )

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            contour = contour.copy()
            contour[:, :, 0] += bx
            contour[:, :, 1] += by

            area_px = float(cv2.contourArea(contour))
            if area_px <= 0:
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            cx, cy, cw, ch = cv2.boundingRect(contour)
            blob = np.zeros((ch, cw), np.uint8)
            cv2.drawContours(blob, [contour], -1, 255, -1, offset=(-cx, -cy))
            mean_intensity = float(cv2.mean(gray[cy : cy + ch, cx : cx + cw], mask=blob)[0])

            digest = hashlib.sha1(f"{frame_id}:det:{index}:{detection.box}".encode()).hexdigest()[:10]
            candidates.append(
                Candidate(
                    candidate_id=f"c_{digest}",
                    frame_id=frame_id,
                    bbox=(int(cx), int(cy), int(cw), int(ch)),
                    area_px=area_px,
                    solidity=round(area_px / hull_area, 4) if hull_area > 0 else 0.0,
                    aspect_ratio=round(max(cw, ch) / max(min(cw, ch), 1), 4),
                    mean_intensity=round(mean_intensity, 2),
                    contour=contour,
                )
            )

        candidates.sort(key=lambda c: c.area_px, reverse=True)
        return candidates


def get_proposer(name: str = "classical", **kwargs) -> Proposer:
    if name == "classical":
        return ClassicalProposer()
    if name == "detector":
        return DetectorProposer(**kwargs)
    raise ValueError(f"unknown proposer: {name}")
