"""Stage 0 - privacy scrubbing, before anything is persisted.

Dashcam footage of public roads incidentally records pedestrians and vehicle
registration plates. A system that uploads that to cloud storage and keeps it
has created a surveillance dataset as a side effect of counting potholes, so
scrubbing runs first, on the frame, before any artifact is written to disk or
to S3.

The module fails closed. If face blurring is enabled and the detector model is
missing, it raises rather than passing the frame through unblurred: silently
skipping the blur is the one outcome that must not be possible, because
nothing downstream would reveal that it happened.

Plates are handled geometrically rather than with a model. A plate is a bright,
high-contrast, strongly horizontal rectangle, and the detector is deliberately
over-inclusive - blurring a road sign costs nothing, missing a plate does.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import PrivacyParams

# Downloaded by scripts/fetch_models.sh from the OpenCV model zoo.
DEFAULT_FACE_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet.onnx"


class FaceModelMissing(RuntimeError):
    """Raised when face blurring is required but the model is unavailable."""


def _blur_region(image: np.ndarray, mask: np.ndarray, min_sigma: float) -> np.ndarray:
    """Blur wherever the mask is set, with a sigma scaled to the region size.

    A fixed kernel leaves a large face recognisable and destroys a small one's
    surroundings, so sigma is derived from the masked area and floored.
    """
    if cv2.countNonZero(mask) == 0:
        return image
    area = cv2.countNonZero(mask)
    sigma = max(min_sigma, float(np.sqrt(area)) / 6.0)
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    result = image.copy()
    result[mask > 0] = blurred[mask > 0]
    return result


def detect_faces(
    image: np.ndarray, params: PrivacyParams, model_path: Path | None = None
) -> list[tuple[int, int, int, int]]:
    """Detect faces with YuNet. Raises FaceModelMissing if the model is absent."""
    path = Path(model_path) if model_path else DEFAULT_FACE_MODEL
    if not path.exists():
        raise FaceModelMissing(
            f"face detection model not found at {path}. Run scripts/fetch_models.sh, "
            "or set privacy.enabled=False to acknowledge that frames will not be "
            "scrubbed. Face blurring is never skipped silently."
        )

    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(path),
        "",
        (width, height),
        params.face_score_threshold,
        params.face_nms_threshold,
        5000,
    )
    detector.setInputSize((width, height))
    _, detections = detector.detect(image)
    if detections is None:
        return []
    boxes = []
    for detection in detections:
        x, y, w, h = detection[:4].astype(int)
        boxes.append((max(0, x), max(0, y), int(w), int(h)))
    return boxes


def detect_plates(image: np.ndarray, params: PrivacyParams) -> list[tuple[int, int, int, int]]:
    """Find plate-like regions: bright, high-contrast, strongly horizontal.

    Over-inclusive within bounds. A missed plate is the failure that matters, so
    the filters are loose - but they are bounded above as well as below, because
    an unbounded privacy filter is not a safe one. Without an upper bound the
    morphological closing below merged the entire road surface into one
    966x346 px region that satisfied every other test, and blurring it took the
    lane markings with it, dropping rectification quality from 0.94 to 0.00.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # A plate's dark characters on a bright backing give it a dense cluster of
    # vertical edges, which is what separates it from a plain bright surface.
    gradient = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close horizontally so neighbouring characters merge into one plate block.
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    )

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = float(gray.shape[0] * gray.shape[1])
    max_area = params.plate_max_area_frac * frame_area
    max_width = params.plate_max_width_frac * gray.shape[1]

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0 or w * h < params.plate_min_area_px:
            continue
        # Bounded above: a region this large is scenery, not a plate, and
        # blurring it would destroy the features the pipeline needs.
        if w * h > max_area or w > max_width:
            continue
        aspect = w / h
        if not (params.plate_min_aspect <= aspect <= params.plate_max_aspect):
            continue
        boxes.append((x, y, w, h))
    return boxes


def scrub(
    image: np.ndarray,
    params: PrivacyParams,
    face_model_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Blur faces and plates. Returns the scrubbed frame and what was blurred.

    The counts are returned so a run can report how much it redacted, which is
    the evidence that scrubbing actually happened.
    """
    if not params.enabled:
        return image, {"faces": 0, "plates": 0, "enabled": 0}

    result = image
    mask = np.zeros(image.shape[:2], np.uint8)

    faces = detect_faces(image, params, face_model_path)
    for x, y, w, h in faces:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    plates = detect_plates(image, params)
    for x, y, w, h in plates:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    result = _blur_region(result, mask, params.min_blur_sigma)
    return result, {"faces": len(faces), "plates": len(plates), "enabled": 1}
