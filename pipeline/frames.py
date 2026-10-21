"""Stage 1 - frame sampling and sharpness gating.

Measuring a defect from a motion-blurred frame produces a confidently wrong
number, which is worse than no number. The Laplacian-variance gate drops those
frames before they reach any measurement stage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import FrameParams
from .types import Frame

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: the standard, cheap focus measure."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _fit_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= long_edge:
        return image
    scale = long_edge / longest
    # INTER_AREA is the correct filter for downscaling, and resize is one of the
    # three operations COOL accelerates on Graviton.
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _frame_id(source: str, index: int) -> str:
    digest = hashlib.sha1(f"{source}:{index}".encode()).hexdigest()[:10]
    return f"f_{digest}"


def sample_video(path: Path, params: FrameParams) -> Iterator[Frame]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, round(native_fps / max(params.sample_fps, 1e-6)))
        index = 0
        kept = 0
        while kept < params.max_frames:
            ok, image = capture.read()
            if not ok:
                break
            if index % stride == 0:
                image = _fit_long_edge(image, params.resize_long_edge)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                sharp = sharpness(gray)
                if sharp >= params.min_sharpness:
                    yield Frame(
                        frame_id=_frame_id(str(path), index),
                        index=index,
                        timestamp_s=index / native_fps,
                        sharpness=sharp,
                        image=image,
                        source=str(path),
                    )
                    kept += 1
            index += 1
    finally:
        capture.release()


def sample_images(path: Path, params: FrameParams) -> Iterator[Frame]:
    """Image directories are first-class input, not a fallback.

    Public road-damage datasets ship stills, so the pipeline treats an ordered
    directory of images as a sequence with synthetic timestamps.
    """
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    kept = 0
    for index, file in enumerate(files):
        if kept >= params.max_frames:
            break
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = _fit_long_edge(image, params.resize_long_edge)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharp = sharpness(gray)
        if sharp < params.min_sharpness:
            continue
        yield Frame(
            frame_id=_frame_id(str(file), 0),
            index=index,
            timestamp_s=index / max(params.sample_fps, 1e-6),
            sharpness=sharp,
            image=image,
            source=str(file),
        )
        kept += 1


def sample(source: str | Path, params: FrameParams) -> list[Frame]:
    """Sample frames from a video file, an image directory, or a single image."""
    path = Path(source)
    if path.is_dir():
        return list(sample_images(path, params))
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return list(sample_video(path, params))
    if suffix in IMAGE_SUFFIXES:
        return _single(path, params)
    raise ValueError(f"unsupported input: {path}")


def _single(path: Path, params: FrameParams) -> list[Frame]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    image = _fit_long_edge(image, params.resize_long_edge)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [
        Frame(
            frame_id=_frame_id(str(path), 0),
            index=0,
            timestamp_s=0.0,
            sharpness=sharpness(gray),
            image=image,
            source=str(path),
        )
    ]
