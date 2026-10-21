"""Evaluate the pipeline on RDD2022 — real road imagery.

The first question is not accuracy, it is **applicability**. Rectification needs
two lane markings that converge to a vanishing point, and a large share of real
roads - unmarked rural lanes, worn urban streets, anything photographed from a
motorbike - do not have them. If rectification fails on most real frames, then
no amount of measurement accuracy matters, and that is a finding worth more
than a decimal place on a figure from synthetic data.

So this reports, in order:

1. How often rectification succeeds, and why it fails when it does.
2. Detection recall against the annotated damage boxes, on the frames where it
   succeeded.
3. Whether the classifier's costed/not-costed decision agrees with the
   annotation class.

Area error is **not** reported here. The annotations are bounding boxes, not
measured areas, and deriving a truth area from a box would systematically
overestimate irregular defects. Real-world area error is established separately
by hand-measurement.

Run:  python -m eval.real_eval --root data/rdd2022/India --limit 200
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from eval.real_images import (
    CRACK_CLASSES,
    POTHOLE_CLASSES,
    RealImage,
    load_dataset,
    summarise,
)
from pipeline.classify import classify
from pipeline.config import (
    CameraGeometry,
    PipelineConfig,
    PrivacyParams,
    ProposeParams,
    RectifyParams,
)
from pipeline.propose import propose
from pipeline.rectify import rectify
from pipeline.types import Frame

COSTED = {"pothole", "crack"}


@dataclass
class ImageResult:
    name: str
    rectified: bool
    quality: float
    method: str
    note: str = ""
    n_annotations: int = 0
    n_candidates: int = 0
    matched: int = 0
    labels: list[str] = field(default_factory=list)
    annotation_labels: list[str] = field(default_factory=list)


def _bev_to_source(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float] | None:
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None
    mapped = cv2.perspectiveTransform(
        np.array([[[point[0], point[1]]]], dtype=np.float64), inverse
    )[0][0]
    return float(mapped[0]), float(mapped[1])


def evaluate_image(record: RealImage, config: PipelineConfig) -> ImageResult:
    image = record.load()
    if image is None:
        return ImageResult(record.name, False, 0.0, "unreadable", "cv2 could not decode")

    frame = Frame(record.name, 0, 0.0, 999.0, image, str(record.path))
    rectification, bev = rectify(frame, config.rectify)

    damage = record.damage
    result = ImageResult(
        name=record.name,
        rectified=bev is not None,
        quality=rectification.quality,
        method=rectification.method,
        note=rectification.notes,
        n_annotations=len(damage),
        annotation_labels=[a.label for a in damage],
    )
    if bev is None or rectification.homography is None:
        return result

    candidates, _ = propose(bev, record.name, config.propose)
    result.n_candidates = len(candidates)

    # A candidate matches an annotation when its centroid, mapped back into the
    # source image, falls inside the annotated box. Centroid-in-box rather than
    # IoU because the annotation is a loose box around an irregular defect, so
    # an IoU threshold would mostly measure how boxy the annotation is.
    matched_annotations: set[int] = set()
    for candidate in candidates:
        if candidate.contour is None:
            continue
        moments = cv2.moments(candidate.contour)
        if moments["m00"] <= 0:
            continue
        centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        source_point = _bev_to_source(centroid, rectification.homography)
        if source_point is None:
            continue
        classification = classify(bev, candidate)
        result.labels.append(classification.label)
        for index, annotation in enumerate(damage):
            if (
                annotation.xmin <= source_point[0] <= annotation.xmax
                and annotation.ymin <= source_point[1] <= annotation.ymax
            ):
                matched_annotations.add(index)
                break
    result.matched = len(matched_annotations)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/rdd2022/India")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--camera-height", type=float, default=0.0,
        help="metres. Enables calibration-only rectification, which does not need lane markings.",
    )
    parser.add_argument("--vfov", type=float, default=55.0)
    parser.add_argument("--pitch", type=float, default=5.0)
    args = parser.parse_args()

    records = load_dataset(args.root, limit=args.limit)
    stats = summarise(records)
    print("dataset sample")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Privacy scrubbing is off because RDD2022 is already scrubbed by its
    # authors - faces and plates were blurred before release.
    #
    # Calibration here is ASSUMED, not measured: this is someone else's dashcam
    # and its height, field of view and tilt are unknown. That is fine for the
    # questions this script asks - does rectification apply at all, and are
    # defects found - but it means any AREA derived from it is untrustworthy,
    # and none is reported.
    camera = None
    if args.camera_height > 0:
        camera = CameraGeometry(
            height_m=args.camera_height, vfov_deg=args.vfov, pitch_deg=args.pitch
        )
        print(
            f"\nassumed calibration: height {camera.height_m} m, vfov {camera.vfov_deg} deg, "
            f"pitch {camera.pitch_deg} deg"
        )
        print("  ASSUMED, not measured. Areas from this would not be trustworthy and are not reported.")

    config = PipelineConfig(
        rectify=RectifyParams(camera=camera),
        propose=ProposeParams(),
        privacy=PrivacyParams(enabled=False),
    )

    results = [evaluate_image(record, config) for record in records]

    rectified = [r for r in results if r.rectified]
    print(f"\n=== 1. Applicability: does a real frame yield usable geometry? ===")
    print(f"  rectified: {len(rectified)}/{len(results)} ({100*len(rectified)/max(len(results),1):.1f}%)")
    methods = Counter(r.method for r in rectified)
    if methods:
        print(f"  by method: {dict(methods)}")
    failure_reasons = Counter(
        r.note.replace("fallback: ", "") for r in results if not r.rectified
    )
    for reason, count in failure_reasons.most_common(5):
        print(f"    {count:4d}  {reason[:70]}")
    if rectified:
        qualities = np.array([r.quality for r in rectified])
        print(f"  quality on success: median {np.median(qualities):.3f} "
              f"range [{qualities.min():.3f}, {qualities.max():.3f}]")

    with_damage = [r for r in rectified if r.n_annotations]
    total_annotations = sum(r.n_annotations for r in with_damage)
    total_matched = sum(r.matched for r in with_damage)
    total_candidates = sum(r.n_candidates for r in rectified)
    print(f"\n=== 2. Detection, on frames that rectified ===")
    print(f"  frames with annotated damage: {len(with_damage)}")
    print(f"  annotated defects: {total_annotations}")
    print(f"  candidates proposed: {total_candidates}")
    if total_annotations:
        print(f"  recall (annotation hit by a candidate): {total_matched}/{total_annotations} "
              f"= {total_matched/total_annotations:.3f}")
    if total_candidates:
        print(f"  candidates landing on an annotation: {total_matched}/{total_candidates} "
              f"= {total_matched/total_candidates:.3f}")
        print("    (candidates off an annotation are not necessarily false: RDD2022")
        print("     annotates damage, not shadows, water or undamaged surface)")

    print(f"\n=== 3. Classification of proposed candidates ===")
    label_counts = Counter(label for r in rectified for label in r.labels)
    total_labels = sum(label_counts.values())
    for label, count in label_counts.most_common():
        marker = "costed" if label in COSTED else "not costed"
        print(f"  {label:10s} {count:5d}  {100*count/max(total_labels,1):5.1f}%  ({marker})")

    annotation_counts = Counter(
        label for r in results for label in r.annotation_labels
    )
    print(f"\n  annotated classes in the sample: {dict(annotation_counts)}")
    print(f"    D40 is pothole; D00/D10/D20 are cracks of different kinds")

    if args.out:
        payload = {
            "dataset": stats,
            "rectified": len(rectified),
            "total": len(results),
            "recall": (total_matched / total_annotations) if total_annotations else None,
            "results": [r.__dict__ for r in results],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
