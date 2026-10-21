"""Load and evaluate against RDD2022 — real road imagery.

Every quantitative claim before this file came from the synthetic renderer.
This is where the pipeline meets real roads, and the first thing to establish
is not accuracy but *applicability*: how often does a real frame even yield
usable road geometry? Rectification depends on finding two lane markings that
converge, and a great many real roads - particularly unmarked rural and urban
roads - simply do not have them.

What RDD2022 can and cannot validate:

* **Can**: whether rectification succeeds at all; detection recall against
  annotated damage; whether the classifier calls damage damage and non-damage
  non-damage.
* **Cannot**: area accuracy in square metres. The annotations are Pascal VOC
  bounding boxes with a damage class, not measured areas. Deriving a truth area
  from box dimensions would systematically overestimate irregular defects, so
  area error against real imagery is established separately by hand-measurement
  (see eval/measure_tool.py), not from these labels.
* **Cannot**: anything multi-frame. RDD2022 is stills, so `iqr_ratio`,
  ego-motion, dedupe and the vantage filter have no real data here. Those need
  video.

Damage classes (from the dataset's own label map):
  D00 longitudinal crack · D10 transverse crack · D20 alligator crack
  D40 pothole · Repair (a completed repair)

Note the dataset is already privacy-scrubbed by its authors - faces and plates
were blurred before release - so our own scrubbing is exercised separately.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# The dataset's classes, mapped onto what this system costs.
DAMAGE_CLASSES = {"D00", "D10", "D20", "D40"}
POTHOLE_CLASSES = {"D40"}
CRACK_CLASSES = {"D00", "D10", "D20"}
REPAIR_CLASSES = {"Repair"}


@dataclass
class Annotation:
    """One annotated defect: a box and a class."""

    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.xmin, self.ymin, self.xmax - self.xmin, self.ymax - self.ymin

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def is_damage(self) -> bool:
        return self.label in DAMAGE_CLASSES


@dataclass
class RealImage:
    """One RDD2022 image with its annotations."""

    path: Path
    annotations: list[Annotation] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def damage(self) -> list[Annotation]:
        return [a for a in self.annotations if a.is_damage]

    @property
    def potholes(self) -> list[Annotation]:
        return [a for a in self.annotations if a.label in POTHOLE_CLASSES]

    def load(self) -> np.ndarray | None:
        return cv2.imread(str(self.path), cv2.IMREAD_COLOR)


def parse_annotation(xml_path: Path) -> list[Annotation]:
    """Parse one Pascal VOC file. Malformed boxes are skipped, not guessed."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    annotations = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            continue
        try:
            xmin = int(float(box.findtext("xmin") or 0))
            ymin = int(float(box.findtext("ymin") or 0))
            xmax = int(float(box.findtext("xmax") or 0))
            ymax = int(float(box.findtext("ymax") or 0))
        except (TypeError, ValueError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        annotations.append(Annotation(name, xmin, ymin, xmax, ymax))
    return annotations


def find_dataset(root: Path) -> tuple[Path, Path] | None:
    """Locate the images and annotations directories under an extracted subset.

    The archives nest differently per country, so the directories are found
    rather than assumed.
    """
    images_dir = annotations_dir = None
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        lowered = path.name.lower()
        if lowered in {"images", "jpegimages"} and images_dir is None:
            images_dir = path
        elif lowered == "annotations" and annotations_dir is None:
            # The xmls sometimes sit one level further down.
            nested = path / "xmls"
            annotations_dir = nested if nested.is_dir() else path
    if images_dir is None:
        return None
    return images_dir, (annotations_dir or images_dir)


def load_dataset(
    root: str | Path, limit: int | None = None, seed: int = 0, require_damage: bool = False
) -> list[RealImage]:
    """Load images with their annotations, optionally a random subset."""
    root = Path(root)
    located = find_dataset(root)
    if located is None:
        raise FileNotFoundError(f"no images directory found under {root}")
    images_dir, annotations_dir = located

    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise FileNotFoundError(f"no images in {images_dir}")

    records = []
    for image_path in files:
        xml_path = annotations_dir / f"{image_path.stem}.xml"
        annotations = parse_annotation(xml_path) if xml_path.exists() else []
        if require_damage and not any(a.is_damage for a in annotations):
            continue
        records.append(RealImage(image_path, annotations))

    if limit is not None and len(records) > limit:
        # Sampled with a fixed seed rather than taking the first N: the files
        # are in capture order, so the first N would be one stretch of one road
        # in one lighting condition.
        random.Random(seed).shuffle(records)
        records = records[:limit]
    return records


def summarise(records: list[RealImage]) -> dict:
    """What is actually in this sample."""
    counts: dict[str, int] = {}
    for record in records:
        for annotation in record.annotations:
            counts[annotation.label] = counts.get(annotation.label, 0) + 1
    with_damage = sum(1 for r in records if r.damage)
    with_pothole = sum(1 for r in records if r.potholes)
    return {
        "images": len(records),
        "images_with_damage": with_damage,
        "images_with_pothole": with_pothole,
        "images_without_annotation": sum(1 for r in records if not r.annotations),
        "instances_by_class": dict(sorted(counts.items())),
    }
