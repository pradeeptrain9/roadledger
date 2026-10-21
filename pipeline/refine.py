"""Stage 4 - ROI refinement at full resolution.

The proposal pass runs downscaled and systematically *under*-measures: the
adaptive threshold cuts inside a defect's soft edge, and the morphological
opening erodes it further. Measured on synthetic ground truth the bias is
around -7%, and consistently negative rather than random, which is precisely
the kind of error a second pass can remove.

So nothing is measured from the cheap pass. Each surviving candidate is
re-examined at full bird's-eye resolution inside its own ROI, where contrast
can be equalised locally (CLAHE) and the boundary recovered with a watershed
seeded by the distance transform. Running this only inside candidate ROIs is
what keeps the expensive pass affordable.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import RefineParams
from .types import Candidate, Measurement

# A defect whose footprint in the source image is smaller than this cannot have
# its boundary located well enough for the area to mean anything.
#
# Calibrated against ground truth rather than guessed. At a 231 px footprint
# the measured area was 39.5% low; at 572 px it was 12.4% low. The floor is set
# where the error stops being reportable, and the consequence is deliberate:
# a defect too distant or too small to size is refused outright and escalated,
# rather than being given a confident wrong number. Raising this floor makes
# the system admit ignorance more often, which is the correct direction.
MIN_SOURCE_FOOTPRINT_PX = 300.0

# A contour touching the edge of the bird's-eye view has been cut off by the
# frame, not by the defect. Its area is a truncation and measuring it anyway is
# worse than useless, because the truncation is silent and the number looks
# ordinary: on a drive-past at 6 fps the view taken as the pothole crossed the
# near edge reported 0.087 m2 against a truth of 0.363 - 76% low - and being
# the lowest of three it then set the median. The batch path never exposed
# this because it tracks only a few frames forward from first detection and
# stops before the defect reaches the edge; a live stream follows every defect
# all the way out of view, so it hits this on essentially every defect.
BORDER_MARGIN_PX = 2


def _roi_bounds(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, ...],
    pad_frac: float,
    min_pad_px: int = 12,
) -> tuple[int, int, int, int]:
    """Pad the proposal box, with a floor.

    A tight ROI around a small defect leaves the threshold operator almost no
    surround to compare against, so a minimum absolute pad is enforced as well
    as the proportional one.
    """
    x, y, w, h = bbox
    pad = max(min_pad_px, int(round(max(w, h) * pad_frac)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(shape[1], x + w + pad)
    y1 = min(shape[0], y + h + pad)
    return x0, y0, x1, y1


def _source_footprint_px(
    contour: np.ndarray, homography: np.ndarray | None
) -> float:
    """Area this bird's-eye contour occupied in the original image.

    The bird's-eye view magnifies the far field enormously, so a BEV area says
    nothing about how much real evidence underpins it. Mapping the contour back
    through the inverse homography gives the number of source pixels the
    measurement actually rests on.
    """
    if homography is None or contour is None or len(contour) < 3:
        return 0.0
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return 0.0
    points = contour.astype(np.float64).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(points, inverse)
    return float(abs(cv2.contourArea(mapped.astype(np.float32))))


def _touches_border(contour: np.ndarray, shape: tuple[int, ...], margin: int) -> bool:
    """Is any part of this contour against the edge of the view?"""
    xs = contour[:, :, 0]
    ys = contour[:, :, 1]
    return bool(
        xs.min() <= margin
        or ys.min() <= margin
        or xs.max() >= shape[1] - 1 - margin
        or ys.max() >= shape[0] - 1 - margin
    )


def _watershed_region(
    roi_colour: np.ndarray, binary: np.ndarray, seed: tuple[int, int], params: RefineParams
) -> np.ndarray | None:
    """Isolate the blob containing `seed`, splitting it from touching blobs."""
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if distance.max() <= 0:
        return None
    _, sure_foreground = cv2.threshold(
        distance, params.watershed_fg_frac * distance.max(), 255, cv2.THRESH_BINARY
    )
    sure_foreground = sure_foreground.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_background = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_background, sure_foreground)

    count, markers = cv2.connectedComponents(sure_foreground)
    if count < 2:
        return None
    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(roi_colour, markers)

    label = markers[seed[1], seed[0]]
    if label <= 1:
        # The seed landed on a watershed boundary or in background. Taking the
        # largest region instead would measure whichever blob happens to be
        # biggest, which is how a 0.07 m2 defect gets reported as 0.23 m2.
        return None
    return (markers == label).astype(np.uint8) * 255


def refine(
    bev: np.ndarray,
    candidate: Candidate,
    params: RefineParams,
    m_per_px_x: float,
    m_per_px_y: float,
    homography: np.ndarray | None = None,
) -> Measurement:
    """Measure one candidate at full resolution. One frame, one measurement."""
    gray = bev if bev.ndim == 2 else cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    colour = bev if bev.ndim == 3 else cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)

    x0, y0, x1, y1 = _roi_bounds(candidate.bbox, gray.shape, params.roi_pad_frac)
    roi = gray[y0:y1, x0:x1]
    roi_colour = colour[y0:y1, x0:x1]
    if roi.size == 0 or min(roi.shape[:2]) < 3:
        return Measurement(
            candidate_id=candidate.candidate_id,
            frame_id=candidate.frame_id,
            area_m2=0.0,
            perimeter_m=0.0,
            convexity=0.0,
            max_width_m=0.0,
            texture_energy=0.0,
            measurable=False,
            note="ROI degenerate",
        )

    # Local contrast equalisation. A defect in shade and one in sunlight have
    # very different absolute intensities but similar local structure.
    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip_limit,
        tileGridSize=(params.clahe_tile_grid, params.clahe_tile_grid),
    )
    equalised = clahe.apply(roi)

    if params.use_otsu:
        _, binary = cv2.threshold(
            equalised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        # The adaptive window must stay well inside the ROI or it degenerates.
        usable = max(3, (min(roi.shape[:2]) // 3) | 1)
        block_size = max(3, min(int(params.block_size) | 1, usable))
        binary = cv2.adaptiveThreshold(
            equalised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            params.c_constant,
        )

    # Remove speckle, then reconnect the interior. Both matter and in this
    # order: opening clears isolated noise, closing bridges the holes that a
    # rough defect interior leaves behind, which would otherwise fragment the
    # defect and leave the seeded component measuring a fraction of it.
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    if params.morph_close_kernel > 1:
        size = max(3, int(params.morph_close_kernel) | 1)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        )

    seed_x = int(np.clip(candidate.bbox[0] + candidate.bbox[2] // 2 - x0, 0, roi.shape[1] - 1))
    seed_y = int(np.clip(candidate.bbox[1] + candidate.bbox[3] // 2 - y0, 0, roi.shape[0] - 1))

    # Take the component under the seed first. Watershed is only consulted if
    # that component is large enough to suggest two defects have merged.
    count, labels = cv2.connectedComponents(binary)
    seed_label = labels[seed_y, seed_x]
    region: np.ndarray | None = None
    if seed_label != 0:
        region = (labels == seed_label).astype(np.uint8) * 255
        seeded_area = float(cv2.countNonZero(region))
        merged = (
            candidate.area_px > 0
            and seeded_area > params.watershed_trigger_growth * candidate.area_px
        )
        if params.use_watershed and merged:
            split = _watershed_region(roi_colour, binary, (seed_x, seed_y), params)
            if split is not None:
                region = split

    if region is None:
        label = seed_label
        if label == 0:
            return Measurement(
                candidate_id=candidate.candidate_id,
                frame_id=candidate.frame_id,
                area_m2=0.0,
                perimeter_m=0.0,
                convexity=0.0,
                max_width_m=0.0,
                texture_energy=0.0,
                measurable=False,
                note="refined threshold did not retain the candidate",
            )

    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return Measurement(
            candidate_id=candidate.candidate_id,
            frame_id=candidate.frame_id,
            area_m2=0.0,
            perimeter_m=0.0,
            convexity=0.0,
            max_width_m=0.0,
            texture_energy=0.0,
            measurable=False,
            note="no contour after refinement",
        )
    contour = max(contours, key=cv2.contourArea)

    # Back into bird's-eye coordinates: every measurement below is expressed in
    # that one frame of reference.
    contour = contour.copy()
    contour[:, :, 0] += x0
    contour[:, :, 1] += y0

    stage = "refine"
    notes: list[str] = []

    # Refinement is allowed to sharpen a boundary, not to discover a different,
    # much larger defect. Unbounded growth means the region leaked into the
    # surrounding surface, and the biased-but-bounded proposal is then the
    # better measurement. Caught a 0.07 m2 defect being reported as 0.23 m2.
    refined_area_px = float(cv2.contourArea(contour))
    if (
        candidate.area_px > 0
        and refined_area_px > params.max_area_growth * candidate.area_px
        and candidate.contour is not None
    ):
        notes.append(
            f"refinement grew the region {refined_area_px / candidate.area_px:.1f}x beyond the "
            f"proposal and was rejected; reporting the proposal boundary instead"
        )
        contour = candidate.contour
        stage = "propose_fallback"

    # Moments give the area of the filled region; contourArea on the same
    # contour agrees, and m00 is used so centroid and area come from one source.
    moments = cv2.moments(contour)
    area_px = float(moments["m00"])
    if area_px <= 0:
        area_px = float(cv2.contourArea(contour))

    perimeter_px = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_perimeter = float(cv2.arcLength(hull, True))
    convexity = hull_perimeter / perimeter_px if perimeter_px > 0 else 0.0

    _, radius_px = cv2.minEnclosingCircle(contour)

    # Anisotropic scales: one bird's-eye pixel is not square in metres.
    area_m2 = area_px * m_per_px_x * m_per_px_y
    mean_m_per_px = (m_per_px_x + m_per_px_y) / 2.0
    perimeter_m = perimeter_px * mean_m_per_px
    max_width_m = 2.0 * radius_px * mean_m_per_px

    texture_energy = float(cv2.Laplacian(roi, cv2.CV_64F).var())

    footprint = _source_footprint_px(contour, homography)
    measurable = footprint >= MIN_SOURCE_FOOTPRINT_PX or homography is None
    if not measurable:
        notes.append(
            f"footprint {footprint:.0f} source px is below the {MIN_SOURCE_FOOTPRINT_PX:.0f} px "
            "floor; too distant to size reliably"
        )

    # Clipping is checked last and overrides everything above: a truncated
    # boundary makes the area meaningless however good the footprint is.
    if _touches_border(contour, gray.shape, BORDER_MARGIN_PX):
        measurable = False
        notes.append(
            "the boundary reaches the edge of the rectified view, so the defect is cut off "
            "rather than enclosed and its area is a truncation, not a measurement"
        )

    return Measurement(
        candidate_id=candidate.candidate_id,
        frame_id=candidate.frame_id,
        area_m2=round(area_m2, 6),
        perimeter_m=round(perimeter_m, 4),
        convexity=round(convexity, 4),
        max_width_m=round(max_width_m, 4),
        texture_energy=round(texture_energy, 2),
        stage=stage,
        source_footprint_px=round(footprint, 1),
        measurable=measurable,
        note="; ".join(notes),
    )
