"""Every tunable in the pipeline, in one place.

The agent adjusts pipeline behaviour by producing modified copies of these
dataclasses (via `dataclasses.replace`) and re-running a stage. Nothing in the
pipeline reads a magic number that is not declared here, so an agent trace can
always be replayed from the recorded parameter set.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

SurfaceClass = Literal["pothole", "crack", "patch", "shadow", "wet_stain", "unknown"]


@dataclass(frozen=True)
class CameraGeometry:
    """Optional true camera calibration.

    Without this, the longitudinal (depth) scale of the bird's-eye view is an
    assumption and is flagged as such on every measurement. With it, both the
    lateral and longitudinal scales follow from pinhole geometry over a flat
    road, and the lane-width prior becomes an independent cross-check rather
    than the only source of scale.
    """

    height_m: float          # camera height above the road surface
    vfov_deg: float          # vertical field of view of the sensor
    # Downward tilt. A dashcam is usually aimed slightly down, which moves the
    # horizon up the frame; zero means the optical axis is level.
    pitch_deg: float = 0.0

    def focal_px(self, image_height_px: int) -> float:
        import math

        return (image_height_px / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)

    def horizon_px(self, image_height_px: int) -> float:
        """Image row of the horizon, from the calibration alone.

        This is what makes rectification possible without lane markings: the
        horizon is where the vanishing point of the road plane must lie, and
        for a calibrated camera it follows from the pitch rather than from
        anything visible in the scene.
        """
        import math

        return image_height_px / 2.0 - self.focal_px(image_height_px) * math.tan(
            math.radians(self.pitch_deg)
        )


@dataclass(frozen=True)
class RectifyParams:
    """Road-plane rectification to a metric bird's-eye view."""

    # Lane-marking detection feeding the vanishing-point estimate.
    canny_low: int = 60
    canny_high: int = 180
    hough_threshold: int = 55
    hough_min_line_length: int = 60
    hough_max_line_gap: int = 25

    # Only lines in this absolute-slope band can be lane markings receding to
    # the horizon; near-horizontal lines are usually shadows or road furniture.
    min_abs_slope: float = 0.35
    max_abs_slope: float = 6.0

    # Metric scale prior. Indian/US urban lane widths cluster near 3.5 m; this
    # is the single assumption that converts pixels to metres, so it is
    # reported alongside every measurement rather than buried.
    lane_width_m: float = 3.5

    # Bird's-eye output geometry.
    bev_width_px: int = 640
    bev_height_px: int = 640
    bev_metres_across: float = 7.0  # two lanes wide
    bev_metres_ahead: float = 14.0

    # Fraction of image height, measured up from the bottom, that is treated as
    # near-field road surface worth rectifying.
    road_band_frac: float = 0.45

    # Supplying this promotes the depth scale from assumption to measurement.
    camera: "CameraGeometry | None" = None

    # Lateral scale from the lane prior and from camera geometry should agree.
    # Disagreement beyond this fraction means one of them is wrong, and the
    # agent is told so rather than silently trusting either.
    max_scale_disagreement: float = 0.25

    @property
    def m_per_px_x(self) -> float:
        return self.bev_metres_across / self.bev_width_px

    @property
    def m_per_px_y(self) -> float:
        return self.bev_metres_ahead / self.bev_height_px

    @property
    def m2_per_px(self) -> float:
        return self.m_per_px_x * self.m_per_px_y


@dataclass(frozen=True)
class ProposeParams:
    """Cheap first pass: the COOL-accelerated core workload.

    Built deliberately from the three operations COOL names as accelerated on
    Graviton - resize, adaptive Gaussian threshold, contour detection - so the
    benchmark measures the workload that actually carries the product.
    """

    # Downscale factor applied before thresholding. 0.5 means the proposal pass
    # runs at quarter the pixel count.
    scale: float = 0.5

    # adaptiveThreshold(ADAPTIVE_THRESH_GAUSSIAN_C). Must be odd and > 1.
    block_size: int = 51
    c_constant: float = 7.0

    morph_open_kernel: int = 3
    morph_close_kernel: int = 7

    # Candidate geometry filters, in bird's-eye pixels at full BEV scale.
    min_area_px: float = 180.0
    max_area_px: float = 90_000.0
    min_solidity: float = 0.35
    max_aspect_ratio: float = 8.0

    def normalised(self) -> "ProposeParams":
        """adaptiveThreshold rejects even or <3 block sizes; clamp rather than crash."""
        bs = max(3, int(self.block_size) | 1)
        return replace(self, block_size=bs)


@dataclass(frozen=True)
class RefineParams:
    """Expensive second pass, run only inside candidate ROIs at full resolution."""

    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8

    # Otsu, not adaptive thresholding, inside the ROI. The ROI is by
    # construction one defect plus its surround - a genuinely bimodal problem,
    # which is what Otsu is for. An adaptive window sized comparably to the ROI
    # degenerates and floods the region.
    use_otsu: bool = True
    # Only used when Otsu is disabled.
    block_size: int = 31
    c_constant: float = 5.0

    # Refinement may sharpen a boundary; it must not discover a defect several
    # times larger than the one proposed. Beyond this growth the refinement is
    # rejected in favour of the proposal, which is biased but bounded.
    max_area_growth: float = 2.5

    # Closing kernel applied after thresholding, in bird's-eye pixels.
    #
    # A pothole interior is not uniformly dark - it is dark *and rough*, so
    # thresholding leaves speckle holes that fragment it. Measured on one
    # near-field 0.50 m2 defect, the mask broke into seven components and the
    # largest alone accounted for only 64% of the true area. Closing at 9 px
    # reconnects the interior and brings that to 95.5%.
    morph_close_kernel: int = 9

    use_watershed: bool = True
    # Distance-transform fraction used to seed sure-foreground for watershed.
    watershed_fg_frac: float = 0.45
    # Watershed exists to split two defects that touch, so it is only worth its
    # risk when there is evidence of that - a seeded region much larger than
    # the proposal. Running it unconditionally on a fragmented mask was
    # catastrophic: the distance transform seeded every speckle, watershed
    # split the defect into scraps, and the seed landed in a five-pixel one,
    # reporting a 0.50 m2 pothole as 0.003 m2.
    watershed_trigger_growth: float = 1.6

    # Padding added around the proposal box before refining, as a fraction of
    # the box's larger side, so the boundary is not clipped by a tight proposal.
    roi_pad_frac: float = 0.25


@dataclass(frozen=True)
class TrackParams:
    """Multi-frame association: independent measurements of the same defect."""

    # How many additional frames to pull on the first tracking attempt, and the
    # larger value the agent escalates to when agreement is poor.
    k_frames: int = 4
    k_frames_escalated: int = 10

    lk_win_size: int = 21
    lk_max_level: int = 3
    lk_max_corners: int = 400
    lk_quality_level: float = 0.01
    lk_min_distance: int = 7

    # A defect must be measured in at least this many frames before its spread
    # is treated as meaningful.
    min_measurements: int = 3
    # Forward-backward LK error above this (pixels) discards a tracked point.
    max_fb_error_px: float = 2.0


@dataclass(frozen=True)
class ScoreConfig:
    """Deterministic severity, priority and cost. No model touches these numbers."""

    # Repair unit rates, currency per square metre, by surface class.
    unit_rate_per_m2: dict[str, float] = field(
        default_factory=lambda: {
            "pothole": 1850.0,
            "crack": 420.0,
            "patch": 0.0,  # already repaired; recorded, never billed
        }
    )
    currency: str = "INR"

    # Minimum billable area - crews do not mobilise for a 2 cm blemish.
    min_billable_m2: float = 0.02
    # Fixed mobilisation charge added to any emitted work order.
    mobilisation_cost: float = 650.0

    # Priority 1 (urgent) at or above this area, priority 2 above the next.
    priority1_min_m2: float = 0.25
    priority2_min_m2: float = 0.08


@dataclass(frozen=True)
class AgentThresholds:
    """The numbers that make the loop agentic.

    Each threshold converts a vision measurement into a decision about which
    tool to call next. They are the contract between the pipeline and the agent.
    """

    # Rectification below this quality score is not trustworthy enough to
    # measure from; the agent re-rectifies or skips the frame.
    homography_min_quality: float = 0.35

    # More candidates than this in one frame means the threshold is too loose -
    # the agent raises block_size/C and re-proposes rather than refining noise.
    max_candidates: int = 25
    # Fewer than this, with a frame that looked textured, suggests too tight.
    min_candidates_for_textured_frame: int = 1

    # Relative spread (IQR / median) of the per-frame area measurements above
    # which the measurement is not yet trustworthy.
    max_area_iqr_ratio: float = 0.20
    # Spread still above this after escalated tracking goes to a human.
    escalate_to_human_iqr_ratio: float = 0.35

    # Top-class probability below this is an ambiguous classification.
    min_class_confidence: float = 0.55

    # Priority-1 work orders always require human approval, regardless of
    # confidence. Fail closed: no measurement, no work order.
    always_approve_priority: int = 1

    # Hard ceiling on agent tool calls per video, so a confused loop cannot
    # burn the compute budget.
    max_tool_calls: int = 60


@dataclass(frozen=True)
class FrameParams:
    """Frame sampling and quality gating."""

    sample_fps: float = 2.0
    # Laplacian variance below this is too blurry to measure from.
    min_sharpness: float = 55.0
    max_frames: int = 400
    resize_long_edge: int = 1280


@dataclass(frozen=True)
class PrivacyParams:
    """Faces and plates are blurred before anything is persisted."""

    enabled: bool = True
    face_score_threshold: float = 0.55
    face_nms_threshold: float = 0.3
    # Gaussian sigma is derived from the region size so small faces are still
    # unrecoverable; this is the floor.
    min_blur_sigma: float = 6.0
    # Plates are not detected by YuNet; they are found as bright high-contrast
    # horizontal rectangles, then blurred. Over-inclusive on purpose - a blurred
    # road sign costs nothing, a missed plate does - but bounded ABOVE as well
    # as below.
    #
    # The upper bounds are not cosmetic. Without them the morphological closing
    # merged the whole road's edge structure into a single 966x346 px region
    # with aspect 2.8, which passed every filter, and blurring it destroyed the
    # lane markings that rectification depends on: quality fell from 0.94 to
    # 0.00 with privacy enabled. An over-inclusive privacy filter is not
    # automatically a safe one.
    plate_min_aspect: float = 2.0
    plate_max_aspect: float = 6.0
    plate_min_area_px: int = 400
    # A real plate is a small part of a frame. Anything larger is scenery.
    plate_max_area_frac: float = 0.02
    plate_max_width_frac: float = 0.25


@dataclass(frozen=True)
class PipelineConfig:
    """The full parameter set for one run. Serialised into every trace record."""

    frames: FrameParams = field(default_factory=FrameParams)
    privacy: PrivacyParams = field(default_factory=PrivacyParams)
    rectify: RectifyParams = field(default_factory=RectifyParams)
    propose: ProposeParams = field(default_factory=ProposeParams)
    refine: RefineParams = field(default_factory=RefineParams)
    track: TrackParams = field(default_factory=TrackParams)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    agent: AgentThresholds = field(default_factory=AgentThresholds)


DEFAULT = PipelineConfig()
