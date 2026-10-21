"""Invariants the pipeline must hold.

These are the properties that would make the output unsafe to act on if they
broke, not a restatement of current numbers. Several of them encode bugs that
were actually found and fixed, so they are regression tests in the literal
sense.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from eval.accuracy import default_config, measure_one
from pipeline.classify import classify, load_model
from pipeline.config import (
    AgentThresholds,
    CameraGeometry,
    PipelineConfig,
    PrivacyParams,
    ProposeParams,
    RectifyParams,
    ScoreConfig,
    TrackParams,
)
from pipeline.privacy import FaceModelMissing, detect_plates, scrub
from pipeline.propose import paint_threshold, propose
from pipeline.rectify import rectify
from pipeline.runner import Session
from pipeline.score import score_defect
from pipeline.track import aggregate, bev_shift_lk
from pipeline.types import Aggregate, Classification, Frame
from tests.ground import build_ground, render_drive, render_frame
from tests.synthetic import SceneCamera, render_road


@pytest.fixture(scope="module")
def camera() -> SceneCamera:
    return SceneCamera()


@pytest.fixture(scope="module")
def config(camera: SceneCamera) -> PipelineConfig:
    return default_config(camera)


# -- rectification -----------------------------------------------------------


def test_vanishing_point_recovered_on_consistent_scene(camera):
    image, truth = render_road(camera)
    frame = Frame("f", 0, 0.0, 999.0, image, "synthetic")
    rectification, bev = rectify(
        frame, RectifyParams(camera=CameraGeometry(camera.height_m, camera.vfov_deg))
    )
    assert bev is not None
    truth_vp = truth["vanishing_point"]
    error = math.hypot(
        rectification.vanishing_point[0] - truth_vp[0],
        rectification.vanishing_point[1] - truth_vp[1],
    )
    assert error < 5.0, f"vanishing point off by {error:.1f} px"


def test_scale_cross_check_flags_a_wrong_camera_height(camera):
    """Two independent routes to scale must disagree loudly when one is wrong."""
    image, _ = render_road(camera)
    frame = Frame("f", 0, 0.0, 999.0, image, "synthetic")

    good, _ = rectify(
        frame, RectifyParams(camera=CameraGeometry(camera.height_m, camera.vfov_deg))
    )
    assert "agrees" in good.notes

    bad, _ = rectify(frame, RectifyParams(camera=CameraGeometry(2.6, camera.vfov_deg)))
    assert "disagree" in bad.notes


def test_rectification_fails_closed_without_lane_geometry(camera):
    """No geometry must mean no bird's-eye view, not a guessed one."""
    blank = np.full((camera.height, camera.width, 3), 70, np.uint8)
    frame = Frame("f", 0, 0.0, 999.0, blank, "synthetic")
    rectification, bev = rectify(frame, RectifyParams())
    assert bev is None
    assert rectification.quality == 0.0
    assert rectification.method == "static_fallback"


def test_depth_scale_is_linear_in_ground_distance(camera):
    """Bird's-eye y must be linear in forward distance, or areas are nonsense."""
    from tests.ground import ground_to_image_homography

    ground = build_ground(seed=5)
    image, _ = render_frame(ground, camera, 0.0, seed=1)
    frame = Frame("f", 0, 0.0, 999.0, image, "synthetic")
    rectification, _ = rectify(
        frame, RectifyParams(camera=CameraGeometry(camera.height_m, camera.vfov_deg))
    )
    homography = ground_to_image_homography(ground, camera, 0.0)

    rows = []
    for forward in (3.0, 5.0, 7.0, 9.0, 12.0):
        u, v = ground.to_px(0.0, forward)
        image_point = cv2.perspectiveTransform(
            np.array([[[u, v]]], dtype=np.float64), homography
        )[0][0]
        bev_point = cv2.perspectiveTransform(
            np.array([[[image_point[0], image_point[1]]]], dtype=np.float64),
            rectification.homography,
        )[0][0]
        rows.append((forward, bev_point[1]))

    array = np.array(rows)
    design = np.vstack([array[:, 0], np.ones(len(array))]).T
    slope, intercept = np.linalg.lstsq(design, array[:, 1], rcond=None)[0]
    residual = np.abs(array[:, 1] - (slope * array[:, 0] + intercept)).max()
    assert residual < 3.0, f"bird's-eye depth is nonlinear: {residual:.2f} px residual"

    actual = abs(1.0 / slope)
    assert abs(actual - rectification.m_per_px_y) / actual < 0.1


# -- proposal ----------------------------------------------------------------


def test_paint_threshold_is_robust_to_surface_noise():
    """A ratio test flags a fifth of noisy asphalt as paint; this must not."""
    rng = np.random.default_rng(0)
    asphalt = np.clip(rng.normal(72, 6, 20000), 0, 255)
    threshold = paint_threshold(asphalt)
    assert threshold > asphalt.max() * 0.9
    assert (asphalt > threshold).mean() < 0.01


def test_no_spurious_candidates_beside_lane_paint(camera, config):
    """The adaptive-threshold halo around bright paint is not a defect.

    Regression: unmasked, this produced two 7 m2 "defects" hugging the lane
    markings, each larger than any real pothole in the scene.
    """
    ground = build_ground(defects=[], seed=5)
    image, _ = render_frame(ground, camera, 0.0, seed=1)
    frame = Frame("f", 0, 0.0, 999.0, image, "synthetic")
    rectification, bev = rectify(frame, config.rectify)
    assert bev is not None
    candidates, _ = propose(bev, "f", ProposeParams())
    m2_per_px = rectification.m_per_px_x * rectification.m_per_px_y
    huge = [c for c in candidates if c.area_px * m2_per_px > 1.0]
    assert not huge, f"{len(huge)} implausibly large candidates on a clean road"


# -- classification ----------------------------------------------------------


def test_classifier_model_matches_the_cue_contract():
    """A model fitted on different cues than the extractor emits must refuse."""
    model = load_model()
    assert model is not None, "run: python -m eval.fit_classifier"
    from pipeline.cues import CUE_NAMES

    assert model.feature_names == CUE_NAMES


@pytest.mark.parametrize("kind", ["shadow", "wet"])
def test_confusers_are_never_classified_as_damage(camera, config, kind):
    """Dispatching a crew to a shadow is the failure that matters most.

    The assertion is deliberately on the *costed* decision, not the exact
    label. Confusing a shadow for a wet stain changes nothing a crew does -
    neither is costed - and the classifier does confuse those two (12 cases in
    314 at last fit). Confusing either for a pothole spends public money on
    nothing, and that is what this guards.

    Several seeds and distances are tried, and the test fails rather than skips
    if no confuser is ever proposed: a skip here would silently stop protecting
    the most important property in the system.
    """
    damage_classes = {"pothole", "crack"}
    checked = 0
    for seed in (9, 17, 25, 31):
        # Confusers are only surfaced by the proposal pass in this band; see
        # the note in eval/sampling.py.
        for distance in (6.0, 7.5, 9.0):
            ground = build_ground(
                defects=[], confusers=[(0.0, distance, 0.32, kind)], seed=seed
            )
            image, _ = render_frame(ground, camera, 0.0, seed=seed + 1)
            frame = Frame("f", 0, 0.0, 999.0, image, "synthetic")
            _, bev = rectify(frame, config.rectify)
            if bev is None:
                continue
            candidates, _ = propose(bev, "f", ProposeParams())
            for candidate in candidates:
                classification = classify(bev, candidate)
                checked += 1
                assert classification.label not in damage_classes, (
                    f"a rendered {kind} at {distance} m was classified as "
                    f"'{classification.label}' (confidence "
                    f"{classification.confidence}, cues {classification.cues})"
                )

    assert checked, f"no {kind} was ever proposed across the tried seeds/distances"


def test_a_rendered_pothole_is_classified_as_damage(camera, config):
    """The other half of the binary decision: real damage must be costed."""
    damage_classes = {"pothole", "crack"}
    checked = correct = 0
    for seed in (9, 17, 25):
        for distance in (6.0, 7.5, 9.0):
            ground = build_ground(defects=[(0.0, distance, 0.32)], seed=seed)
            image, _ = render_frame(ground, camera, 0.0, seed=seed + 1)
            frame = Frame("f", 0, 0.0, 999.0, image, "synthetic")
            _, bev = rectify(frame, config.rectify)
            if bev is None:
                continue
            for candidate in propose(bev, "f", ProposeParams())[0]:
                checked += 1
                correct += classify(bev, candidate).label in damage_classes

    assert checked, "no pothole was ever proposed"
    assert correct / checked > 0.9, f"only {correct}/{checked} potholes called damage"


# -- tracking and aggregation ------------------------------------------------


def test_ego_motion_recovered_from_asphalt_not_markings(camera, config):
    """Lane markings report zero motion; the estimate must not follow them."""
    ground = build_ground(seed=5)
    images, _ = render_drive(ground, camera, n_frames=6, step_m=0.6)
    session = Session(config=config)
    session.load_images(images)
    for frame in session.frames:
        session.rectify_frame(frame.frame_id)

    errors = []
    for previous, current in zip(session.frames, session.frames[1:]):
        if previous.frame_id not in session.bevs or current.frame_id not in session.bevs:
            continue
        (_, dy), agreement = bev_shift_lk(
            session.bevs[previous.frame_id], session.bevs[current.frame_id], TrackParams()
        )
        forward = dy * session.rectifications[current.frame_id].m_per_px_y
        errors.append(abs(forward - 0.6))

    assert errors, "no frame pairs were comparable"
    good = sum(1 for error in errors if error < 0.1)
    assert good >= len(errors) - 1, f"only {good}/{len(errors)} pairs within 0.1 m"


def test_no_measurements_means_maximum_distrust():
    """An unmeasured defect must not read as perfect agreement."""
    result = aggregate([], TrackParams())
    assert result.n_measurements == 0
    assert not math.isfinite(result.iqr_ratio)


def test_odometry_puts_one_defect_at_one_station(camera, config):
    """The same pothole from several frames is one defect, not several."""
    ground = build_ground(defects=[(-0.5, 9.0, 0.36)], seed=13)
    images, _ = render_drive(ground, camera, n_frames=8, step_m=0.6)
    session = Session(config=config)
    session.load_images(images)
    session.build_odometry()

    stations = []
    for frame in session.frames[:5]:
        if frame.frame_id not in session.bevs:
            continue
        for candidate in session.propose_candidates(frame.frame_id):
            if classify(session.bevs[frame.frame_id], candidate).label == "pothole":
                position = session.ground_position(candidate.candidate_id)
                if position:
                    stations.append(position[1])

    assert len(stations) >= 3
    assert max(stations) - min(stations) < 0.4, f"station drifted: {stations}"


# -- scoring: the gate must fail closed --------------------------------------


def test_unmeasured_defect_is_never_billed():
    result = Aggregate("c", 0, 0.0, 0.0, math.inf, 0.0, 0.0, [])
    classification = Classification("c", "unknown", 0.0, {}, {})
    defect = score_defect(result, classification, ScoreConfig(), AgentThresholds())
    assert not defect.billable
    assert defect.cost_estimate == 0.0
    assert defect.needs_human_approval


def test_priority_one_always_requires_approval():
    """However confident the measurement, urgent public spend is reviewed."""
    result = Aggregate("c", 6, 0.42, 0.01, 0.024, 0.41, 0.43, ["a", "b", "c"])
    classification = Classification("c", "pothole", 0.99, {"pothole": 0.99}, {})
    defect = score_defect(result, classification, ScoreConfig(), AgentThresholds())
    assert defect.priority == 1
    assert defect.needs_human_approval


def test_shadows_are_recorded_but_never_costed():
    result = Aggregate("c", 5, 0.30, 0.01, 0.033, 0.29, 0.31, ["a", "b", "c"])
    classification = Classification("c", "shadow", 0.95, {"shadow": 0.95}, {})
    defect = score_defect(result, classification, ScoreConfig(), AgentThresholds())
    assert defect.cost_estimate == 0.0
    assert not defect.billable


# -- privacy: must fail closed ----------------------------------------------


def test_face_blurring_never_skips_silently(monkeypatch):
    """A missing model must raise, not pass the frame through unscrubbed."""
    import pipeline.privacy as privacy

    monkeypatch.setattr(privacy, "DEFAULT_FACE_MODEL", privacy.Path("/nonexistent.onnx"))
    image = np.full((200, 300, 3), 90, np.uint8)
    with pytest.raises(FaceModelMissing):
        privacy.scrub(image, PrivacyParams(enabled=True))


def test_privacy_scrubbing_does_not_destroy_lane_markings(camera, config):
    """An over-inclusive privacy filter is not automatically a safe one.

    Regression: unbounded plate detection merged the whole road surface into a
    single 966x346 px "plate", and blurring it took the lane markings with it.
    Rectification quality went from 0.94 to 0.00 with privacy enabled.
    """
    from dataclasses import replace as dataclass_replace

    ground = build_ground(defects=[(-0.5, 9.0, 0.34)], seed=13)
    images, _ = render_drive(ground, camera, n_frames=3, step_m=0.6)

    qualities = {}
    for name, enabled in (("on", True), ("off", False)):
        session = Session(
            config=dataclass_replace(config, privacy=PrivacyParams(enabled=enabled))
        )
        session.load_images(images)
        qualities[name] = session.rectify_frame(session.frames[0].frame_id).quality

    assert qualities["off"] > 0.5, "baseline rectification failed; test is not meaningful"
    assert qualities["on"] > 0.8 * qualities["off"], (
        f"privacy scrubbing degraded rectification: {qualities['off']:.2f} -> "
        f"{qualities['on']:.2f}"
    )


def test_plate_detection_ignores_frame_sized_regions(camera):
    """Nothing occupying half the frame is a licence plate."""
    ground = build_ground(seed=13)
    image, _ = render_frame(ground, camera, 0.0, seed=1)
    frame_area = image.shape[0] * image.shape[1]
    for x, y, w, h in detect_plates(image, PrivacyParams()):
        assert w * h < 0.05 * frame_area, f"box {w}x{h} is scenery, not a plate"


def test_plate_like_region_is_redacted():
    image = np.full((400, 600, 3), 90, np.uint8)
    cv2.rectangle(image, (240, 300), (330, 325), (235, 235, 235), -1)
    for index in range(6):
        cv2.rectangle(
            image, (247 + index * 13, 306), (253 + index * 13, 320), (30, 30, 30), -1
        )
    assert detect_plates(image, PrivacyParams())
    scrubbed, counts = scrub(image, PrivacyParams())
    assert counts["plates"] >= 1
    before = image[300:325, 240:330].std()
    after = scrubbed[300:325, 240:330].std()
    assert after < before / 4, f"plate detail survived: {before:.1f} -> {after:.1f}"


# -- end to end --------------------------------------------------------------


def test_a_measured_pothole_lands_within_a_quarter_of_truth():
    """Loose on purpose: the tight figures live in docs/accuracy.md.

    This guards against a regression that breaks measurement outright, which is
    what a 99% error from the watershed bug looked like.
    """
    row = measure_one(0.38)
    assert row.detected
    assert row.views >= 3
    assert abs(row.error_pct) < 25.0, f"error {row.error_pct:+.1f}%"


# -- pluggable proposal backends ---------------------------------------------


def test_detector_proposed_candidates_flow_through_unchanged(camera, config):
    """A detection must become a Candidate the rest of the pipeline can measure.

    Detection is the weak link on real roads (recall 0.429 in India, 0.092 in
    Czechia), so the proposal stage is an interface. This guards the contract:
    everything downstream works on a Candidate and must not care how it was
    proposed. The detector itself is stubbed, because what is under test is the
    plumbing from a box to a measurement, not the model.
    """
    import math

    from pipeline.config import RefineParams
    from pipeline.proposers import Detection, DetectorProposer

    class StubDetector(DetectorProposer):
        def __init__(self, boxes):
            self.name = "detector"
            self._boxes = boxes
            self.refine_params = RefineParams()

        def detect(self, image):
            return [Detection(box, 0.9, 3) for box in self._boxes]

    truth_r = 0.34
    ground = build_ground(defects=[(-0.4, 7.0, truth_r)], seed=13)
    images, truths = render_drive(ground, camera, n_frames=6, step_m=0.6)

    defect = truths[0]["defects"][0]
    cx, cy = defect["centre_px"]
    lateral_px = truth_r * camera.focal_px / defect["distance_m"]
    box = (
        int(cx - lateral_px * 1.4), int(cy - lateral_px * 0.9),
        int(lateral_px * 2.8), int(lateral_px * 1.9),
    )

    session = Session(config=config, proposer=StubDetector([box]))
    session.load_images(images)
    frame_id = session.frames[0].frame_id
    session.rectify_frame(frame_id)

    candidates = session.propose_candidates(frame_id)
    assert candidates, "the detector's box produced no candidate"

    candidate = candidates[0]
    measurement = session.refine_candidate(candidate.candidate_id)
    assert measurement.measurable
    truth_area = math.pi * truth_r**2
    error = abs(measurement.area_m2 - truth_area) / truth_area
    assert error < 0.25, f"detector-proposed area off by {error:.1%}"

    # The box is a region of interest, never the defect's extent: a bounding
    # box around an irregular pothole overstates its area badly.
    box_area_m2 = box[2] * box[3] * 1.0
    assert measurement.area_m2 < box_area_m2

    assert session.classify(candidate.candidate_id).label in {"pothole", "crack"}
    aggregate = session.track_across_frames(candidate.candidate_id, k=4)
    assert aggregate.n_measurements >= 2
    assert session.score(candidate.candidate_id).priority in {1, 2, 3}


def test_detector_backend_refuses_without_a_model():
    from pipeline.proposers import get_proposer

    with pytest.raises(FileNotFoundError, match="export"):
        get_proposer("detector", model_path="models/does-not-exist.onnx")


def test_calibration_solver_recovers_a_known_camera():
    """Two markers at known distances must pin down focal length and horizon."""
    import math

    from scripts.calibrate import solve

    image_height = 1080
    height_m, vfov, pitch = 1.27, 62.0, 6.5
    focal = (image_height / 2) / math.tan(math.radians(vfov) / 2)
    horizon = image_height / 2 - focal * math.tan(math.radians(pitch))
    rows = [horizon + height_m * focal / d for d in (3.0, 6.0)]

    result = solve(height_m, 3.0, 6.0, rows[0], rows[1], image_height)
    assert abs(result["vfov_deg"] - vfov) < 0.01
    assert abs(result["pitch_deg"] - pitch) < 0.01


def test_calibration_solver_rejects_swapped_markers():
    """The nearer marker sits lower in the frame; the reverse is a user error."""
    from scripts.calibrate import solve

    with pytest.raises(SystemExit, match="LOWER"):
        solve(1.27, 3.0, 6.0, 600.0, 800.0, 1080)
