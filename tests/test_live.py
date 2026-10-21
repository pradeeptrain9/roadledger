"""Tests for live capture: streaming measurement, geography, and calibration.

The endpoint functions are exercised directly rather than over HTTP. Starlette's
test client needs `httpx`, which is not in the pinned requirements and would be
a dependency carried solely for a test; the actual HTTP round trip - JPEG
encoding, base64 framing, the concurrency lock - is covered end to end by
`scripts/simulate_phone.py`, which speaks the same API the phone does.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.calibration import (
    CalibrationError,
    scale_to_frame,
    solve_from_horizon_and_marker,
    solve_from_two_markers,
    warnings_for,
)
from pipeline.config import PipelineConfig, PrivacyParams, RectifyParams, CameraGeometry
from pipeline.geo import (
    GeoFix,
    GeoLedger,
    Sighting,
    bearing_deg,
    haversine_m,
    locate,
    metres_per_degree,
    offset_to_latlon,
    track_heading,
)
from pipeline.live import LiveSession
from tests.ground import build_ground, render_drive
from tests.synthetic import SceneCamera

START_LAT, START_LON = 17.4401, 78.3489


def _config(camera: SceneCamera) -> PipelineConfig:
    return PipelineConfig(
        rectify=RectifyParams(camera=CameraGeometry(camera.height_m, camera.vfov_deg)),
        privacy=PrivacyParams(enabled=False),
    )


def _drive(defects, n_frames=36, step_m=0.8, fps=4.0, seed=5, scrub=False):
    """Stream a synthetic drive through the live session, as the phone would."""
    camera = SceneCamera()
    ground = build_ground(defects=defects, seed=seed, length_m=90.0)
    images, _ = render_drive(ground, camera, n_frames=n_frames, step_m=step_m, seed=seed + 16)
    per_lat, _ = metres_per_degree(START_LAT)
    session = LiveSession(config=_config(camera), run_id=f"test_{seed}")
    results = []
    for seq, image in enumerate(images):
        fix = GeoFix(
            lat=START_LAT + (seq * step_m) / per_lat,
            lon=START_LON,
            accuracy_m=6.0,
            heading_deg=0.0,
            speed_mps=step_m * fps,
            timestamp_s=seq / fps,
        )
        results.append(session.ingest(image, fix, seq, scrub_frame=scrub))
    session.finish()
    return session, results


# -- geography -----------------------------------------------------------


def test_ground_offset_lands_where_the_bearing_and_distance_say_it_should():
    """A defect 8 m ahead and 1 m right of a north-facing camera is where it is."""
    fix = GeoFix(lat=START_LAT, lon=START_LON, accuracy_m=5.0, heading_deg=0.0, speed_mps=8.0)
    lat, lon = offset_to_latlon(fix, ahead_m=8.0, lateral_m=1.0, heading_deg=0.0)
    assert haversine_m(fix.lat, fix.lon, lat, lon) == pytest.approx(math.hypot(8.0, 1.0), abs=0.05)
    # Mostly north, slightly east.
    assert bearing_deg(fix.lat, fix.lon, lat, lon) == pytest.approx(
        math.degrees(math.atan2(1.0, 8.0)), abs=0.5
    )

    # Facing east, the same offset goes east and slightly south.
    lat_e, lon_e = offset_to_latlon(fix, ahead_m=8.0, lateral_m=1.0, heading_deg=90.0)
    assert lon_e > fix.lon and lat_e < fix.lat


def test_a_defect_with_no_usable_heading_is_placed_at_the_camera_and_says_so():
    """Degrading to the camera position is correct; inventing a heading is not."""
    stationary = GeoFix(lat=START_LAT, lon=START_LON, accuracy_m=5.0, heading_deg=90.0, speed_mps=0.2)
    assert not stationary.has_usable_heading

    position = locate(stationary, ahead_m=9.0, lateral_m=0.5)
    assert position.method == "camera_only"
    assert (position.lat, position.lon) == (stationary.lat, stationary.lon)
    # The accuracy must absorb the distance the defect lies ahead, not ignore it.
    assert position.accuracy_m == pytest.approx(5.0 + 9.0)
    assert "no usable heading" in position.note

    moving = GeoFix(lat=START_LAT, lon=START_LON, accuracy_m=5.0, heading_deg=90.0, speed_mps=8.0)
    placed = locate(moving, ahead_m=9.0, lateral_m=0.5)
    assert placed.method == "gps_heading"
    assert placed.accuracy_m < position.accuracy_m
    assert haversine_m(moving.lat, moving.lon, placed.lat, placed.lon) > 8.0


def test_track_heading_refuses_fixes_too_close_together_to_define_one():
    """Two fixes inside their own uncertainty define noise, not a course."""
    per_lat, _ = metres_per_degree(START_LAT)
    jitter = [
        GeoFix(lat=START_LAT + d / per_lat, lon=START_LON, accuracy_m=10.0)
        for d in (0.0, 0.4, 0.9)
    ]
    assert track_heading(jitter) is None

    travelling = [
        GeoFix(lat=START_LAT + d / per_lat, lon=START_LON, accuracy_m=6.0)
        for d in (0.0, 12.0, 24.0)
    ]
    assert track_heading(travelling) == pytest.approx(0.0, abs=1.0)


def test_cross_drive_matches_are_candidates_not_merges():
    """GPS cannot tell one pothole from its neighbour, so it is not allowed to."""
    ledger = GeoLedger()
    position = locate(
        GeoFix(lat=START_LAT, lon=START_LON, accuracy_m=6.0, heading_deg=0.0, speed_mps=8.0),
        ahead_m=5.0,
        lateral_m=0.0,
    )
    first = Sighting("drive_a", "d_1", 1000.0, 0.30, "pothole", 1, position)
    entry, created = ledger.merge_or_add(first, same_drive=True)
    assert created and len(ledger.entries) == 1

    # A second drive sees something 3 m away. Inside GPS error, so reported -
    # but as a candidate, and the ledger still holds two separate entries.
    near = locate(
        GeoFix(
            lat=entry.lat + 3.0 / metres_per_degree(entry.lat)[0],
            lon=entry.lon,
            accuracy_m=6.0,
            heading_deg=0.0,
            speed_mps=8.0,
        ),
        ahead_m=0.0,
        lateral_m=0.0,
    )
    candidates = ledger.revisit_candidates(
        near.lat, near.lon, near.accuracy_m, exclude_run_id="drive_b"
    )
    assert len(candidates) == 1
    assert candidates[0]["confirmed"] is False
    assert candidates[0]["distance_m"] == pytest.approx(3.0, abs=0.4)

    second = Sighting("drive_b", "d_2", 2000.0, 0.34, "pothole", 1, near)
    _, created_again = ledger.merge_or_add(second, same_drive=True)
    assert created_again, "a 3 m GPS-only match must not be merged into one defect"
    assert len(ledger.entries) == 2


def test_repeat_sightings_of_one_defect_carry_an_area_trend():
    ledger = GeoLedger()
    position = locate(
        GeoFix(lat=START_LAT, lon=START_LON, accuracy_m=6.0, heading_deg=0.0, speed_mps=8.0),
        ahead_m=4.0,
        lateral_m=0.0,
    )
    entry, _ = ledger.merge_or_add(Sighting("a", "d1", 1.0, 0.30, "pothole", 1, position))
    assert entry.area_trend_m2 is None, "one sighting cannot establish a trend"
    ledger.merge_or_add(Sighting("b", "d2", 2.0, 0.42, "pothole", 1, position))
    assert entry.area_trend_m2 == pytest.approx(0.12, abs=1e-6)
    assert len(ledger.to_geojson()["features"]) == 1


# -- calibration ---------------------------------------------------------


def test_both_calibration_routes_recover_a_known_camera_exactly():
    image_height, height_m, vfov, pitch = 720, 1.27, 62.0, 6.5
    focal = (image_height / 2.0) / math.tan(math.radians(vfov) / 2.0)
    horizon = image_height / 2.0 - focal * math.tan(math.radians(pitch))
    row = lambda d: horizon + height_m * focal / d

    two = solve_from_two_markers(height_m, 3.0, 6.0, row(3.0), row(6.0), image_height)
    one = solve_from_horizon_and_marker(height_m, horizon, 5.0, row(5.0), image_height)
    for solved in (two, one):
        assert solved["vfov_deg"] == pytest.approx(vfov, abs=1e-3)
        assert solved["pitch_deg"] == pytest.approx(pitch, abs=1e-3)
        assert solved["focal_px"] == pytest.approx(focal, abs=0.05)


def test_the_horizon_route_is_less_sensitive_to_reading_error():
    """The claim in the module docstring, checked rather than asserted."""
    image_height, height_m, vfov = 720, 1.27, 62.0
    focal = (image_height / 2.0) / math.tan(math.radians(vfov) / 2.0)
    horizon = image_height / 2.0
    row = lambda d: horizon + height_m * focal / d

    # The same 2 px of reading error applied to each route's own inputs.
    two = solve_from_two_markers(height_m, 3.0, 6.0, row(3.0) + 2, row(6.0) - 2, image_height)
    one = solve_from_horizon_and_marker(height_m, horizon + 2, 5.0, row(5.0) - 2, image_height)
    assert abs(one["horizon_px"] - horizon) < abs(two["horizon_px"] - horizon)


def test_calibration_refuses_a_marker_at_or_above_the_horizon():
    with pytest.raises(CalibrationError, match="above the horizon"):
        solve_from_horizon_and_marker(1.3, 400.0, 5.0, 390.0, 720)
    with pytest.raises(CalibrationError, match="wrong way round"):
        solve_from_two_markers(1.3, 3.0, 6.0, 400.0, 500.0, 720)


def test_centimetres_entered_as_metres_are_caught_before_any_frame_is_measured():
    """Nothing downstream can detect this; it would scale every area by 10000."""
    solved = solve_from_horizon_and_marker(0.0127, 360.0, 5.0, 600.0, 720)
    problems = warnings_for(solved)
    assert any("10000" in problem for problem in problems)


def test_rescaling_a_calibration_keeps_the_lens_and_scales_the_pixels():
    """Field of view and tilt belong to the mount; focal and horizon do not."""
    solved = solve_from_horizon_and_marker(1.27, 360.0, 5.0, 600.0, 720)
    bigger = scale_to_frame(solved, 1080)
    assert bigger["vfov_deg"] == solved["vfov_deg"]
    assert bigger["pitch_deg"] == solved["pitch_deg"]
    assert bigger["focal_px"] == pytest.approx(solved["focal_px"] * 1.5, abs=0.05)
    assert bigger["horizon_px"] == pytest.approx(solved["horizon_px"] * 1.5, abs=0.05)


# -- streaming measurement ----------------------------------------------


def test_a_streamed_drive_confirms_a_defect_from_several_views():
    """The whole point of the live path: evidence accumulates as you approach."""
    truth = math.pi * 0.34**2
    session, results = _drive([(-0.45, 12.0, 0.34)])

    confirmed = [d for d in session.defects.values() if d["confirmed"]]
    assert confirmed, "a defect driven past for 36 frames should confirm"
    best = max(confirmed, key=lambda d: d["n_views"])
    assert best["n_views"] >= session.config.track.min_measurements
    assert math.isfinite(best["area_iqr_ratio"]) and best["area_iqr_ratio"] >= 0
    # Not a tight accuracy assertion - that is what eval/live_eval.py is for -
    # but the measurement must be the right order of magnitude.
    assert 0.4 * truth < best["area_m2"] < 2.0 * truth
    assert best["position"] is not None
    assert best["position"]["method"] == "gps_heading"

    # The ego-motion must come from the speed the phone reported, not be
    # re-derived from pixels on every frame.
    sources = {r.step_source for r in results if r.accepted and r.step_m > 0}
    assert sources == {"gps_speed"}


def test_evidence_strengthens_rather_than_appearing_all_at_once():
    """Provisional first, confirmed later: the number visibly firms up."""
    session, results = _drive([(-0.45, 12.0, 0.34)])
    kinds = [event["type"] for result in results for event in result.events]
    assert "defect_provisional" in kinds
    assert "defect_confirmed" in kinds
    assert kinds.index("defect_provisional") < kinds.index("defect_confirmed")


def test_confirmation_is_allowed_to_go_backwards():
    """A confirmed defect whose evidence weakens must stop reading as confirmed."""
    from pipeline.live import LiveTrack
    from pipeline.types import Measurement

    session = LiveSession(config=_config(SceneCamera()), run_id="weaken")
    track = LiveTrack(track_id="t_x", bev_xy=(100.0, 100.0))
    session.tracks[track.track_id] = track

    def add(area: float, footprint: float) -> None:
        track.measurements.append(
            Measurement(
                candidate_id=track.track_id, frame_id="f", area_m2=area, perimeter_m=1.0,
                convexity=1.0, max_width_m=0.5, texture_energy=1.0,
                source_footprint_px=footprint, measurable=True,
            )
        )

    from pipeline.live import LiveFrameResult

    result = LiveFrameResult(seq=0, accepted=True)
    for area in (0.30, 0.31, 0.32):
        add(area, 1000.0)
    session._report(track, result)
    assert track.confirmed, "three comparable views establish a spread"

    # A much closer view arrives. The vantage filter now discards the three
    # distant ones, leaving a single comparable measurement.
    add(0.33, 40_000.0)
    session._report(track, result)
    assert not track.confirmed
    assert result.events[-1]["type"] == "defect_weakened"
    assert result.events[-1]["needs_human_approval"]


def test_a_proposal_never_measurable_is_counted_not_reported():
    """A 0.000 m2 row on the operator's screen is noise, not a finding."""
    from pipeline.live import LiveFrameResult, LiveTrack
    from pipeline.types import Measurement

    session = LiveSession(config=_config(SceneCamera()), run_id="empty")
    track = LiveTrack(track_id="t_e", bev_xy=(10.0, 10.0))
    session.tracks[track.track_id] = track
    track.measurements.append(
        Measurement(
            candidate_id="t_e", frame_id="f", area_m2=0.0, perimeter_m=0.0, convexity=0.0,
            max_width_m=0.0, texture_energy=0.0, measurable=False, note="too distant",
        )
    )
    result = LiveFrameResult(seq=1, accepted=True)
    session._close_track(track, result)
    assert result.events == []
    assert session.defects == {}
    assert session.stats()["proposals_without_measurement"] == 1


def test_a_defect_clipped_by_the_view_edge_is_refused():
    """A truncated boundary is not a small defect, and must not be measured."""
    from pipeline.config import RefineParams
    from pipeline.refine import refine
    from pipeline.types import Candidate

    bev = np.full((240, 240, 3), 180, np.uint8)
    # A dark region running off the bottom edge: cut off by the frame.
    bev[170:240, 90:160] = 30
    contour = np.array(
        [[[90, 170]], [[159, 170]], [[159, 239]], [[90, 239]]], dtype=np.int32
    )
    candidate = Candidate(
        candidate_id="c_clip", frame_id="f", bbox=(90, 170, 70, 70),
        area_px=4900.0, solidity=1.0, aspect_ratio=1.0, mean_intensity=30.0, contour=contour,
    )
    measurement = refine(bev, candidate, RefineParams(), 0.01, 0.02, np.eye(3))
    assert not measurement.measurable
    assert "truncation" in measurement.note


def test_a_blurred_frame_is_refused_rather_than_measured():
    camera = SceneCamera()
    session = LiveSession(config=_config(camera), run_id="blur")
    flat = np.full((camera.height, camera.width, 3), 90, np.uint8)
    result = session.ingest(
        flat, GeoFix(lat=START_LAT, lon=START_LON, accuracy_m=6.0), 0, scrub_frame=False
    )
    assert not result.accepted
    assert "blurry" in result.reason
    assert session.stats()["frames_skipped_blurry"] == 1


# -- durability ----------------------------------------------------------


def test_the_ledger_survives_the_process_that_wrote_it():
    """The worker is designed to be stopped; the cross-drive record is not."""
    from pipeline.store import MemoryLedgerStore

    store = MemoryLedgerStore()
    position = locate(
        GeoFix(lat=START_LAT, lon=START_LON, accuracy_m=6.0, heading_deg=0.0, speed_mps=8.0),
        ahead_m=5.0,
        lateral_m=0.0,
    )
    first = GeoLedger(store=store)
    first.merge_or_add(Sighting("drive_a", "d_1", 1000.0, 0.30, "pothole", 1, position))

    # A second process, sharing only the store - which is what a restarted
    # worker is.
    second = GeoLedger(store=store)
    assert len(second.entries) == 1
    candidates = second.revisit_candidates(
        position.lat, position.lon, position.accuracy_m, exclude_run_id="drive_b"
    )
    assert len(candidates) == 1
    assert candidates[0]["last_area_m2"] == 0.30


def test_a_lookup_reads_a_bounded_number_of_cells_not_the_whole_ledger():
    """Lookup cost must not grow with the ledger; this runs while driving."""
    from pipeline.store import MemoryLedgerStore, cell_key, neighbour_cells

    store = MemoryLedgerStore()
    per_lat, per_lon = metres_per_degree(START_LAT)
    # Defects spread over several kilometres of road.
    for index in range(300):
        lat = START_LAT + (index * 40.0) / per_lat
        position = locate(
            GeoFix(lat=lat, lon=START_LON, accuracy_m=6.0, heading_deg=0.0, speed_mps=8.0),
            ahead_m=0.0,
            lateral_m=0.0,
        )
        GeoLedger(store=store).merge_or_add(
            Sighting(f"r{index}", f"d{index}", 1.0, 0.3, "pothole", 1, position)
        )
    assert len(store.rows) == 300

    near = store.entries_near(START_LAT, START_LON)
    assert len(near) <= 9, "a lookup must not return the whole ledger"
    assert all(row["cell"] in set(neighbour_cells(START_LAT, START_LON)) for row in near)
    # And the one actually at that point is among them.
    assert any(row["cell"] == cell_key(START_LAT, START_LON) for row in near)


def test_a_defect_just_over_a_cell_boundary_is_still_found():
    """The reason a lookup reads nine cells and not one."""
    from pipeline.store import CELL_DEG, MemoryLedgerStore, cell_key

    store = MemoryLedgerStore()
    per_lat, _ = metres_per_degree(START_LAT)
    # Sit just below a cell boundary, and place a defect just above it.
    below = math.floor(START_LAT / CELL_DEG) * CELL_DEG + CELL_DEG - 1e-7
    above = below + 2e-7
    assert cell_key(below, START_LON) != cell_key(above, START_LON)

    position = locate(
        GeoFix(lat=above, lon=START_LON, accuracy_m=6.0, heading_deg=0.0, speed_mps=8.0),
        ahead_m=0.0,
        lateral_m=0.0,
    )
    GeoLedger(store=store).merge_or_add(Sighting("a", "d", 1.0, 0.3, "pothole", 1, position))

    found = GeoLedger(store=store).nearest(below, START_LON)
    assert found is not None
    assert found[1] < 1.0, "centimetres apart, across a cell boundary"


def test_an_unset_ledger_table_falls_back_to_memory_rather_than_failing():
    """Losing cross-drive history is bad; refusing to measure is worse."""
    import os

    from pipeline.store import MemoryLedgerStore, store_from_environment

    previous = os.environ.pop("ROADLEDGER_LEDGER_TABLE", None)
    try:
        assert isinstance(store_from_environment(), MemoryLedgerStore)
    finally:
        if previous is not None:
            os.environ["ROADLEDGER_LEDGER_TABLE"] = previous
