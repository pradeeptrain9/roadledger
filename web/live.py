"""Live capture from a phone or dashcam: the endpoint the vehicle talks to.

Deliberately plain HTTP, one POST per frame, rather than a WebSocket. The
reasons are all practical:

* **The request/response shape is the semantics.** "Measure this frame and tell
  me what you found" is a call that returns a value. Every event a frame
  produces comes back in that frame's response, so there is nothing to push.
* **It is self-limiting.** The phone does not send frame N+1 until N comes
  back, so the capture rate settles at whatever the server can actually
  measure. A socket would let the phone fill a queue it cannot drain, and a
  queued frame is not a late measurement - by the time it is processed the
  vehicle is thirty metres past the defect.
* **It deploys.** This has to run behind API Gateway and CloudFront on a free
  tier, where WebSocket support is a separate product with separate limits.
  Plain POST works through anything.
* **No new dependency.** `websockets` is not in the pinned requirements and a
  submission that must be reproducible from `requirements.txt` is better off
  without one it does not need.

The frame rate is not fixed. How fast the phone must send is a geometric
consequence of how fast the vehicle is moving, because a defect has to be seen
at least three times from comparable distances before its spread means
anything - and the window in which that is possible is a couple of metres of
road. So the server computes the interval from the measured speed and returns
it with every frame, and says plainly when the vehicle is simply going too fast
to measure from. A tool that reports areas it cannot support would be worse
than one that says "slow down".
"""

from __future__ import annotations

import base64
import binascii
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from pipeline.calibration import (
    CalibrationError,
    scale_to_frame,
    solve_from_horizon_and_marker,
    solve_from_two_markers,
    warnings_for,
)
from pipeline.config import (
    AgentThresholds,
    CameraGeometry,
    FrameParams,
    PipelineConfig,
    PrivacyParams,
    RectifyParams,
)
from pipeline.geo import GeoFix, GeoLedger
from pipeline.store import store_from_environment
from pipeline.live import LiveSession

router = APIRouter(prefix="/api/live", tags=["live"])

# Target frame spacing, in metres of travel between frames. Everything about
# the recommended capture rate follows from this one number and the vehicle's
# speed, so it is measured rather than chosen.
#
# The sweep in eval/live_eval.py confirmed 8 of 8 defects at 1.2 m spacing and
# 0 of 8 at 2.0 m, which made 1.2 m look like the answer. Driving the real
# endpoint at 1.2 m spacing then showed it is the *edge* of the cliff rather
# than a safe distance from it: repeated runs over the same three defects
# confirmed between one and three of them depending on which views happened to
# survive the vantage filter. At 0.8 m the same sweep used 3.9 comparable views
# per defect instead of 3.2, which is the margin that turns a coin flip into a
# measurement. The cost is a higher frame rate, and the honest consequence is
# that a phone on mobile data measures reliably at survey speed rather than at
# traffic speed. That limit is reported to the operator instead of being hidden
# behind confident-looking areas.
MAX_FRAME_SPACING_M = 0.8

# Bounds on what the phone is asked to do. Below the floor the upload cannot
# keep up on mobile data; above the ceiling the operator is waiting around.
MIN_INTERVAL_S = 0.12
MAX_INTERVAL_S = 1.0

# Resource ceilings. This endpoint is public during judging and runs on a free
# tier, so every one of them is enforced rather than assumed.
MAX_SESSIONS = 8
MAX_FRAMES_PER_SESSION = 4000
MAX_JPEG_BYTES = 2_000_000
SESSION_TTL_S = 3 * 3600

# Raw frames are never written to disk. Only the privacy-scrubbed evidence crop
# of a measured defect is retained, and only in memory for the session's life.
MAX_CROPS = 400

# The persistent geographic record, shared across capture sessions. This is what
# makes a second drive down the same road recognise what the first one found,
# so it is backed by a store rather than by this process's memory: the worker
# is designed to be stopped, and an in-memory ledger on a machine that is
# supposed to be stopped is not a cross-drive ledger at all. DynamoDB when
# ROADLEDGER_LEDGER_TABLE is set by the stack, memory otherwise.
LEDGER = GeoLedger(store=store_from_environment())


class Calibration(BaseModel):
    """How the camera is mounted. Either solved on the phone or given directly."""

    height_m: float = Field(gt=0.05, lt=10.0)
    # Route 1: the operator drags a line onto the horizon and taps one marker.
    horizon_px: float | None = None
    marker_distance_m: float | None = Field(default=None, gt=0.2, lt=100.0)
    marker_row_px: float | None = None
    # Route 2: two markers at known distances, no horizon needed.
    near_distance_m: float | None = Field(default=None, gt=0.2, lt=100.0)
    near_row_px: float | None = None
    far_distance_m: float | None = Field(default=None, gt=0.2, lt=100.0)
    far_row_px: float | None = None
    # Route 3: already-known lens geometry.
    vfov_deg: float | None = Field(default=None, gt=10.0, lt=170.0)
    pitch_deg: float | None = None
    # The frame height the rows above were read at.
    image_height_px: int = Field(gt=0, le=4320)

    def solve(self) -> dict[str, Any]:
        if self.horizon_px is not None and self.marker_distance_m and self.marker_row_px is not None:
            return solve_from_horizon_and_marker(
                self.height_m,
                self.horizon_px,
                self.marker_distance_m,
                self.marker_row_px,
                self.image_height_px,
            )
        if (
            self.near_distance_m
            and self.far_distance_m
            and self.near_row_px is not None
            and self.far_row_px is not None
        ):
            return solve_from_two_markers(
                self.height_m,
                self.near_distance_m,
                self.far_distance_m,
                self.near_row_px,
                self.far_row_px,
                self.image_height_px,
            )
        if self.vfov_deg is not None:
            focal = (self.image_height_px / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)
            pitch = self.pitch_deg or 0.0
            horizon = self.image_height_px / 2.0 - focal * math.tan(math.radians(pitch))
            return {
                "height_m": round(self.height_m, 4),
                "vfov_deg": round(self.vfov_deg, 3),
                "pitch_deg": round(pitch, 3),
                "focal_px": round(focal, 2),
                "horizon_px": round(horizon, 2),
                "image_height_px": self.image_height_px,
                "solved_from": {"method": "given_lens_geometry"},
            }
        raise CalibrationError(
            "give either a horizon row plus one marker, or two markers at known distances, "
            "or the lens field of view directly"
        )


class StartRequest(BaseModel):
    calibration: Calibration
    label: str = Field(default="", max_length=120)
    # Scrubbing faces and plates costs real time per frame. It defaults on
    # because evidence crops are retained; an operator capturing a closed test
    # track can turn it off and the response says that it is off.
    privacy: bool = True


class FrameRequest(BaseModel):
    seq: int = Field(ge=0)
    jpeg_b64: str = Field(min_length=16)
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    accuracy_m: float = Field(default=15.0, ge=0.0, le=10_000.0)
    heading_deg: float | None = None
    speed_mps: float | None = Field(default=None, ge=0.0, le=120.0)
    t: float = Field(default=0.0, ge=0.0)


class ApprovalRequest(BaseModel):
    approved: bool
    approver: str = Field(default="", max_length=120)
    note: str = Field(default="", max_length=500)


@dataclass
class LiveState:
    """One connected vehicle."""

    run_id: str
    session: LiveSession
    calibration: dict[str, Any]
    label: str = ""
    created_s: float = field(default_factory=time.time)
    last_seen_s: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    finished: bool = False
    approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Wall-clock cost of measuring a frame, most recent first; used to tell the
    # phone what rate the server can actually sustain.
    recent_ms: list[float] = field(default_factory=list)
    # Wall-clock gaps between frames actually arriving. The server cannot
    # measure the phone's uplink, but the arrival interval already contains it,
    # along with the phone's own encode time and whatever the network did. It
    # is the only honest measure of the rate this vehicle can really sustain.
    recent_gap_s: list[float] = field(default_factory=list)
    last_arrival_s: float = 0.0
    frame_height_px: int = 0
    rescaled_note: str = ""

    def note_cost(self, ms: float) -> None:
        self.recent_ms.insert(0, ms)
        del self.recent_ms[12:]

    def note_arrival(self) -> None:
        now = time.time()
        if self.last_arrival_s:
            gap = now - self.last_arrival_s
            if 0.0 < gap < 10.0:
                self.recent_gap_s.insert(0, gap)
                del self.recent_gap_s[12:]
        self.last_arrival_s = now

    @property
    def sustainable_fps(self) -> float:
        """The rate this vehicle is really achieving, not the one it could.

        Server processing time is a lower bound on the frame interval and is
        usually not the binding one: measuring a frame takes 10-30 ms, while
        encoding a 1280 px JPEG on a phone and pushing 50 KB over mobile data
        takes considerably longer. So the limit is whichever is slower, and the
        upload side is measured from how fast frames actually arrive.

        Medians, not means: one frame delayed by a garbage-collection pause or
        a momentary cell handover should not permanently lower the rate the
        operator is told to use.
        """
        by_processing = (
            1000.0 / max(float(np.median(self.recent_ms)), 1.0)
            if self.recent_ms
            else 1.0 / MIN_INTERVAL_S
        )
        if len(self.recent_gap_s) < 3:
            return by_processing
        by_arrival = 1.0 / max(float(np.median(self.recent_gap_s)), MIN_INTERVAL_S)
        return min(by_processing, by_arrival)


_sessions: dict[str, LiveState] = {}
_sessions_lock = threading.Lock()


def _evict_stale() -> None:
    now = time.time()
    for run_id, state in list(_sessions.items()):
        if now - state.last_seen_s > SESSION_TTL_S:
            _sessions.pop(run_id, None)


def _state(run_id: str) -> LiveState:
    with _sessions_lock:
        state = _sessions.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"no such capture session: {run_id}")
    return state


def _config_for(solved: dict[str, Any], privacy: bool) -> PipelineConfig:
    """Build the pipeline configuration a live stream needs.

    Two differences from the batch configuration, both forced by the live case:
    the camera geometry comes from the phone's own calibration, and the
    sharpness floor is lower. A windscreen-mounted phone at speed produces
    frames that are genuinely softer than a still from a dataset, and gating
    them at the batch threshold would reject most of a drive. The gate still
    exists, because a measurement from a smeared frame is confidently wrong.
    """
    camera = CameraGeometry(
        height_m=solved["height_m"],
        vfov_deg=solved["vfov_deg"],
        pitch_deg=solved["pitch_deg"],
    )
    return PipelineConfig(
        rectify=RectifyParams(camera=camera),
        frames=FrameParams(sample_fps=6.0, min_sharpness=18.0, max_frames=MAX_FRAMES_PER_SESSION),
        privacy=PrivacyParams(enabled=privacy),
        agent=AgentThresholds(),
    )


def _capture_advice(state: LiveState, speed_mps: float | None) -> dict[str, Any]:
    """What rate to send at, and whether the vehicle is too fast to measure.

    The interval is the time to cover `MAX_FRAME_SPACING_M`, so the spacing
    between frames stays inside the window where a defect can be measured
    repeatedly from comparable distances. The speed ceiling is the other side
    of the same equation: given what the server is actually managing per frame,
    there is a speed above which the required frames cannot be delivered, and
    saying so is more use than quietly producing unconfirmable defects.
    """
    sustainable = state.sustainable_fps
    max_speed_mps = MAX_FRAME_SPACING_M * sustainable
    if speed_mps is None or speed_mps <= 0.2:
        interval = MAX_INTERVAL_S
        note = "stationary or no speed reported; capturing slowly"
        too_fast = False
    else:
        wanted = MAX_FRAME_SPACING_M / speed_mps
        # Too fast is not about the clamp on the interval. It is about whether
        # the frames needed at this speed can actually be delivered: the phone
        # and the link have a ceiling, and above it the frames arrive further
        # apart than the measurement needs however small an interval is asked
        # for.
        too_fast = speed_mps > max_speed_mps * 1.05
        interval = min(MAX_INTERVAL_S, max(MIN_INTERVAL_S, wanted))
        if too_fast:
            note = (
                f"at {speed_mps * 3.6:.0f} km/h frames land {speed_mps / max(sustainable, 1e-6):.1f} m "
                f"apart, and a defect needs several views from similar distances to be sized at "
                f"all. This phone and link are managing {sustainable:.1f} frames a second, so "
                f"about {max_speed_mps * 3.6:.0f} km/h is the fastest this setup can measure from. "
                "Defects will still be reported, marked unconfirmed and not costed."
            )
        else:
            note = (
                f"{1.0 / interval:.1f} frames per second keeps frames {MAX_FRAME_SPACING_M} m apart, "
                f"which is the spacing that confirmed 8 of 8 defects in the evaluation sweep"
            )
    return {
        "recommended_interval_s": round(interval, 3),
        "recommended_fps": round(1.0 / interval, 2),
        "max_frame_spacing_m": MAX_FRAME_SPACING_M,
        "sustainable_fps": round(sustainable, 2),
        "max_speed_kmh": round(max_speed_mps * 3.6, 1),
        "too_fast_to_measure": too_fast,
        "note": note,
    }


def _decode(jpeg_b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(jpeg_b64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"frame is not valid base64: {error}")
    if len(raw) > MAX_JPEG_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"frame is {len(raw)} bytes, above the {MAX_JPEG_BYTES} byte limit",
        )
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="frame could not be decoded as an image")
    return image


# -- endpoints -----------------------------------------------------------


@router.post("/sessions")
def start(request: StartRequest) -> dict[str, Any]:
    """Open a capture session. Calibration is solved and checked here.

    Refusing a bad calibration at this point is the whole purpose of the step.
    Every area the session goes on to report is scaled by these numbers, and
    nothing downstream can detect that the camera height was entered in
    centimetres - it would simply report every pothole as four square
    millimetres and look entirely confident about it.
    """
    try:
        solved = request.calibration.solve()
    except CalibrationError as error:
        raise HTTPException(status_code=422, detail=str(error))

    problems = warnings_for(solved)

    with _sessions_lock:
        _evict_stale()
        if len(_sessions) >= MAX_SESSIONS:
            raise HTTPException(
                status_code=429,
                detail=f"{MAX_SESSIONS} capture sessions are already open; finish one first",
            )
        run_id = f"live_{uuid.uuid4().hex[:10]}"
        state = LiveState(
            run_id=run_id,
            session=LiveSession(
                config=_config_for(solved, request.privacy), run_id=run_id, ledger=LEDGER
            ),
            calibration=solved,
            label=request.label,
        )
        _sessions[run_id] = state

    from pipeline.calibration import reprojection_check

    return {
        "run_id": run_id,
        "calibration": solved,
        "calibration_warnings": problems,
        "reprojection_check": reprojection_check(solved),
        "privacy_scrubbing": request.privacy,
        "capture": _capture_advice(state, None),
        "min_views_to_confirm": state.session.config.track.min_measurements,
        "note": (
            "Areas are measured from the image geometry and do not depend on GPS. Positions "
            "come from GPS and are only as good as it is; each one carries its own accuracy."
        ),
    }


@router.post("/sessions/{run_id}/frames")
def ingest_frame(run_id: str, request: FrameRequest) -> dict[str, Any]:
    """Measure one captured frame. Everything it found comes back in the reply."""
    state = _state(run_id)
    if state.finished:
        raise HTTPException(status_code=409, detail="this capture session has been finished")
    if state.session.frames_received >= MAX_FRAMES_PER_SESSION:
        raise HTTPException(
            status_code=429,
            detail=f"session reached its {MAX_FRAMES_PER_SESSION} frame limit",
        )

    image = _decode(request.jpeg_b64)

    # One frame at a time per vehicle. The session carries forward tracking
    # state, so two frames measured concurrently would corrupt it; and a frame
    # that arrives while the previous one is still being measured is better
    # dropped than queued, because by the time a queued frame is measured the
    # vehicle has driven past what was in it.
    if not state.lock.acquire(blocking=False):
        state.session.frames_dropped_behind += 1
        return {
            "type": "dropped",
            "seq": request.seq,
            "reason": "the previous frame is still being measured; this one was dropped rather "
            "than queued, because a queued frame measures road the vehicle has already passed",
            "capture": _capture_advice(state, request.speed_mps),
            "stats": state.session.stats(),
        }

    try:
        if state.frame_height_px == 0:
            state.frame_height_px = int(image.shape[0])
            declared = state.calibration["image_height_px"]
            if state.frame_height_px != declared:
                # The phone calibrated against the preview and is uploading at a
                # different size. Field of view and pitch survive that; a pixel
                # focal length and horizon do not, so they are rescaled and the
                # rescaling is reported rather than done quietly.
                state.calibration = scale_to_frame(state.calibration, state.frame_height_px)
                state.rescaled_note = (
                    f"calibration was read at {declared} px high and frames arrive at "
                    f"{state.frame_height_px} px; the focal length and horizon were rescaled "
                    "accordingly"
                )

        fix = GeoFix(
            lat=request.lat,
            lon=request.lon,
            accuracy_m=request.accuracy_m,
            heading_deg=request.heading_deg,
            speed_mps=request.speed_mps,
            timestamp_s=request.t or (time.time() - state.created_s),
        )
        result = state.session.ingest(image, fix, request.seq)
        state.note_cost(result.processing_ms)
        state.note_arrival()
        state.last_seen_s = time.time()

        if len(state.session.crops) > MAX_CROPS:
            for key in list(state.session.crops)[:-MAX_CROPS]:
                state.session.crops.pop(key, None)

        payload = result.to_dict()
        for event in payload["events"]:
            event["approval"] = state.approvals.get(event["defect_id"])
        if state.rescaled_note:
            payload["warnings"] = list(payload["warnings"]) + [state.rescaled_note]
        payload["capture"] = _capture_advice(state, request.speed_mps)
        payload["stats"] = state.session.stats()
        return payload
    finally:
        state.lock.release()


@router.post("/sessions/{run_id}/finish")
def finish(run_id: str) -> dict[str, Any]:
    """Close the session and every track still open.

    A track still open has not reached its closest approach, so its measurement
    is the weakest it will ever be. It is closed and recorded as unconfirmed
    rather than discarded, because "we saw something here and could not size
    it" is a useful thing for a road authority to be told.
    """
    state = _state(run_id)
    with state.lock:
        events = state.session.finish()
        state.finished = True
        for event in events:
            event["approval"] = state.approvals.get(event["defect_id"])
        return {
            "run_id": run_id,
            "closing_events": events,
            "stats": state.session.stats(),
            "defects": list(state.session.defects.values()),
        }


@router.get("/sessions/{run_id}")
def get_session(run_id: str) -> dict[str, Any]:
    state = _state(run_id)
    defects = []
    for record in state.session.defects.values():
        item = dict(record)
        item["approval"] = state.approvals.get(item["defect_id"])
        defects.append(item)
    defects.sort(key=lambda d: (d["priority"], -d["area_m2"]))
    return {
        "run_id": run_id,
        "label": state.label,
        "finished": state.finished,
        "calibration": state.calibration,
        "defects": defects,
        "stats": state.session.stats(),
        "capture": _capture_advice(state, None),
    }


@router.get("/sessions/{run_id}/defects/{defect_id}/crop")
def get_crop(run_id: str, defect_id: str) -> Response:
    """The privacy-scrubbed evidence crop: what was measured, with its outline."""
    state = _state(run_id)
    crop = state.session.crops.get(defect_id)
    if not crop:
        raise HTTPException(status_code=404, detail=f"no evidence crop for {defect_id}")
    return Response(content=crop, media_type="image/jpeg")


@router.post("/sessions/{run_id}/defects/{defect_id}/approval")
def set_approval(run_id: str, defect_id: str, request: ApprovalRequest) -> dict[str, Any]:
    """The human gate, unchanged by being live.

    Urgent work is still not dispatched on the system's own judgement. Being in
    a moving vehicle is a reason for the gate to exist, not a reason to drop it:
    the operator can approve at the next stop, and nothing is committed until
    they do.
    """
    state = _state(run_id)
    record = state.session.defects.get(defect_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no such defect: {defect_id}")
    approval = {
        "approved": request.approved,
        "approver": request.approver or "unattributed",
        "note": request.note,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    state.approvals[defect_id] = approval
    record["approval"] = approval
    return {"defect_id": defect_id, "approval": approval, "dispatched": request.approved}


@router.get("/sessions/{run_id}/geojson")
def session_geojson(run_id: str) -> dict[str, Any]:
    """The work order, as something a GIS or a works-management system can read.

    Only approved defects are exported as dispatchable. Anything awaiting
    approval is exported too, flagged, so the export is a full picture rather
    than a filtered one.
    """
    state = _state(run_id)
    features = []
    for record in state.session.defects.values():
        position = record.get("position")
        if not position:
            continue
        approval = state.approvals.get(record["defect_id"])
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [position["lon"], position["lat"]],
                },
                "properties": {
                    "defect_id": record["defect_id"],
                    "label": record["label"],
                    "area_m2": record["area_m2"],
                    "priority": record["priority"],
                    "cost_estimate": record["cost_estimate"],
                    "currency": record["currency"],
                    "n_views": record["n_views"],
                    "area_iqr_ratio": record["area_iqr_ratio"],
                    "confirmed": record["confirmed"],
                    "needs_human_approval": record["needs_human_approval"],
                    "dispatchable": bool(approval and approval["approved"]),
                    "position_accuracy_m": position["accuracy_m"],
                    "position_method": position["method"],
                    "reasons": record["reasons"],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


@router.get("/ledger/geojson")
def ledger_geojson() -> dict[str, Any]:
    """Every located defect across every drive, with how its area has changed."""
    return LEDGER.to_geojson()


@router.get("/sessions")
def list_sessions() -> dict[str, Any]:
    with _sessions_lock:
        _evict_stale()
        return {
            "sessions": [
                {
                    "run_id": state.run_id,
                    "label": state.label,
                    "finished": state.finished,
                    "frames_measured": state.session.frames_measured,
                    "defects": len(state.session.defects),
                    "started": round(state.created_s, 1),
                }
                for state in _sessions.values()
            ],
            "ledger_entries": len(LEDGER.entries),
        }
