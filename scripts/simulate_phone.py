"""Drive the live capture endpoint exactly as the phone does.

The phone page and this script speak the same API, which is the point: the live
path can be exercised, demonstrated and regression-tested without a vehicle, a
mount or daylight, and a judge can reproduce the whole thing from a laptop.

It also tests what a unit test cannot. The pipeline is tested directly
elsewhere; what is at stake here is the *contract* - the JPEG round trip, the
base64 framing, the calibration being solved server-side, the one-frame-at-a-
time lock, the capture rate the server asks for, and the geographic ledger
surviving across two separate drives down the same road.

    python -m scripts.simulate_phone --base http://127.0.0.1:8061
    python -m scripts.simulate_phone --base http://127.0.0.1:8061 --realtime --speed-kmh 22
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
import urllib.error
import urllib.request

import cv2

from pipeline.geo import metres_per_degree
from tests.ground import build_ground, render_drive
from tests.synthetic import SceneCamera

START_LAT, START_LON = 17.4401, 78.3489


def post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{url} -> {error.code}: {error.read().decode()[:400]}")


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8061")
    parser.add_argument("--speed-kmh", type=float, default=22.0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--radius-m", type=float, default=None,
                        help="bend the driven path into an arc of this radius")
    parser.add_argument("--realtime", action="store_true",
                        help="honour the interval the server asks for, as the phone does")
    parser.add_argument("--jpeg-quality", type=int, default=62)
    parser.add_argument("--upload-width", type=int, default=1280,
                        help="frame width sent to the server; see docs/live-capture.md")
    parser.add_argument("--label", default="simulated phone")
    parser.add_argument("--quiet", action="store_true", help="per-frame lines off")
    args = parser.parse_args()

    camera = SceneCamera()
    # The frame the phone would upload: 640 px wide, same aspect as the camera.
    upload_w = args.upload_width
    upload_h = round(camera.height * upload_w / camera.width)

    speed_mps = args.speed_kmh / 3.6
    # The server will ask for whatever spacing the measurement needs; start from
    # its own published constant so the first few frames are not wasted.
    from web.live import MAX_FRAME_SPACING_M

    step_m = MAX_FRAME_SPACING_M
    fps = speed_mps / step_m

    print(f"{args.speed_kmh:.0f} km/h -> {step_m} m between frames -> {fps:.1f} fps")

    ground = build_ground(
        defects=[(-0.45, 11.0, 0.34), (0.8, 20.0, 0.21)],
        confusers=[(0.3, 15.0, 0.30, "shadow")],
        seed=5,
        length_m=90.0,
    )
    images, _ = render_drive(
        ground, camera, n_frames=args.frames, step_m=step_m, seed=21, radius_m=args.radius_m
    )

    # Calibration, read off the frame the way the operator reads it off the
    # preview: the horizon row and one marker at a measured distance.
    focal_full = camera.focal_px
    horizon_full = camera.horizon_y
    marker_d = 5.0
    marker_row_full = horizon_full + camera.height_m * focal_full / marker_d
    scale = upload_h / camera.height
    session = post(
        f"{args.base}/api/live/sessions",
        {
            "calibration": {
                "height_m": camera.height_m,
                "horizon_px": round(horizon_full * scale, 1),
                "marker_distance_m": marker_d,
                "marker_row_px": round(marker_row_full * scale, 1),
                "image_height_px": upload_h,
            },
            "label": args.label,
            "privacy": False,
        },
    )
    run_id = session["run_id"]
    print(f"session {run_id}")
    print("  solved:", {k: session["calibration"][k] for k in ("vfov_deg", "pitch_deg", "focal_px", "horizon_px")})
    if session["calibration_warnings"]:
        print("  WARNINGS:", session["calibration_warnings"])
    advice = session["capture"]

    per_lat, _ = metres_per_degree(START_LAT)
    confirmed_seen: set[str] = set()

    for seq, image in enumerate(images):
        frame = cv2.resize(image, (upload_w, upload_h), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        if not ok:
            continue
        started = time.perf_counter()
        payload = post(
            f"{args.base}/api/live/sessions/{run_id}/frames",
            {
                "seq": seq,
                "jpeg_b64": base64.b64encode(buffer.tobytes()).decode(),
                "lat": START_LAT + (seq * step_m) / per_lat,
                "lon": START_LON,
                "accuracy_m": 6.0,
                "heading_deg": 0.0,
                "speed_mps": speed_mps,
                "t": seq / fps,
            },
        )
        advice = payload.get("capture", advice)
        kb = len(buffer) / 1024
        events = payload.get("events", [])
        line = (
            f"  seq {seq:3d} {kb:5.1f} KB  q={payload.get('rectification_quality', 0):.2f} "
            f"step={payload.get('step_m', 0):.2f}m  {payload.get('processing_ms', 0):5.0f} ms"
        )
        for event in events:
            kind = event["type"].replace("defect_", "")
            line += f"  [{kind} {event['area_m2']:.3f} m2 /{event['n_views']}v]"
            if event.get("confirmed") and event["defect_id"] not in confirmed_seen:
                confirmed_seen.add(event["defect_id"])
        if payload.get("type") == "dropped":
            line += "  DROPPED"
        if not args.quiet:
            print(line)
        for warning in payload.get("warnings", []):
            print("      warn:", warning)
        if advice.get("too_fast_to_measure"):
            print("      too fast:", advice["note"])
        if args.realtime:
            remaining = advice["recommended_interval_s"] - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)

    final = post(f"{args.base}/api/live/sessions/{run_id}/finish", {})
    print("\nwork order")
    for defect in sorted(final["defects"], key=lambda d: (d["priority"], -d["area_m2"])):
        position = defect.get("position") or {}
        print(
            f"  {defect['label']:8s} {defect['area_m2']:.3f} m2  p{defect['priority']}  "
            f"{defect['n_views']} views  iqr={defect['area_iqr_ratio']}  "
            f"{'CONFIRMED' if defect['confirmed'] else 'unconfirmed'}  "
            f"cost={defect['cost_estimate']} {defect['currency']}"
        )
        if position:
            print(
                f"           at {position['lat']:.6f},{position['lon']:.6f} "
                f"+/-{position['accuracy_m']:.0f} m via {position['method']}"
            )
        for reason in defect["reasons"]:
            print("           -", reason)

    stats = final["stats"]
    print("\nstats")
    for key in (
        "frames_received", "frames_measured", "frames_dropped_behind",
        "frames_skipped_blurry", "frames_skipped_unrectifiable",
        "distance_m", "defects_total", "defects_confirmed",
        "defects_needing_approval", "total_cost", "ledger_entries",
    ):
        print(f"  {key:30s} {stats[key]}")

    geojson = get(f"{args.base}/api/live/sessions/{run_id}/geojson")
    print(f"\ngeojson: {len(geojson['features'])} features")
    ledger = get(f"{args.base}/api/live/ledger/geojson")
    print(f"ledger:  {len(ledger['features'])} located defects across every drive so far")
    for feature in ledger["features"]:
        properties = feature["properties"]
        if properties.get("n_sightings", 0) > 1 or properties.get("area_trend_m2") is not None:
            print(
                f"  {properties['entry_id']}: seen {properties['n_sightings']}x, "
                f"area trend {properties['area_trend_m2']} m2"
            )


if __name__ == "__main__":
    main()
