# Live capture: measuring from a phone, while driving

Open `/capture` on a phone mounted in the vehicle. Calibrate in about a minute,
start driving, and defects are measured, located, priced and shown as you pass
them — with the number firming up on screen as each defect is seen again from
closer in.

The phone is a sensor. Every figure it displays was computed server-side by the
OpenCV pipeline; nothing is measured in the browser.

---

## 1. What the operator does

**Mount and measure (once).** Mount the phone so it looks down the road and
will not move again. Measure the **lens height above the road** with a tape.
This is the one input that cannot be recovered later, and area scales as its
square: 5% out on the height is 10% out on every area, permanently and
invisibly.

**Calibrate (about a minute).** The page freezes a still. Drag the cyan line
onto the horizon, drag the amber line onto a point on the road at a distance
you have measured, and type that distance. That is two observations, which is
exactly enough to solve the two unknowns:

```
row = horizon + height · focal / distance   →   focal = (row − horizon) · distance / height
vfov = 2·atan((H/2)/focal)      pitch = atan((H/2 − horizon)/focal)
```

The server solves it, prints the recovered field of view and tilt, and shows
where the solution claims the road is at 2, 5, 10 and 20 m so the operator can
sanity-check it against what they can see. Implausible geometries are named,
including the one failure nothing downstream can catch — a camera height typed
in centimetres, which would scale every area by ten thousand and look entirely
confident doing it.

**Drive.** Press start. The page sends frames at whatever rate the server asks
for, and shows defects as they are measured.

### Why the horizon route rather than two markers

`scripts/calibrate.py` solves from two markers at known distances and needs no
horizon. That is the better route for a recorded clip, but on a phone the
horizon can simply be pointed at, and pointing at it is more accurate than
extrapolating to it:

```
y_horizon = y₁ − (y₁ − y₂)/(1 − d₁/d₂)
```

With markers at 3 m and 6 m the factor on the row difference is 2, so 2 px of
reading error on each row puts the horizon out by 4–6 px. A horizon line is a
boundary hundreds of pixels wide and is judged to 1–2 px. Measured on a
forward-simulated camera with ±2 px of error applied to each route's own
inputs: focal length off by 3.15% from two markers, 2.63% from horizon plus one
marker, and the horizon itself off by 6.0 px against 2.0 px. Both routes are
available; the page uses the second and `scripts/calibrate.py` uses the first.

---

## 2. Why this is not the batch pipeline with a different input

The batch session in `pipeline/runner.py` is **pull**-based: to track a defect
it reaches forward into frames that have already been decoded. A live stream
cannot do that — the frames that would confirm a measurement have not been
recorded yet.

`pipeline/live.py` inverts it. State is carried forward rather than fetched
backward: each arriving frame is rectified and proposed once, open tracks are
predicted into it from the vehicle's own motion, candidates are matched to
those predictions, and each match appends one more independent measurement to a
track that is already open.

Three things follow that the batch path cannot do.

**The number visibly firms up.** A defect is reported the moment it is seen, as
*provisional*, with one view behind it. Each further view re-reports it with a
tighter spread, and at three comparable views it becomes *confirmed*. In the
batch path the same process is invisible because it finishes before anything is
displayed.

Confirmation is also allowed to go **backwards**. It was sticky at first, which
produced a defect labelled confirmed whose own reasons said its spread could
not be established — a later, closer view had been discarded by the vantage
filter, dropping the comparable count from three to two, and the earlier
confirmation stayed on the record contradicting it. Evidence that weakens is
now displayed as weaker, as `defect_weakened`.

**Ego-motion comes from the speedometer.** Frame-to-frame forward distance is
the fix's own speed times the interval. It is Doppler-derived, it does not care
whether the asphalt has texture, and it is exact on a synthetic drive (10.2 m
measured against 10.2 m truth). The image-shift estimate is kept as a
cross-check rather than discarded, because a disagreement between the two is
the earliest available signal that **the mount has moved** — which invalidates
the calibration and therefore every area measured after it. It runs every fifth
frame rather than every frame: at 11.9 ms it was the single most expensive
operation in the budget, more than rectification, proposal and refinement
combined, and a mount-movement check that runs once a second catches a moved
mount just as well as one that runs five times a second.

**Turning is handled.** A defect ahead of a turning vehicle sweeps sideways by
its forward distance times the course change. At 8 m ahead a 10° turn moves it
1.4 m — three times the 45 cm match radius — so prediction that ignores yaw
loses the defect after one frame and silently destroys the multi-view
agreement. Course change comes from consecutive GPS fixes, is capped at 30° per
frame so a bad fix cannot throw every track sideways, and the synthetic rig in
`tests/ground.py` was extended with a `CameraPose` and `arc_path` so curved
driving can be rendered and the prediction tested at all.

---

## 3. The speed limit, and why it is a geometric fact

A defect must be seen at least three times **from comparable distances** before
its spread means anything. Only views whose source footprint is within
`VANTAGE_FOOTPRINT_FRACTION` of the best view's contribute, and projected
ground area falls as roughly the cube of distance, so a fraction *v* admits a
distance band of *v*^(−1/3). That band, near the bird's-eye view's own near
edge, is a couple of metres of road. Everything about the capture rate follows
from it.

`eval/live_eval.py` sweeps it — 128 drives over four defect sizes, four frame
spacings and two noise seeds:

| vantage | band | bias at 1.2 m spacing | confirmed |
|---|---|---|---|
| 0.10 | 2.15× | −18.7% | 8/8 |
| **0.18** | **1.77×** | **−17.9%** | **8/8** |
| 0.25 (was) | 1.59× | −20.0% | 7/8 |
| 0.40 | 1.36× | −20.0% | 1/8 |

Accuracy is flat across the whole range — within 3 points — while the
confirmation rate collapses. The old 0.25 had been calibrated against the
*first* synthetic rig, which regenerated surface noise every frame and so made
distant views look catastrophic; on the persistent ground plane that replaced
it they are only mildly worse. **Changed to 0.18.**

At 2.0 m spacing nothing confirms at any vantage. 1.2 m confirmed 8/8 in the
sweep but proved to be the edge of the cliff rather than a safe distance from
it: repeated runs through the real endpoint over the same three defects
confirmed between one and three of them depending on which views happened to
survive the filter. **The target is 0.8 m**, which used 3.9 comparable views per
defect instead of 3.2 — the margin that turns a coin flip into a measurement.

So the capture rate is `0.8 m ÷ speed`, computed per frame and returned to the
phone, and the honest consequence is stated rather than hidden:

| speed | frames needed | at 1280 px / 50 KB |
|---|---|---|
| 10 km/h | 3.5 fps | 175 KB/s |
| 15 km/h | 5.2 fps | 260 KB/s |
| 25 km/h | 8.7 fps | 435 KB/s |

**A phone on mobile data measures reliably at survey speed, not traffic
speed.** The server measures what the link is actually achieving — from the
arrival intervals of frames, which contain the upload time that processing time
alone does not — and tells the operator the speed this particular setup can
measure from. Above it, defects are still reported, marked unconfirmed, and not
costed.

### Upload resolution decides whether anything confirms at all

Resolution does not merely sharpen a boundary. A measurement is only accepted
once the defect's footprint in the source frame clears a floor, and footprint
falls as the *square* of the upload width — so halving the width quarters it
and collapses the band over which a defect is measurable. One drive past two
potholes and a shadow at 22 km/h, through the real endpoint:

| upload width | views per defect | confirmed | spread on the larger pothole |
|---|---|---|---|
| 640 | 2 | 0 of 3 | could not be established |
| 800 | 3 | 1 of 3 | 3.7% |
| 960 | 3 | 1 of 3 | 2.8% |
| **1280** | **5** | **3 of 3** | **0.9%** |

JPEG *quality*, swept from 30 to 70 at 1280 px, moved the measured areas by
less than the run-to-run variation. Fidelity is not the constraint; resolution
is. The page uploads 1280 px wide at quality 0.5, about 50 KB a frame.
Dropping the resolution to save bandwidth does not buy a rougher measurement —
it buys none.

---

## 4. Position: measured differently from area, reported separately

Area comes from image geometry and does not involve GPS. Position comes from a
consumer GNSS chip behind a windscreen and is good to somewhere between 4 and
20 m. Conflating the two accuracies is the easiest way to make the system
untrustworthy, so every position carries its own figure and its own provenance:

- `gps_heading` — the fix's own course, used only above 1.5 m/s, since a course
  reported while stationary is stale or noise.
- `track_heading` — derived from consecutive fixes, for a chip that never
  reports course. Two fixes only define a direction if they are further apart
  than their own uncertainty; otherwise `None` is returned and the next route
  is used.
- `camera_only` — no usable heading, so the defect is placed **at the camera**
  and its accuracy widened by the distance it lies ahead. Still useful: a crew
  looking for a 0.4 m pothole within 20 m of a point on a road will find it.
  Inventing a heading to make the number look tighter would not be.

With a heading, the accuracy is `hypot(fix accuracy, ahead × 8°)` — the lever
arm of an angular error at the defect's distance.

### Two kinds of duplicate, handled differently

**Within one drive**, positions are related by rectified image geometry and
ego-motion, which is sub-metre. Two detections within 0.6 m are the same defect
and are merged.

**Across drives**, only GNSS relates them. A nearby defect is reported as a
*revisit candidate*, with the distance and the radius the two fixes can
actually resolve, and is **never silently merged**. Merging two real potholes
three metres apart because the GPS was off by five would quietly halve a road
authority's defect count. A confirmed revisit carries an area trend, which is a
different maintenance decision — a pothole that grew 40% in a month — and
something a one-shot detector cannot produce.

`GET /api/live/ledger/geojson` returns the cross-drive record;
`GET /api/live/sessions/{id}/geojson` returns one drive's work order, with
every defect flagged `dispatchable` only once a human has approved it.

---

## 5. Honest limits

**Measurement accuracy on hard cases is poor.** On a held-out grid of 15 drives
over five radii (0.16–0.38 m), three lateral offsets, three start distances and
three unused noise seeds: **bias −0.7%, MAE 18.5%, worst case 56%**, with 11 of
12 measurable defects confirmable at 0.8 m spacing. The easier holdout in
`accuracy.md` reports 6.5% MAE; small and distant defects are substantially
worse than that, and a large defect and a small one in the same drive have come
back −12% and +81%.

The closing kernel was investigated as the cause and looked like a 3.5×
improvement in MAE on a development grid. It did not replicate on held-out
data and **was not changed** — see `negative-results.md` §7. That is the third
tuning change in this project to look strong on a development grid and fail on
held-out data, which is itself the finding: per-configuration variance on the
synthetic rig is large compared with the effects being chased. **No further
synthetic tuning is planned.** The remaining question — whether the error is
the algorithm or the rig's own projection chain — cannot be answered by the rig
under suspicion, so the binding requirement is real footage with hand-measured
areas.

**Privacy.** Faces and plates are blurred server-side before any evidence crop
is retained, and raw frames are never written to disk. TLS protects the frame
in transit; the scrub protects what is stored and shown. The retained crop is
the scrubbed bird's-eye patch around the defect with its measured outline
drawn on it — enough to audit the measurement, not a photograph of the street.

**A secure origin is required.** Browsers give neither camera nor location to a
page served over plain HTTP, except on `localhost`. Deployed behind CloudFront
this is automatic; for a laptop test, use `localhost` on the same device or an
HTTPS tunnel.

**Curves are predicted but not yet validated on real curved footage.** The yaw
term is correct on the synthetic arc rig and is exercised by
`scripts/simulate_phone.py --radius-m`, but a tight real junction has not been
driven.

---

## 6. Trying it without a vehicle

`scripts/simulate_phone.py` speaks the same API the phone does — same JPEG
round trip, same base64 framing, same server-side calibration, same
one-frame-at-a-time lock, same adaptive rate — so the live path can be
demonstrated and regression-tested from a laptop:

```bash
python -m scripts.simulate_phone --base http://127.0.0.1:8061 --speed-kmh 12
python -m scripts.simulate_phone --base http://127.0.0.1:8061 --speed-kmh 35   # refuses to confirm, and says why
python -m scripts.simulate_phone --base http://127.0.0.1:8061 --radius-m 40 --realtime
```

Run it twice and the second drive reports revisit candidates against the first,
which is the cross-drive ledger working.

---

## 7. API

| method | path | purpose |
|---|---|---|
| `POST` | `/api/live/sessions` | solve and check the calibration, open a session |
| `POST` | `/api/live/sessions/{id}/frames` | measure one frame; returns everything it found |
| `POST` | `/api/live/sessions/{id}/finish` | close every open track and record it |
| `GET` | `/api/live/sessions/{id}` | the drive so far |
| `GET` | `/api/live/sessions/{id}/defects/{did}/crop` | scrubbed evidence crop with the measured outline |
| `POST` | `/api/live/sessions/{id}/defects/{did}/approval` | the human gate |
| `GET` | `/api/live/sessions/{id}/geojson` | this drive's work order |
| `GET` | `/api/live/ledger/geojson` | every located defect, across every drive |

Plain HTTP, one POST per frame, no WebSocket. The request/response shape *is*
the semantics — "measure this frame and tell me what you found" is a call that
returns a value — and it is self-limiting: the phone does not send frame N+1
until N returns, so the rate settles at what the server can actually measure. A
socket would let the phone fill a queue it cannot drain, and by the time a
queued frame is measured the vehicle is thirty metres past the defect. It also
deploys behind API Gateway and CloudFront without a second product, and adds no
dependency to a submission that must be reproducible from `requirements.txt`.

Caps are enforced rather than assumed: 8 concurrent sessions, 4000 frames per
session, 2 MB per frame, 400 retained crops, 3-hour session expiry.
