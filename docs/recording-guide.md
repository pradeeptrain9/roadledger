# Recording your own footage

This is the highest-value missing input. With a camera whose geometry you have
actually measured, rectification is exact rather than assumed, which unlocks
three things nothing else can:

* **Real multi-frame agreement** — `iqr_ratio`, ego-motion, vantage filtering
  and deduplication currently have no real data at all, because RDD2022 is
  stills.
* **Real area ground truth** — RDD2022 annotates boxes, not areas. With known
  calibration you can hand-measure a defect and compare.
* **Honest demo footage** — the video shows the system on a real road rather
  than on a simulation.

Budget about 30 minutes, including the calibration.

---

## A. Record the calibration clip first (5 minutes, do not skip)

This is what converts every later measurement from an assumption into a
measurement. You need a tape measure and two objects that sit flat and are
clearly visible — a book, a shoe, a water bottle on its side.

1. Park on flat ground, pointing along the road. **Mount the phone where it
   will stay for the whole session** — a windscreen or dashboard mount,
   landscape orientation. Once mounted, do not move or re-aim it.
2. Measure the height of the **phone's lens** above the road surface with the
   tape. Write it down to the nearest centimetre. This is the single most
   important number: an error here scales every area by its square.
3. Place object A on the road **exactly 3.00 m** ahead of the lens, in the
   middle of the view. Measure along the ground from directly below the lens.
4. Place object B **exactly 6.00 m** ahead, same line.
5. Record 10 seconds with both objects in frame, stationary.
6. Remove the objects. **Do not adjust the phone.**

Two distances at known ranges are enough to solve for both the focal length and
the horizon row, so the field of view and the downward tilt do not have to be
looked up or guessed.

## B. Record the driving clip (10–15 minutes)

* **Same mount, untouched.** If the phone moves, the calibration is void and
  section A has to be repeated.
* **1080p, 30 fps**, landscape. Turn off any "stabilisation" the camera app
  offers — it warps the frame geometry and invalidates the rectification.
* Drive **slowly, 20–30 km/h**, on a road you know has potholes or broken
  surface. Slow is important: the system measures the same defect from several
  frames as you approach, and at speed there are too few.
* **Straight stretches.** Ego-motion between frames is estimated as a
  translation, so sharp turns break the chain. Gentle curves are fine.
* **Daylight, dry, no low sun straight ahead.** Wet road is worth a second clip
  if you can — standing water is one of the confusers the classifier is
  supposed to reject, and real examples would be genuinely useful.
* **2–3 minutes is plenty.** At 30 fps that is thousands of frames; the system
  samples a few per second.
* Note roughly where the worst potholes were, so they can be found in the clip.

## C. Hand-measure a few defects (10 minutes, optional but valuable)

For real area ground truth, pick 10–20 defects you drove over and measure them
on foot with the tape: longest width, and the width perpendicular to it. Note
which part of the clip each one is in. That gives a genuine real-world area
error figure, which is the project's headline claim and currently has no real
data behind it at all.

Do this only where it is safe to stand on the road.

---

## Then

Drop both files in `roadledger/data/footage/` and tell me. Then:

```bash
# 1. Calibrate: click the two markers in the calibration frame
python -m scripts.calibrate --video data/footage/calibration.mp4 \
    --camera-height 1.27 --distances 3.0 6.0

# 2. Run the full pipeline and the agent loop on the driving clip
python -m eval.real_video --video data/footage/drive.mp4 --calibration data/footage/calibration.json
```

## What can go wrong

| Problem | Consequence | Avoid by |
|---|---|---|
| Phone moved between clips | Calibration void, areas wrong by an unknown factor | One mount, untouched |
| Stabilisation left on | Frame geometry warps per frame; rectification invalid | Turn it off before recording |
| Driving too fast | Too few views per defect, so no spread and everything escalates | 20–30 km/h |
| Lens height guessed | Every area wrong by the square of the error | Use the tape |
| Portrait orientation | Much less road in frame, far worse coverage | Landscape |
| Heavy rain or dusk | Detection is poor enough in daylight | Dry daylight |
