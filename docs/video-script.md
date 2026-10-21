# Five-minute video script

Requirement: shows the team, the application working, its architecture, and the
principal results. Hard limit 5:00 — this runs about 4:40, leaving room.

Record the browser at 1280×800 with the pane at desktop width. Have the server
running and the "Two potholes, a shadow and a wet stain" scenario loaded.

---

## 0:00–0:20 — Who, and the problem

> On camera, briefly.

"I'm [name]. A road authority doesn't need to be told a pothole exists — its
inspectors already know. What it can't produce cheaply is the thing that
actually governs spending: a ranked list of defects with quantities. How big,
how urgent, what it costs, and how much to trust the number.

So RoadLedger measures road defects instead of just detecting them."

---

## 0:20–1:10 — The application working

> Screen: the live endpoint. Scroll to the route plan.

"This is a dashcam pass over a stretch of road. Circles are what the system
measured; crosses are the ground truth.

Two real potholes, measured at 0.38 and 0.14 square metres against a truth of
0.41 and 0.13. Position within 15 centimetres.

There's also a shadow and a patch of standing water in this footage. Neither is
costed — and that matters, because dispatching a crew to a shadow is the most
expensive mistake a detection-only system makes."

> Scroll to the ledger. Point at the per-view numbers under each area.

"Each area shows the individual measurements behind it. The first pothole was
measured five times from five distances and they agree to within 4.6%. The
second disagrees by 24% — and the system says so rather than hiding it."

---

## 1:10–2:10 — Why it's an agent, not a pipeline

> Scroll to the decision trace.

"This is the part that makes it a loop rather than a pipeline. Every one of
these decisions thresholds a number the vision stages produced, and changes
which tool gets called next.

Here: `measurement_spread_too_wide` — the views disagreed by 24%, over the 20%
threshold, so the agent re-tracked across ten frames instead of four. It bought
more evidence because the measurement told it to.

Here: a candidate whose source footprint was 23 pixels. Below the floor, so it
is deferred — no area is reported for it at all. The system would rather admit
it can't size something than hand you a confident wrong number.

And here: priority one. However well those five views agreed, an urgent repair
is never dispatched unreviewed."

> Click Approve, with a name in the field.

"That's the human gate. Now it's a work order, and who approved it is on the
record."

---

## 2:10–2:50 — Architecture

> Screen: docs/architecture.svg.

"Video lands in S3. A Lambda starts an ephemeral Graviton worker running the
Cloud Optimized OpenCV AMI, which serves the pipeline as eleven MCP tools. The
agent calls those tools; results and the full trace go to DynamoDB; the
reviewer sees it through CloudFront.

The worker is stopped between runs, so the endpoint answers from cached
results. A demo that only works while an instance is burning isn't a demo.

The green path is where COOL executes the core image workload on Arm."

---

## 2:50–3:50 — COOL, and two findings

> Screen: docs/cool-benchmark.md, the two tables.

"Two things I found building the benchmark changed how I report it.

First — the standard pip OpenCV 5 wheel for arm64 **already bundles KleidiCV**.
COOL's advertised twenty to seventy percent was measured against OpenCV 4.12,
which predates that. So pip-OpenCV-5-on-Graviton is already accelerated, and
COOL against it is a much harder comparison than COOL against OpenCV 4. The
benchmark has four arms, not three, and I report both.

Second, and more important — when I first measured it, the three operations
COOL accelerates were **six percent** of the core pass. A seventy percent
speedup on them would have moved the product by four percent.

So I profiled. The pass was dominated by `np.median` at thirty-eight percent, a
fifty-one pixel dilation, and a full-frame mask allocated per candidate. None
of that is vision work. Replacing the median with an exact histogram and fixing
the other two made the pass **seven times faster** — and took the
COOL-accelerated operations from six percent of it to **forty-six**.

That's the point. Measuring COOL now measures the workload that actually
carries the product. That evidence didn't exist before profiling."

---

## 3:50–4:25 — Results, honestly

> Screen: docs/accuracy.md, the two tables.

"Fifteen runs against exact ground truth. Mean absolute error on area is
sixteen percent.

But the number I care about is this one. Split by what the system decided about
itself: where it reported confidence, error is eleven percent, worst case
fourteen. Where it escalated instead, eighteen percent, worst case thirty-five.

**The confidence signal is calibrated.** The cases it flags for review really
are the less accurate ones. That's what makes acting on the spread legitimate
rather than decorative."

---

## 4:25–4:40 — What I got wrong

> Screen: docs/negative-results.md.

"Six hypotheses tested and rejected, documented, because a report that lists
only what worked isn't evidence.

My ego-motion estimator failed three times. The cause, found by brute-force
correlation scanning: lane markings dominate the view and don't move along the
direction of travel. Driving along a solid line looks like standing still.
Excluding paint took it from three correct frame pairs out of thirteen to
twelve.

And my privacy filter merged the whole road into one 'licence plate' and blurred
it — rectification quality went from 0.94 to zero. An over-inclusive privacy
filter is not automatically a safe one.

All of this is synthetic so far. Re-establishing it on RDD2022 is the next step,
and I've said so everywhere the numbers appear."

---

## Shot list

| Time | Screen |
|---|---|
| 0:00 | presenter |
| 0:20 | live endpoint, route plan |
| 0:50 | ledger, per-view measurements |
| 1:10 | decision trace, three rule firings |
| 1:55 | approval gate click |
| 2:10 | architecture.svg |
| 2:50 | cool-benchmark.md tables |
| 3:50 | accuracy.md tables |
| 4:25 | negative-results.md |

## Do not

* Claim OpenCV 5 performance for COOL if the AMI reports 4.x — show the split.
* Show a benchmark arm that hasn't been run. Say which are outstanding.
* Describe any accuracy figure as being from real road footage.
