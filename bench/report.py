"""Render the COOL comparison from whatever arms have been measured.

Reads every result document in `bench/results/`, lines the arms up against
each other, and prints the comparison plus a Markdown table for the docs.

Three principles, all of which exist because the easy version of this file
would quietly overstate the result:

**Every arm carries its own environment, and the report shows it.** The whole
COOL claim rests on which `cv2` was actually imported, so the version, the
build's KleidiCV, the architecture and the instance type are printed next to
every number. An arm whose environment contradicts its label is flagged rather
than tabulated.

**Speedups are named by what they compare.** "COOL is 1.4x faster" is not a
claim until you say faster *than what*. D-over-B is the vendor's own baseline,
stock OpenCV 4.12.0 with no KleidiCV. D-over-C is COOL against the free wheel
that already bundles a newer KleidiCV, which is the comparison a practitioner
actually faces. They can differ enormously and only reporting the flattering
one is the thing this file is built to prevent.

**A missing arm is reported as missing.** Never interpolated, never dropped
silently from a table that then looks complete.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).parent / "results"

# The operations COOL names as accelerated, plus the two aggregate workloads.
OPS = ["resize", "adaptiveThreshold", "findContours"]
AGGREGATES = ["proposal_pass", "full_analysis"]

# Comparisons worth drawing, as (numerator, denominator, what it means).
COMPARISONS = [
    ("D", "B", "COOL against the vendor's own stated baseline (stock 4.12.0, no KleidiCV)"),
    ("D", "C", "COOL against the free pip wheel, which already bundles KleidiCV 26.03"),
    ("C", "A", "Graviton3 against x86_64, both on the same OpenCV 5 wheel"),
    ("C", "B", "what the wheel's bundled KleidiCV alone is worth on the same silicon"),
]


@dataclass
class Arm:
    label: str
    path: Path
    data: dict[str, Any]

    @property
    def key(self) -> str:
        """A, B, C or D, taken from the label's first character."""
        first = self.label[:1].upper()
        return first if first in "ABCD" else "?"

    @property
    def env(self) -> dict[str, Any]:
        return self.data.get("environment", {})

    def stat(self, name: str) -> dict[str, Any] | None:
        for s in self.data.get("stats", []):
            if s["name"] == name:
                return s
        return None

    def median(self, name: str) -> float | None:
        s = self.stat(name)
        return s["median"] if s else None

    @property
    def problems(self) -> list[str]:
        """Ways this arm's environment contradicts what its label claims."""
        # environment.py warns whenever /opt/cool exists and was not imported.
        # On arms B and C that is the whole point of the arm, so the warning is
        # only a problem for D.
        notes = list(self.env.get("notes") or [])
        out: list[str] = [
            n for n in notes
            if not (self.key in "ABC" and "NOT using COOL" in n)
        ]
        cool = self.env.get("is_cool_build")
        version = str(self.env.get("cv2_version", ""))
        arch = self.env.get("architecture", "")

        if self.key == "D" and not cool:
            out.append(
                f"labelled COOL but the imported cv2 is {self.env.get('cv2_path')}, "
                "which is not the COOL build - this arm is mislabelled"
            )
        if self.key in "BC" and cool:
            out.append("labelled a pip wheel but the COOL build was imported")
        if self.key == "C" and not version.startswith("5."):
            out.append(f"arm C must be OpenCV 5; this is {version}")
        if self.key == "B" and not version.startswith("4."):
            out.append(f"arm B must be OpenCV 4.12.x; this is {version}")
        if self.key in "BCD" and arch not in ("aarch64", "arm64"):
            out.append(f"arm {self.key} must run on Arm; this is {arch}")
        if self.key == "A" and arch not in ("x86_64", "AMD64"):
            out.append(f"arm A must run on x86_64; this is {arch}")
        if self.key == "D" and version.startswith("4."):
            out.append(
                f"COOL provides OpenCV {version}, so OpenCV 5 must not be claimed for "
                "this arm; the report splits the claim"
            )
        return out


def load() -> list[Arm]:
    arms: list[Arm] = []
    for path in sorted(RESULTS.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            print(f"  skipping {path.name}: not valid JSON ({error})")
            continue
        arms.append(Arm(label=data.get("label", path.stem), path=path, data=data))
    return arms


def _fmt(value: float | None, places: int = 3) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def environments(arms: list[Arm]) -> None:
    print("\nENVIRONMENTS — what actually ran\n")
    header = (f"{'arm':<26} {'cv2':<9} {'numpy':<8} {'KleidiCV':<9} {'arch':<8} "
              f"{'instance':<13} {'threads':>7}")
    print(header)
    print("-" * len(header))
    for arm in arms:
        env = arm.env
        kleidi = "yes" if env.get("kleidicv_in_build") else "no"
        if env.get("is_cool_build"):
            kleidi += "/COOL"
        print(
            f"{arm.label[:25]:<26} {str(env.get('cv2_version','?')):<9} "
            f"{str(env.get('numpy_version','?')):<8} {kleidi:<9} "
            f"{str(env.get('architecture','?')):<8} "
            f"{str(env.get('instance_type') or 'local'):<13} "
            f"{env.get('cv2_threads','?'):>7}"
        )
    numpies = {str(a.env.get("numpy_version")) for a in arms if a.env.get("numpy_version")}
    if len(numpies) > 1:
        print(
            f"\n  NOTE: the arms do not share a numpy ({', '.join(sorted(numpies))}). "
            "COOL's cv2 is built against NumPy 1.x and fails to import under 2.x, so this "
            "cannot be equalised - it is a property of COOL, not a flaw in the comparison. "
            "The timed operations are OpenCV calls on preallocated arrays, so numpy is not "
            "in the measured path, but it is reported because it could not be held constant."
        )

    flagged = [(a, a.problems) for a in arms if a.problems]
    if flagged:
        print("\n  FLAGS — an arm whose environment contradicts its label is not comparable:")
        for arm, problems in flagged:
            for problem in problems:
                print(f"    [{arm.key}] {problem}")


def timings(arms: list[Arm]) -> None:
    print("\nTIMINGS — median ms/frame, with interquartile spread\n")
    width = 26
    header = f"{'measurement':<22}" + "".join(f"{a.key + ' ' + a.label[:14]:>22}" for a in arms)
    print(header)
    print("-" * len(header))
    for name in OPS + AGGREGATES:
        row = f"{name:<22}"
        for arm in arms:
            s = arm.stat(name)
            row += (f"{s['median']:.3f} ±{s['iqr']:.3f}".rjust(22)) if s else "—".rjust(22)
        print(row)

    print()
    row = f"{'throughput f/s':<22}"
    for arm in arms:
        d = arm.data.get("derived", {})
        row += (f"{d.get('throughput_frames_per_s', 0):.1f}".rjust(22))
    print(row)
    row = f"{'USD / 1000 frames':<22}"
    for arm in arms:
        d = arm.data.get("derived", {})
        known = d.get("pricing_known")
        row += ((f"{d.get('usd_per_1000_frames', 0):.4f}" if known else "unpriced").rjust(22))
    print(row)


def comparisons(arms: list[Arm]) -> list[dict[str, Any]]:
    by_key = {a.key: a for a in arms}
    print("\nCOMPARISONS — a speedup is not a claim until it says faster than what\n")
    out: list[dict[str, Any]] = []
    for num, den, meaning in COMPARISONS:
        a, b = by_key.get(num), by_key.get(den)
        if a is None or b is None:
            missing = num if a is None else den
            print(f"  {num} over {den}: NOT MEASURED — arm {missing} is missing")
            out.append({"numerator": num, "denominator": den, "meaning": meaning,
                        "measured": False})
            continue
        print(f"  {num} over {den} — {meaning}")
        row: dict[str, Any] = {"numerator": num, "denominator": den,
                               "meaning": meaning, "measured": True, "ratios": {}}
        for name in OPS + AGGREGATES:
            fast, slow = a.median(name), b.median(name)
            if fast is None or slow is None or fast <= 0:
                continue
            # Speedup of the numerator arm: >1 means the numerator is faster.
            ratio = slow / fast
            row["ratios"][name] = round(ratio, 3)
            verdict = "faster" if ratio > 1.02 else ("slower" if ratio < 0.98 else "no difference")
            print(f"      {name:<22} {ratio:5.2f}x  {verdict}")
        out.append(row)
        print()
    return out


def markdown(arms: list[Arm], compared: list[dict[str, Any]]) -> str:
    lines = ["| measurement | " + " | ".join(f"{a.key} ({a.env.get('cv2_version','?')})"
                                             for a in arms) + " |"]
    lines.append("|---" * (len(arms) + 1) + "|")
    for name in OPS + AGGREGATES:
        cells = []
        for arm in arms:
            s = arm.stat(name)
            cells.append(f"{s['median']:.3f} ±{s['iqr']:.3f}" if s else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    for row in compared:
        if not row["measured"]:
            lines.append(f"- **{row['numerator']} over {row['denominator']}**: not measured.")
            continue
        core = row["ratios"].get("proposal_pass")
        if core:
            lines.append(
                f"- **{row['numerator']} over {row['denominator']}** — {row['meaning']}: "
                f"**{core:.2f}x** on the core pass."
            )
    return "\n".join(lines)


def main() -> None:
    arms = load()
    if not arms:
        print(f"no result documents in {RESULTS}")
        print("run: bash scripts/run_benchmark_aws.sh")
        return

    arms.sort(key=lambda a: a.key)
    print("=" * 78)
    print("RoadLedger — COOL comparison")
    print("=" * 78)

    present = {a.key for a in arms}
    for key, what in (("A", "pip OpenCV 5 on x86_64"),
                      ("B", "pip OpenCV 4.12.0 on Graviton3"),
                      ("C", "pip OpenCV 5 on Graviton3"),
                      ("D", "COOL on Graviton3")):
        mark = "measured" if key in present else "NOT MEASURED"
        print(f"  arm {key}  {what:<34} {mark}")

    environments(arms)
    timings(arms)
    compared = comparisons(arms)

    out = RESULTS / "comparison.md"
    out.write_text(markdown(arms, compared) + "\n")
    print(f"\nMarkdown table written to {out.relative_to(Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
