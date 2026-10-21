"""Record exactly what the benchmark ran on.

The COOL award asks for the COOL version, the AWS deployment configuration, and
evidence that COOL executes the claimed core workload. None of that is
trustworthy if it is typed into a report by hand, so it is captured by the
benchmark itself and written into its own output file.

The COOL detection is deliberately blunt: the Marketplace AMI installs to
/opt/cool, so an imported cv2 whose module file lives under that prefix is the
COOL build and one that does not, is not. `claims_cool` is never set from a
flag the operator passes.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

COOL_PREFIX = "/opt/cool"


@dataclass
class Environment:
    hostname: str
    architecture: str
    platform: str
    python_version: str
    cv2_version: str
    cv2_path: str
    numpy_version: str
    cpu_count: int
    cv2_threads: int
    # True only if the imported cv2 actually lives under the COOL prefix.
    is_cool_build: bool = False
    cool_prefix_present: bool = False
    kleidicv_in_build: bool = False
    cpu_features: str = ""
    instance_type: str = ""
    instance_region: str = ""
    instance_id: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def arm(self) -> bool:
        return self.architecture.lower() in {"aarch64", "arm64"}

    def label(self) -> str:
        """Short name for this arm of the comparison."""
        build = "COOL" if self.is_cool_build else "pip-opencv"
        where = self.instance_type or ("local-" + self.architecture)
        return f"{build} / {'arm64' if self.arm else 'x86_64'} / {where}"


def _imds(path: str, timeout: float = 0.4) -> str:
    """Query EC2 instance metadata, IMDSv2 first. Empty string off EC2."""
    try:
        import urllib.request

        token_request = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(token_request, timeout=timeout) as response:
            token = response.read().decode()
        data_request = urllib.request.Request(
            f"http://169.254.169.254/latest/meta-data/{path}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(data_request, timeout=timeout) as response:
            return response.read().decode().strip()
    except Exception:
        return ""


def detect() -> Environment:
    build_info = cv2.getBuildInformation()
    cpu_lines = [
        line.strip()
        for line in build_info.splitlines()
        if "Baseline:" in line or "requested:" in line
    ]

    cv2_path = getattr(cv2, "__file__", "") or ""
    is_cool = cv2_path.startswith(COOL_PREFIX)

    environment = Environment(
        hostname=platform.node(),
        architecture=platform.machine(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        cv2_version=cv2.__version__,
        cv2_path=cv2_path,
        numpy_version=np.__version__,
        cpu_count=os.cpu_count() or 0,
        cv2_threads=cv2.getNumThreads(),
        is_cool_build=is_cool,
        cool_prefix_present=os.path.isdir(COOL_PREFIX),
        kleidicv_in_build="kleidicv" in build_info.lower(),
        cpu_features=" | ".join(cpu_lines)[:300],
        instance_type=_imds("instance-type"),
        instance_region=_imds("placement/region"),
        instance_id=_imds("instance-id"),
    )

    # The honesty checks that decide what may be claimed.
    if environment.cool_prefix_present and not is_cool:
        environment.notes.append(
            f"{COOL_PREFIX} exists but the imported cv2 is {cv2_path} - this run is "
            "NOT using COOL. Activate /opt/cool/venvs/python_3.12 or set "
            "LD_LIBRARY_PATH and PYTHONPATH per the Marketplace AMI instructions."
        )
    if is_cool and not environment.cv2_version.startswith("5"):
        environment.notes.append(
            f"COOL build reports OpenCV {environment.cv2_version}, not 5.x. The "
            "competition requires OpenCV 5 for the substantive analysis, so this "
            "must be reported as a split: COOL accelerates the core ops on Arm "
            "while OpenCV 5 performs the analysis. Do not claim OpenCV 5 for COOL."
        )
    if is_cool and not environment.arm:
        environment.notes.append(
            "COOL build detected on a non-Arm machine, which the award's Arm-path "
            "requirement does not permit. Check the instance family."
        )
    return environment


def cool_version() -> str:
    """Best-effort COOL version string from the AMI's own files."""
    for candidate in ("VERSION", "version.txt", "cool_version.txt"):
        path = os.path.join(COOL_PREFIX, candidate)
        if os.path.isfile(path):
            try:
                with open(path) as handle:
                    return handle.read().strip()
            except OSError:
                continue
    # The Marketplace AMI records its version in the image description rather
    # than on disk, so fall back to the AMI id, which is what identifies it.
    ami = _imds("ami-id")
    return f"ami:{ami}" if ami else "unknown"


if __name__ == "__main__":
    detected = detect()
    print(json.dumps({**detected.to_dict(), "cool_version": cool_version()}, indent=2))
