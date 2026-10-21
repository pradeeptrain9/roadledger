"""Fetch a slice of RDD2022 without downloading 13 GB.

The per-country archives the dataset's README links to are no longer public -
every one of those S3 URLs returns 403 - and the figshare mirror offers only
the full 13.2 GB bundle. With 2.9 GB of free disk that is not an option, and it
would be a wasteful way to obtain the few hundred images an evaluation needs.

The figshare host does honour HTTP range requests, so this reads the zip
remotely: the End of Central Directory record from the tail, then the central
directory, then only the byte ranges of the members actually wanted. A few
hundred images cost a few tens of megabytes instead of 13 GB.

Three wrinkles handled. The archive is ZIP64 (it exceeds the 4 GB classic
limit). Figshare's download URL redirects to a presigned S3 URL valid for about
ten seconds, so it is re-signed per request rather than cached. And the bundle
is an archive of per-country archives - but those inner zips are STORED rather
than deflated, so their bytes are contiguous in the outer file and the same
range-reading works one level down, against a window into the outer archive.

Usage:
    python scripts/fetch_rdd2022.py --country India --limit 300
"""

from __future__ import annotations

import argparse
import io
import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

NDOWNLOADER = "https://ndownloader.figshare.com/files/38030910"

EOCD_SIG = b"PK\x05\x06"
ZIP64_EOCD_LOCATOR_SIG = b"PK\x06\x07"
ZIP64_EOCD_SIG = b"PK\x06\x06"
CENTRAL_SIG = b"PK\x01\x02"


def presign(url: str = NDOWNLOADER) -> str:
    """Resolve the redirect to a presigned URL. Valid for ~10 seconds."""
    result = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-D", "-", url],
        capture_output=True, text=True, timeout=60,
    )
    match = re.search(r"^[Ll]ocation:\s*(\S+)", result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("no redirect from figshare; the mirror may have changed")
    return match.group(1).strip()


def fetch_range(start: int, end: int, retries: int = 3) -> bytes:
    """Fetch bytes [start, end] inclusive, re-signing each time."""
    last_error = ""
    for _ in range(retries):
        url = presign()
        result = subprocess.run(
            ["curl", "-s", "--fail", "-r", f"{start}-{end}", url],
            capture_output=True, timeout=300,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
        last_error = f"curl exit {result.returncode}"
    raise RuntimeError(f"range {start}-{end} failed: {last_error}")


class Window:
    """A byte range of the remote file, addressed from zero.

    Lets the same zip parsing run against the outer archive or against a
    nested one, since the inner archives are stored uncompressed and so are
    just a contiguous window of the outer file.
    """

    def __init__(self, base: int = 0, length: int | None = None) -> None:
        self.base = base
        self._length = length

    def fetch(self, start: int, end: int) -> bytes:
        return fetch_range(self.base + start, self.base + end)

    @property
    def length(self) -> int:
        if self._length is None:
            self._length = total_size()
        return self._length

    def nested(self, entry: dict) -> "Window":
        """A window onto a STORED member of this archive."""
        if entry["method"] != 0:
            raise RuntimeError(
                f"{entry['name']} is compressed; cannot range into it without "
                "decompressing the whole member"
            )
        header = self.fetch(entry["local_offset"], entry["local_offset"] + 29)
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {entry['name']}")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        start = entry["local_offset"] + 30 + name_len + extra_len
        return Window(self.base + start, entry["compressed"])


def total_size() -> int:
    url = presign()
    result = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-D", "-", "-r", "0-0", url],
        capture_output=True, text=True, timeout=60,
    )
    match = re.search(r"[Cc]ontent-[Rr]ange:\s*bytes\s+\d+-\d+/(\d+)", result.stdout)
    if not match:
        raise RuntimeError("server did not report a total size; no range support")
    return int(match.group(1))


def central_directory(window: Window) -> tuple[int, int]:
    """Locate the central directory, following the ZIP64 records."""
    size = window.length
    tail_len = min(65557, size)
    tail = window.fetch(size - tail_len, size - 1)

    eocd_at = tail.rfind(EOCD_SIG)
    if eocd_at < 0:
        raise RuntimeError("no End of Central Directory record found")
    cd_size, cd_offset = struct.unpack("<II", tail[eocd_at + 12 : eocd_at + 20])

    # 0xFFFFFFFF in either field means the real values live in the ZIP64 record.
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        locator_at = tail.rfind(ZIP64_EOCD_LOCATOR_SIG)
        if locator_at < 0:
            raise RuntimeError("archive claims ZIP64 but has no locator")
        (zip64_eocd_offset,) = struct.unpack("<Q", tail[locator_at + 8 : locator_at + 16])
        zip64 = window.fetch(zip64_eocd_offset, zip64_eocd_offset + 55)
        if not zip64.startswith(ZIP64_EOCD_SIG):
            raise RuntimeError("ZIP64 EOCD signature missing")
        cd_size, cd_offset = struct.unpack("<QQ", zip64[40:56])

    return cd_offset, cd_size


def parse_central(data: bytes) -> list[dict]:
    """Parse central directory entries into name/offset/size records."""
    entries: list[dict] = []
    position = 0
    while position + 46 <= len(data):
        if data[position : position + 4] != CENTRAL_SIG:
            break
        (
            method, _t, _d, _crc, compressed, uncompressed,
            name_len, extra_len, comment_len, _disk, _ia, _ea, local_offset,
        ) = struct.unpack("<HHHIIIHHHHHII", data[position + 10 : position + 46])

        name = data[position + 46 : position + 46 + name_len].decode("utf-8", "replace")
        extra = data[position + 46 + name_len : position + 46 + name_len + extra_len]

        # ZIP64 extended information: present when a field is maxed out. The
        # order of fields in the extra block depends on which were overflowed.
        if 0xFFFFFFFF in (compressed, uncompressed, local_offset):
            offset = 0
            while offset + 4 <= len(extra):
                header_id, data_size = struct.unpack("<HH", extra[offset : offset + 4])
                if header_id == 0x0001:
                    values = extra[offset + 4 : offset + 4 + data_size]
                    cursor = 0
                    if uncompressed == 0xFFFFFFFF and cursor + 8 <= len(values):
                        (uncompressed,) = struct.unpack("<Q", values[cursor : cursor + 8]); cursor += 8
                    if compressed == 0xFFFFFFFF and cursor + 8 <= len(values):
                        (compressed,) = struct.unpack("<Q", values[cursor : cursor + 8]); cursor += 8
                    if local_offset == 0xFFFFFFFF and cursor + 8 <= len(values):
                        (local_offset,) = struct.unpack("<Q", values[cursor : cursor + 8]); cursor += 8
                    break
                offset += 4 + data_size

        entries.append({
            "name": name, "method": method, "compressed": compressed,
            "uncompressed": uncompressed, "local_offset": local_offset,
        })
        position += 46 + name_len + extra_len + comment_len
    return entries


def extract(window: Window, entry: dict) -> bytes:
    """Fetch and decompress one member."""
    # The local header's name and extra lengths differ from the central
    # directory's, so the data offset must be read from the local header.
    header = window.fetch(entry["local_offset"], entry["local_offset"] + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {entry['name']}")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = entry["local_offset"] + 30 + name_len + extra_len
    raw = window.fetch(start, start + entry["compressed"] - 1)

    if entry["method"] == 0:
        return raw
    if entry["method"] == 8:
        return zlib.decompress(raw, -15)
    raise RuntimeError(f"unsupported compression method {entry['method']}")


def window_for(outer: "Window", entry: dict) -> "Window":
    return outer.nested(entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="India")
    parser.add_argument("--limit", type=int, default=300, help="images to fetch")
    parser.add_argument(
        "--blocks", type=int, default=12,
        help="how many places in the dataset to sample from",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/rdd2022")
    parser.add_argument("--index-only", action="store_true", help="list members, fetch nothing")
    args = parser.parse_args()

    print("resolving archive...", flush=True)
    outer = Window()
    print(f"  bundle is {outer.length/1e9:.2f} GB")

    cd_offset, cd_size = central_directory(outer)
    outer_entries = parse_central(outer.fetch(cd_offset, cd_offset + cd_size - 1))
    print(f"  {len(outer_entries)} per-country archives in the bundle")

    # The bundle holds one zip per country. They are STORED, so the same range
    # reading works one level in without pulling the whole 13 GB.
    wanted = f"{args.country}.zip"
    inner_entry = next(
        (e for e in outer_entries if e["name"].endswith(wanted)), None
    )
    if inner_entry is None:
        available = ", ".join(Path(e["name"]).stem for e in outer_entries)
        raise SystemExit(f"no archive for {args.country}. Available: {available}")

    print(f"  opening {inner_entry['name']} ({inner_entry['compressed']/1e6:.0f} MB)", flush=True)
    window = window_for(outer, inner_entry)

    inner_cd_offset, inner_cd_size = central_directory(window)
    print(f"  inner central directory: {inner_cd_size/1e6:.1f} MB", flush=True)
    entries = parse_central(window.fetch(inner_cd_offset, inner_cd_offset + inner_cd_size - 1))
    print(f"  {len(entries)} members")

    prefix = f"{args.split}/"
    images = sorted(
        (e for e in entries
         if prefix in e["name"] and e["name"].lower().endswith((".jpg", ".jpeg", ".png"))),
        key=lambda e: e["name"],
    )
    annotations = {
        Path(e["name"]).stem: e
        for e in entries
        if prefix in e["name"] and e["name"].lower().endswith(".xml")
    }
    print(f"  {args.country}/{args.split}: {len(images)} images, {len(annotations)} annotations")

    if args.index_only:
        for entry in images[:10]:
            print("   ", entry["name"], entry["uncompressed"], "bytes")
        return

    # Prefer images that have an annotation: an image with no label can neither
    # confirm a detection nor contradict one.
    annotated = [e for e in images if Path(e["name"]).stem in annotations]
    print(f"  {len(annotated)} images carry annotations")

    # Sampled as several contiguous blocks rather than one stretch or evenly
    # spaced singles. One stretch would be a single road in a single lighting
    # condition; evenly spaced singles would mean a separate ranged request per
    # file, and each one needs its own presigned URL. Blocks give variety
    # across the dataset while keeping each fetch one contiguous span.
    per_block = max(1, args.limit // max(1, args.blocks))
    blocks = []
    if len(annotated) <= args.limit:
        blocks = [annotated]
    else:
        stride = len(annotated) // args.blocks
        for index in range(args.blocks):
            start = index * stride
            blocks.append(annotated[start : start + per_block])
    selected = [entry for block in blocks for entry in block][: args.limit]
    print(f"  taking {len(selected)} from {len(blocks)} places in the dataset", flush=True)

    images_dir = Path(args.out) / args.country / "images"
    annotations_dir = Path(args.out) / args.country / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    fetched = bytes_written = 0
    for block_index, block in enumerate(blocks, 1):
        if not block:
            continue
        members = []
        for entry in block:
            members.append(entry)
            annotation = annotations.get(Path(entry["name"]).stem)
            if annotation is not None:
                members.append(annotation)

        # One request for the whole span, then sliced locally. The members of a
        # block sit close together in the archive, so this trades a little
        # wasted bandwidth for far fewer round trips.
        span_start = min(m["local_offset"] for m in members)
        last = max(members, key=lambda m: m["local_offset"])
        span_end = last["local_offset"] + 30 + 512 + last["compressed"]
        span_end = min(span_end, window.length - 1)

        try:
            span = window.fetch(span_start, span_end)
        except Exception as exception:
            print(f"  ! block {block_index}: {exception}", file=sys.stderr)
            continue

        for member in members:
            stem = Path(member["name"]).stem
            is_xml = member["name"].lower().endswith(".xml")
            target = (annotations_dir / f"{stem}.xml") if is_xml else (images_dir / Path(member["name"]).name)
            if target.exists():
                continue
            local = member["local_offset"] - span_start
            if local + 30 > len(span) or span[local : local + 4] != b"PK\x03\x04":
                continue
            name_len, extra_len = struct.unpack("<HH", span[local + 26 : local + 30])
            start = local + 30 + name_len + extra_len
            raw = span[start : start + member["compressed"]]
            if len(raw) < member["compressed"]:
                continue
            try:
                payload = raw if member["method"] == 0 else zlib.decompress(raw, -15)
            except zlib.error:
                continue
            target.write_bytes(payload)
            bytes_written += len(payload)
            if not is_xml:
                fetched += 1

        print(f"  block {block_index}/{len(blocks)}: {fetched} images, {bytes_written/1e6:.1f} MB", flush=True)

    print(f"\nfetched {fetched} images into {images_dir} ({bytes_written/1e6:.1f} MB)")
    print("Dataset: RDD2022, CC BY 4.0. Cite Arya et al., Geoscience Data Journal 11(4), 2024.")


if __name__ == "__main__":
    main()
