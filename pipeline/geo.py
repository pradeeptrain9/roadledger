"""Where the defect is, in coordinates a road authority can dispatch against.

A measured area is useless without a location, and a phone gives us one for
free - but a far worse one than it gives us for area. This module keeps those
two accuracies apart, because conflating them is the single easiest way to make
this system untrustworthy:

* **Area** is measured from the image geometry. On the synthetic holdout its
  error is a few percent, and it does not depend on GPS at all.
* **Position** is measured by a consumer GNSS chip behind a windscreen, and is
  good to something between 4 and 20 metres. No amount of image processing
  improves it.

So every emitted position carries its own accuracy figure, derived from the
fix's reported accuracy plus the geometric lever arm of projecting forward from
the camera to the defect. And the two kinds of duplicate are handled
differently: within one drive, relative geometry is sub-metre and duplicates
are *merged*; across two drives, only GPS relates them, so a nearby defect is
*flagged as a possible revisit* and never silently merged. Merging two real
potholes three metres apart because the GPS was off by five would quietly
halve a road authority's defect count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Within one drive, defect positions are related by rectified image geometry
# and ego-motion, which is sub-metre. Two detections this close are the same
# physical defect seen twice.
SAME_DRIVE_RADIUS_M = 0.6

# Across drives there is nothing but GNSS relating the two, so the floor is set
# by GPS error rather than by how far apart two potholes can be. Anything
# inside this is a *candidate* revisit for a human or a later visual check to
# confirm, never an automatic merge.
MIN_REVISIT_RADIUS_M = 4.0

# A GNSS course-over-ground is noisy at low speed; below this the heading is
# not used to project the defect forward.
MIN_SPEED_FOR_HEADING_MPS = 1.5

# Assumed 1-sigma error on a usable course-over-ground, in degrees. Used only
# to convert the forward projection into a position-accuracy contribution.
HEADING_SIGMA_DEG = 8.0

# How far from a query point the in-process cache is consulted. Comfortably
# wider than the nine-cell store lookup it supplements, so the cache can never
# be the reason a nearby entry is missed.
CACHE_LOOKUP_RADIUS_M = 400.0

# The cache exists to cover the gap between writing a sighting and being able
# to read it back, not to hold a city. A long drive would otherwise accumulate
# every defect it had ever seen and turn each lookup back into a linear scan.
MAX_CACHED_ENTRIES = 512


def metres_per_degree(latitude_deg: float) -> tuple[float, float]:
    """Local scale of the graticule, (per degree latitude, per degree longitude).

    The standard series expansion of the WGS84 meridional and parallel arc
    lengths. Accurate to well under a metre per degree, which is three orders
    of magnitude better than the GPS fixes it is applied to, and it keeps this
    module dependency-free.
    """
    phi = math.radians(latitude_deg)
    per_lat = 111_132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    per_lon = 111_412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
    return per_lat, max(per_lon, 1e-6)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    radius = 6_371_008.8  # mean Earth radius
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from the first point to the second, degrees from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


@dataclass(frozen=True)
class GeoFix:
    """One GNSS fix as a browser's Geolocation API reports it.

    `heading` and `speed` are genuinely optional: the spec allows null for both
    and a phone returns null for heading whenever it is not moving, which is
    exactly when a driver is most likely to be looking at the screen.
    """

    lat: float
    lon: float
    accuracy_m: float = 15.0
    heading_deg: float | None = None
    speed_mps: float | None = None
    timestamp_s: float = 0.0

    @property
    def has_usable_heading(self) -> bool:
        if self.heading_deg is None or not math.isfinite(self.heading_deg):
            return False
        # A heading reported while stationary is the last one the chip had, or
        # noise. Either way it should not steer a projection.
        if self.speed_mps is not None and self.speed_mps < MIN_SPEED_FOR_HEADING_MPS:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "accuracy_m": round(self.accuracy_m, 1),
            "heading_deg": None if self.heading_deg is None else round(self.heading_deg, 1),
            "speed_mps": None if self.speed_mps is None else round(self.speed_mps, 2),
            "timestamp_s": round(self.timestamp_s, 3),
        }


def offset_to_latlon(
    fix: GeoFix, ahead_m: float, lateral_m: float, heading_deg: float
) -> tuple[float, float]:
    """Project a camera-relative ground offset onto the graticule.

    `ahead_m` is along the heading, `lateral_m` is positive to the right of it,
    matching the bird's-eye view's own axes.
    """
    theta = math.radians(heading_deg)
    north = ahead_m * math.cos(theta) - lateral_m * math.sin(theta)
    east = ahead_m * math.sin(theta) + lateral_m * math.cos(theta)
    per_lat, per_lon = metres_per_degree(fix.lat)
    return fix.lat + north / per_lat, fix.lon + east / per_lon


@dataclass
class GeoPosition:
    """A located defect, with the provenance and accuracy of that location."""

    lat: float
    lon: float
    accuracy_m: float
    method: str  # gps_heading | track_heading | camera_only
    ahead_m: float
    lateral_m: float
    heading_deg: float | None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "accuracy_m": round(self.accuracy_m, 1),
            "method": self.method,
            "ahead_m": round(self.ahead_m, 2),
            "lateral_m": round(self.lateral_m, 2),
            "heading_deg": None if self.heading_deg is None else round(self.heading_deg, 1),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeoPosition":
        return cls(
            lat=float(data["lat"]),
            lon=float(data["lon"]),
            accuracy_m=float(data["accuracy_m"]),
            method=str(data["method"]),
            ahead_m=float(data.get("ahead_m", 0.0)),
            lateral_m=float(data.get("lateral_m", 0.0)),
            heading_deg=(
                None if data.get("heading_deg") is None else float(data["heading_deg"])
            ),
            note=str(data.get("note", "")),
        )

    def to_geojson_feature(self, properties: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(self.lon, 7), round(self.lat, 7)]},
            "properties": {**properties, "position": self.to_dict()},
        }


def locate(
    fix: GeoFix,
    ahead_m: float,
    lateral_m: float,
    track_heading_deg: float | None = None,
) -> GeoPosition:
    """Place a defect, degrading honestly when the heading is unavailable.

    Three routes, in descending order of trust:

    1. the fix's own course-over-ground, when the vehicle is moving fast enough
       for it to mean something;
    2. a heading derived from the recent track of fixes, which survives a chip
       that simply never reports course;
    3. no heading at all - in which case the defect is placed *at the camera*
       and its accuracy is inflated by the forward distance, because that is
       genuinely all that is known. It is still a useful position: a crew
       looking for a 0.4 m pothole within 20 m of a point on a road will find
       it. Inventing a heading to make the number look tighter would not.
    """
    heading: float | None = None
    method = "camera_only"
    note = ""

    if fix.has_usable_heading:
        heading = fix.heading_deg
        method = "gps_heading"
    elif track_heading_deg is not None and math.isfinite(track_heading_deg):
        heading = track_heading_deg
        method = "track_heading"
        note = "course derived from consecutive fixes; the device reported none"

    if heading is None:
        note = (
            f"no usable heading, so the defect is placed at the camera and its accuracy "
            f"widened by the {ahead_m:.1f} m it lies ahead of it"
        )
        return GeoPosition(
            lat=fix.lat,
            lon=fix.lon,
            accuracy_m=fix.accuracy_m + abs(ahead_m),
            method=method,
            ahead_m=ahead_m,
            lateral_m=lateral_m,
            heading_deg=None,
            note=note,
        )

    lat, lon = offset_to_latlon(fix, ahead_m, lateral_m, heading)
    # The lever arm: an angular error in the heading becomes a cross-track
    # error proportional to how far ahead the defect is.
    lever = abs(ahead_m) * math.radians(HEADING_SIGMA_DEG)
    accuracy = math.hypot(fix.accuracy_m, lever)
    return GeoPosition(
        lat=lat,
        lon=lon,
        accuracy_m=accuracy,
        method=method,
        ahead_m=ahead_m,
        lateral_m=lateral_m,
        heading_deg=heading,
        note=note,
    )


@dataclass
class Sighting:
    """One drive's observation of one physical defect."""

    run_id: str
    defect_id: str
    at_epoch_s: float
    area_m2: float
    label: str
    priority: int
    position: GeoPosition

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "defect_id": self.defect_id,
            "at_epoch_s": round(self.at_epoch_s, 1),
            "area_m2": self.area_m2,
            "label": self.label,
            "priority": self.priority,
            "position": self.position.to_dict(),
        }


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sighting":
        return cls(
            run_id=str(data["run_id"]),
            defect_id=str(data["defect_id"]),
            at_epoch_s=float(data["at_epoch_s"]),
            area_m2=float(data["area_m2"]),
            label=str(data["label"]),
            priority=int(data["priority"]),
            position=GeoPosition.from_dict(data["position"]),
        )


@dataclass
class GeoEntry:
    """A physical defect in the persistent ledger, and every time it was seen."""

    entry_id: str
    lat: float
    lon: float
    sightings: list[Sighting] = field(default_factory=list)

    @property
    def latest(self) -> Sighting:
        return self.sightings[-1]

    @property
    def area_trend_m2(self) -> float | None:
        """Change in measured area between the first and latest sighting.

        Only meaningful with two or more drives, which is the point: a pothole
        that grew 40% in a month is a different maintenance decision from one
        that did not, and that is a thing this ledger can show and a one-shot
        detector cannot.
        """
        if len(self.sightings) < 2:
            return None
        return round(self.sightings[-1].area_m2 - self.sightings[0].area_m2, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "n_sightings": len(self.sightings),
            "first_seen_epoch_s": round(self.sightings[0].at_epoch_s, 1) if self.sightings else None,
            "latest": self.latest.to_dict() if self.sightings else None,
            "area_trend_m2": self.area_trend_m2,
            "sightings": [s.to_dict() for s in self.sightings],
        }

    def to_storage(self) -> dict[str, Any]:
        """The shape the store persists: everything needed to rebuild this."""
        from .store import cell_key

        return {
            "entry_id": self.entry_id,
            "cell": cell_key(self.lat, self.lon),
            "lat": self.lat,
            "lon": self.lon,
            "sightings": [s.to_dict() for s in self.sightings],
        }

    @classmethod
    def from_storage(cls, data: dict[str, Any]) -> "GeoEntry":
        return cls(
            entry_id=str(data["entry_id"]),
            lat=float(data["lat"]),
            lon=float(data["lon"]),
            sightings=[Sighting.from_dict(s) for s in data.get("sightings", [])],
        )


@dataclass
class GeoLedger:
    """The persistent, cross-drive record of located defects.

    Two separate questions, deliberately answered separately:

    * `merge_or_add` is for one drive. Positions inside it are related by
      rectified geometry and ego-motion, so two detections within
      `SAME_DRIVE_RADIUS_M` really are one defect and are merged.
    * `revisit_candidates` is for a later drive. It reports what is nearby and
      how far, and leaves the merge to a human or to a visual check, because
      GPS alone cannot tell one pothole from its neighbour.

    Backed by a `LedgerStore`, which is memory by default and DynamoDB in the
    deployed stack. Both are addressed by latitude/longitude cell, so a lookup
    reads a fixed nine partitions rather than scanning every defect ever
    recorded - this runs on the measuring path, several times a second.
    """

    store: Any = None
    # A read-through cache of entries this process has touched, so repeated
    # lookups during one drive do not re-read the same partitions.
    _cache: dict[str, GeoEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.store is None:
            from .store import MemoryLedgerStore

            self.store = MemoryLedgerStore()

    def _remember(self, entry: GeoEntry) -> None:
        """Cache an entry, evicting the oldest once the cache is full.

        Eviction is safe: everything in the cache has already been written to
        the store, so a dropped entry is re-read rather than lost.
        """
        self._cache.pop(entry.entry_id, None)
        self._cache[entry.entry_id] = entry
        while len(self._cache) > MAX_CACHED_ENTRIES:
            self._cache.pop(next(iter(self._cache)))

    def _entry_id(self, lat: float, lon: float) -> str:
        import hashlib

        digest = hashlib.sha1(f"{lat:.6f}:{lon:.6f}".encode()).hexdigest()[:12]
        return f"g_{digest}"

    def _near(self, lat: float, lon: float) -> list[GeoEntry]:
        entries: dict[str, GeoEntry] = {}
        for row in self.store.entries_near(lat, lon):
            entry = GeoEntry.from_storage(row)
            entries[entry.entry_id] = entry
        # Anything written in this process wins over what was read back: a
        # sighting appended moments ago may not be visible to a fresh query yet,
        # and on DynamoDB a read is eventually consistent by default.
        for entry_id, entry in self._cache.items():
            if haversine_m(lat, lon, entry.lat, entry.lon) <= CACHE_LOOKUP_RADIUS_M:
                entries[entry_id] = entry
        return list(entries.values())

    @property
    def entries(self) -> dict[str, GeoEntry]:
        """Every entry, for export and for tests. Not used on the hot path."""
        out = {
            row["entry_id"]: GeoEntry.from_storage(row)
            for row in self.store.all_entries()
        }
        out.update(self._cache)
        return out

    def nearest(self, lat: float, lon: float) -> tuple[GeoEntry, float] | None:
        best: tuple[GeoEntry, float] | None = None
        for entry in self._near(lat, lon):
            distance = haversine_m(lat, lon, entry.lat, entry.lon)
            if best is None or distance < best[1]:
                best = (entry, distance)
        return best

    def merge_or_add(self, sighting: Sighting, same_drive: bool = True) -> tuple[GeoEntry, bool]:
        """Record a sighting. Returns the entry and whether it was newly created."""
        lat, lon = sighting.position.lat, sighting.position.lon
        nearest = self.nearest(lat, lon)
        radius = SAME_DRIVE_RADIUS_M if same_drive else 0.0
        if nearest is not None and radius > 0 and nearest[1] <= radius:
            entry, _ = nearest
            # Same drive, same place: another view of the same defect. The
            # newest measurement supersedes, the older one is kept as evidence.
            entry.sightings.append(sighting)
            self._remember(entry)
            self.store.put_entry(entry.to_storage())
            return entry, False

        entry = GeoEntry(entry_id=self._entry_id(lat, lon), lat=lat, lon=lon)
        entry.sightings.append(sighting)
        self._remember(entry)
        self.store.put_entry(entry.to_storage())
        return entry, True

    def revisit_candidates(
        self, lat: float, lon: float, accuracy_m: float, exclude_run_id: str = ""
    ) -> list[dict[str, Any]]:
        """Entries from *other* drives close enough to possibly be this defect.

        The radius grows with the GPS accuracy of both fixes, so a 20 m fix
        reports more candidates and a 4 m fix reports fewer - which is the
        correct behaviour and is also why these are candidates rather than
        merges.
        """
        out: list[dict[str, Any]] = []
        for entry in self._near(lat, lon):
            if not entry.sightings:
                continue
            if exclude_run_id and all(s.run_id == exclude_run_id for s in entry.sightings):
                continue
            radius = max(
                MIN_REVISIT_RADIUS_M, accuracy_m + entry.latest.position.accuracy_m
            )
            distance = haversine_m(lat, lon, entry.lat, entry.lon)
            if distance <= radius:
                out.append(
                    {
                        "entry_id": entry.entry_id,
                        "distance_m": round(distance, 1),
                        "radius_m": round(radius, 1),
                        "last_area_m2": entry.latest.area_m2,
                        "last_label": entry.latest.label,
                        "last_seen_epoch_s": round(entry.latest.at_epoch_s, 1),
                        "confirmed": False,
                        "note": (
                            f"within {distance:.0f} m of a defect recorded earlier, which is "
                            f"inside the {radius:.0f} m these two GPS fixes can resolve; "
                            "treat as a possible revisit, not a confirmed match"
                        ),
                    }
                )
        return sorted(out, key=lambda item: item["distance_m"])

    def to_geojson(self) -> dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": [
                entry.latest.position.to_geojson_feature(
                    {
                        "entry_id": entry.entry_id,
                        "label": entry.latest.label,
                        "area_m2": entry.latest.area_m2,
                        "priority": entry.latest.priority,
                        "n_sightings": len(entry.sightings),
                        "area_trend_m2": entry.area_trend_m2,
                    }
                )
                for entry in self.entries.values()
                if entry.sightings
            ],
        }


def track_heading(fixes: list[GeoFix]) -> float | None:
    """Course from the recent track of fixes, when it is resolvable.

    Two fixes only define a direction if they are further apart than their own
    uncertainty; otherwise the bearing between them is noise. Returning None is
    the honest answer and the caller degrades accordingly.
    """
    if len(fixes) < 2:
        return None
    last = fixes[-1]
    for earlier in reversed(fixes[:-1]):
        distance = haversine_m(earlier.lat, earlier.lon, last.lat, last.lon)
        if distance >= max(2.0, 0.5 * (earlier.accuracy_m + last.accuracy_m)):
            return bearing_deg(earlier.lat, earlier.lon, last.lat, last.lon)
    return None
