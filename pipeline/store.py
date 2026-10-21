"""Where the geographic ledger lives when the server is not running.

The Graviton worker is deliberately ephemeral - started for a drive or a demo
and stopped afterwards, because leaving it up spends the budget on nothing. That
is fine for a batch run, whose output is a file, and fatal for the cross-drive
ledger, whose entire value is that it survives between drives. An in-memory
ledger on an instance that is designed to be stopped means the second drive down
a road never recognises what the first one found, which is the one capability a
one-shot detector cannot have.

So the ledger is backed by a store, with two implementations behind one
interface: memory for tests and for running offline, DynamoDB for deployment.

**Lookups are cell-indexed, not scanned.** `GeoLedger.nearest` originally
compared against every entry. That is correct and irrelevant at a hundred
defects and is a full table scan at a hundred thousand - per defect, on a server
measuring six frames a second. Entries are therefore keyed by a fixed
latitude/longitude cell and a lookup reads the nine cells around a point. The
cell is about 110 m on a side, so a 3x3 neighbourhood covers a 330 m window,
two orders of magnitude more than any radius this system uses, and the number of
rows read per lookup stays constant as the ledger grows.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

# Cell size in degrees. 0.001 degrees of latitude is about 111 m; longitude
# cells narrow towards the poles, which only makes the neighbourhood smaller in
# metres and is handled by the distance test that follows the lookup.
CELL_DEG = 0.001


def cell_key(lat: float, lon: float) -> str:
    return f"{math.floor(lat / CELL_DEG):d}:{math.floor(lon / CELL_DEG):d}"


def neighbour_cells(lat: float, lon: float) -> list[str]:
    """The nine cells around a point, which bound any radius used here."""
    row = math.floor(lat / CELL_DEG)
    col = math.floor(lon / CELL_DEG)
    return [
        f"{row + dr:d}:{col + dc:d}" for dr in (-1, 0, 1) for dc in (-1, 0, 1)
    ]


class LedgerStore(Protocol):
    """Durable storage for located defects, addressed by cell."""

    def entries_near(self, lat: float, lon: float) -> list[dict[str, Any]]: ...

    def put_entry(self, entry: dict[str, Any]) -> None: ...

    def all_entries(self, limit: int = 5000) -> list[dict[str, Any]]: ...


@dataclass
class MemoryLedgerStore:
    """The default. Holds the same shape as DynamoDB so the swap is exact."""

    rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    def entries_near(self, lat: float, lon: float) -> list[dict[str, Any]]:
        wanted = set(neighbour_cells(lat, lon))
        return [row for row in self.rows.values() if row["cell"] in wanted]

    def put_entry(self, entry: dict[str, Any]) -> None:
        self.rows[entry["entry_id"]] = entry

    def all_entries(self, limit: int = 5000) -> list[dict[str, Any]]:
        return list(self.rows.values())[:limit]


@dataclass
class DynamoLedgerStore:
    """DynamoDB, partitioned by cell so a lookup reads nine partitions.

    Numbers are stored as strings and converted on the way out. DynamoDB's
    native number type is a decimal and boto3 hands back `Decimal` objects,
    which are not JSON-serialisable and compare badly against floats; carrying
    them into the measurement path would be a quiet source of type errors far
    from here.
    """

    table_name: str
    region_name: str = ""
    _table: Any = None

    def __post_init__(self) -> None:
        import boto3

        kwargs = {"region_name": self.region_name} if self.region_name else {}
        self._table = boto3.resource("dynamodb", **kwargs).Table(self.table_name)

    @staticmethod
    def _out(item: dict[str, Any]) -> dict[str, Any]:
        import json

        return json.loads(item["body"]) | {
            "entry_id": item["entry_id"],
            "cell": item["cell"],
        }

    def entries_near(self, lat: float, lon: float) -> list[dict[str, Any]]:
        from boto3.dynamodb.conditions import Key

        found: list[dict[str, Any]] = []
        for cell in neighbour_cells(lat, lon):
            response = self._table.query(KeyConditionExpression=Key("cell").eq(cell))
            found.extend(self._out(item) for item in response.get("Items", []))
        return found

    def put_entry(self, entry: dict[str, Any]) -> None:
        import json

        self._table.put_item(
            Item={
                "cell": entry["cell"],
                "entry_id": entry["entry_id"],
                "updated_at": int(time.time()),
                # The whole entry as one JSON blob. It is small, it is written
                # and read as a unit, and keeping it opaque means adding a field
                # to a sighting never needs a table migration.
                "body": json.dumps(entry),
            }
        )

    def all_entries(self, limit: int = 5000) -> list[dict[str, Any]]:
        """A full scan. Only for the map export, never on the measuring path."""
        found: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {"Limit": min(limit, 1000)}
        while len(found) < limit:
            response = self._table.scan(**kwargs)
            found.extend(self._out(item) for item in response.get("Items", []))
            key = response.get("LastEvaluatedKey")
            if not key:
                break
            kwargs["ExclusiveStartKey"] = key
        return found[:limit]


def store_from_environment() -> LedgerStore:
    """DynamoDB when the stack says so, memory otherwise.

    Deliberately an environment lookup rather than a flag threaded through the
    call sites: the same code runs on a laptop with no AWS credentials and on
    the Graviton worker inside the stack, and neither should need a different
    entry point. A missing or unreachable table falls back to memory with a
    warning rather than taking the capture endpoint down, because losing the
    cross-drive history is bad and refusing to measure at all is worse.
    """
    import os

    table = os.environ.get("ROADLEDGER_LEDGER_TABLE", "").strip()
    if not table:
        return MemoryLedgerStore()
    try:
        return DynamoLedgerStore(table, os.environ.get("AWS_REGION", ""))
    except Exception as error:  # pragma: no cover - needs a broken environment
        import logging

        logging.getLogger(__name__).warning(
            "ledger table %s is unusable (%s); falling back to in-memory storage, "
            "so cross-drive history will not survive a restart",
            table,
            error,
        )
        return MemoryLedgerStore()
