"""Operating-band lookup: the planner names bands, the doc agent reads their rows through sql-mcp.

Band edges are structured facts in `spec_operating_bands` (migration 0013), so they are looked up
instead of searched. The band labels come from the planner's LLM output: they are validated against
a strict pattern before they go into the SQL text, which therefore can contain nothing else.
"""

import re
from typing import Any

from rflens.agents.sql_tools import SqlMcpTools
from rflens.graph.spec_rules import LTE_SPEC, NR_SPEC, SPECS

BAND_RE = re.compile(r"^n?\d{1,3}$")
MAX_BANDS = 6
COLUMNS = (
    "spec, version, band, ul_low_mhz, ul_high_mhz, dl_low_mhz, dl_high_mhz, duplex_mode, "
    "has_uplink, has_downlink, source_chunk_id, notes"
)


def clean_bands(raw: Any) -> list[dict[str, str]]:
    """The planner's bands[] as [{"spec", "band"}]: bad entries dropped, duplicates removed. A
    missing spec is inferred from the label (an n-number is NR, a plain number is LTE)."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        band = str(item.get("band") or "").strip().lower().removeprefix("band").strip()
        if not BAND_RE.match(band):
            continue
        spec = item.get("spec") if item.get("spec") in SPECS else None
        if spec is None or (band.startswith("n") != (spec == NR_SPEC)):
            spec = NR_SPEC if band.startswith("n") else LTE_SPEC  # the label decides
        entry = {"spec": spec, "band": band}
        if entry not in out:
            out.append(entry)
    return out[:MAX_BANDS]


def lookup_sql(bands: list[dict[str, str]]) -> str:
    """One SELECT for all bands. Safe by construction: spec is in SPECS, band matches BAND_RE."""
    if not bands:
        raise ValueError("no bands to look up")
    clauses = []
    for b in bands:
        if b["spec"] not in SPECS or not BAND_RE.match(b["band"]):
            raise ValueError(f"refusing to build SQL for {b!r}")
        clauses.append(f"(spec = '{b['spec']}' AND band = '{b['band']}')")
    return (
        f"SELECT {COLUMNS} FROM spec_operating_bands WHERE {' OR '.join(clauses)} "
        "ORDER BY spec, band, version DESC"
    )


def _num(value: Any) -> float | None:
    return None if value is None else float(value)


def lookup_bands(
    tools: SqlMcpTools, bands: list[dict[str, str]], question: str
) -> tuple[list[dict], dict]:
    """(rows, info): the newest row per (spec, band) in the order the bands were named, and
    {"query_id", "requested", "found", "missing"} (label lists are 'spec band' strings)."""
    outcome = tools.run_query(lookup_sql(bands), f"band lookup for: {question}")
    if not outcome.ok:
        raise BandLookupError(f"sql-mcp refused the band lookup: {outcome.error}")
    by_key: dict[tuple[str, str], dict] = {}
    for values in outcome.rows:
        row = dict(zip(outcome.columns, values, strict=True))
        key = (row["spec"], row["band"])
        if key in by_key:  # rows come newest version first
            continue
        for name in ("ul_low_mhz", "ul_high_mhz", "dl_low_mhz", "dl_high_mhz"):
            row[name] = _num(row[name])
        by_key[key] = row
    rows = [by_key[(b["spec"], b["band"])] for b in bands if (b["spec"], b["band"]) in by_key]
    label = lambda b: f"{b['spec']} {b['band']}"  # noqa: E731
    info = {
        "query_id": outcome.query_id,
        "requested": [label(b) for b in bands],
        "found": [label(b) for b in bands if (b["spec"], b["band"]) in by_key],
        "missing": [label(b) for b in bands if (b["spec"], b["band"]) not in by_key],
    }
    return rows, info


class BandLookupError(Exception):
    pass


def describe_row(row: dict) -> str:
    """One line for the model: the edges in MHz, or N/A for a side the band does not have."""

    def side(low, high):
        return "N/A" if low is None else f"{low:g}-{high:g} MHz"

    uplink = side(row["ul_low_mhz"], row["ul_high_mhz"])
    downlink = side(row["dl_low_mhz"], row["dl_high_mhz"])
    return (
        f"TS {row['spec']} band {row['band']}: uplink {uplink}, downlink {downlink}, "
        f"{row['duplex_mode']}"
    )
