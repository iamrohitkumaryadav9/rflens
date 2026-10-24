"""Smoke test for sql-mcp over HTTP, using the official MCP client (mcp 2.x).

Start the server first (make sql-mcp), then:  uv run python scripts/sql_mcp_smoke.py [URL]

Checks: the three tools are listed; describe_table("v_uls_frequencies") shows the ASSUMED / UNKNOWN
unit notes; run_query for the top 5 active licences by frequency rows, written against the views,
returns the same rows as the ULS step 3 acceptance query on the base tables; a cross join is
rejected for cost with the cost shown; DELETE FROM uls_hd is rejected at parse; uls_en is refused;
every one of these calls is in sql_query_log with the right status. Needs the ULS cell service
loaded (make load-uls SERVICE=cell). Exits non-zero if any check fails. Log rows written by this
script are tagged with a unique question text and left in the log (it is an audit log).
"""

import asyncio
import sys
import uuid

from mcp import Client
from sqlalchemy import text

from rflens.db import get_engine

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8102/mcp"
TAG = f"smoke-{uuid.uuid4().hex[:12]}"

# The ULS step 3 acceptance query, on the base tables: top 5 active licences by FR rows.
BASE_TABLE_QUERY = """
SELECT h.call_sign, e.entity_name, count(*) AS n
FROM uls_hd h
JOIN uls_en e ON e.service = h.service AND e.usi = h.usi AND e.entity_type = 'L'
JOIN uls_fr f ON f.service = h.service AND f.usi = h.usi
WHERE h.license_status = 'A'
GROUP BY h.service, h.usi, h.call_sign, e.entity_name
ORDER BY n DESC, h.call_sign
LIMIT 5
"""
# The same question through the views.
VIEW_QUERY = """
SELECT call_sign, licensee_name, count(*) AS n
FROM v_uls_frequencies
WHERE license_status = 'A'
GROUP BY service, usi, call_sign, licensee_name
ORDER BY n DESC, call_sign
LIMIT 5
"""
CROSS_JOIN = "SELECT count(*) FROM uls_fr a, uls_lo b"


def check(ok: bool, message: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
    if not ok:
        raise SystemExit(1)


def logged(statuses: dict[str, str]) -> None:
    """Each (call text -> expected status) has a row tagged with this run's question."""
    with get_engine().connect() as c:
        rows = c.execute(
            text(
                "SELECT tool, sql_text, status, plan_cost, rejection_reason FROM sql_query_log "
                "WHERE question = :q ORDER BY id"
            ),
            {"q": TAG},
        ).all()
    by_sql = {r.sql_text: r for r in rows}
    for sql, expected in statuses.items():
        row = by_sql.get(sql)
        shown = sql.strip().splitlines()[0][:60]
        ok = row is not None and row.status == expected
        detail = f"{row.status}, cost {row.plan_cost}, {row.rejection_reason}" if row else "MISSING"
        check(ok, f"logged: {shown!r} -> {expected} ({detail})")


async def main() -> None:
    async with Client(URL) as client:
        print(f"connected to {URL}; protocol {client.protocol_version}; question tag {TAG}")

        print("\n1. tools/list")
        tools = (await client.list_tools()).tools
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:100]}...")
        check(
            {t.name for t in tools} == {"list_tables", "describe_table", "run_query"},
            "exactly list_tables, describe_table, run_query",
        )

        print("\n2. list_tables")
        listed = (await client.call_tool("list_tables", {"question": TAG})).structured_content
        names = [t["name"] for t in listed["tables"]]
        print(f"  {len(names)} relations: {', '.join(names)}")
        check(len(names) == 12 and "uls_en" not in names, "9 tables + 3 views, no uls_en")

        print('\n3. describe_table("v_uls_frequencies")')
        described = await client.call_tool(
            "describe_table", {"table": "v_uls_frequencies", "question": TAG}
        )
        desc = described.structured_content
        for col in desc["columns"]:
            print(f"  {col['name']:22} {col['type']:10} {col['description'][:70]}")
        by_col = {c["name"]: c["description"] for c in desc["columns"]}
        check("ASSUMED" in by_col["frequency_assigned"], "frequency unit is marked ASSUMED")
        check(
            all("UNKNOWN" in by_col[c] for c in ("power_erp", "power_output", "eirp")),
            "power / EIRP units are marked UNKNOWN",
        )

        print("\n4. top 5 active licences by frequency rows, through the views")
        out = await client.call_tool("run_query", {"sql": VIEW_QUERY, "question": TAG})
        check(not out.is_error, f"run_query accepted: {out.content[0].text[:100]}")
        got = out.structured_content
        print(f"  plan cost {got['plan_cost']:,.0f}, {got['duration_ms']} ms; executed:")
        print(f"  {got['executed_sql']}")
        for row in got["rows"]:
            print(f"  {row}")
        with get_engine().connect() as c:
            want = [[r[0], r[1], r[2]] for r in c.execute(text(BASE_TABLE_QUERY))]
        check(got["rows"] == want, "rows equal the ULS step 3 query on the base tables")
        print("\n5. the cross join")
        cross = await client.call_tool("run_query", {"sql": CROSS_JOIN, "question": TAG})
        message = cross.content[0].text
        print(f"  is_error={cross.is_error}: {message}")
        check(cross.is_error and "rejected (cost)" in message, "cross join rejected for cost")
        check(
            "estimated planner cost" in message and "limit of 500,000" in message,
            "the message shows the cost and the limit",
        )

        print("\n6. DELETE FROM uls_hd")
        delete = await client.call_tool("run_query", {"sql": "DELETE FROM uls_hd", "question": TAG})
        print(f"  is_error={delete.is_error}: {delete.content[0].text}")
        check(
            delete.is_error and "rejected (not_select)" in delete.content[0].text,
            "DELETE rejected at parse",
        )

        print("\n7. uls_en is not available")
        for sql in ("SELECT * FROM uls_en", 'SELECT * FROM "uls_en"'):
            res = await client.call_tool("run_query", {"sql": sql, "question": TAG})
            check(
                res.is_error and "rejected (table_not_allowed)" in res.content[0].text,
                f"{sql} rejected",
            )

    print("\n8. sql_query_log")
    logged(
        {
            "list_tables()": "ok",
            "describe_table('v_uls_frequencies')": "ok",
            VIEW_QUERY: "ok",
            CROSS_JOIN: "rejected",
            "DELETE FROM uls_hd": "rejected",
            "SELECT * FROM uls_en": "rejected",
            'SELECT * FROM "uls_en"': "rejected",
        }
    )
    with get_engine().connect() as c:
        row = c.execute(
            text(
                "SELECT plan_cost, rejection_reason FROM sql_query_log "
                "WHERE question = :q AND sql_text = :s"
            ),
            {"q": TAG, "s": CROSS_JOIN},
        ).one()
    check(
        row.plan_cost is not None and row.rejection_reason.startswith("cost:"),
        f"the cross join's log row has its plan cost ({row.plan_cost:,.0f})",
    )

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
