"""Smoke test for docs-mcp over HTTP, using the official MCP client (mcp 2.x).

Start the server first (make docs-mcp), then:  uv run python scripts/mcp_smoke.py [URL]

Checks: the two tools are listed; search_docs finds Table 6.2.1-1 for a power-class question;
get_chunk returns the related notes text of a split table; k=500 is capped at 20; a 600-character
query is rejected with a clear error. Exits non-zero if any check fails.
3GPP text is printed to the terminal only, never written to disk.
"""

import asyncio
import json
import sys

from mcp import Client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8101/mcp"
NOTES_QUERIES = [
    "requirements for spurious emissions for UE co-existence protected band",
    "NR CA configurations and bandwidth combination sets",
    "UE co-existence spurious emission table notes",
]


def check(ok: bool, message: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
    if not ok:
        raise SystemExit(1)


def payload(result) -> dict:
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


async def main() -> None:
    async with Client(URL) as client:
        print(f"connected to {URL}")
        print(f"negotiated protocol version: {client.protocol_version}")
        print(f"server: {client.server_info}")

        print("\n1. tools/list")
        tools = (await client.list_tools()).tools
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:110]}...")
        check(
            {t.name for t in tools} == {"search_docs", "get_chunk"},
            "exactly search_docs and get_chunk",
        )

        print('\n2. search_docs("maximum output power class 3 band n78", spec="38.101-1")')
        res = await client.call_tool(
            "search_docs",
            {"query": "maximum output power class 3 band n78", "spec": "38.101-1"},
        )
        data = payload(res)
        print(f"  mode={data['mode']} k={data['k']}")
        for i, r in enumerate(data["results"], 1):
            print(
                f"  {i}. {r['chunk_id']}  clause {r['clause']}  table {r['table_number']}  "
                f"part {r['table_part']}  score {r['score']}"
            )
            print(f"       {r['snippet'][:110]}")
        top5 = data["results"][:5]
        check(
            any(r["table_number"] == "6.2.1-1" for r in top5),
            "a Table 6.2.1-1 chunk is in the top 5",
        )
        check(
            all(len(r["snippet"]) <= 300 for r in data["results"]), "snippets are at most 300 chars"
        )

        print("\n3. get_chunk on a result that points to a separate notes chunk")
        source = None
        for query in NOTES_QUERIES:
            found = payload(
                await client.call_tool("search_docs", {"query": query, "spec": "38.101-1", "k": 20})
            )
            source = next((r for r in found["results"] if r["related_chunks"]), None)
            if source:
                break
        check(source is not None, "found a search result with related_chunks")
        print(
            f"  result: {source['chunk_id']} (table {source['table_number']}, "
            f"part {source['table_part']}) -> related_chunks {source['related_chunks']}"
        )
        chunk = payload(
            await client.call_tool(
                "get_chunk", {"chunk_id": source["chunk_id"], "include_related": True}
            )
        )
        print(
            f"  get_chunk: {chunk['chunk_id']} ({len(chunk['text'])} chars), "
            f"related ids {chunk['related_chunk_ids']}"
        )
        for rel in chunk["related"]:
            print(f"  related {rel['chunk_id']} [{rel['kind']}, part {rel['table_part']}]")
            print(f"    first 200 chars: {rel['text'][:200]!r}")
        check(bool(chunk["related"]), "related notes text returned")
        check(
            [r["chunk_id"] for r in chunk["related"]] == chunk["related_chunk_ids"],
            "related ids match",
        )
        bare = payload(
            await client.call_tool(
                "get_chunk", {"chunk_id": source["chunk_id"], "include_related": False}
            )
        )
        check(
            bare["related"] == [] and bare["related_chunk_ids"],
            "include_related=false returns ids only",
        )

        print("\n4. guardrails")
        capped = payload(
            await client.call_tool(
                "search_docs", {"query": "output power", "spec": "38.101-1", "k": 500}
            )
        )
        n_results = len(capped["results"])
        print(f"  k=500 -> k={capped['k']}, {n_results} results, notice: {capped['notice']!r}")
        check(capped["k"] == 20 and len(capped["results"]) == 20, "k=500 is capped at 20")
        long_query = await client.call_tool("search_docs", {"query": "x" * 600})
        message = long_query.content[0].text
        print(f"  600-char query -> is_error={long_query.is_error}: {message}")
        check(
            long_query.is_error and "maximum is 500" in message,
            "a 600-char query is rejected with a clear error",
        )
        unknown = await client.call_tool("get_chunk", {"chunk_id": "38.101-1@19.7.0/6.2.1/99999"})
        print(f"  unknown chunk_id -> is_error={unknown.is_error}: {unknown.content[0].text}")
        check(
            unknown.is_error and "unknown chunk_id" in unknown.content[0].text,
            "unknown chunk id is a clear error",
        )
        bad = await client.call_tool("get_chunk", {"chunk_id": "1'; DROP TABLE chunks;--"})
        print(f"  malformed chunk_id -> is_error={bad.is_error}: {bad.content[0].text[:110]}")
        check(
            bad.is_error and "invalid chunk_id" in bad.content[0].text,
            "malformed chunk id rejected before SQL",
        )

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
