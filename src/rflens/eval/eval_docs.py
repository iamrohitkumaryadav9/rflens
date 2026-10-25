"""make eval-docs: retrieval recall@5 for the doc questions of the golden set, through docs-mcp.

    make docs-mcp          # in another terminal (needs Ollama for the vector and hybrid modes)
    make eval-docs

Free: no LLM is called (queries are embedded locally by Ollama). For every doc question it runs
search_docs in vector, keyword and hybrid mode, each without and with the spec filter, and a hit
is a top-5 chunk of the question's spec whose clause equals an expected clause exactly (a
sub-clause does not count) or whose table number is an expected table. Recall is reported
separately for the human-written and the assistant-written questions, and one row per group is
appended to eval/RESULTS.md. Agents reach documents only through docs-mcp, and so does this.
"""

import argparse
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from mcp import Client

from rflens.config import get_agent_settings
from rflens.eval.results_log import append_docs_rows, git_commit
from rflens.retrieval.eval_smoke import DEFAULT_FILE, Question, load_questions

MODES = ("vector", "keyword", "hybrid")
K = 5


class DocsMcpUnavailable(Exception):
    pass


class DocsMcpError(Exception):
    """docs-mcp answered with a tool error (for example: the embedding service is down)."""


class DocsSearchClient:
    """search_docs over MCP: `target` is docs-mcp's URL or, in tests, an in-process server."""

    def __init__(self, target: Any):
        self.target = target
        self.calls = 0

    def search(self, query: str, spec: str | None, mode: str, k: int = K) -> list[dict]:
        arguments: dict[str, Any] = {"query": query, "mode": mode, "k": k}
        if spec:
            arguments["spec"] = spec

        async def run():
            async with Client(self.target) as client:
                return await client.call_tool("search_docs", arguments)

        try:
            result = anyio.run(run)
        except Exception as e:  # connection refused, protocol error, ...
            raise DocsMcpUnavailable(
                f"cannot reach docs-mcp at {self.target!r} ({type(e).__name__}: {e}); "
                "start it with `make docs-mcp`"
            ) from e
        self.calls += 1
        if result.is_error:
            raise DocsMcpError(result.content[0].text)
        return result.structured_content["results"]


def rank_of_hit(question: Question, results: list[dict]) -> int | None:
    """1-based rank of the first result that answers the question, or None."""
    for rank, r in enumerate(results, start=1):
        if r["spec"] == question.spec and (
            r["clause"] in question.expected_clauses
            or (r.get("table_number") is not None and r["table_number"] in question.expected_tables)
        ):
            return rank
    return None


Config = tuple[str, bool]  # (mode, spec filter on)
CONFIGS: list[Config] = [(mode, filtered) for mode in MODES for filtered in (False, True)]


@dataclass
class DocsEval:
    questions: list[Question]
    ranks: dict[str, dict[Config, int | None]]  # question id -> config -> rank of first hit
    searches: int

    def group(self, author: str) -> list[Question]:
        return [q for q in self.questions if q.author == author]

    def hits(self, author: str, config: Config) -> int:
        return sum(self.ranks[q.id][config] is not None for q in self.group(author))


def run_eval(client: DocsSearchClient, questions: list[Question], k: int = K) -> DocsEval:
    ranks: dict[str, dict[Config, int | None]] = defaultdict(dict)
    for q in questions:
        for mode, filtered in CONFIGS:
            results = client.search(q.question, q.spec if filtered else None, mode, k)
            ranks[q.id][(mode, filtered)] = rank_of_hit(q, results)
    return DocsEval(questions, dict(ranks), client.calls)


def group_cells(ev: DocsEval, author: str) -> dict[str, tuple[int, int, int]]:
    """{mode: (hits without filter, hits with filter, question count)} for one author group."""
    n = len(ev.group(author))
    return {
        mode: (ev.hits(author, (mode, False)), ev.hits(author, (mode, True)), n) for mode in MODES
    }


def render(ev: DocsEval, k: int = K) -> str:
    def cell(rank: int | None) -> str:
        return str(rank) if rank else "-"

    head = f"{'id':7} {'spec':9} {'author':9} " + " ".join(
        f"{mode[:3] + ('+f' if f else ''):>5}" for mode, f in CONFIGS
    )
    lines = [f"hit rank within the top {k} (- = miss; +f = with spec filter)", head]
    for q in ev.questions:
        cells = " ".join(f"{cell(ev.ranks[q.id][c]):>5}" for c in CONFIGS)
        lines.append(f"{q.id:7} {q.spec:9} {q.author:9} {cells}  {q.question[:46]}")
    lines.append("")
    lines.append(f"recall@{k}      " + "".join(f"{m + ' (no filter / filter)':>30}" for m in MODES))
    for author in ("human", "assistant"):
        n = len(ev.group(author))
        if not n:
            continue
        cells = group_cells(ev, author)
        text = "".join(f"{f'{p}/{n} / {f}/{n}':>30}" for p, f, _ in cells.values())
        lines.append(f"{author + ' (' + str(n) + ')':14}" + text)
    return "\n".join(lines)


def misses(ev: DocsEval, config: Config) -> list[Question]:
    return [q for q in ev.questions if ev.ranks[q.id][config] is None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval recall@5 on the golden doc questions")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--note", default="", help="one line for the last column of RESULTS.md")
    parser.add_argument("--no-record", action="store_true", help="do not append to RESULTS.md")
    args = parser.parse_args(argv)

    questions = load_questions(args.file)
    client = DocsSearchClient(get_agent_settings().docs_mcp_url)
    try:
        ev = run_eval(client, questions)
    except (DocsMcpUnavailable, DocsMcpError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"{len(questions)} doc questions, {ev.searches} searches through docs-mcp (free)\n")
    print(render(ev))
    for config in ((m, f) for m in MODES for f in (False, True)):
        missed = misses(ev, config)
        if missed:
            label = f"{config[0]}{' + filter' if config[1] else ''}"
            print(f"\nmisses, {label}: " + ", ".join(q.id for q in missed))
    if not args.no_record:
        run_id = f"eval-docs-{uuid.uuid4().hex[:8]}"
        groups = {
            author: group_cells(ev, author) for author in ("human", "assistant") if ev.group(author)
        }
        note = " ".join(p for p in (f"{ev.searches} searches, free.", args.note) if p)
        append_docs_rows(run_id=run_id, commit=git_commit(), groups=groups, note=note)
        print(f"\nrows appended to eval/RESULTS.md ({run_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
