"""make eval-docs: retrieval recall@5 for the doc questions of the golden set, through docs-mcp.

    make docs-mcp          # in another terminal (needs Ollama for the vector and hybrid modes)
    make eval-docs

Free: no LLM is called (queries are embedded locally by Ollama). For every doc question it runs
search_docs in vector, keyword and hybrid mode, each without and with the spec filter, and a hit
is a top-5 chunk of the question's spec whose clause equals an expected clause exactly (a
sub-clause does not count) or whose table number is an expected table. Recall is reported
separately for the human-written and the assistant-written questions, and one row per group is
appended to eval/RESULTS.md. Agents reach documents only through docs-mcp, and so does this.

`--queries-from RUN_ID` (repeatable) compares the three ways the doc agent can search, with the
planner's real `doc_query` of each question taken from a saved end-to-end run (still free): the
doc_query alone, the user's question alone, and both merged by reciprocal rank fusion before the
variant rule and the cut (`also_query`), each with the spec the agent had for that question and
with no spec filter. Vector mode, as the orchestrator uses.
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

    def search(
        self,
        query: str,
        spec: str | None,
        mode: str,
        k: int = K,
        context: str | None = None,
        also_query: str | None = None,
    ) -> list[dict]:
        arguments: dict[str, Any] = {"query": query, "mode": mode, "k": k}
        if spec:
            arguments["spec"] = spec
        if context:
            arguments["context"] = context
        if also_query:
            arguments["also_query"] = also_query

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


# --- doc_query versus question versus both ----------------------------------------------------

WAYS = ("doc_query", "question", "both")
SPEC_SETTINGS = ("agent spec", "no filter")


@dataclass
class QueryInput:
    """What the doc agent had for one question in a saved run."""

    doc_query: str  # the planner's doc_query with the clause / table references put back
    spec: str | None  # the spec the agent searched with (rule or hint), None when unknown


def saved_query_inputs(run_id: str, questions: list[Question]) -> dict[str, QueryInput]:
    """The planner's doc_query and the spec of each doc question in a saved end-to-end run."""
    from rflens.eval.eval_e2e import thread_state
    from rflens.graph.app import open_checkpointer
    from rflens.graph.nodes import _query_with_refs

    found: dict[str, QueryInput] = {}
    with open_checkpointer() as saver:
        for q in questions:
            state = thread_state(saver, f"{run_id}-{q.id}")
            if not state or not (state.get("plan") or {}).get("doc_query"):
                continue  # not routed to documents in that run
            found[q.id] = QueryInput(
                _query_with_refs(state["plan"]["doc_query"], q.question), state.get("spec")
            )
    return found


def search_way(
    client: DocsSearchClient, q: Question, given: QueryInput, way: str, spec: str | None, k: int
) -> list[dict]:
    if way == "doc_query":
        return client.search(given.doc_query, spec, "vector", k, context=q.question)
    if way == "question":
        return client.search(q.question, spec, "vector", k)
    return client.search(
        given.doc_query, spec, "vector", k, context=q.question, also_query=q.question
    )


def run_ways_eval(
    client: DocsSearchClient, questions: list[Question], inputs: dict[str, QueryInput], k: int = K
) -> dict[str, dict[tuple[str, str], int | None]]:
    """question id -> (way, spec setting) -> rank of the first hit."""
    ranks: dict[str, dict[tuple[str, str], int | None]] = {}
    for q in questions:
        given = inputs.get(q.id)
        if given is None:
            continue
        ranks[q.id] = {}
        for way in WAYS:
            for setting in SPEC_SETTINGS:
                spec = given.spec if setting == "agent spec" else None
                results = search_way(client, q, given, way, spec, k)
                ranks[q.id][(way, setting)] = rank_of_hit(q, results)
    return ranks


def ways_table(questions: list[Question], ranks: dict[str, dict[tuple[str, str], int | None]]):
    """{(way, setting): {author: (hits, n)}} over the questions that have saved inputs."""
    out: dict[tuple[str, str], dict[str, tuple[int, int]]] = {}
    for way in WAYS:
        for setting in SPEC_SETTINGS:
            cells = {}
            for author in ("human", "assistant"):
                group = [q for q in questions if q.author == author and q.id in ranks]
                hits = sum(ranks[q.id][(way, setting)] is not None for q in group)
                cells[author] = (hits, len(group))
            out[(way, setting)] = cells
    return out


def merged_verdict(table: dict[tuple[str, str], dict[str, tuple[int, int]]], setting: str) -> bool:
    """The merged search is kept only if, for BOTH author groups, it finds at least as many as the
    better of the two single queries."""
    return all(
        table[("both", setting)][a][0]
        >= max(table[("doc_query", setting)][a][0], table[("question", setting)][a][0])
        for a in ("human", "assistant")
    )


def render_ways(
    run_id: str, questions: list[Question], ranks: dict[str, dict[tuple[str, str], int | None]]
) -> str:
    table = ways_table(questions, ranks)
    lines = [
        f"planner doc_queries and specs from {run_id} ({len(ranks)} doc questions), vector mode"
    ]
    n = {a: sum(q.author == a and q.id in ranks for q in questions) for a in ("human", "assistant")}
    human, assistant = f"human ({n['human']})", f"assistant ({n['assistant']})"
    lines.append(f"{'way':10} {'spec':11} {human:>12} {assistant:>16}")
    for way in WAYS:
        for setting in SPEC_SETTINGS:
            c = table[(way, setting)]
            lines.append(
                f"{way:10} {setting:11} {c['human'][0]:>8}/{c['human'][1]:<3} "
                f"{c['assistant'][0]:>12}/{c['assistant'][1]:<3}"
            )
    for setting in SPEC_SETTINGS:
        lines.append(
            f"merged >= better single query for both groups ({setting}): "
            + ("YES" if merged_verdict(table, setting) else "NO")
        )
    lines.append("")
    lines.append(
        f"{'id':7} {'author':9} "
        + " ".join(f"{w[:3]}/{s[:3]:>3}" for w in WAYS for s in SPEC_SETTINGS)
    )
    for q in questions:
        if q.id in ranks:
            cells = " ".join(
                f"{str(ranks[q.id][(w, s)] or '-'):>7}" for w in WAYS for s in SPEC_SETTINGS
            )
            lines.append(f"{q.id:7} {q.author:9} {cells}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval recall@5 on the golden doc questions")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--note", default="", help="one line for the last column of RESULTS.md")
    parser.add_argument("--no-record", action="store_true", help="do not append to RESULTS.md")
    parser.add_argument(
        "--queries-from",
        action="append",
        default=[],
        metavar="RUN_ID",
        help="compare doc_query / question / both with the planner's queries of a saved e2e run",
    )
    args = parser.parse_args(argv)

    questions = load_questions(args.file)
    client = DocsSearchClient(get_agent_settings().docs_mcp_url)
    if args.queries_from:
        try:
            for run_id in args.queries_from:
                inputs = saved_query_inputs(run_id, questions)
                ranks = run_ways_eval(client, questions, inputs)
                print(render_ways(run_id, questions, ranks) + "\n")
        except (DocsMcpUnavailable, DocsMcpError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        print(f"{client.calls} searches through docs-mcp (free)")
        return 0
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
