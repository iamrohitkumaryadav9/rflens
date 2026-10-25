"""Smoke eval: recall@5 of vector-only, keyword-only and hybrid search (make eval-smoke).

Questions are the type-doc questions of eval/golden.yaml (the first ten were the smoke set; their
expected clauses were reviewed by a human). This is the quick in-process check; `make eval-docs`
runs the same questions through docs-mcp and records the result in eval/RESULTS.md. Run it after
the full embed; with a partial embed the numbers mean little and a warning says so. A hit needs
the right spec and an exactly matching clause (a sub-clause does not count) or listed table.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import Connection, text

from rflens.config import get_retrieval_settings
from rflens.db import get_engine
from rflens.retrieval.embedder import Embedder, OllamaEmbedder
from rflens.retrieval.hybrid import MODES, Hit, search

DEFAULT_FILE = Path(__file__).resolve().parents[3] / "eval" / "golden.yaml"


@dataclass(frozen=True)
class Question:
    id: str
    spec: str
    question: str
    expected_clauses: tuple[str, ...]
    expected_tables: tuple[str, ...]
    author: str = ""


def load_questions(path: Path, author: str | None = None) -> list[Question]:
    """The type-doc questions of a golden file (optionally only those by one author)."""
    data = yaml.safe_load(path.read_text("utf-8"))
    return [
        Question(
            q["id"],
            q["spec"],
            q["question"],
            tuple(q.get("expected_clauses") or ()),
            tuple(q.get("expected_tables") or ()),
            q.get("author", ""),
        )
        for q in data["questions"]
        if q.get("type") == "doc" and author in (None, q.get("author"))
    ]


def is_hit(question: Question, hit: Hit) -> bool:
    return hit.spec == question.spec and (
        hit.clause in question.expected_clauses
        or (hit.table_number is not None and hit.table_number in question.expected_tables)
    )


def recall_at_k(
    conn: Connection,
    embedder: Embedder,
    questions: list[Question],
    *,
    k: int = 5,
    filter_spec: bool = False,
) -> dict[str, list[tuple[Question, int | None]]]:
    """Per mode, per question: the 1-based rank of the first matching hit in the top k, or None.

    With filter_spec each question's own spec is passed to the search as a filter; without it
    the search runs over every spec (the default).
    """
    results: dict[str, list[tuple[Question, int | None]]] = {}
    for mode in MODES:
        rows = []
        for q in questions:
            hits = search(
                conn,
                embedder,
                q.question,
                mode=mode,
                top_k=k,
                spec=q.spec if filter_spec else None,
            )
            rank = next((i for i, h in enumerate(hits, start=1) if is_hit(q, h)), None)
            rows.append((q, rank))
        results[mode] = rows
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@5 smoke eval for 3GPP retrieval")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--spec", help="only the questions about this spec, e.g. 38.101-1")
    parser.add_argument("--author", choices=("human", "assistant"), help="only one author")
    parser.add_argument(
        "--filter-spec",
        action="store_true",
        help="pass each question's spec to the search as a filter (default: search all specs)",
    )
    args = parser.parse_args()

    settings = get_retrieval_settings()
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embed_model)
    questions = [q for q in load_questions(args.file, args.author) if args.spec in (None, q.spec)]
    with get_engine().connect() as conn:
        done, total = conn.execute(
            text("SELECT count(*) FILTER (WHERE embedding_model = :m), count(*) FROM chunks"),
            {"m": embedder.model_id},
        ).one()
        print(f"model {embedder.model_id}: {done}/{total} chunks embedded")
        if not total or done < total:
            print("WARNING: not all chunks are embedded; recall numbers are not meaningful.")
        results = recall_at_k(conn, embedder, questions, k=args.top_k, filter_spec=args.filter_spec)
    print(
        f"\n{args.file.name} ({len(questions)} doc questions); spec filter: "
        f"{'ON (each question searches only its own spec)' if args.filter_spec else 'off'}; "
        f"hit rank within top {args.top_k}:"
    )
    print(f"{'id':5} {'spec':9} " + " ".join(f"{m:>8}" for m in MODES) + "  question")
    for i, q in enumerate(questions):
        ranks = [results[m][i][1] for m in MODES]
        cells = " ".join(f"{(str(r) if r else '-'):>8}" for r in ranks)
        print(f"{q.id:5} {q.spec:9} {cells}  {q.question[:60]}")
    print(f"\nrecall@{args.top_k} over {len(questions)} questions:")
    for mode in MODES:
        hits = sum(1 for _, r in results[mode] if r is not None)
        print(f"  {mode:8} {hits}/{len(questions)} = {hits / len(questions):.2f}")


if __name__ == "__main__":
    main()
