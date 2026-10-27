"""The verifier: deterministic checks of the writer's draft against the evidence it was given.

No LLM, no database. `verify(draft, ...)` returns a JSON-able dict:

    {"ok": bool, "failures": [{"type", "sentence", "detail"}], "evidence_map": [...]}

Evidence is the SQL result (`[S1]`), the retrieved spec chunks (`[D1]`, `[D2]`, ...) and a
statistical analysis (`[A1]`, the stats route). Checks:

- citations: every `[Sn]` / `[Dn]` must exist (dangling_citation); a sentence with a number that is
  not just the question's own, or with a quotation, needs a citation (missing_citation); a draft
  with evidence must cite something.
- numbers: each number must be one of the SQL cells, a number in a cited chunk, a number in the
  question or in the executed SQL, or derived from the SQL cells of one column (a sum, difference,
  ratio or percentage of two cells, a share of the total or of the row count). A draft number is
  matched at its own precision, so 12.35 -> 12.3 / 12.4 / 12 all count as rounded, and "1,234"
  equals 1234 (unsupported_number).
- units: a number followed by a unit needs the unit in the question, a cited chunk, a column name
  or comment, or the executed SQL; a column whose comment says UNKNOWN carries no unit at all
  (invented_unit).
- assumptions: a column whose comment says ASSUMED means the draft must say so
  (missing_assumption_wording).
- quotations: quoted text must occur in a cited chunk, a SQL cell or the question
  (unsupported_quote).
- band rows: when the doc agent supplied band rows, at least one of their chunks must be cited
  (missing_band_citation).
- companies: a SQL that groups by licensee_name together with licensee_frn, or by licensee_name
  alone while v_uls_licences exposes licensee_frn, splits one company across its name spellings
  (frn_name_grouping; the schema comments say to group by licensee_frn alone).
- statistics (stats route): a number must be a numeric leaf of the cited analysis result or its
  parameters, a run count, a percentage of a share, a bound such as "p < 0.001" that a p- or
  q-value in the result satisfies, or a sum, difference or ratio of leaves of one result row.
  Feature values have no unit (invented_unit). Three rules of the statistics: "significant" needs
  a p- or q-value in the result below 0.05 (unsupported_stat_claim; "not significant" needs one at
  or above it); cause words are refused (causal_claim: SECOM is observational); a p-value in the
  answer needs the number of values behind it (missing_n); an n stated in a sentence about a test
  must be one of that test's group sizes or their total, not another number of the result such as
  the runs of the extract (wrong_n).
- mixed questions: each numeric literal that the SQL agent compared with a frequency column in a
  WHERE clause (a band edge) must appear in the text of a cited chunk (unsupported_sql_literal).
  Zero is exempt: it is a guard, not an edge.
"""

import itertools
import re
from typing import Any

import sqlglot
from sqlglot import exp

CITE_RE = re.compile(r"\[([SDA])(\d+)\]")
NUM_RE = re.compile(r"(?<![\w.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")
UNIT_NAMES = r"dBm|dBi|dBc|dBW|dB|MHz|GHz|kHz|Hz|mW|kW|ms|km"
UNIT_AFTER_NUMBER_RE = re.compile(rf"(?:\d)\s*({UNIT_NAMES})(?![A-Za-z])", re.I)
UNIT_ANYWHERE_RE = re.compile(rf"(?<![A-Za-z])({UNIT_NAMES})(?![A-Za-z])", re.I)
QUOTE_RE = re.compile(r"\"([^\"]{3,})\"|“([^”]{3,})”")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“\[(])")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
MAX_DERIVED_CELLS = 60
TRIVIAL_SOURCES = {"question", "sql-text"}

Failure = dict[str, Any]


def clean_numbers(text: str) -> str:
    return text.replace("−", "-")


def parse_number(token: str) -> tuple[float, int]:
    """(value, decimals the writer gave it). 1.5e-06 has 7 decimals: 1.5 shifted by the exponent."""
    plain = token.replace(",", "")
    mantissa, _, exponent = plain.lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    if exponent:
        decimals = max(decimals - int(exponent), 0)
    return float(plain), decimals


def numbers_in(text: str) -> set[float]:
    found: set[float] = set()
    for m in NUM_RE.finditer(clean_numbers(text)):
        value = parse_number(m.group(0))[0]
        found.update((value, abs(value)))
    return found


def units_in(text: str) -> set[str]:
    return {m.group(1).lower() for m in UNIT_ANYWHERE_RE.finditer(text)}


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = LIST_MARKER_RE.sub("", line.strip())
        out += [p.strip() for p in SENTENCE_SPLIT_RE.split(line) if p.strip()]
    return out


def _norm(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


def _close(candidate: float, value: float, decimals: int) -> bool:
    """`value` is `candidate` rounded to `decimals`. The float slack is relative: a fixed 1e-9 would
    make every tiny number (a p-value of 1e-12) match any other."""
    return abs(candidate - value) <= 0.5 * 10**-decimals + 1e-9 * max(abs(candidate), abs(value))


def _sql_cells(sql: dict | None) -> list[list[Any]]:
    """The result as columns of cells: [[cells of column 0], [cells of column 1], ...]."""
    if not sql or sql.get("status") != "answered":
        return []
    rows = sql.get("rows") or []
    return [[row[i] for row in rows if i < len(row)] for i in range(len(sql.get("columns") or []))]


def _as_float(cell: Any) -> float | None:
    if isinstance(cell, bool) or cell is None:
        return None
    if isinstance(cell, int | float):
        return float(cell)
    if isinstance(cell, str):
        try:
            return float(cell.replace(",", ""))
        except ValueError:
            return None
    return None


def sql_numbers(sql: dict | None) -> tuple[set[float], set[float]]:
    """(the cell values, numbers derived from them)."""
    cells = _sql_cells(sql)
    values: set[float] = set()
    derived: set[float] = set()
    row_count = float(sql.get("row_count") or 0) if sql else 0.0
    for column in cells:
        numeric = [v for v in map(_as_float, column) if v is not None][:MAX_DERIVED_CELLS]
        values.update(numeric)
        total = sum(numeric)
        if len(numeric) > 1:
            derived.update((total, total / len(numeric)))
        for a in numeric:
            if total:
                derived.add(a / total * 100)
            if row_count:
                derived.add(a / row_count * 100)
        for a, b in itertools.combinations(numeric, 2):
            derived.update((a + b, abs(a - b)))
            if b:
                derived.update((a / b, a / b * 100))
            if a:
                derived.update((b / a, b / a * 100))
    return values, derived


def sql_text(sql: dict | None) -> str:
    return "\n".join(str(c) for col in _sql_cells(sql) for c in col if isinstance(c, str))


P_BOUNDS = (0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0.000001)
MAX_ROW_LEAVES = 12
# the keys that hold the number of values a statistic rests on (not the runs of the extract)
SAMPLE_KEYS = {"n", "n_pass", "n_fail", "n_pairs"}
P_KEYS = {"p_value", "q_value"}
NO_COMBINE = P_KEYS | {"alpha"}  # never combined arithmetically (their combinations are noise)
MIN_DERIVED = 0.01  # a derived number below this is noise (0.05 / 97), not something to state


def _combos(values: list[float], out: set[float]) -> None:
    """Sums, differences and ratios (also as percentages) of the values of one result row."""
    for a, b in itertools.combinations(values[:MAX_ROW_LEAVES], 2):
        found = {a + b, abs(a - b)}
        if b:
            found.update((a / b, a / b * 100))
        if a:
            found.update((b / a, b / a * 100))
        if a + b:  # each as a share of the two together
            found.update((a / (a + b), a / (a + b) * 100, b / (a + b), b / (a + b) * 100))
        out.update(v for v in found if abs(v) >= MIN_DERIVED)


def stats_evidence(stats: dict) -> dict[str, Any]:
    """What the answer may say about an analysis result [A1]: its numeric leaves (result,
    parameters, run counts), numbers derived within one result row, the bounds that its p- and
    q-values satisfy, and which leaves are p/q-values and group sizes."""
    leaves: list[tuple[str, float]] = []
    derived: set[float] = set()
    totals: set[float] = set()  # group sizes added up: the total tested n of a comparison

    def numeric(v: Any) -> bool:
        return isinstance(v, int | float) and not isinstance(v, bool)

    def walk(node: Any, key: str = "") -> None:
        if numeric(node):
            leaves.append((key, float(node)))
        elif isinstance(node, dict):
            if numeric(node.get("n_pass")) and numeric(node.get("n_fail")):
                totals.add(float(node["n_pass"]) + float(node["n_fail"]))
            sizes = [
                float(c["n"]) for c in node.values() if isinstance(c, dict) and numeric(c.get("n"))
            ]
            if len(sizes) > 1:
                totals.add(sum(sizes))
            row = [float(v) for k, v in node.items() if numeric(v) and k not in NO_COMBINE]
            if len(row) > 1:
                _combos(row, derived)
            children = [v for v in node.values() if isinstance(v, dict)]
            if len(children) > 1:  # sibling dicts (groups): same key across them
                for k in {k for c in children for k in c} - NO_COMBINE:
                    same = [float(c[k]) for c in children if numeric(c.get(k))]
                    if len(same) > 1:
                        _combos(same, derived)
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)

    walk(stats.get("result"))
    walk(stats.get("params"))
    leaves += [
        ("n_runs_used", float(stats.get("n_runs_used") or 0)),
        ("n_runs_in_extract", float(stats.get("n_runs_in_extract") or 0)),
    ]
    values = {v for _, v in leaves}
    p_values = [v for k, v in leaves if k in P_KEYS]
    for k, v in leaves:
        if k not in NO_COMBINE and 0 <= v <= 1:
            derived.add(v * 100)  # a share written as a percentage
    for p in p_values:
        derived.update(t for t in P_BOUNDS if p < t)  # "p < 0.001" is true
    sample_sizes = {v for k, v in leaves if k in SAMPLE_KEYS} | totals
    runs = {v for k, v in leaves if k in ("n_runs_used", "n_runs_in_extract")}
    return {
        "values": values,
        "derived": derived,
        "p_values": p_values,
        # what a statistic can rest on; the runs of the extract only if nothing else is stated
        "n_values": sample_sizes or runs,
        "sample_sizes": sample_sizes,
    }


SIGNIFICANT_RE = re.compile(r"\bsignifican(?:t|tly|ce)\b", re.I)
NOT_SIGNIFICANT_RE = re.compile(
    r"\b(?:not|no|non|nor|never|isn't|aren't|wasn't|weren't)\b[\w\s,-]{0,40}\bsignifican|"
    r"\binsignificant|\bnon-significant",
    re.I,
)
P_MENTION_RE = re.compile(r"\b[pq][- ]?values?\b|\b[pq]\s*[=<>≤≥]", re.I)
CAUSAL_RE = re.compile(
    r"\b(?:cause[sd]?|causing|caused\s+by|leads?\s+to|led\s+to|due\s+to|results?\s+in|"
    r"resulting\s+in|because\s+of|drives?|driven\s+by|responsible\s+for)\b",
    re.I,
)
TEST_RE = re.compile(
    r"\b(?:tests?|mann-whitney|welch|t-test|spearman|pearson|correlation|effect\s+size|"
    r"rank-biserial|significan\w*|p-values?|q-values?)\b|\b[pq]\s*[=<>≤≥]",
    re.I,
)
# a number stated as a sample size: "n=104", "n of 1,456", "1456 passing", "104 failing values"
N_STATED_RES = (
    re.compile(r"\bn\s*(?:=|:|of)\s*(-?[\d,]+(?:\.\d+)?)", re.I),
    re.compile(
        r"(?<![\w.,])(-?[\d,]*\d)\s+(?:(?:passing|failing|pass|fail|valid|non-missing|paired)\s+)?"
        r"(?:values?|runs?|samples?|observations?|measurements?|pairs?|wafers?|lots?)\b",
        re.I,
    ),
    re.compile(r"(?<![\w.,])(-?[\d,]*\d)\s+(?:passing|failing|pass|fail)\b", re.I),
)
NEGATION_RE = re.compile(r"\b(?:not|no|cannot|can't|without|neither|nor)\b|n't\b", re.I)


def stats_claim_failures(draft: str, evidence: dict[str, Any]) -> list[Failure]:
    """unsupported_stat_claim, causal_claim, wrong_n and missing_n over the sentences of `draft`."""
    failures: list[Failure] = []
    p_values = evidence["p_values"]
    sentences = split_sentences(draft)
    first_p_sentence: int | None = None
    for index, raw in enumerate(sentences):
        sentence = CITE_RE.sub("", raw)
        if P_MENTION_RE.search(sentence) and first_p_sentence is None:
            first_p_sentence = index
        if SIGNIFICANT_RE.search(sentence):
            if NOT_SIGNIFICANT_RE.search(sentence):
                if not any(p >= 0.05 for p in p_values):
                    failures.append(
                        _fail(
                            "unsupported_stat_claim",
                            index,
                            "'not significant' needs a p- or q-value of 0.05 or more in the result",
                        )
                    )
            elif not any(p < 0.05 for p in p_values):
                why = "the analysis has none" if not p_values else "all are 0.05 or more"
                failures.append(
                    _fail(
                        "unsupported_stat_claim",
                        index,
                        f"'significant' needs a p- or q-value below 0.05 in the result ({why})",
                    )
                )
            elif not P_MENTION_RE.search(draft) and not any(
                abs(p - v) <= 0.5 * 10**-d
                for m in NUM_RE.finditer(clean_numbers(draft))
                for v, d in [parse_number(m.group(0))]
                for p in p_values
            ):
                failures.append(
                    _fail(
                        "unsupported_stat_claim",
                        index,
                        "'significant' must come with the p- or q-value it rests on",
                    )
                )
        sizes = evidence["sample_sizes"]
        if sizes and TEST_RE.search(sentence):
            reported: set[float] = set()  # the patterns overlap: one failure per stated number
            for pattern in N_STATED_RES:
                for m in pattern.finditer(clean_numbers(sentence)):
                    stated = float(m.group(1).replace(",", ""))
                    if stated in reported:
                        continue
                    reported.add(stated)
                    if not any(abs(stated - s) < 0.5 for s in sizes):
                        shown = ", ".join(f"{s:g}" for s in sorted(sizes)[:8])
                        failures.append(
                            _fail(
                                "wrong_n",
                                index,
                                f"n = {m.group(1)} is not a number of values the test used "
                                f"(its group sizes and total: {shown})",
                            )
                        )
        if CAUSAL_RE.search(sentence) and not NEGATION_RE.search(sentence):
            failures.append(
                _fail(
                    "causal_claim",
                    index,
                    "the analysis shows an association in observational data, not a cause: "
                    f"remove '{CAUSAL_RE.search(sentence).group(0)}'",
                )
            )
    # a p-value (stated as such, or as a number equal to one) needs the n behind it
    numbers = [
        parse_number(m.group(0)) for m in NUM_RE.finditer(clean_numbers(CITE_RE.sub("", draft)))
    ]
    states_p = first_p_sentence is not None or any(
        abs(p - v) <= 0.5 * 10**-d and p < 1 for v, d in numbers for p in p_values
    )
    has_n = any(abs(n - v) < 1e-9 for v, _ in numbers for n in evidence["n_values"])
    if states_p and not has_n:
        failures.append(
            _fail(
                "missing_n",
                first_p_sentence,
                "a p-value is stated without the number of values (n) it rests on",
            )
        )
    return failures


def sql_band_edges(executed_sql: str | None) -> list[tuple[str, float]]:
    """(column, literal) for each numeric literal compared with a frequency column in a WHERE."""
    if not executed_sql:
        return []
    try:
        tree = sqlglot.parse_one(executed_sql, read="postgres")
    except sqlglot.errors.SqlglotError:
        return []

    def literal(node: exp.Expression) -> float | None:
        if isinstance(node, exp.Paren):
            return literal(node.this)
        sign = 1.0
        if isinstance(node, exp.Neg):
            sign, node = -1.0, node.this
        if isinstance(node, exp.Cast):
            node = node.this
        if isinstance(node, exp.Literal) and not node.is_string:
            try:
                return sign * float(node.name)
            except ValueError:
                return None
        return None

    def freq_column(node: exp.Expression) -> str | None:
        for column in node.find_all(exp.Column):
            if "freq" in column.name.lower():
                return column.name
        return None

    edges: list[tuple[str, float]] = []
    for where in tree.find_all(exp.Where):
        for node in where.find_all(exp.Binary):
            if not isinstance(node, exp.GT | exp.GTE | exp.LT | exp.LTE | exp.EQ | exp.NEQ):
                continue
            for column_side, literal_side in ((node.left, node.right), (node.right, node.left)):
                value, column = literal(literal_side), freq_column(column_side)
                if value is not None and column is not None:
                    edges.append((column, value))
        for node in where.find_all(exp.Between):
            column = freq_column(node.this)
            for bound in (node.args.get("low"), node.args.get("high")):
                value = literal(bound) if bound is not None else None
                if value is not None and column is not None:
                    edges.append((column, value))
    return edges


def _fail(kind: str, sentence: int | None, detail: str) -> Failure:
    return {"type": kind, "sentence": sentence, "detail": detail}


def verify(
    draft: str,
    *,
    question: str,
    route: str,
    sql: dict | None,
    chunks: list[dict],
    sql_question: str = "",
    stats: dict | None = None,
) -> dict:
    failures: list[Failure] = []
    evidence_map: list[dict] = []
    sql_ok = bool(sql) and sql.get("status") == "answered"
    stats_ok = bool(stats) and stats.get("status") == "answered"
    stats_ev = stats_evidence(stats) if stats_ok else None
    if not draft.strip():
        return {
            "ok": False,
            "failures": [_fail("empty_answer", None, "the draft is empty")],
            "evidence_map": [],
        }

    cell_values, derived_values = sql_numbers(sql)
    executed = (sql or {}).get("executed_sql") or ""
    flags: dict[str, dict] = (sql or {}).get("column_flags") or {}
    question_numbers = numbers_in(question)
    sql_literals = numbers_in(executed)
    # the SQL result's row count (zero included) is a number the answer may state; without a SQL
    # result there is none, or a tiny number such as a p-value would match the "0 rows" of nothing
    row_numbers = (
        {float(sql.get("row_count") or 0), float(len(sql.get("rows") or []))} if sql_ok else set()
    )
    cited = {(m.group(1), int(m.group(2))) for m in CITE_RE.finditer(draft)}
    cited_chunks = [c for i, c in enumerate(chunks, start=1) if ("D", i) in cited]
    chunk_numbers: dict[int, set[float]] = {
        i: numbers_in(c["text"]) | numbers_in(_chunk_header(c))
        for i, c in enumerate(chunks, start=1)
    }

    # --- dangling citations ---------------------------------------------------------------------
    valid = {("D", i) for i in range(1, len(chunks) + 1)} | ({("S", 1)} if sql_ok else set())
    valid |= {("A", 1)} if stats_ok else set()
    for kind, n in sorted(cited - valid):
        failures.append(_fail("dangling_citation", None, f"[{kind}{n}] does not exist"))
    if (sql_ok or chunks or stats_ok) and not (cited & valid):
        failures.append(_fail("missing_citation", None, "the answer cites no source"))

    # --- the band rows the doc agent supplied must be cited --------------------------------------
    band_chunks = [i for i, c in enumerate(chunks, start=1) if c.get("band_lines")]
    if band_chunks and not any(("D", i) in cited for i in band_chunks):
        tags = ", ".join(f"[D{i}]" for i in band_chunks)
        failures.append(
            _fail(
                "missing_band_citation",
                None,
                f"the band edges come from {tags}; the answer must cite at least one of them",
            )
        )

    # --- units that the evidence carries --------------------------------------------------------
    question_units = units_in(question)
    chunk_units = (
        set().union(*(units_in(c["text"]) for c in cited_chunks)) if cited_chunks else set()
    )
    column_units = units_in(" ".join((sql or {}).get("columns") or [])) | units_in(executed)
    for flag in flags.values():
        column_units.update(flag.get("units") or [])
    unknown_columns = [name for name, f in flags.items() if f.get("unknown")]
    corpus = _norm(" ".join([question, sql_text(sql)] + [c["text"] for c in cited_chunks]))

    for index, sentence in enumerate(split_sentences(draft)):
        cites = [(m.group(1), int(m.group(2))) for m in CITE_RE.finditer(sentence)]
        body = clean_numbers(CITE_RE.sub("", sentence))
        sentence_chunk_cites = [n for k, n in cites if k == "D" and 1 <= n <= len(chunks)]
        numbers: list[dict] = []
        need_citation = False

        for m in NUM_RE.finditer(body):
            value, decimals = parse_number(m.group(0))
            source = _number_source(
                value,
                decimals,
                cell_values,
                derived_values,
                chunk_numbers,
                cited,
                question_numbers,
                sql_literals | row_numbers,
                stats_ev,
            )
            numbers.append({"number": m.group(0), "source": source[0], "match": source[1]})
            if source[0] == "unsupported":
                failures.append(
                    _fail("unsupported_number", index, f"{m.group(0)} is in no evidence")
                )
                need_citation = True
            elif source[0] not in TRIVIAL_SOURCES:
                need_citation = True

        quotes = [q for m in QUOTE_RE.finditer(body) for q in m.groups() if q]
        for quote in quotes:
            need_citation = True
            if _norm(quote) not in corpus:
                failures.append(
                    _fail("unsupported_quote", index, f"quoted text {quote!r} not found")
                )

        if need_citation and not cites:
            failures.append(_fail("missing_citation", index, "a sourced claim has no citation"))

        for um in UNIT_AFTER_NUMBER_RE.finditer(body):
            unit = um.group(1).lower()
            only_sql = not sentence_chunk_cites and ("S", 1) in cites
            only_stats = (
                stats_ok
                and ("A", 1) in cites
                and not sentence_chunk_cites
                and ("S", 1) not in cites
            )
            if only_stats:
                if unit not in question_units:
                    failures.append(
                        _fail(
                            "invented_unit",
                            index,
                            f"unit {um.group(1)} on a SECOM feature value, which has no known unit",
                        )
                    )
            elif unknown_columns and only_sql:
                if unit not in question_units:
                    failures.append(
                        _fail(
                            "invented_unit",
                            index,
                            f"unit {um.group(1)} on a value whose unit is UNKNOWN "
                            f"({', '.join(unknown_columns)})",
                        )
                    )
            elif unit not in (question_units | chunk_units | column_units):
                failures.append(
                    _fail("invented_unit", index, f"unit {um.group(1)} is in no evidence")
                )

        evidence_map.append(
            {
                "sentence": index,
                "text": sentence[:140],
                "cites": [f"{k}{n}" for k, n in cites],
                "numbers": numbers,
            }
        )

    # --- statistics: significance words, causes, the n behind a p-value ---------------------------
    if stats_ev is not None:
        failures += stats_claim_failures(draft, stats_ev)

    # --- assumptions ----------------------------------------------------------------------------
    assumed = [name for name, f in flags.items() if f.get("assumed")]
    if assumed and not re.search(r"assum", draft, re.I):
        failures.append(
            _fail(
                "missing_assumption_wording",
                None,
                f"the schema marks {', '.join(assumed)} as ASSUMED; the answer must say so",
            )
        )

    # --- band edges in the SQL (mixed questions) ------------------------------------------------
    band_edges: list[dict] = []
    if route == "mixed" and sql_ok:
        for column, value in sql_band_edges(executed):
            if value == 0:
                continue
            supplier = next(
                (
                    f"D{i}"
                    for i, c in enumerate(chunks, start=1)
                    if ("D", i) in cited
                    and any(abs(value - n) < 1e-9 for n in numbers_in(c["text"]))
                ),
                None,
            )
            band_edges.append(
                {
                    "column": column,
                    "value": value,
                    "cited_chunk": supplier,
                    "chunk_id": chunks[int(supplier[1:]) - 1]["chunk_id"] if supplier else None,
                }
            )
            if supplier is None:
                failures.append(
                    _fail(
                        "unsupported_sql_literal",
                        None,
                        f"the SQL compares {column} with {value:g}: in no cited chunk",
                    )
                )

    # --- companies are identified by licensee_frn, not by name (schema convention) -------------
    if sql_ok:
        problem = frn_name_grouping(executed)
        if problem:
            failures.append(_fail("frn_name_grouping", None, problem))

    # --- band table joins: which band, and does the side match the question? --------------------
    band_joins: list[dict] = []
    if route == "mixed" and sql_ok:
        uses_table, used_sides, band_joins = band_table_use(executed)
        asked = asked_sides(f"{question}\n{sql_question}")
        if uses_table and used_sides and asked and (asked - used_sides):
            failures.append(
                _fail(
                    "edge_side_mismatch",
                    None,
                    f"the question asks for the {_sides(asked)} range but the SQL uses only the "
                    f"{_sides(used_sides)} columns of spec_operating_bands",
                )
            )

    return {
        "ok": not failures,
        "failures": failures,
        "evidence_map": evidence_map,
        "band_edges": band_edges,
        "band_joins": band_joins,
    }


FRN_VIEWS = frozenset({"v_uls_licences"})  # the relations that expose licensee_frn


def _group_columns(group: exp.Group) -> set[str]:
    """Column names inside a GROUP BY, with ordinals (GROUP BY 1) and output aliases resolved to
    the SELECT expression they stand for."""
    select = group.parent if isinstance(group.parent, exp.Select) else None
    outputs = list(select.expressions) if select is not None else []
    by_alias = {e.alias: e.this for e in outputs if isinstance(e, exp.Alias)}
    names: set[str] = set()
    for item in group.expressions:
        targets = [item]
        if isinstance(item, exp.Literal) and not item.is_string and item.name.isdigit():
            index = int(item.name) - 1
            if 0 <= index < len(outputs):
                chosen = outputs[index]
                targets = [chosen.this if isinstance(chosen, exp.Alias) else chosen]
        elif isinstance(item, exp.Column) and not item.table and item.name in by_alias:
            targets = [by_alias[item.name]]
        for target in targets:
            names.update(c.name.lower() for c in target.find_all(exp.Column))
    return names


def frn_name_grouping(executed_sql: str) -> str | None:
    """A message when the SQL groups companies by licensee_name together with licensee_frn, or by
    licensee_name alone while licensee_frn is available in the queried view; else None. One company
    (one FRN) appears under several name spellings, so this splits it."""
    try:
        tree = sqlglot.parse_one(executed_sql, read="postgres")
    except sqlglot.errors.SqlglotError:
        return None
    has_frn_view = any(t.name in FRN_VIEWS for t in tree.find_all(exp.Table))
    for group in tree.find_all(exp.Group):
        names = _group_columns(group)
        if "licensee_name" not in names:
            continue
        if "licensee_frn" in names:
            return (
                "the SQL groups by licensee_name together with licensee_frn, which splits one "
                "company across its name spellings: group by licensee_frn alone and show "
                "min(licensee_name)"
            )
        if has_frn_view:
            return (
                "the SQL groups companies by licensee_name although licensee_frn is available: "
                "group by licensee_frn alone and show min(licensee_name)"
            )
    return None


SIDE_WORDS = {
    "ul": re.compile(r"\b(uplink|UL)\b", re.I),
    "dl": re.compile(r"\b(downlink|DL)\b", re.I),
}
SIDE_COLUMNS = {
    "ul_low_mhz": "ul",
    "ul_high_mhz": "ul",
    "dl_low_mhz": "dl",
    "dl_high_mhz": "dl",
}


def _sides(sides: set[str]) -> str:
    return " and ".join({"ul": "uplink", "dl": "downlink"}[s] for s in sorted(sides, reverse=True))


def asked_sides(text: str) -> set[str]:
    """Which direction(s) the question names: {'ul'}, {'dl'} or both."""
    return {side for side, pattern in SIDE_WORDS.items() if pattern.search(text)}


def band_table_use(executed_sql: str) -> tuple[bool, set[str], list[dict]]:
    """(the SQL reads spec_operating_bands, the sides of its edge columns it uses, the
    (spec, band) literals it filters the table on)."""
    try:
        tree = sqlglot.parse_one(executed_sql, read="postgres")
    except sqlglot.errors.SqlglotError:
        return False, set(), []
    if not any(t.name == "spec_operating_bands" for t in tree.find_all(exp.Table)):
        return False, set(), []
    sides = {SIDE_COLUMNS[c.name] for c in tree.find_all(exp.Column) if c.name in SIDE_COLUMNS}
    found: dict[str, list[str]] = {"spec": [], "band": []}
    for node in tree.find_all(exp.EQ):
        for column, value in ((node.left, node.right), (node.right, node.left)):
            if (
                isinstance(column, exp.Column)
                and column.name in found
                and isinstance(value, exp.Literal)
                and value.is_string
            ):
                found[column.name].append(value.name)
    joins = [
        {"spec": found["spec"][i] if i < len(found["spec"]) else None, "band": band}
        for i, band in enumerate(found["band"])
    ]
    return True, sides, joins


def _chunk_header(chunk: dict) -> str:
    return " ".join(
        str(chunk.get(k) or "")
        for k in ("chunk_id", "spec", "version", "clause", "clause_title", "table_number")
    )


def _number_source(
    value: float,
    decimals: int,
    cells: set[float],
    derived: set[float],
    chunk_numbers: dict[int, set[float]],
    cited: set[tuple[str, int]],
    question_numbers: set[float],
    sql_text_numbers: set[float],
    stats_ev: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """(source, match) for one draft number; source 'unsupported' when no evidence supports it."""
    if any(abs(c - value) < 1e-9 for c in cells):
        return "sql", "exact"
    for i in sorted(n for k, n in cited if k == "D"):
        if any(abs(c - value) < 1e-9 for c in chunk_numbers.get(i, ())):
            return f"doc:D{i}", "exact"
    if any(abs(c - value) < 1e-9 for c in question_numbers):
        return "question", "exact"
    if any(abs(c - value) < 1e-9 for c in sql_text_numbers):
        return "sql-text", "exact"
    if any(_close(c, value, decimals) for c in cells):
        return "sql", "rounded"
    if any(_close(c, value, decimals) for c in derived):
        return "sql", "derived"
    if stats_ev is not None:  # relative to the writer's own precision, never a fixed epsilon
        if any(c == value for c in stats_ev["values"]):
            return "stats", "exact"
        if any(_close(c, value, decimals) for c in stats_ev["values"]):
            return "stats", "rounded"
        if any(_close(c, value, decimals) for c in stats_ev["derived"]):
            return "stats", "derived"
    return "unsupported", ""
