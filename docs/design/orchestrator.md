# Orchestrator design (source of truth for `src/rflens/graph/`)

Status: approved in week 4 step 1; step 2 built it. Where the code differs from this document, this
document wins unless the difference is listed under "Build differences".

This file was written after the build, from the approved decisions as they were restated in the
step 2 and step 2b requests. Sections marked **(to confirm)** are the parts those requests did not
restate in detail; compare them with the original step-1 proposal.

## Hard rules

1. Every LLM call goes through `rflens.llm` (ledger, budget cap, loop guards). No LangChain
   chat-model wrapper and no provider SDK in the graph; `langchain_core` and `langchain_protocol`
   are present only because langgraph requires them (tested).
2. Agents reach data only through sql-mcp and docs-mcp over MCP. The checkpointer is the only
   direct database connection of the graph.
3. Tracing is off: importing `rflens.graph` sets `LANGSMITH_TRACING=false` and refuses to import if
   a tracing switch is on. `LANGGRAPH_STRICT_MSGPACK=true`.
4. The stats agent / analytics-mcp is week 5: the graph has no node for it, only room for one.

## State schema (`graph/state.py`)

JSON types only (checkpoints use strict msgpack; tuples would come back as lists).

| key | meaning |
|---|---|
| `question_id`, `question` | thread id (= question id) and the user's text |
| `route`, `plan` | `sql` / `doc` / `mixed` / `refuse`; plan = spec hint, `doc_query`, `sql_question`, refuse reason, rule matches, planned worst-case USD |
| `spec`, `spec_source` | spec filter (`36.101`, `38.101-1` or none) and where it came from: `rule`, `llm-hint`, `human`, `ambiguous`, `none` |
| `approval` | `{needed, reasons, decision{approve, by}}` |
| `chunks` | retrieved spec chunks `D1..Dn` (id, spec, version, clause, title, table, text, cut) |
| `sql` | SQL agent result `S1`: status, SQL, executed SQL, `query_id`, columns, rows (at most `ORCH_MAX_STATE_ROWS`), notes, `column_flags` (UNKNOWN / ASSUMED units) |
| `draft`, `writer_attempts` | writer output and attempt count |
| `draft_history` | every writer draft with its verification failures (`{attempt, draft, ok, failures}`), kept in the checkpoint |
| `verification` | `{ok, failures[], evidence_map[], band_edges[]}` (+ `redacted`, `withheld_sql_query_id` after redaction) |
| `status`, `answer`, `sources`, `halt` | outcome: answered / partial / refused / failed / cancelled |
| `node_stats` | per node `{calls, cost_usd, runs}`, merged by a reducer |

## Graph and routing

```
START -> planner -> approval -> doc_agent -> (mixed) sql_agent -> writer -> verifier -> finalize
                       |            |                                 ^         |
                       |            +--(doc) writer                   +--retry--+ (<= 2 retries)
                       +--(sql) sql_agent                                         v
         planner --(refuse / halt)--> finalize                                  redact -> finalize
```

- **planner** — one LLM call returning `route`, `spec` hint, `doc_query`, `sql_question`,
  `refuse_reason`. `doc_query` names the specification concept to find, with spec vocabulary and no
  database-side words (e.g. "E-UTRA operating band 5 uplink downlink frequency range").
  `sql_question` restates only the database part of a mixed question.
- **doc_agent** — no LLM. `search_docs(doc_query, spec filter, mode=ORCH_RETRIEVAL_MODE, k=5)`, then
  `get_chunk` for the top `ORCH_DOC_FULL_CHUNKS` (3). The doc agent searches with `doc_query`, never
  with the full question (falls back to the question only if the planner left it empty).
- **sql_agent** — wraps `answer_question`. For a mixed question it asks the planner's
  `sql_question` and receives the retrieved chunks as context ("take numeric limits such as band
  edges from these excerpts and nowhere else"). A SQL question uses the user's own text.
- **writer** — JSON `{answer}` with citations; up to `1 + ORCH_MAX_WRITER_RETRIES` (2) drafts, each
  retry shown the verifier's failures.
- **verifier** — deterministic, below. **redact** keeps only the sentences that verify (or falls back
  to the evidence itself) and marks the answer `partial`. **finalize** sets the status and builds
  the Sources section.
- Stops of one question (its call cap, a failed model call, an unreachable MCP server) set `halt`
  and end the question `failed`. Stops of the whole command (run cap, budget, loop detector,
  paid-call block, ledger failure) propagate; the thread keeps its last checkpoint.

## Spec inference

Deterministic rules first (`graph/spec_rules.py`): NR / 5G / FR1 / `nNN` / 38.101 -> `38.101-1`;
LTE / E-UTRA / 36.101 -> `36.101`. The planner's hint decides only when the rules have no answer.

## Approval triggers

The approval node calls `interrupt()` (nothing before it has side effects) when any of these holds;
it resumes with `{"approve": bool, "spec": "36.101" | "38.101-1" | null}`.

- (a) `ASK=1` (the demo flag): always ask.
- (b) The planned worst-case cost of the rest of the run exceeds `APPROVAL_COST_USD` (default 0.05).
- (c) The remaining budget (`LLM_BUDGET_USD` minus the ledger total) is below twice the planned
  worst case.
- (d) For doc and mixed questions: the deterministic spec rule and the planner's hint conflict.
- (e) For doc and mixed questions: the wording names both NR and LTE (added in the build; the spec
  is then left open for the human to choose).

`approval_mode="auto"` (smoke scripts, tests) approves and records `by: auto`. Planned worst case =
(1 + 2 corrections) SQL generations + (1 + 2 retries) writer calls, each at its input estimate and
full output cap, from `config/llm_prices.yaml`; it is 0 for an unpaid provider.

## Budget and per-question cap

`LLM_BUDGET_USD` is the hard cap (ledger, worst-case reservation). Inside the graph the
per-question call cap is `ORCH_MAX_CALLS_PER_QUESTION` = 10 (planner, SQL agent, corrections, writer
and retries share it). The counter is seeded from `llm_calls` by question id
(`LLMRun.seed_question`) in every LLM node, so a resumed or restarted question cannot exceed the cap
by starting a new process.

## Checkpoint store and roles

LangGraph 1.2.17, `langgraph-checkpoint-postgres` 3.2.0, `durability="sync"`, thread id = question
id. Migration 0012 creates schema `checkpoints` (the library's own `setup()`, run under the owner)
and two roles:

- `rflens_checkpointer` (`CHECKPOINTER_PASSWORD`): DML on the four library tables,
  `search_path = checkpoints`, nothing else. The runtime never calls `setup()`; at start it checks
  `checkpoint_migrations` against the installed library and refuses to start if older.
- `rflens_llm_ledger` (`LLM_LEDGER_PASSWORD`): SELECT / INSERT / UPDATE on `llm_calls` only; the
  gateway ledger connects as this role.

sql-mcp `run_query` returns `query_id` (its `sql_query_log` row) so the answer's `[S1]` source is
the logged query.

## Citation format

`[S1]` = the SQL result (one query per question); `[D1]`, `[D2]`, ... = retrieved chunks in
retrieval order. The writer cites; the **Sources section is built by code** from the tags that
remain in the final text: `[S1] sql-mcp query #<id> (<n> row(s)): <executed SQL>` and
`[D<n>] TS <spec> V<version>, clause <c>, <title>, table <t> (chunk <chunk_id>)`.

## Verifier rules (`graph/verifier.py`)

No LLM, no database. Failure types:

- `dangling_citation`, `missing_citation` (a sentence with a sourced number or a quotation, or an
  answer with evidence but no citation at all).
- `unsupported_number`: a number must be a SQL cell, a number in a cited chunk, a number in the
  question, the executed SQL or the row count, or derived from one SQL column (sum, difference,
  ratio, percentage, share of total). Matching is at the draft's own precision (rounded values
  pass), and thousands separators are read. Derived is a floor, not proof.
- `invented_unit`: a number followed by a unit needs the unit in the question, a cited chunk, a
  column name or comment, or the SQL; a column whose comment says UNKNOWN allows no unit.
- `missing_assumption_wording`: a column whose comment says ASSUMED means the answer must say so.
- `unsupported_quote`: quoted text must occur in a cited chunk, a SQL cell or the question.
- `unsupported_sql_literal` (mixed questions only): every numeric literal that the SQL compares with
  a frequency column in a WHERE clause (a band edge; `>`, `>=`, `<`, `<=`, `=`, `<>`, `BETWEEN`,
  also inside COALESCE) must appear in the text of a **cited** chunk. **Zero is exempt**: it is a
  guard (`> 0`), not an edge. Failure withholds the SQL result (`partial`, SQL dropped). The
  verification records each edge's supplier in `band_edges`: `{column, value, cited_chunk,
  chunk_id}`.

- `edge_side_mismatch` (mixed questions only): if the question or the planner's `sql_question`
  says uplink / UL and the SQL reads only `dl_*` columns of `spec_operating_bands` (or the
  reverse), it fails; if both directions are asked, both sides must appear. Like
  `unsupported_sql_literal` it is a failure of the SQL, so a surviving one withholds the SQL
  result.
- `band_joins` (information, not a failure): the `(spec, band)` literals the SQL filters
  `spec_operating_bands` on.

After `1 + 2` failed drafts, `redact` applies. Statuses: `answered`, `partial`, `refused`,
`failed`, `cancelled`.

## Kill-and-resume demo (step 3, not built)

`make orchestrate Q="<mixed question>" THREAD=demo-1 ASK=1` runs the planner, prints the plan, saves
a checkpoint and waits at the approval prompt. From another terminal, hard-kill that process; show
thread `demo-1` in `checkpoints.checkpoints`. `make orchestrate-resume THREAD=demo-1 DECISION=approve`
in a brand-new process runs the remaining nodes and prints the final answer with Sources.
Evidence: `llm_calls` for that question_id shows two run ids and exactly one planner call. Also
runnable offline with the fake provider.

## End-to-end scoring (not built)

`make eval-e2e` runs all golden questions with approval auto, through the checkpointer.

- **sql:** the cited SQL result matches the reference under the relaxed `compare()`, and the answer
  text contains every reference value (ranked questions in reference order).
- **doc:** a cited chunk hits the expected clause/table and verification passed; evidence recall is
  reported separately.
- **mixed:** doc hit and SQL match, plus `must_mention`.
- **refuse:** status `refused` with a reason, and no SQL or doc search run.

Results by author group are appended to `eval/RESULTS.md`. The first full run needs explicit
approval and has a $0.40 stop.

## Band lookup (structural fix for mixed retrieval)

Why: semantic search over the large operating-band tables does not find the band row (mixed-01: the
expected chunk was not in the top 20 even for the planner's `doc_query`). The band edges are
structured facts, so they are looked up, not searched.

- **Table `spec_operating_bands`** (migration 0013, local database only: 3GPP-derived, never
  committed). Key `(spec, version, band)`; columns `ul_low_mhz, ul_high_mhz, dl_low_mhz,
  dl_high_mhz` (MHz as stated in the spec table; NULL for an N/A side), `duplex_mode` (exactly as
  the spec labels it: FDD for LTE bands 29, 32, 67, 69, 75, 76 although they have no uplink),
  `has_uplink` and `has_downlink` (generated: edges present), `source_chunk_id`, `notes`. All
  columns have comments for `describe_table`. 141 rows: 70 in TS 36.101 V19.6.0 (Table 5.5-1) and 71
  in TS 38.101-1 V19.7.0 (Table 5.2-1).
- **Parsing and loading:** `ingest/spec_bands.py` parses the raw HTML of every piece of the two
  tables from `chunks.jsonl`, strictly (any row that does not parse stops the load; reserved bands
  and the ellipsis row are skipped and reported); `make load-spec-bands` replaces the rows of both
  specs in one transaction (a second load leaves the same rows) and refuses rows whose source
  chunk is not in `chunks`.
- **Footnotes:** the footnote *texts* are absent from the corpus: each table's notes chunk holds
  only its heading and the converted Markdown has no footnote paragraphs after the table. Only the
  markers are stored (in `notes`, with "UL: N/A" / "DL: N/A" and a remark where the source lost a
  dash, as in LTE band 41). The column comment says so.
- **sql-mcp:** `spec_operating_bands` is the tenth allow-listed table; `rflens_sql_reader` has
  SELECT (migration 0013); `sql_guard.ALLOWED_TABLES` lists it.
- **Planner:** for mixed questions it also returns `bands[]`, items `{spec, band}` ("36.101" with
  "5", "38.101-1" with "n5"). The code validates every label against `^n?\d{1,3}$` (the label
  decides the spec, at most 6 bands, duplicates dropped) before it goes into SQL text.
- **Doc agent:** when `bands[]` is non-empty it reads the rows through sql-mcp (one logged
  `run_query`, newest version per band), then `get_chunk` on each distinct `source_chunk_id`; the
  `[D]` citation is the chunk that holds the table row. The SQL agent receives the looked-up rows
  as context (not the whole table text); the writer receives the rows and the chunk text. It falls
  back to `search_docs(doc_query)` only when no band is named. A named band that is not in the table
  is reported to the writer ("Band lookup: X is not in the operating-band tables."); if none of the
  named bands exists the question ends `failed` with that message, with no search.
- **Verifier:** unchanged. `band_edges` shows which cited chunk supplied each SQL literal. It proves
  provenance, not meaning: a literal can come from the right chunk and still be the wrong edge (for
  example a downlink edge used for an uplink question), which only the end-to-end reference
  comparison (`make eval-e2e`) catches.

## Provenance versus meaning

The verifier proves that every number, quotation and band edge in the answer comes from the
evidence. It does not prove the SQL means what the question asks. Two live failures showed the gap
(mixed-01 used the downlink upper edge 894 for an uplink question; mixed-02 dropped "active" and the
whole-block rule), and both passed verification because every number was in the cited chunk. The
causes were fixed where they arose, and the remaining gap is stated:

- **Band edges by join, not literals.** When the doc agent looked bands up, the SQL agent's context
  tells it to join `spec_operating_bands` by `spec` and `band` and use the `ul_*` / `dl_*` columns
  of the side the question asks for, instead of typing numbers. The SQL prompt has the rule and a
  worked example (LTE band 13, not a golden question). Literal edges stay allowed when no band row
  exists; `unsupported_sql_literal` still checks them. Because a join carries no literal, the
  edge's provenance is the band row's `source_chunk_id`, which is the `[D]` citation.
- **`edge_side_mismatch`** (above) catches the wrong side when the join is used.
- **No sub-question loss.** For a mixed question the SQL agent receives the *original* question as
  its question (and `sql_query_log` records it), plus the planner's `sql_question` as context under
  the rule that the original's constraints (status filters, inclusion rules, counts, ranks) take
  precedence over the restatement. The SQL prompt also says to keep every constraint and that a
  frequency record is "inside" a range when its whole assigned block lies within it.
- **`draft_history`** keeps every writer draft and its failures in the checkpoint, so a redacted
  answer can be explained after the fact.
- **What is still not checked:** meaning in general. Only the reference comparison
  (`make eval-e2e`, not built) can say whether an answer is *right*.

## Build differences (flagged)

- Approval triggers: the step 2 build had only (b), (d) and (e), with the threshold named
  `ORCH_APPROVAL_USD`; this revision adds (a) and (c) and names the threshold `APPROVAL_COST_USD`.
- The planner's single call now also returns `doc_query` and `sql_question` (the step 2 build
  searched with the full question text).
- Mixed questions look up named bands in `spec_operating_bands` instead of searching (the step 2b `doc_query`-only retrieval did not find the expected chunks).
- The SQL agent of a mixed question is given the original question plus `sql_question` as context (step 2b used `sql_question` alone, which dropped constraints).
