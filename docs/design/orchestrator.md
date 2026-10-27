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
4. The stats agent reaches the analyses only through analytics-mcp, which runs a fixed menu in a
   Docker sandbox (see "Stats route (week 5)" below and [analytics.md](analytics.md)). The model
   chooses an analysis name and JSON parameters and never writes code, SQL or a path.

## State schema (`graph/state.py`)

JSON types only (checkpoints use strict msgpack; tuples would come back as lists).

| key | meaning |
|---|---|
| `question_id`, `question` | thread id (= question id) and the user's text |
| `route`, `plan` | `sql` / `doc` / `mixed` / `stats` / `refuse`; plan = spec hint, `doc_query`, `sql_question`, `stats_question`, refuse reason, rule matches, planned worst-case USD |
| `spec`, `spec_source` | spec filter (`36.101`, `38.101-1` or none) and where it came from: `rule`, `llm-hint`, `human`, `ambiguous`, `none` |
| `approval` | `{needed, reasons, decision{approve, by}}` |
| `chunks` | retrieved spec chunks `D1..Dn` (id, spec, version, clause, title, table, text, cut) |
| `stats` | stats agent result `A1`: status, analysis, params (validated, defaults filled in), `call_id` (its `analytics_call_log` row), `analysis_id`, result (JSON, at most 64 KB), artifacts (plot ids), dataset (extract SHA-256), runs used, missing-value policy, warnings, notes |
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
                       +--(stats)--> stats_agent --> writer (as above)
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

`[S1]` = the SQL result (one query per question); `[A1]` = the statistical analysis (one per
question); `[D1]`, `[D2]`, ... = retrieved chunks in retrieval order. The writer cites; the **Sources section is built by code** from the tags that
remain in the final text: `[S1] sql-mcp query #<id> (<n> row(s)): <executed SQL>`, `[A1] analytics-mcp call #<id>:
<analysis>(<params>) on SECOM, <n> run(s), extract <sha-prefix>` (plus one line per plot
artifact) and
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
- `frn_name_grouping` (SQL-side, any route): the SQL groups by `licensee_name` together with
  `licensee_frn`, or by `licensee_name` alone while `v_uls_licences` (which exposes
  `licensee_frn`) is queried. One company (one FRN) appears under several name spellings, so such a
  grouping splits it; the schema comments say to group by `licensee_frn` alone and show
  `min(licensee_name)`. GROUP BY ordinals and output aliases are resolved, subqueries and CTEs are
  checked. It is routed like the other SQL-side failures (one SQL retry, then redact).
- `band_joins` (information, not a failure): the `(spec, band)` literals the SQL filters
  `spec_operating_bands` on.

## Retry routing after a failed verification

Which node tries again depends on what failed:

| failures | next | why |
|---|---|---|
| none | finalize | |
| only SQL-side (`edge_side_mismatch`, `unsupported_sql_literal`, `frn_name_grouping`) and the SQL has not been retried | **sql_agent**, once, with the failures and the previous SQL in its context | the writer cannot fix a wrong SQL |
| only SQL-side, and the retry did not fix it | redact (the SQL result is withheld) | another writer pass changes nothing |
| prose (`unsupported_number`, `unsupported_quote`, `invented_unit`, `missing_assumption_wording`, citation failures), or SQL-side mixed with prose | writer, up to `ORCH_MAX_WRITER_RETRIES` (2) | the prose is what can change |

After a SQL retry the new SQL gets a fresh writer chain (`writer_attempts` back to 0, no old draft,
no correction message about the old SQL). `sql_retries` and `sql_feedback` are state keys;
`draft_history` numbers drafts across both chains. The per-question call cap (10) still bounds all
of it: a retry that would exceed it ends the question `failed`.

After `1 + 2` failed drafts, `redact` applies. Statuses: `answered`, `partial`, `refused`,
`failed`, `cancelled`.

## Kill-and-resume demo (built in step 3)

`make orchestrate Q="<mixed question>" THREAD=demo-1 ASK=1` runs the planner, prints the plan, saves
a checkpoint and waits at the approval prompt. From another terminal, hard-kill that process; show
thread `demo-1` in `checkpoints.checkpoints`. `make orchestrate-resume THREAD=demo-1 DECISION=approve`
in a brand-new process runs the remaining nodes and prints the final answer with Sources.
Evidence: `llm_calls` for that question_id shows two run ids and exactly one planner call. Also
runnable offline with the fake provider.

Built as `rflens.graph.cli` (`run`, `resume`, `status`; `make orchestrate`,
`make orchestrate-resume`, `make orchestrate-status`):

- `run` prints the plan (route, spec, bands, `doc_query`, `sql_question`, estimated worst-case cost
  against the remaining budget, and why approval is needed), prints its pid, and waits at
  `approve / edit / cancel`. The checkpoint is written before the prompt (`durability="sync"`).
  Without a terminal (stdin closed) it exits with code 3 and leaves the thread saved. A thread id
  that already has a checkpoint is refused.
- `resume` (`DECISION=approve|cancel`, optional `SPEC=`) opens the saved thread in the new process:
  at the approval interrupt it resumes with the decision, if the process died between nodes it runs
  the next node from the last checkpoint (`invoke(None)`), if the thread is finished it reports it.
  It prints the SQL, the final answer with Sources, the verification result, the comparison with the
  golden reference when the question is a golden question, and calls and cost per node.
- `status` prints the number of checkpoints, the latest checkpoint, the next node(s), the pending
  approval reasons, the plan, and from `llm_calls` the run ids that made calls for the question id
  with calls and cost per node.
- Offline: `LLM_PROVIDER=fake`, `LLM_FALLBACK=` and `RFLENS_FAKE_SCRIPT=<json list of answers>`
  script the model (the MCP servers are still real). `tests/test_orchestrate_cli.py` does exactly
  this with real subprocesses: it kills the process at the prompt (the interpreter that printed its
  pid), resumes in a new one and asserts one planner call, two run ids and a final answer.
- A hard kill must hit the interpreter that printed its pid. On Windows the `.venv` python is a
  launcher in front of the real interpreter; killing only the launcher is not a hard kill.

## End-to-end scoring (built in step 5)

`make eval-e2e` runs all golden questions with approval auto, through the checkpointer.

- **sql:** the cited SQL result matches the reference under the relaxed `compare()`, and the answer
  text contains every reference value (ranked questions in reference order).
- **doc:** a cited chunk hits the expected clause/table and verification passed; evidence recall is
  reported separately.
- **mixed:** doc hit and SQL match, plus `must_mention`.
- **refuse:** status `refused` with a reason, and no SQL or doc search run.

Results by author group are appended to `eval/RESULTS.md` (table "End-to-end runs"). The first full
run needs explicit approval and has a $0.40 stop.

Details of the build (`rflens.eval.eval_e2e`, scoring in `rflens.eval.e2e_score`):

- One thread per question, id = run id + question id; approval automatic; per-question cap 10;
  the run call cap is questions x 10 and fallback providers are off. Before any paid call it prints
  the question count, the worst-case calls (questions x 10) and the worst-case cost (all 10 calls of
  every question at the dearest kind of call), then proceeds. `--stop-usd` (default 0.40): once the
  cost of the run has reached it, the question in progress is finished and the run stops with the
  results so far. A run-level stop (budget, loop, unreachable server) also ends the run there.
- **sql:** the answer cites [S1]; the SQL result matches the reference under the relaxed
  `compare()`; the answer text contains every reference value (ranked: in reference order; a number
  matches a number written at its own precision, thousands separators are read, zero also matches
  the words zero / no / none, NULL needs no text, strings match case-insensitively); `must_mention`
  words are in the answer text.
- **doc:** a cited chunk hits the target (right spec; exact clause or listed table) and the status
  is `answered` with verification passed. **Evidence recall** is separate: the target among the
  search results (`doc_hits` in the state) and among the chunks the writer saw.
- **mixed:** doc part and SQL part, plus `must_mention`. **refuse:** `refused` with a reason and no
  SQL query (`query_id`) and no document search or band lookup in the state.
- Per question it records the final status, whether verification passed, writer and SQL retries (from
  `draft_history`), calls, cost and the failure types seen. A first guess at the cause of each miss
  (retrieval miss, wrong SQL, writer/verifier redaction, wrong route, refusal error, scorer issue)
  is made from the state; a human reads every miss.
- Full per-question results go to `data/eval/eval_e2e_<run>.json` (gitignored).

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

## Project convention: "inside" a range

A frequency record is **inside** a range when its whole assigned block lies within it:
`frequency_assigned >= low AND COALESCE(frequency_upper_band, frequency_assigned) <= high` (for a
band, `low` / `high` are the `ul_*` or `dl_*` columns of `spec_operating_bands`). This is rule 10 of
the SQL agent prompt (with "keep every constraint of the question": status filters such as
"active", counts, ranks) and it is the convention of the golden mixed questions. It is a project
decision, not a fact about the data: an engineer who means "overlaps" must say so, and the agent
states the interpretation in `assumptions`.

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

## Command line, API and UI (step 4)

**`rflens ask "<question>" [--thread ID] [--ask] [--show-drafts] [--json]`** (console script of
`rflens.graph.cli`) runs one question to the end in the calling process and prints the answer with
Sources, the verification summary, and calls and cost per node. When approval is needed it asks at
the prompt (or, without a terminal, exits 3 and leaves the thread saved for
`make orchestrate-resume`). `--show-drafts` prints `draft_history` with each draft's failures;
`make orchestrate` / `orchestrate-resume` print it by themselves when the writer needed more than
one draft. `--json` prints one JSON document (the thread view below) and sends the gateway's run
summary to stderr.

**API** (`rflens.api`, FastAPI, `make api`, 127.0.0.1 only; the host is not configurable):

| route | |
|---|---|
| `POST /ask` `{question, ask, thread_id?}` | runs the question; returns the thread view: the final answer, or `pending` (the approval payload) |
| `POST /threads/{id}/resume` `{decision: approve\|cancel\|edit, spec?}` | `edit` is approve with a changed spec filter and needs `spec` (`36.101`, `38.101-1`, `none`); `spec` with another decision is refused; unknown fields are refused |
| `GET /threads/{id}` | status, badge, plan, answer, sources (text and structured items), verification, `draft_history`, cost per node, ledger run ids |
| `GET /threads` | the recent threads, newest first |
| `GET /chunks/{chunk_id}` | one spec chunk through docs-mcp (the UI's source panel); the id is validated against the chunk id format |
| `GET /health` | open |

Every route but `/health` needs `X-API-Key` equal to `API_KEY` in `.env` (constant-time compare; a
server without a key answers 503 to everything protected) and counts against a sliding 60 s window
of `API_RATE_LIMIT_PER_MINUTE` (30) per key (429 with `Retry-After`; wrong keys never reach the
limiter). Question length is capped at `API_MAX_QUESTION_CHARS` (2000); thread ids match
`[A-Za-z0-9._-]{1,64}`. 404 unknown thread or chunk, 409 thread exists / not waiting, 422 invalid
input, 503 for a model-side stop (budget, caps, loop, paid-call block: the thread keeps its last
checkpoint) or an unreachable data service. Each request opens its own gateway run (its own run id),
checkpointer connection and graph, and closes them again, so a request that dies leaves a resumable
thread. The JSON view of a thread is `graph/view.py` (also used by `rflens ask --json`).

**UI** (`rflens.ui`, Streamlit, `make ui`, 127.0.0.1:8502) calls only the API (with the key): question
box and an "ask for approval" box; when approval is pending, a plan panel (reasons, route, spec,
bands, `doc_query`, `sql_question`, estimated cost) with approve / edit (spec) / cancel; the answer with
a verification badge (verified / partial / refused / failed / cancelled / pending approval), sources
as expanders (the SQL text and rows for [S1]; the clause and text of the chunk, read through the API
from docs-mcp, for [D1]), the verification details and drafts, calls and cost per node; the sidebar
lists recent threads that can be reopened and resumed. **Why Streamlit:** it installs and imports
(with pyarrow) under this machine's Application Control policy (checked in a throwaway venv before
it was added; streamlit 1.66.1, pyarrow 26.0.0), so no hand-written page was needed. It sends no
usage statistics (`.streamlit/config.toml`).

## Writer failures and loop scope (after the first end-to-end run)

- **`writer_truncated`:** a writer call whose output is not a complete JSON answer (a MAX_TOKENS
  finish, or unparseable output) is a verification failure of type `writer_truncated`, not an empty
  answer. The writer's output allowance is `ORCH_WRITER_MAX_TOKENS` = 1500.
- **Retry prompts never repeat:** every writer retry sends the previous output (or a placeholder
  when there was none) and a feedback message that names the retry number and the failures; for
  `writer_truncated` it says to shorten the answer (summarise long lists, keep the citations).
- **Raw output:** `draft_history` entries carry `raw`, the writer's raw output cut to 2000 characters.
- **Loop scope:** the repeated-prompt and repeated-error detectors count per question when the call
  names one. A loop within one question fails that question (status `failed`, failure type
  `loop_detected` in its `draft_history` / verification) and an end-to-end run continues with the
  next question. The run call cap, the budget and the paid-call block still stop the whole run.

## Improvement round after the first full baseline (2026-10-26)

**Temperature.** Until now the gateway sent no `temperature`, `topP` or `topK` to Gemini: its
`generationConfig` held only `maxOutputTokens`, `thinkingConfig` (level `minimal`) and the JSON
schema fields, so the API defaults applied and two runs of the same question could differ.
`LLM_TEMPERATURE` (unset: not sent) now sets it for Gemini and Ollama; the `eval-*` commands use
0 unless `LLM_TEMPERATURE` is set. `eval-sql` also has `--only`, `--repeat` and `--stop-usd`, and
prints how many answers differ between repeated runs.

**Clause lookup** (`CLAUSE_LOOKUP`, default on). A question that says "clause 6.6.2.1" or "Table
7.3.1-1" gets those chunks (the first 3 of each, in document order) in front of the search results,
in the one spec when it is known and in both otherwise; results carry `via: lookup`. Only a number
after clause / subclause / section or Table counts. docs-mcp does it inside `search_docs`; the doc
agent keeps a named clause in its query when the planner's rewrite dropped it and passes the
question as `context`.

**Variant-clause rule** (`VARIANT_DOWNRANK`, default on). Clause numbers with a letter in them (5.5H,
6.2.4A.7, 7.5.1H, 6.3.2F, 6.2.3E) belong to variants of a requirement and repeat the wording of the
base clause, so in a vector search they displace it: 75% of the TS 38.101-1 chunks and 26% of the
TS 36.101 chunks belong to a variant clause. In the top 50 candidates they are moved behind the
other clauses unless the question mentions the topic of every letter in the number. What each letter
stands for, read off the clause titles of the loaded specs (annex clauses such as A.3.9.2 are not
variants):

| letter | TS 36.101 (LTE) | TS 38.101-1 (NR FR1) |
|---|---|---|
| A | carrier aggregation (CA_NS_xx) | carrier aggregation |
| B | UL-MIMO; spatial-multiplexing performance | NR-DC (dual connectivity) |
| C | spatial multiplexing (2 Tx / 4Rx) | SUL (supplementary uplink) |
| D | ProSe (sidelink) | UL MIMO |
| E | category M1 / M2 (and category 0, 1bis) | V2X and sidelink |
| F | NB-IoT (category NB1 / NB2) | "additional" requirements (NS_xx regions) |
| G | V2X | Tx diversity |
| H | LTE based 5G terrestrial broadcast | inter-band UL CA with UL MIMO |
| I, J | spatial-multiplexing performance | (e)RedCap (I); ATG, air-to-ground (J) |
| K | aerial UE | aerial UE (UAV) |
| L, M | - | UL CA with Tx diversity (L); LP-WUS / WUR (M) |

The keyword lists are in `retrieval/variants.py` (`TOPICS`). F in NR is the least certain reading.

**Writer depth.** The writer gets the top 5 chunks (`ORCH_DOC_FULL_CHUNKS` 5, was 3;
`ORCH_EVIDENCE_CHARS` 18000, was 9000).

**NULL keys** (migration 0014 and SQL prompt rule 11). The comments of `fcc_grantees.country`,
`fcc_test_firms.country` and `v_uls_licences.licensee_frn` say that rankings and counts of
entities leave out NULL keys unless the question asks about missing values.

**`missing_band_citation`** (verifier, a prose failure sent to the writer): when the doc agent
supplied band rows, the answer must cite at least one of the chunks that hold them.

## Stats route (week 5)

Design and sandbox: [analytics.md](analytics.md). In the graph:

- **planner** has a fourth source, `stats`: descriptive analyses of the SECOM measurements from the
  fixed menu (which features differ between pass and fail, distributions, correlations, outliers,
  missing values, plots). It returns `stats_question` (the question restated for the analysis
  step). Training a model, predicting or classifying unseen data, causes, custom code or formulas
  are `refuse` at the planner; the stats agent can refuse too.
- **stats_agent** — one LLM call (`purpose=stats-agent`, temperature as configured) returns
  `{refuse, refuse_reason, analysis, params_json}`; `params_json` is a string because a free-form
  object is not a reliable structured-output type. analytics-mcp validates the parameters again;
  a rejection (invalid parameters, or an analysis the data cannot answer such as too few values)
  goes back to the model once, so at most 2 calls. A sandbox failure (timeout, docker error) is
  not corrected: the question ends `failed`. The planner's `stats_question` is appended to the
  question text the agent sees; `sql_query_log`-style traceability is `analytics_call_log`, which
  records the original question.
- **writer** gets `[A1]` as evidence: analysis, parameters, runs used, missing-value policy,
  warnings, the result JSON (cut at 7,000 characters) and plot artifact ids. Its prompt says to
  state the method, the numbers of values behind each statistic, effect sizes and p-values from
  the evidence, never to use cause words, and to write feature values without a unit.
- **verifier**: `[A1]` is a number source (see "Statistics rules" below). The three new failures
  are prose failures, so they go to the writer (up to 2 retries) and then `redact`; there is no
  second stats-agent attempt (a build difference from the first design).
- **approval**: the planned worst case of a stats question is 2 stats-agent calls plus the writer
  calls. `finalize`: a stats-agent refusal ends the question `refused`, a failed analysis
  `failed` (no writer call).
- `GraphDeps.analytics_tools` (an `AnalyticsMcpTools`) is `None` unless the caller supplies it; a
  stats question then stops with "analytics-mcp is not configured". The CLI, the API service and
  `make eval-e2e` supply it (`ANALYTICS_MCP_URL`, default `http://127.0.0.1:8103/mcp`).
- API: `GET /artifacts/{id}` serves a plot by its analysis id (64 hex characters, validated; never
  a path), behind the API key. The UI shows `[A1]` with its parameters, the result and the plot.

### Statistics rules (verifier)

- **Numbers.** A number in the answer must be a numeric leaf of the cited result or of its
  parameters, a run count, a share of one written as a percentage, a bound such as "p < 0.001"
  that a p- or q-value in the result satisfies (0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001,
  0.000001), or a sum, difference, ratio or share of leaves of one result row (or of the same key
  across sibling groups, such as the pass and fail medians). p-values, q-values and alpha are never
  combined, and combinations below 0.01 are dropped, because such tiny numbers would make almost
  any small number the writer invents look derived. Matching is at the draft's own precision and
  scientific notation is read (`3.4e-10` has 11 decimals). It is a floor, not proof.
- `unsupported_stat_claim`: "significant" needs a p- or q-value below 0.05 in the result and the
  p- or q-value (or its number) in the answer; "not significant" needs one at or above 0.05.
- `causal_claim`: cause, causes, caused by, leads to, due to, results in, because of, drives,
  responsible for, unless the sentence is negated. SECOM is observational.
- `missing_n`: a stated p- or q-value needs one of the numbers of values behind the result in the
  answer. These are the sample sizes of the result (`n`, `n_pass`, `n_fail`, `n_pairs`) and the
  group sizes added up; the runs of the extract count only when the result has no sample size.
- `wrong_n` (prose failure, to the writer): an n stated in a sentence about a test (`n=104`,
  `n of 1,456`, `1456 passing`, `104 failing values`, `1567 runs`) must be one of those sample
  sizes or their total. Found live: stats-01 said "Mann-Whitney U test (n=1567 ...)", the total
  number of runs, while the tests rested on 1,456 or 1,461 passing and 104 failing values.
- `invented_unit`: a value cited only to `[A1]` carries no unit unless the question has it.
- Two fixes made while building this, both general: the float slack of the rounding match is
  relative (a fixed 1e-9 made every tiny number match another), and a SQL row count of zero is
  only a known number when a SQL result exists (without one, `0` matched every tiny value).

### Fixes after the first full run with the stats route (eval-e2e-aecd54a7, 41/51 like-for-like)

- **Writer prompt.** doc-03, doc-08 and doc-13 ended `partial` on `missing_citation`. Every failing
  sentence ended in a combined bracket (`[D1, D2]`, `[D3, D5]`), which the verifier does not read
  as a citation (it never did). The verifier was ruled out: the verifier of the previous run and
  today's give the same verdict on every draft of both runs. doc-13 had the same question, the
  same `doc_query` and the same five chunks in both runs; only the writer system prompt differed,
  and its draft changed from `[D1], [D2]` to `[D1, D2]`. So the prompt change (the statistics
  rules written into the citation rules, and `[A1]` in the tag list) caused doc-13; doc-03 and
  doc-08 also had a slightly different `doc_query` (the planner prompt changed too), so their
  evidence differed. Fix: the statistics rules are no longer part of `WRITER_SYSTEM`; they are
  appended (`STATS_WRITER_RULES`, `writer_system(stats)`) only when the evidence holds an analysis,
  so the prompt of every other route is the previous one plus one sentence that spells out the
  failure: one tag per bracket, never `[D1, D2]`. The wrong_n guidance stays in the statistics
  rules. The verifier was not changed.
- **Routing boundary.** sql-03 and sql-04 (missing-value count, average) went to `stats`. The
  planner prompt now says that simple aggregates over any table (counts, sums, averages, minimum /
  maximum, top-N by a count or a value, missing-value counts) are `sql`, and that statistical
  tests, effect sizes, association rankings, correlations, distributions with bins or quartiles,
  histograms, outliers and plots are `stats`, with six examples (none a golden question;
  `PLANNER_EXAMPLES`). `graph/route_rules.py` settles the boundary deterministically when the
  rules are decisive, like the spec rules: only for a SECOM question, only between `sql` and
  `stats`, a statistics word beats an aggregate word, and the plan records `route_rule` when it
  overrode the model.
- **stats-03** no longer requires n in the answer text (the question does not ask for it); n is
  still checked in the result. **stats-02** likewise no longer requires the group sizes (it asks
  for medians and the effect size).

### Multi-query retrieval (after eval-e2e-94d99ed1)

The planner's `doc_query` wording decides retrieval, and prompt changes move it (doc-04 and doc-14
missed when the routing fix reworded their queries). The doc agent now searches with the planner's
`doc_query` AND the user's question (`search_docs(query=doc_query, also_query=question,
context=question)`, spec filter when known): each query is searched in the same mode and the two
ranked lists are merged by reciprocal rank fusion (k=60) before the variant-clause rule and the cut
to the top 5. Clause and table lookups (named in either text) stay first. `ORCH_MULTI_QUERY` (default
on) switches it off. Measured free with `make eval-docs` / `python -m rflens.eval.eval_docs
--queries-from RUN_ID`, with the planner's real queries of three saved full runs, vector mode: the
merged search found as many as the better single query in both author groups in every case, and
never more (recall@5 hits, human 10 / assistant 12 doc questions; spec as the agent had it):

| planner queries from | doc_query only | question only | both merged |
|---|---|---|---|
| eval-e2e-94d99ed1 (latest) | 9/10, 9/12 | 9/10, 8/12 | 9/10, 9/12 |
| eval-e2e-aecd54a7 | 10/10, 10/12 | 9/10, 8/12 | 10/10, 10/12 |
| eval-e2e-559aacfa (44/51) | 10/10, 9/12 | 9/10, 8/12 | 10/10, 9/12 |

With no spec filter the picture is the same (merged equals the better single query). The merge
protects against a poorer single query but cannot rescue a question where both miss: doc-04 and
doc-14 miss with the planner's latest wording and with the question alone (and hit with the
earlier wording, where the merged search hits too).

## Build differences (flagged)

- Stats route (week 5): no second stats-agent attempt after a verifier failure (the first design
  had one); the distribution analysis also returns n, missing count and quartiles, so one call
  answers "distribution, quartiles and missing values"; `params_json` is a string.

- Approval triggers: the step 2 build had only (b), (d) and (e), with the threshold named
  `ORCH_APPROVAL_USD`; this revision adds (a) and (c) and names the threshold `APPROVAL_COST_USD`.
- The planner's single call now also returns `doc_query` and `sql_question` (the step 2 build
  searched with the full question text).
- Mixed questions look up named bands in `spec_operating_bands` instead of searching (the step 2b `doc_query`-only retrieval did not find the expected chunks).
- The SQL agent of a mixed question is given the original question plus `sql_question` as context (step 2b used `sql_question` alone, which dropped constraints).
- Retry routing: SQL-side failures go back to the SQL agent once instead of the writer (step 3); the writer retries remain for prose failures.
- `frn_name_grouping` is a fourth verifier rule on the SQL side (step 4); the CLI, API and UI were added (step 4).
