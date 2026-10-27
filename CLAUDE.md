# RFLens

RFLens is an agentic data platform that lets a wireless engineer ask plain-English questions across real RF compliance data and test reports, and get answers that cite the SQL rows and document passages they came from. LangGraph agents call three MCP servers (sql-mcp, docs-mcp, analytics-mcp) over Postgres with pgvector. Data comes from FCC equipment authorizations, 3GPP specs, spectrum licences, and signal and manufacturing-yield datasets. It is a public-data analogue of the Smart Data Ecosystem problem and uses no Apple data. Full blueprint: docs/blueprint.pdf.

## Stack

Python 3.12, uv for dependencies, FastAPI, Postgres 16 + pgvector, SQLAlchemy 2 + Alembic, pytest, ruff.

## Repository and CI

The repo has a GitHub remote (`origin`, https://github.com/iamrohitkumaryadav9/rflens.git) and
CI (`.github/workflows/ci.yml`) is green on `main`. Treat CI as working; do not report it as
unrun. The user reviews and commits; do not commit unless asked.

## Tests and the test database

Tests run on their own database, `rflens_test`, in the same Postgres server (`TEST_DATABASE_URL` in
`.env`, `.env.example` and CI); they never touch the development database. `make test` first runs
`python -m rflens.testdb` (creates the database if missing, `alembic upgrade head`; idempotent),
then pytest. `tests/conftest.py` replaces `DATABASE_URL` with `TEST_DATABASE_URL` and refuses to run
unless the connected database name ends with `_test`. The roles from migrations 0007 and 0008 are
cluster-wide and keep their passwords; their grants are applied per database by the migrations, so
they work in the test database too. CI has only `TEST_DATABASE_URL`, no `DATABASE_URL`.

Tests must not assume CI is unset or set; pin environment variables with monkeypatch when behaviour
depends on them. (GitHub Actions always sets `CI=true`; ci #10 to #13 failed because one test
assumed it was unset and expected the "pytest" block reason where CI reports its own.)

`make ci-local` is the check to run before pushing changes to tests, the workflow, migrations or
settings. It runs the CI job in Docker (`src/rflens/ci_local.py`): a fresh pgvector Postgres from
the workflow's `services:` block, a clean copy of the working tree (tracked and untracked files git
does not ignore, never `.env`, `data/` or `.venv`), the workflow's `env:` plus `CI=true` and
`GITHUB_ACTIONS=true`, no Ollama, no sql-mcp or docs-mcp, no API keys, and the workflow's own `run:`
steps in order, stopping at the first failure. It reads `ci.yml` itself, so it cannot drift from CI.
Limits: it does not run `uses:` steps (the copy replaces checkout, uv comes from pip), and the runner
is `python:3.12-bookworm` (Debian), not GitHub's Ubuntu image. `--plan` prints what it would run
without Docker. Workflow actions must be on Node 24 majors (checkout v7, setup-uv v10.3.2: setup-uv
has no floating major tag since v8, so it is pinned to the exact release).

## Repo layout

The repo root is the project root (no nested rflens/ folder).

```
src/rflens/__init__.py
src/rflens/config.py      # pydantic-settings, reads .env
src/rflens/db.py          # SQLAlchemy engine + session; analyze() (loaders call it after commit)
src/rflens/testdb.py      # create + migrate the test database; the _test name guard
src/rflens/ci_local.py    # make ci-local: the CI job in Docker (see Tests)
src/rflens/agents/        # sql_agent.py (+ sql_prompt.py, sql_tools.py: MCP client)
src/rflens/eval/          # golden.py, scorer.py, eval_sql.py, eval_docs.py, results_log.py
src/rflens/llm/           # LLM gateway (see Conventions); config/llm_prices.yaml; make llm-spend
scripts/llm_smoke.py      # live gateway check (needs LLM_ALLOW_PAID=true)
src/rflens/api/main.py    # FastAPI app, GET /health
src/rflens/ingest/        # secom.py (UCI SECOM), uls.py (FCC ULS, one service per run), fcc_eas.py (FCC grantees, test firms),
                          # spec_fetch.py + spec_chunker.py (3GPP specs -> data/interim/3gpp/<spec>/chunks.jsonl)
src/rflens/retrieval/     # embedder.py (Ollama), index.py (load + embed chunks), hybrid.py
                          # (vector + keyword + RRF), embed_3gpp.py, eval_smoke.py
eval/                     # golden.yaml (all question types), RESULTS.md (eval run log + golden counts)
src/rflens/api/           # FastAPI: main.py (routes), service.py (orchestrator per request), security.py, server.py
src/rflens/ui/            # Streamlit UI on the API (client.py, app.py)
src/rflens/graph/         # orchestrator (LangGraph): state, planner/approval/doc/sql/stats/writer nodes, verifier
src/rflens/analytics/     # menu.py (parameter models of the fixed analyses), extract.py, sandbox.py (Docker), reference.py
src/rflens/mcp/           # docs_server.py: docs-mcp (search_docs, get_chunk), make docs-mcp, port 8101
                          # sql_server.py + sql_guard.py: sql-mcp (list_tables, describe_table,
                          # run_query), make sql-mcp, port 8102
                          # analytics_server.py: analytics-mcp (fixed menu of analyses), make analytics-mcp, port 8103
analytics_runner/         # the sandbox image: Dockerfile, analyses.py, runner.py, reference.py, tests/ (run in the image)
tests_docker/             # container tests (make test-analytics); not in `make test`
artifacts/                # gitignored; plots and result JSON of analyses, by analysis id
scripts/mcp_smoke.py      # official MCP client over HTTP against a running docs-mcp
scripts/sql_mcp_smoke.py  # same, against a running sql-mcp (needs the ULS cell data loaded)
migrations/               # Alembic
tests/                    # tests/fixtures/uls: small real ULS subset, committed
docs/blueprint.pdf
data/                     # gitignored; raw downloads (data/raw/secom, data/raw/uls/<service>)
```

## Conventions

- `data_sources.row_count` is the count of the source's primary entity: runs for `uci_secom`,
  HD rows (`uls_hd` rows: licences and spectrum leases, see open question j) for `uls_<service>`. Per-table counts live elsewhere (for ULS,
  `uls_file_counts`, which also records the zip's `sha256` and `snapshot_date`).
- Every loader (secom, uls, fcc_eas) runs `ANALYZE` on the tables it loaded at the end of the same
  run, after its transaction commits (`db.analyze`). Stale planner statistics skewed estimates
  (uls_fr was estimated at 178,449 rows against 236,958 actual), and sql-mcp's cost gate reads
  those estimates.

- 3GPP specs are copyrighted (3GPP terms of use: no reproduction or distribution without ETSI's
  consent). Raw files, converted Markdown and chunks live only under gitignored `data/` (and later
  in the local database); nothing from 3GPP is committed. Test fixtures for the spec parser are
  synthetic, hand-written Markdown. Golden-set questions cite clause numbers, never quoted spec
  text. Specs in use: TS 36.101 19.6.0 and TS 38.101-1 19.7.0 (Release 19); no FR2 for now.
- Spec chunk sizes are config (`ChunkSettings` in `config.py`) and apply to the embedding `text`;
  `raw_text` keeps the original Markdown or HTML. Env: `CHUNK_SOFT_CHARS` 3000,
  `CHUNK_HARD_CHARS` 6000, `CHUNK_TABLE_HARD_CHARS` 3600, `CHUNK_MAX_TOKENS` 1840. To be tuned
  later against recall@5.
- **Token rule (measured, not guessed).** nomic-embed-text has a hard 2,048-token context under
  Ollama (`num_ctx` is clamped). Characters alone cannot respect it: a character cap calibrated on
  clause 6 tables (about 2.4 chars/token) failed on carrier-aggregation tables. Measured with
  Ollama on 170 table chunks (85 densest by the estimator, 85 random, all over 2,800 chars):
  chars/token min 1.28, 5th percentile 1.51, median 1.81 (dense group median 1.65, random group
  2.28); 66 of 170 overflowed (up to about 2,750 tokens) and 94 were over 1,840. Dense tables are
  band-combination and CA tables, mainly clauses 6.2.5, 7.3.1, 5.5A.3.x, 5.5B, 6.2A.4.2.x and
  5.2A.2.x. Rule: `estimate_tokens()` in `ingest/spec_chunker.py` (letter runs and digit runs
  count as word pieces, each punctuation mark as one token, times `TOKEN_SCALE` 1.20) and no
  chunk may estimate above `CHUNK_MAX_TOKENS`; tables are split by row groups on that estimate,
  in addition to the character caps. The raw estimate is within 0.87 to 1.145 of the real count;
  1.20 made the estimate at or above the real count for all 170 calibration chunks (est/real 1.05
  to 1.38) and for 103 later re-checked chunks, and the largest of those re-checked chunks is
  1,638 real tokens. A lower character cap alone was rejected: it needed 2,000 chars to reach zero
  estimated overflows, adding 1,030 chunks (+15%) with 32 mid-row forced splits, against +130
  chunks (+1.9%) and no forced splits for the token rule. If the model, prefix or tokenizer
  changes, re-measure and recalibrate `TOKEN_SCALE`.
- The embedder stores each chunk's real token count (`prompt_eval_count`) in
  `chunks.metadata.tokens` and prints the max and the number of chunks above 1,840 after every
  embed run, so the full embed doubles as the complete measurement. A chunk embedded without a
  token count is embedded again. Any chunk above 1,840 real tokens means recalibrate.
- Embeddings: Ollama's HTTP API (`nomic-embed-text`, 768 dimensions), never torch. Chunks are sent
  with the `search_document: ` prefix and queries with `search_query: ` (both defined once in
  `retrieval/embedder.py`); every request sends `truncate=false`, so an over-long input stops the
  run naming the chunk instead of being cut silently. `chunks.embedding_model` holds tag plus
  digest (e.g. `nomic-embed-text:latest@0a109f422b47`); a chunk embedded by a different model
  counts as not embedded. The full embed (~3 h on this CPU-only machine) is run by the user
  from their own terminal: `make embed-3gpp` (resumable; `SPEC=`, `CLAUSE=`, `LIMIT=` for subsets).
- Full-text search: `chunks.tsv` is generated from `rflens_fts_norm(text)` with the `english`
  config; queries must use the same function. `rflens_fts_norm` splits glued units ("23dBm",
  "dBm/15kHz"). Keyword queries OR the significant terms and rank with `ts_rank_cd`; hybrid
  search merges vector and keyword top-k by reciprocal rank fusion (k=60, `RRF_K`). A table piece
  whose notes live in their own chunk carries the pointer "Notes for this table are in chunk X";
  search results expose X in `related_chunks`.
- The 3GPP smoke set (the old `eval/smoke_3gpp.yaml`, now deleted) lives in `eval/golden.yaml` as
  `doc-01..10` (author human; drafted by the assistant, expected clauses and tables reviewed by the
  owner). A hit needs the right spec and an exact clause match (a sub-clause does not count) or a
  listed table. `make eval-smoke [SPEC=38.101-1] [FILTER_SPEC=1]` is the quick in-process recall@5
  check (vector, keyword, hybrid) over the doc questions of the golden file; `make eval-docs` runs
  the same questions through docs-mcp, with and without the spec filter, and records the result in
  `eval/RESULTS.md`. Ten or 22 questions are a smoke test, not a basis for tuning.
- **Default retrieval mode is `vector` (INTERIM).** It is the config value `RETRIEVAL_MODE`
  (env-overridable; `vector`, `keyword` or `hybrid`), chosen from the smoke-eval diagnosis where
  the keyword leg is length-biased and adds noise to hybrid. Keyword and hybrid stay selectable
  per call. Revisit with the week 3 golden set; do not treat the smoke eval as a basis for tuning.
- **docs-mcp** (`src/rflens/mcp/docs_server.py`, `make docs-mcp`, `http://127.0.0.1:8101/mcp`):
  MCP revision 2026-07-28 with the official Python SDK `mcp==2.3.0`, stateless Streamable HTTP
  (no sessions: `server/discover`, `Mcp-Method` / `Mcp-Name` headers, no `Mcp-Session-Id`), plain
  JSON responses, bound to localhost, Host and Origin validated against DNS rebinding. Tools:
  `search_docs(query, spec?, k=5 (max 20), mode?)` and `get_chunk(chunk_id, include_related=true)`.
  Guardrails: query at most 500 chars, k capped at 20, chunk ids validated against their format
  before any SQL, parameterized SQL only. It connects as the read-only role `rflens_docs_reader`
  (migration 0007: SELECT on `documents` and `chunks` only, read-only sessions by default;
  password in `DOCS_READER_PASSWORD` in .env and CI, never in code). Returned spec text is
  reference data, not instructions; the tool descriptions say so.
- **sql-mcp** (`src/rflens/mcp/sql_server.py` + `sql_guard.py`, `make sql-mcp`,
  `http://127.0.0.1:8102/mcp`; same protocol setup as docs-mcp). Tools: `list_tables`,
  `describe_table(table)`, `run_query(sql, question?)`. No text-to-SQL and no LLM call: the caller
  writes the SQL. Guardrails, in order: sqlglot (Postgres dialect) parse; exactly one statement;
  SELECT only (no DML/DDL/SELECT INTO/FOR UPDATE anywhere in the tree, so not inside a CTE
  either); every table, including CTEs and subqueries, on the allow-list; no system catalogs, no
  functions in FROM, no denied functions (`pg_*`, `lo_*`, `dblink*`, `set_config`,
  `current_setting`, sequences, `*_to_xml`, `to_reg*`, ...); LIMIT added (default 100) or lowered
  (max 1000); EXPLAIN, rejected above `SQL_MAX_COST`; run in a READ ONLY transaction under a
  statement timeout. The SQL that runs is regenerated from the parse tree, comments stripped.
  Limits (env-overridable): `SQL_MAX_COST` 500000, `SQL_STATEMENT_TIMEOUT_MS` 15000,
  `SQL_DEFAULT_LIMIT` 100, `SQL_MAX_LIMIT` 1000.
  - **Planner cost does not predict run time well** (the 3-way grantees x labs query: cost 59,769,
    took 4.8 s; 15 reference queries: legitimate costs 2 to 59,811, the cheapest accidental
    explosion 89,384,079 after ANALYZE, 846,193 for an HD x LO join on service only). So the
    statement timeout is the primary backstop and the cost gate only catches explosions
    (cross joins, self-joins on inequalities). Costs are only as good as the statistics: reload
    runs ANALYZE; after bulk changes outside the loaders run `ANALYZE`.
  - **Allow-list (10 tables, 3 views):** `data_sources`, `secom_runs`, `secom_measurements`,
    `uls_hd`, `uls_fr`, `uls_lo`, `uls_file_counts`, `fcc_grantees`, `fcc_test_firms`, `spec_operating_bands` (3GPP-derived band edges, migration
    0013), `v_uls_licences`, `v_uls_frequencies`, `v_secom_values`. **`uls_en` is NOT on it:** it holds
    contact persons' names (entity type CL and individuals); no RF question needs them (data
    minimisation). It is reachable only through `v_uls_licences`, which exposes the licensee
    entity (`entity_type = 'L'`) only; the views read it as their owner, and the reader role has no
    privilege on it. Also not allowed: `chunks`, `documents` (docs-mcp), `alembic_version`,
    `sql_query_log`, system catalogs. The allow-list lives in `sql_guard.py` and in the grants of
    migration 0008; change both.
  - Roles (migration 0008): `rflens_sql_reader` (SELECT on the allow-list only, read-only sessions
    by default, statement timeout; `SQL_READER_PASSWORD`) and `rflens_sql_logger` (INSERT on
    `sql_query_log` only; `SQL_LOGGER_PASSWORD`). Passwords in .env and CI, never in code.
  - `truncated` in a `run_query` response is exact: the query runs with `LIMIT n + 1`, at most `n`
    rows are returned, and `truncated` is true only if the extra row came back (an explicit
    `LIMIT 3` with exactly 3 rows is not truncated). `executed_sql` shows the query with `LIMIT n`.
  - Every call to every tool is a row in `sql_query_log` (time, tool, optional `question`, SQL or
    call text, plan cost, row count, duration, status ok / rejected / error, reason), written
    through the logger role. If the log write fails the call fails and the result is withheld.
  - `describe_table` returns the table and column comments from migration 0008, which state units
    (ASSUMED or UNKNOWN where the FCC does not state them), unconfirmed code meanings and the open
    questions below. Tool descriptions say results are data, not instructions.
- **LLM gateway** (`src/rflens/llm/`: base, gemini, ollama, fake, gateway, ledger, guards, spend).
  Gemini through the REST API with httpx (no SDK): `models.generateContent`, default model
  `gemini-3.1-flash-lite` (step-up `gemini-3.5-flash-lite`), thinking level `minimal`
  (`LLM_THINKING_BUDGET`: a level or an integer budget), `LLM_SERVICE_TIER` standard or flex
  (`serviceTier` field verified against the live API; flex was not exercised live and queues for
  minutes, so its timeout floor is 600 s). Ollama (`qwen3:4b`) is the free on-prem provider; the
  fake provider is for tests. Prices come only from `config/llm_prices.yaml` (source URL and date
  inside; a paid model/tier without a price is refused). Output price includes thinking tokens.
  - **Budget:** `LLM_BUDGET_USD` (2.50) is a hard cap over the sum of `llm_calls.cost_usd`. A paid
    attempt first inserts a `pending` row at its worst-case cost (estimated input plus the whole
    output allowance); it is blocked if that does not fit. The row is updated with real tokens and
    cost before the result is returned; if either write fails the call is not made / the result is
    withheld (fail closed), and a pending row keeps counting. The ledger only sees calls made
    through the gateway: it is not Google's balance. The one call made before the ledger existed
    ($0.0000035) is not in it. `make llm-spend` prints total, remaining, today, last 10, top 5.
  - **Paid calls are opt-in and never happen in tests or CI:** they need `LLM_ALLOW_PAID=true`, and
    the check at the single place a Gemini request leaves the process refuses when `CI` is set or
    pytest is loaded, whatever the settings say. Tests use the fake provider and
    `httpx.MockTransport` (which cannot reach the network).
  - **Loop protection** (each stop names the run_id): `LLM_MAX_CALLS_PER_RUN` 30 and
    `LLM_MAX_CALLS_PER_QUESTION` 4 count attempts, so retries and fallbacks count (Ollama too);
    retries only on timeout / 429 / 5xx, at most `LLM_MAX_RETRIES` (2), exponential backoff, never
    on 400/401/403/404; the third identical prompt in a run, or the same error on three calls in a
    row (counted per call, not per retry), stops the run; `LLM_MAX_CALLS_PER_MINUTE` 20 across the
    process (it sleeps); fallback at most once per call, only after a retryable failure, never back
    to the primary, and not on a budget block.
  - Every command that makes calls uses `gateway.run(name)` and prints a summary at the end.
    `scripts/llm_smoke.py` is the live check (3 paid Gemini calls + 1 Ollama call, retries and
    fallback off).
- **SQL agent** (`src/rflens/agents/sql_agent.py`, a plain function `answer_question`; week 4 wraps
  it as a LangGraph node). Agents never connect to Postgres: it reaches data only through sql-mcp
  over MCP (`sql_tools.py`; URL in `SQL_MCP_URL`, default `http://127.0.0.1:8102/mcp`). Schema
  context (list_tables + describe_table, with the column comments) is built once per run. The
  gateway returns JSON `{refuse, refuse_reason, sql, tables_used, assumptions}`; sql-mcp rejections
  and Postgres errors go back to the model, at most 2 corrections (so at most 3 generations); the
  gateway's per-question cap (4) is the hard limit and ends that question as `failed`, while run
  cap, budget, loop detector and ledger failures stop the command. The question text is passed to
  `run_query`, so `sql_query_log` links SQL to questions. Prompt rules (sql_prompt.py): rankings
  return the entity and its measure and break ties by key ascending; never guess units, report
  ASSUMED/UNKNOWN ones in notes; refuse data changes, personal contact data and out-of-scope
  questions. Seven worked examples, none from the golden set (a test enforces it).
- **Golden set** `eval/golden.yaml`, all question types, 55 questions (`eval/RESULTS.md` has the
  counts per type and author, regenerated by `make results-header`; a test fails when stale):
  `sql-01..19`, `doc-01..22`, `mixed-01, 02, 04, 05` (mixed-03 was dropped in review),
  `stats-01..04` (assistant, reviewed: ranking of the two
  features most associated with failing runs, a pass/fail comparison, a distribution, and a
  refusal; `reference_analysis` + `reference_stats` come from `analytics_runner/reference.py`,
  independent of the analysis code, `make golden-stats-refresh`), `refuse-01..06`. Every question has an `author`:
  `human` (the project owner edited and reviewed it; status `reviewed`) or `assistant` (drafted by
  the assistant, reviewed by the owner on 2026-10-25 and 2026-10-27: all 25 are `reviewed`). Human: 17 sql, 10 doc, 3 refuse. Assistant: 2 sql
  (sql-18, sql-19: generalisation checks that need a schema-comment fact the question does not
  mention), 12 doc (doc-11..22, six per spec, each with exactly one expected clause or table and
  `confirmed_chunks` that exist in the corpus), 4 mixed (a spec table fact plus a SQL lookup; the
  orchestrator runs them; mixed-01's reference returns the counts only, like sql-18), 3 refuse. sql-16 and
  sql-17 are held-out checks of the FRN guidance. doc and mixed questions have `spec` and
  `expected_clauses` / `expected_tables`; never quote spec text in them. sql and mixed questions
  carry `reference_sql` and a stored
  `reference_result` produced through sql-mcp (`make golden-refresh`, needs `make sql-mcp`);
  `ordered: true` where the question asks for a ranking; `must_mention` where the agent's notes
  must say something (sql-09: the ERP unit is unknown). Scorer (`eval/scorer.py`): column names
  ignored; relaxed column match (every reference column must have an agent column with the same
  values, extra agent columns are fine, the row count must match exactly); order only if
  `ordered`; numbers to 1e-6 relative; NULL equals NULL only; empty results need at least as many
  columns as the reference. `make eval-sql` (PAID; needs
  `LLM_ALLOW_PAID=true` and a running sql-mcp; `DRY=1` prints the plan and worst-case cost and
  calls nothing) runs sql and refuse questions (25 now; the worst case is about $0.32 at 800 output
  tokens), with a per-run cap of questions x 4 and no fallback provider for that command only.
  `make eval-docs` (FREE; needs a running docs-mcp and Ollama) scores recall@5 of the doc
  questions in vector, keyword and hybrid mode with and without the spec filter, for the human and
  the assistant questions separately. Every real eval run appends rows (date, run_id, commit,
  model/tier or question group, accuracy, calls, cost, note) to `eval/RESULTS.md`; `NOTE=...`
  fills the note.
- **analytics-mcp** (`src/rflens/mcp/analytics_server.py` + `src/rflens/analytics/`, `make analytics-mcp`,
  `http://127.0.0.1:8103/mcp`; same protocol setup as the other two). A FIXED menu of nine
  parameterized analyses on SECOM (describe, missing_summary, distribution, group_compare,
  correlation, correlation_top, feature_ranking, outliers, plot) plus `list_analyses`; never
  LLM-written code. Parameters are Pydantic models (`extra="forbid"`, ranges); the analysis runs in
  a Docker container (`rflens-analytics:v1`, `make analytics-image`; numpy, scipy, matplotlib; no
  pandas) with `--network none`, memory/CPU/pids/time limits, a read-only root file system, all
  capabilities dropped, `/in` read-only and `/out` the only writable mount; the data reaches it as
  a CSV extract read through the role `rflens_analytics_reader` (SELECT on the SECOM tables only;
  migration 0015), never a credential. A job past `ANALYTICS_TIMEOUT_S` (60) is killed by the
  container name that call chose. Results (JSON, at most 64 KB) and plots (`artifacts/<analysis_id>.png`,
  the id is a hash of analysis, parameters and data) come back with `call_id`, the row in
  `analytics_call_log` (INSERT-only role `rflens_analytics_logger`; every call logged, a failed log
  write withholds the result). Statistics: Mann-Whitney U two-sided asymptotic, rank-biserial
  effect size, Benjamini-Hochberg q-values, missing values dropped per feature, nothing imputed.
  Design: docs/design/analytics.md. Needs `ANALYTICS_READER_PASSWORD` / `ANALYTICS_LOGGER_PASSWORD`
  (.env and CI). RadioML was skipped (form-gated 21 GB download, CC BY-NC-SA).
  - **Stats route** of the graph (`agents/stats_agent.py`, `agents/analytics_tools.py`): the planner
    routes descriptive analyses of SECOM to `stats`; the stats agent makes one LLM call (one
    correction after a rejection) and calls analytics-mcp; the writer cites `[A1]`; the verifier
    treats the result as a number source and adds `unsupported_stat_claim`, `causal_claim`,
    `wrong_n` (an n stated for a test must be one of its group sizes or their total) and `missing_n` (docs/design/orchestrator.md, "Stats route").
  - **Routing boundary and writer prompt (after eval-e2e-aecd54a7):** `graph/route_rules.py` settles
    sql versus stats for SECOM questions when decisive (aggregates -> sql; tests, effect sizes,
    associations, correlations, distributions, plots -> stats) and the plan records `route_rule`; the
    planner prompt has six examples (`PLANNER_EXAMPLES`). The statistics rules of the writer are
    appended only when there is an analysis (`writer_system`); combined brackets such as `[D1, D2]`
    are not citations, and the prompt says so. Details: docs/design/orchestrator.md.
  - **Container tests** (`make test-analytics`: the analyses inside the image, then
    `tests_docker/`: isolation flags against a real container, the whole stack) need a Docker
    daemon. The GitHub workflow runs them; `make ci-local` cannot (no Docker inside its
    container) and prints them as NOT RUN, so a green ci-local does not cover them.
- **No authentication in local development.** Per the MCP authorization spec (2026-07-28),
  authorization is OPTIONAL, and HTTP-based servers that offer it follow OAuth 2.1 (the server is a
  resource server publishing OAuth 2.0 Protected Resource Metadata, RFC 9728, and accepting only
  tokens issued for it). A remote deployment of docs-mcp or sql-mcp needs that, plus a real
  hostname in the transport-security allow-lists; do not expose ports 8101 and 8102 beyond
  localhost without it. The SDK
  also still answers a plain GET with a keep-alive SSE stream for legacy clients (a modern-only
  server SHOULD answer 405); harmless on localhost, revisit before any remote deployment.
- The `mcp` SDK adds native-code dependencies (`cryptography`, `cffi`, `rpds-py`, `pywin32`);
  all import cleanly under this machine's Application Control policy (checked one by one).
- Spec filter: the doc agent must pass a spec filter to the search when the question implies one
  (NR vs LTE); inference rules belong to the planner (week 4). Measured on the smoke eval with both
  specs embedded: unfiltered vector 8/10, keyword 0/10, hybrid 6/10; filtered 9/10, 2/10, 7/10
  (`make eval-smoke FILTER_SPEC=1`), because 36.101 has analogous clauses that displace NR answers.
- Retrieval diagnosis on the 8 questions about 38.101-1 (full embed, 2,502 chunks): vector 7/8,
  keyword 2/8, hybrid 5/8. No code defect was found: the HNSW leg equals an exact scan, each leg
  passes 50 candidates into RRF (deeper is worse: depth 20/50/100/200 gives hybrid 6/5/4/3 of 8),
  every expected clause exists, and tokenization of identifiers such as NS_04 and n78 is the same
  on the document and query side. What drives the misses is the keyword leg: the OR of the
  question's terms matches 60-90% of all chunks, `ts_rank_cd` (flags 0) has no length
  normalization and no term rarity weighting, so its top results are long tables (median 2,600-3,300
  chars against a corpus median of 718), and equal-weight RRF rewards presence in both lists over a
  rank-2 hit in one. Tuning options were measured but are NOT applied (see the task report).
- **Orchestrator graph** (`src/rflens/graph/`, LangGraph 1.2.17, `langgraph-checkpoint-postgres`
  3.2.0). Nodes: planner (1 LLM call: route sql / doc / mixed / refuse + spec hint) -> approval ->
  doc_agent (docs-mcp, no LLM) and/or sql_agent (wraps `answer_question`; in a mixed question the
  retrieved chunks go to it as context) -> writer (JSON `{answer}` with `[S1]` / `[D1]` citations)
  -> verifier (deterministic, `graph/verifier.py`) -> finalize; after 1 + `ORCH_MAX_WRITER_RETRIES`
  (2) failed drafts, `redact` keeps only the sentences that verify (or falls back to the evidence
  itself) and the status is `partial`. Statuses: answered, partial, refused, failed, cancelled.
  The Sources section is built by code from the cited tags, never by the model.
  - Rules the package keeps (all tested): every LLM call goes through `rflens.llm`; agents reach
    data only through sql-mcp and docs-mcp; the checkpointer is the only direct database
    connection; no `langchain_*` import beyond `langchain_core` / `langchain_protocol` (langgraph
    requires them); importing `rflens.graph` sets `LANGSMITH_TRACING=false` and
    `LANGGRAPH_STRICT_MSGPACK=true` and refuses to import when a tracing switch is on; state holds
    JSON types only; checkpoints are written with `durability="sync"`.
  - Checkpoint store: schema `checkpoints`, tables created by migration 0012 through the library's
    own `setup()` under the owner. Runtime role `rflens_checkpointer` (`CHECKPOINTER_PASSWORD`) has
    DML on the four tables and `search_path = checkpoints`; `open_checkpointer()` never calls
    `setup()` and refuses to start when `checkpoint_migrations` is not at the version the installed
    library expects (run `make migrate`). Thread id = question id.
  - The gateway ledger connects as `rflens_llm_ledger` (`LLM_LEDGER_PASSWORD`; SELECT / INSERT /
    UPDATE on `llm_calls` only). The per-question call cap counts the ledger's rows for the
    question (`LLMRun.seed_question`), so a resumed question cannot restart its allowance.
    Orchestrated cap `ORCH_MAX_CALLS_PER_QUESTION` = 10 (replaces `LLM_MAX_CALLS_PER_QUESTION`
    inside the graph). A question-cap stop ends the question as `failed`; run cap, budget, loop
    and paid-call stops propagate and leave the thread resumable.
  - sql-mcp `run_query` returns `query_id` (its `sql_query_log` row; the logger role cannot SELECT,
    so the id comes from `nextval` on the granted sequence).
  - **Source of truth for the graph: [docs/design/orchestrator.md](docs/design/orchestrator.md)**
    (state, routing, approval triggers, budget and cap, roles, verifier, citations). Where the
    build differs from it, the design wins unless a reason is flagged.
  - Spec inference: deterministic rules first (`graph/spec_rules.py`: NR / 5G / FR1 / nNN /
    38.101 -> 38.101-1; LTE / E-UTRA / 36.101 -> 36.101), the planner's hint only when the rules
    have no answer. Approval interrupts (`interrupt()`, resumed with `{"approve", "spec"}`) on
    (a) `ASK=1`, (b) planned worst case above `APPROVAL_COST_USD` (0.05), (c) remaining budget
    below twice the planned worst case, (d) rule and planner hint conflict, (e) wording names both
    NR and LTE. `approval_mode="auto"` approves and records `by: auto`. The planner's one call also
    returns `doc_query` (the doc agent searches with it) and `sql_question` (asked by the SQL agent
    of a mixed question).
  - Verifier checks: citations (dangling / missing), numbers (exact, rounded at the draft's own
    precision, thousands separators, derived from one SQL column), units (`invented_unit`; a column
    whose comment says UNKNOWN allows no unit), ASSUMED columns require assumption wording,
    quotations, and for mixed questions every numeric literal compared with a frequency column in a
    WHERE clause (band edge; zero exempt) must appear in a cited chunk (`unsupported_sql_literal`,
    which withholds the SQL result). It is a floor, not proof: a number matched through the
    "derived" rule can coincide with an unrelated value.
  - **Band lookup** (`spec_operating_bands`, migration 0013, `make load-spec-bands`; 70 rows from
    TS 36.101 Table 5.5-1 and 71 from TS 38.101-1 Table 5.2-1, parsed from the chunks' raw HTML;
    local database only): the planner names bands in mixed questions, the doc agent reads their
    rows through sql-mcp and cites the chunk that holds the row. Needed because semantic search
    did not find the band-table chunks (mixed-01: expected chunk not in the top 20, even for the
    planner's `doc_query`). Footnote texts are not in the corpus, only markers. `duplex_mode` is
    kept as the spec labels it; `has_uplink` / `has_downlink` are generated. Details in
    docs/design/orchestrator.md.
  - **Provenance versus meaning.** The verifier proves where numbers come from, not that the SQL
    means what was asked (live: mixed-01 used a downlink edge for an uplink question, mixed-02
    dropped "active" and the whole-block rule). Fixes: band edges are joined from
    `spec_operating_bands` (SQL prompt rule 9 and a worked example, the doc agent's context says
    to join), the verifier's `edge_side_mismatch` (uplink/downlink asked vs `ul_*`/`dl_*` used;
    withholds the SQL result), the SQL agent of a mixed question gets the original question plus
    the planner's `sql_question` as context (the original's constraints win; the SQL prompt rule
    10 keeps every constraint and defines "inside" as the whole assigned block), and
    `draft_history` keeps every writer draft and its failures in the checkpoint.
  - **Retry routing:** a verification that fails only with SQL-side failures
    (`edge_side_mismatch`, `unsupported_sql_literal`) sends the question back to the SQL agent once,
    with the failures and the previous SQL; prose failures go to the writer (up to 2 retries); the
    per-question cap of 10 bounds everything. "Inside a range" = the whole assigned block lies
    within it is a project convention (SQL prompt rule 10; docs/design/orchestrator.md).
  - **Command line:** `make orchestrate Q="..." THREAD=<id> ASK=1` (plan, print, checkpoint, wait at
    approve / edit / cancel), `make orchestrate-resume THREAD=<id> DECISION=approve|cancel` (a new
    process continues the thread), `make orchestrate-status THREAD=<id>` (latest checkpoint, next
    node, run ids and calls per node from `llm_calls`); `rflens.graph.cli`. A hard kill must hit
    the pid the program prints (on Windows the venv launcher is a separate process).
    `make servers-up|servers-status|servers-down` start and stop the dev servers (sql-mcp, docs-mcp,
    api, ui) by the PID recorded in the gitignored `run/<name>.pid`, never by port; a taken port
    is reported, not freed (`PORT="api=8010"` picks another).
    `scripts/demo_kill_resume.py` scripts the whole demo; offline it runs with
    `LLM_PROVIDER=fake RFLENS_FAKE_SCRIPT=<json>` (tests/test_orchestrate_cli.py).
  - **CLI / API / UI (step 4).** `rflens ask "<question>" [--thread ID] [--ask] [--show-drafts]
    [--json]` (one shot). `make api` serves the FastAPI app on 127.0.0.1 only (`POST /ask`,
    `POST /threads/{id}/resume`, `GET /threads/{id}`, `GET /threads`, `GET /chunks/{id}`, open
    `GET /health`); every other route needs the `X-API-Key` header = `API_KEY` in `.env` and is
    limited to `API_RATE_LIMIT_PER_MINUTE` (30) per key. `make ui` starts the Streamlit UI
    (127.0.0.1:8502, usage statistics off in `.streamlit/config.toml`; `API_URL` overrides the API
    address) which only calls the API. Streamlit 1.66.1 / pyarrow 26.0.0 import under this
    machine's Application Control policy. Details: docs/design/orchestrator.md. The verifier also
    has `frn_name_grouping` (grouping companies by name, alone or together with the FRN, is a SQL-side
    failure: one SQL retry, then redact).
  - **End-to-end eval:** `make eval-e2e` (PAID; needs `LLM_ALLOW_PAID=true`, `make servers-up` and
    Ollama; `DRY=1` prints the plan and worst case) runs all golden questions through the graph
    (approval auto, one thread each) and scores the final answers (docs/design/orchestrator.md);
    `STOP_USD` (0.40) stops the run after the question in progress. A row goes to the "End-to-end
    runs" table of eval/RESULTS.md. The full-run results and misses are diagnosed, not tuned.
  - **Writer failures and loop scope:** a writer output that is cut off (MAX_TOKENS) or not valid
    JSON is the failure `writer_truncated` (`ORCH_WRITER_MAX_TOKENS` 1500); writer retries always
    carry feedback and a retry number, so they never repeat a prompt; `draft_history` keeps each
    writer call's raw output. Loop detectors count per question when the call names one: a loop
    fails that question (`loop_detected`), the run goes on; run caps and the budget still stop it.
    End-to-end scorer v2 checks the order of rows only (tied rows are a group); see eval/RESULTS.md.
  - **Improvement round (details in docs/design/orchestrator.md):** `LLM_TEMPERATURE` (unset: not
    sent; the gateway sent no temperature, topP or topK before; `eval-*` commands use 0);
    retrieval: `CLAUSE_LOOKUP` and `VARIANT_DOWNRANK` (both on; `retrieval/variants.py`), writer
    depth 5; the SQL prompt and three column comments say NULL keys are not entities (migration
    0014); verifier `missing_band_citation`. `make test` runs `python -m pytest`: this machine's
    Application Control policy blocked the unsigned `pytest.exe` launcher on 2026-10-25.
  - Live smoke (`scripts/graph_smoke.py`, approval auto): sql-01 and doc-02 answered and verified.
    After the fixes mixed-01 (top 5 licensees, band 5 uplink) and mixed-02 (116,525) were
    `answered`, verified and equal to the golden reference under the relaxed compare, each SQL
    joining `spec_operating_bands`, D1 = the band table chunk. One run each: not a score. Only
    `make eval-e2e` (not built) measures correctness across the set.

## Open data questions

Unresolved facts about the loaded data. Do not guess answers; confirm from an FCC source.
Details are for the `cell` service, snapshot complete/l_cell.zip dated 2026-10-18.

- (a) Units of `uls_fr.power_erp`, `power_output` and `eirp` are not stated in the layout document
  (v9.0). Frequency is assumed MHz (cellular values run 824-892). Columns are unitless.
- (b) Meanings of the code values are not defined in the layout document: `license_status`
  (A 3861, C 1122, E 371, T 110), `entity_type` (L, CL, O) and `class_station_code`
  (almost always empty in cell). The ULS code-definitions document has not been found.
- (c) 380 `uls_fr` rows (48 licences) have `power_erp <= 0`: 120 are `.000` and the rest negative
  (down to -1035.0). Loaded as-is. Largest: usi 12443 (28 rows), 12679 (26), 13149 (26).
- (d) Suspect coordinates in `uls_lo`:
  - Data errors (stored as in the source, not corrected):
    - usi 12510 (KNKA454), location 55, Vidor TX: source longitude `3 56 39.8 W`. Stored as
      -3.944; a leading digit looks lost (Vidor is near 94 W).
    - usi 11695 (KNKN718), location 1, PAGO PAGO: longitude `170 41 42.0 E`. Pago Pago is about
      170 W, so the direction letter is wrong (E instead of W). Stored as +170.695.
  - usi 12361 (KNKN792), location 33, LaFollette TN: latitude `36 73 11.3 N` has minutes 73, so
    `latitude` is NULL and the load prints a warning.
  - usi 13056, location 56: latitude seconds are empty, so `latitude` is NULL.
  - 8 rows have seconds of exactly 60.0 (e.g. usi 11913 location 11). Accepted; the arithmetic is
    exact.
- (e) `uls_fr` has no unique natural key. Two groups repeat (usi, location, antenna, freq_seq_id), both
  in usi 13401 (KNKN520): source lines 224503/224504 (same freq 869.04-879.99, power 140.820 vs
  276.000) and 224505/224506 (different frequency bands, 824.04-834.99 and 890.01-891.48). The
  primary key is (service, line_no).

- (f) FCC equipment-authorization snapshots (`fcc_grantees`, `fcc_test_firms`) end 2021-03-22.
  Devices registered later will not join to a grantee. A fresher source has not been found.
  (Report extraction is not pursued: docs/design/report-extraction.md.)
- (g) `fcc_test_firms.designation_number` is not unique: US5037 (Fangguang Inspection & Testing,
  China; Unified Compliance Laboratory, United States) and US1273 (Shanghai Testing & Inspection
  Institute, China; TARANG, India) each belong to two firms. The key is
  `(designation_number, firm_name)`; do not join on designation_number alone.
- (h) All 591 test firms are expired as of today (expiration dates run 2021-03-26 to 2023-03-18).
  The list reflects accreditation around 2021, not current status.
- (i) Placeholder text other than the explicit list (`N/A`, `NA`, `n/a`, blank, which become NULL)
  is kept as written. In kept `fcc_grantees` fields there is exactly 1 `NIL` (2AA23, city) and
  also 1 `nil` (2AJCJ, city), 1 `NONE` (OQ4, city) and 1 `-` (PK4, city). `fcc_test_firms` has
  none. Whether these mean "no value" is unconfirmed.

- (j) **Spectrum leases in `uls_hd`** (resolved 2026-10-24, with unconfirmed details). 3,174 of the
  5,464 HD rows of the cell file have a call sign of the form L + 9 digits (e.g. L000070487): they
  are spectrum LEASES, not licences; the other 2,290 are licences. `v_uls_licences.is_lease` marks
  them (`call_sign ~ '^L[0-9]{9}$'`; never NULL). Evidence: (1) FCC ULS Public Access Database
  Definitions v9.0 (https://www.fcc.gov/sites/default/files/public_access_database_definitions_v9_0.pdf,
  record layouts, p. 74-77): the Lease Dates record [LD] field 5 is "Call Sign (Lease Id)", the
  Lease Link record [LL] holds Call Sign + Lease ID + Licensee Id, and the Lease Classification
  [LC], leased location, antenna and frequency records are keyed by Lease ID; (2) l_cell.zip
  contains ll.dat, lc.dat, l3.dat, a3.dat and f3.dat, and 3,169 of the 3,174 L-pattern HD rows have
  an LL and an LC record linking the lease ID to a licence call sign (e.g. L000070487 -> licence
  KNKA796); (3) an FCC Wireless Telecommunications Bureau public notice (report 20538,
  2026-07-15, https://docs.fcc.gov/public/attachments/DOC-423045A1.pdf) lists "Call Sign or Lead
  Call Sign: L000031726" for a De Facto Transfer Lease with lessee and licensee, and a separate
  "Lease Identifier: L000045030". The layout document does NOT say outright that leases appear as
  HD rows with the lease ID as call sign; that is shown by the data (2). Unconfirmed: 5 pattern
  rows (L000001242-1245, L000002482; status T, 2006-2007) have no LL/LC record; the lease
  classification codes (ML, TL) are not defined in the layout document. Consequences: leases have
  no `uls_fr` or `uls_lo` rows (0 of 3,174; their frequencies and locations are in the lease files,
  not loaded); the `L` entity of a lease row is the LESSEE; `data_sources.row_count` for
  `uls_cell` (5,464) counts HD rows, i.e. licences and leases; active (status A): 1,732 licences
  and 2,129 leases. Use `is_lease = false` for questions about licences.
- (k) Licensee names are not unique per company: the same licensee id / FRN appears with different
  spellings ("NEW CINGULAR WIRELESS PCS, LLC" 177 and "New Cingular Wireless PCS, LLC" 138 active
  licences, same FRN 0003291192; "Cellco Partnership" 418 and "CELLCO PARTNERSHIP" 4). Counting by
  name splits them; counting by `licensee_frn` or `licensee_id` does not. Group by `licensee_frn`
  alone (grouping by FRN and name together splits the company too) and show `min(licensee_name)`.
  Some licences have a NULL FRN (e.g. 41 expired ones under 29 spellings of one operator), which
  forms its own group. The `v_uls_licences` comments say this (migration 0011).

## Working rules

- Scope: do only what the current prompt asks. No extra features, tables, agents or "nice to haves". List ideas under "Suggestions" at the end instead.
- Anti-loop: if the same command or test fails 3 times, STOP. Do not try a 4th variation. Report the exact error, what you tried, and your best guess at the cause, then wait for me.
- Never re-run an identical command expecting a different result. Change something specific or stop.
- Never delete or rewrite working code to fix an unrelated failure.
- Never weaken a test or acceptance check to make it pass. Never mock the database in integration tests.
- If a requirement is ambiguous, pick the simplest reading, state it in one line, and continue.
- Never invent dataset schemas, URLs or file formats. If you need to know one, inspect the real file or ask me.
- Pin dependency versions. Keep secrets only in .env.
- Never stop, kill or restart a process you did not start in this session. Track the PIDs of processes you start and stop only those PIDs. Never stop processes by port, name or command-line pattern. If a port you need is taken, report it and choose another port.
- End every task with a summary: what changed, check results, open issues.
