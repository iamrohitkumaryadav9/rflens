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

## Repo layout

The repo root is the project root (no nested rflens/ folder).

```
src/rflens/__init__.py
src/rflens/config.py      # pydantic-settings, reads .env
src/rflens/db.py          # SQLAlchemy engine + session
src/rflens/testdb.py      # create + migrate the test database; the _test name guard
src/rflens/agents/        # sql_agent.py (+ sql_prompt.py, sql_tools.py: MCP client)
src/rflens/eval/          # golden.py, scorer.py, eval_sql.py; golden set in eval/golden.yaml
src/rflens/llm/           # LLM gateway (see Conventions); config/llm_prices.yaml; make llm-spend
scripts/llm_smoke.py      # live gateway check (needs LLM_ALLOW_PAID=true); analyze() (loaders call it after commit)
src/rflens/api/main.py    # FastAPI app, GET /health
src/rflens/ingest/        # secom.py (UCI SECOM), uls.py (FCC ULS, one service per run), fcc_eas.py (FCC grantees, test firms),
                          # spec_fetch.py + spec_chunker.py (3GPP specs -> data/interim/3gpp/<spec>/chunks.jsonl)
src/rflens/retrieval/     # embedder.py (Ollama), index.py (load + embed chunks), hybrid.py
                          # (vector + keyword + RRF), embed_3gpp.py, eval_smoke.py
eval/                     # smoke_3gpp.yaml (reviewed-labels questions)
src/rflens/mcp/           # docs_server.py: docs-mcp (search_docs, get_chunk), make docs-mcp, port 8101
                          # sql_server.py + sql_guard.py: sql-mcp (list_tables, describe_table,
                          # run_query), make sql-mcp, port 8102
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
- `eval/smoke_3gpp.yaml`: ten questions, status `reviewed-labels` (a human reviewed the expected
  clauses and tables). A hit needs the right spec and an exact clause match (a sub-clause does not
  count) or a listed table. `make eval-smoke [SPEC=38.101-1]` reports recall@5 for vector,
  keyword and hybrid. Eight questions are a smoke test, not a basis for tuning.
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
  - **Allow-list (9 tables, 3 views):** `data_sources`, `secom_runs`, `secom_measurements`,
    `uls_hd`, `uls_fr`, `uls_lo`, `uls_file_counts`, `fcc_grantees`, `fcc_test_firms`,
    `v_uls_licences`, `v_uls_frequencies`, `v_secom_values`. **`uls_en` is NOT on it:** it holds
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
  questions. Six worked examples, none from the golden set (a test enforces it).
- **Golden set** `eval/golden.yaml` (all question types; ids `sql-01..15`, `refuse-01..03`, all
  `reviewed` by the project owner on 2026-10-24). sql questions carry `reference_sql` and a stored
  `reference_result` produced through sql-mcp (`make golden-refresh`, needs `make sql-mcp`);
  `ordered: true` where the question asks for a ranking; `must_mention` where the agent's notes
  must say something (sql-09: the ERP unit is unknown). Scorer (`eval/scorer.py`): column names
  ignored; relaxed column match (every reference column must have an agent column with the same
  values, extra agent columns are fine, the row count must match exactly); order only if
  `ordered`; numbers to 1e-6 relative; NULL equals NULL only; empty results need at least as many
  columns as the reference. `make eval-sql` (PAID; needs
  `LLM_ALLOW_PAID=true` and a running sql-mcp; `DRY=1` prints the plan and worst-case cost and
  calls nothing) runs sql and refuse questions, with a per-run cap of questions x 4 and no fallback
  provider for that command only.
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
  Devices registered later will not join to a grantee. When curating FCC IDs for report
  extraction later, prefer IDs whose grantee code exists in `fcc_grantees`. A fresher source has
  not been found.
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
  name splits them; counting by `licensee_frn` or `licensee_id` does not.

## Working rules

- Scope: do only what the current prompt asks. No extra features, tables, agents or "nice to haves". List ideas under "Suggestions" at the end instead.
- Anti-loop: if the same command or test fails 3 times, STOP. Do not try a 4th variation. Report the exact error, what you tried, and your best guess at the cause, then wait for me.
- Never re-run an identical command expecting a different result. Change something specific or stop.
- Never delete or rewrite working code to fix an unrelated failure.
- Never weaken a test or acceptance check to make it pass. Never mock the database in integration tests.
- If a requirement is ambiguous, pick the simplest reading, state it in one line, and continue.
- Never invent dataset schemas, URLs or file formats. If you need to know one, inspect the real file or ask me.
- Pin dependency versions. Keep secrets only in .env.
- End every task with a summary: what changed, check results, open issues.
