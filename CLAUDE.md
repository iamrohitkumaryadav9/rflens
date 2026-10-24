# RFLens

RFLens is an agentic data platform that lets a wireless engineer ask plain-English questions across real RF compliance data and test reports, and get answers that cite the SQL rows and document passages they came from. LangGraph agents call three MCP servers (sql-mcp, docs-mcp, analytics-mcp) over Postgres with pgvector. Data comes from FCC equipment authorizations, 3GPP specs, spectrum licences, and signal and manufacturing-yield datasets. It is a public-data analogue of the Smart Data Ecosystem problem and uses no Apple data. Full blueprint: docs/blueprint.pdf.

## Stack

Python 3.12, uv for dependencies, FastAPI, Postgres 16 + pgvector, SQLAlchemy 2 + Alembic, pytest, ruff.

## Repository and CI

The repo has a GitHub remote (`origin`, https://github.com/iamrohitkumaryadav9/rflens.git) and
CI (`.github/workflows/ci.yml`) is green on `main`. Treat CI as working; do not report it as
unrun. The user reviews and commits; do not commit unless asked.

## Repo layout

The repo root is the project root (no nested rflens/ folder).

```
src/rflens/__init__.py
src/rflens/config.py      # pydantic-settings, reads .env
src/rflens/db.py          # SQLAlchemy engine + session
src/rflens/api/main.py    # FastAPI app, GET /health
src/rflens/ingest/        # secom.py (UCI SECOM), uls.py (FCC ULS, one service per run), fcc_eas.py (FCC grantees, test firms),
                          # spec_fetch.py + spec_chunker.py (3GPP specs -> data/interim/3gpp/<spec>/chunks.jsonl)
src/rflens/retrieval/     # embedder.py (Ollama), index.py (load + embed chunks), hybrid.py
                          # (vector + keyword + RRF), embed_3gpp.py, eval_smoke.py
eval/                     # smoke_3gpp.yaml (reviewed-labels questions)
migrations/               # Alembic
tests/                    # tests/fixtures/uls: small real ULS subset, committed
docs/blueprint.pdf
data/                     # gitignored; raw downloads (data/raw/secom, data/raw/uls/<service>)
```

## Conventions

- `data_sources.row_count` is the count of the source's primary entity: runs for `uci_secom`,
  licences (`uls_hd` rows) for `uls_<service>`. Per-table counts live elsewhere (for ULS,
  `uls_file_counts`, which also records the zip's `sha256` and `snapshot_date`).

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
