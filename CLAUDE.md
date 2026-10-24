# RFLens

RFLens is an agentic data platform that lets a wireless engineer ask plain-English questions across real RF compliance data and test reports, and get answers that cite the SQL rows and document passages they came from. LangGraph agents call three MCP servers (sql-mcp, docs-mcp, analytics-mcp) over Postgres with pgvector. Data comes from FCC equipment authorizations, 3GPP specs, spectrum licences, and signal and manufacturing-yield datasets. It is a public-data analogue of the Smart Data Ecosystem problem and uses no Apple data. Full blueprint: docs/blueprint.pdf.

## Stack

Python 3.12, uv for dependencies, FastAPI, Postgres 16 + pgvector, SQLAlchemy 2 + Alembic, pytest, ruff.

## Repo layout

The repo root is the project root (no nested rflens/ folder).

```
src/rflens/__init__.py
src/rflens/config.py      # pydantic-settings, reads .env
src/rflens/db.py          # SQLAlchemy engine + session
src/rflens/api/main.py    # FastAPI app, GET /health
src/rflens/ingest/        # secom.py (UCI SECOM), uls.py (FCC ULS, one service per run), fcc_eas.py (FCC grantees, test firms)
migrations/               # Alembic
tests/                    # tests/fixtures/uls: small real ULS subset, committed
docs/blueprint.pdf
data/                     # gitignored; raw downloads (data/raw/secom, data/raw/uls/<service>)
```

## Conventions

- `data_sources.row_count` is the count of the source's primary entity: runs for `uci_secom`,
  licences (`uls_hd` rows) for `uls_<service>`. Per-table counts live elsewhere (for ULS,
  `uls_file_counts`, which also records the zip's `sha256` and `snapshot_date`).

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
