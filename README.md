# RFLens

Agentic RF data platform: ask plain-English questions across public RF compliance data and test reports, and get answers that cite the SQL rows and document passages they came from.

**No Apple data.** RFLens is a public-data analogue of an internal smart-data-ecosystem problem; it uses only public sources (FCC, 3GPP, DeepSig, UCI).

Status: scaffold only (Postgres 16 + pgvector, FastAPI `/health`, Alembic).

Quickstart (needs Docker, uv, make):
`cp .env.example .env` (PowerShell: `Copy-Item .env.example .env`), then `uv sync`, `make up`, `make migrate`, `make test`, `make run`.
`make up` must be running (and `make migrate` applied) before `make test`; the tests use the real Postgres.
3GPP specs are copyrighted: run `make fetch-3gpp` then `make chunk-3gpp` to build the corpus locally; nothing from 3GPP is in this repo.
Check: `curl localhost:8000/health` returns `{"status":"ok","pgvector":true}`.

## Plan: week 5

Report extraction (FCC test reports, vendor datasheets) is not pursued: FCC blocks automated
access, many reports are plot-only, some PDFs disallow copying, and vendor datasheets need a login
(`docs/design/report-extraction.md`; kept as a record: `docs/design/datasheets.md`). The
document-to-SQL piece is the 3GPP operating-band extraction (`spec_operating_bands`, every row
cites its source chunk). Built: analytics-mcp (port 8103) and a stats route: a fixed menu of nine
parameterized analyses on SECOM (rankings of features against pass/fail, comparisons,
distributions, correlations, outliers, plots), never LLM-written code, run in a Docker container
with no network (`docs/design/analytics.md`). Needs Docker: `make analytics-image`, then
`make analytics-mcp` or `make servers-up`; `make test-analytics` runs the container tests (they are
not part of `make ci-local`, which says so).

Future work: report or datasheet value extraction, if a legitimate machine-readable source appears.

## Dev servers

`make servers-up` starts sql-mcp (8102), docs-mcp (8101), analytics-mcp (8103), the API (8000) and the UI (8502), each as
a detached process whose PID is written to `run/<name>.pid` (gitignored) with its output in
`run/<name>.log`. `make servers-status` shows them, and `make servers-down` stops exactly the PIDs
recorded there (and the children of those PIDs). Nothing is ever stopped by port, name or command
line. If a port is taken the server is not started and the command says so; pick another with
`make servers-up SERVERS=api PORT="api=8010"` (clients then need the matching URL, for example
`API_URL`). `SERVERS="api ui"` starts or stops only some. The docs-mcp search needs Ollama running.

