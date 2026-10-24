# RFLens

Agentic RF data platform: ask plain-English questions across public RF compliance data and test reports, and get answers that cite the SQL rows and document passages they came from.

**No Apple data.** RFLens is a public-data analogue of an internal smart-data-ecosystem problem; it uses only public sources (FCC, 3GPP, DeepSig, UCI).

Status: scaffold only (Postgres 16 + pgvector, FastAPI `/health`, Alembic).

Quickstart (needs Docker, uv, make):
`cp .env.example .env` (PowerShell: `Copy-Item .env.example .env`), then `uv sync`, `make up`, `make migrate`, `make test`, `make run`.
`make up` must be running (and `make migrate` applied) before `make test`; the tests use the real Postgres.
Check: `curl localhost:8000/health` returns `{"status":"ok","pgvector":true}`.
