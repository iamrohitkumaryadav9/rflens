.PHONY: up down migrate test-db test lint run load-secom load-uls load-fcc-eas fetch-3gpp chunk-3gpp embed-3gpp eval-smoke docs-mcp sql-mcp llm-spend golden-refresh eval-sql eval-docs load-spec-bands results-header ci-local

up:
	docker compose up -d --wait

down:
	docker compose down

migrate:
	uv run alembic upgrade head

# Creates the test database if missing and migrates it (idempotent); tests never use DATABASE_URL.
test-db:
	uv run python -m rflens.testdb

test: test-db
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

load-secom:
	uv run python -m rflens.ingest.secom

load-uls:
	uv run python -m rflens.ingest.uls $(SERVICE)

load-fcc-eas:
	uv run python -m rflens.ingest.fcc_eas

# Parse the operating-band tables of the spec chunks into spec_operating_bands (needs chunk-3gpp output).
load-spec-bands:
	uv run python -m rflens.ingest.spec_bands --load

fetch-3gpp:
	uv run python -m rflens.ingest.spec_fetch

chunk-3gpp:
	uv run python -m rflens.ingest.spec_chunker

# Optional: SPEC=38.101-1 CLAUSE=6 LIMIT=150 embeds only that subset.
embed-3gpp:
	uv run python -m rflens.retrieval.embed_3gpp $(if $(SPEC),--spec $(SPEC)) $(if $(CLAUSE),--clause $(CLAUSE)) $(if $(LIMIT),--limit $(LIMIT))

# Optional: SPEC=38.101-1 limits the eval to that spec's questions; FILTER_SPEC=1 makes each
# question search only its own spec.
eval-smoke:
	uv run python -m rflens.retrieval.eval_smoke $(if $(SPEC),--spec $(SPEC)) $(if $(FILTER_SPEC),--filter-spec)

docs-mcp:
	uv run python -m rflens.mcp.docs_server

sql-mcp:
	uv run python -m rflens.mcp.sql_server

# Spend so far, from the llm_calls ledger (total, remaining, today, last calls, top purposes).
llm-spend:
	uv run python -m rflens.llm.spend

run:
	uv run uvicorn rflens.api.main:app --host 127.0.0.1 --port 8000

# Store each sql golden question's reference result (needs a running sql-mcp: make sql-mcp).
golden-refresh:
	uv run python -m rflens.eval.golden

# SQL agent on the golden sql + refuse questions. PAID: needs LLM_ALLOW_PAID=true and a running
# sql-mcp. DRY=1 prints the plan and the worst-case cost and makes no LLM call. Every real run
# appends a row to eval/RESULTS.md; NOTE=... fills its last column.
eval-sql:
	uv run python -m rflens.eval.eval_sql $(if $(DRY),--dry-run) $(if $(NOTE),--note "$(NOTE)")

# Retrieval recall@5 of the golden doc questions through docs-mcp (run make docs-mcp first).
# FREE: no LLM call. Appends rows to eval/RESULTS.md; NOTE=... fills the note.
eval-docs:
	uv run python -m rflens.eval.eval_docs $(if $(NOTE),--note "$(NOTE)")

# Refresh the generated golden-set counts at the top of eval/RESULTS.md.
results-header:
	uv run python -m rflens.eval.results_log

# The CI job in Docker: fresh Postgres, clean copy of the working tree (no .env, no data/), the
# workflow's env plus CI=true, no Ollama or MCP servers. Run it before pushing test or CI changes.
ci-local:
	uv run python -m rflens.ci_local
