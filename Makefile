.PHONY: up down migrate test-db test lint run load-secom load-uls load-fcc-eas fetch-3gpp chunk-3gpp embed-3gpp eval-smoke docs-mcp sql-mcp llm-spend

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
