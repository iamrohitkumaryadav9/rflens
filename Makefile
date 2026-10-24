.PHONY: up down migrate test lint run load-secom load-uls load-fcc-eas fetch-3gpp chunk-3gpp embed-3gpp eval-smoke

up:
	docker compose up -d --wait

down:
	docker compose down

migrate:
	uv run alembic upgrade head

test:
	uv run pytest

lint:
	uv run ruff check .

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

eval-smoke:
	uv run python -m rflens.retrieval.eval_smoke

run:
	uv run uvicorn rflens.api.main:app --host 127.0.0.1 --port 8000
