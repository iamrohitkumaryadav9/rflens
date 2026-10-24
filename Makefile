.PHONY: up down migrate test lint run

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

run:
	uv run uvicorn rflens.api.main:app --host 127.0.0.1 --port 8000
