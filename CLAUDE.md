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
migrations/               # Alembic
tests/
docs/blueprint.pdf
data/                     # gitignored; raw downloads go here later
```

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
