"""The local HTTP API (the blueprint's gateway) over the orchestrator.

    POST /ask                    {"question", "ask"}  -> thread view (answer, or the pending plan)
    POST /threads/{id}/resume    {"decision": "approve" | "cancel" | "edit", "spec"}
    GET  /threads/{id}           status, plan, answer, sources, verification, drafts, cost
    GET  /threads                recent threads
    GET  /chunks/{chunk_id}      one spec chunk through docs-mcp (source panel)
    GET  /health                 open

Every route but /health needs the X-API-Key header (API_KEY in .env) and is rate limited per key.
`make api` serves it on 127.0.0.1 only.
"""

from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import text
from sqlalchemy.orm import Session

from rflens.agents.sql_tools import SqlMcpUnavailable
from rflens.api.security import RateLimiter, make_guard
from rflens.api.service import (
    THREAD_ID_RE,
    BadChunkId,
    ChunkNotFound,
    NotWaiting,
    OrchestratorService,
    ThreadExists,
    ThreadNotFound,
)
from rflens.config import ApiSettings, get_api_settings
from rflens.db import get_session
from rflens.graph.docs_tools import DocsMcpError, DocsMcpUnavailable
from rflens.llm.base import LLMError

SessionDep = Annotated[Session, Depends(get_session)]
Spec = Literal["36.101", "38.101-1", "none"]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3)
    ask: bool = False
    thread_id: str | None = Field(default=None, pattern=THREAD_ID_RE.pattern)


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "cancel", "edit"]
    spec: Spec | None = None

    @model_validator(mode="after")
    def _edit_needs_a_spec(self) -> "ResumeRequest":
        if self.decision == "edit" and self.spec is None:
            raise ValueError("decision 'edit' needs a spec (36.101, 38.101-1 or none)")
        if self.decision != "edit" and self.spec is not None:
            raise ValueError("spec is only valid with decision 'edit'")
        return self

    def as_graph_decision(self) -> dict[str, Any]:
        if self.decision == "cancel":
            return {"approve": False}
        if self.decision == "edit":
            return {"approve": True, "spec": None if self.spec == "none" else self.spec}
        return {"approve": True}


def create_app(
    service: OrchestratorService | None = None, settings: ApiSettings | None = None
) -> FastAPI:
    app = FastAPI(title="RFLens", version="0.1.0")
    settings = settings or get_api_settings()
    svc = service or OrchestratorService()
    guard = make_guard(settings, RateLimiter(settings.api_rate_limit_per_minute))
    protected = [Depends(guard)]

    @app.get("/health")
    def health(session: SessionDep) -> dict[str, str | bool]:
        session.execute(text("SELECT 1"))
        pgvector = session.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        ).scalar_one()
        return {"status": "ok", "pgvector": pgvector}

    def run(call, *args, **kwargs):
        """Call the service and map its errors to HTTP statuses."""
        try:
            return call(*args, **kwargs)
        except ThreadNotFound as e:
            raise HTTPException(404, f"no thread {e}") from e
        except ThreadExists as e:
            raise HTTPException(409, f"thread {e} already exists") from e
        except NotWaiting as e:
            raise HTTPException(409, f"thread {e} is not waiting for a decision") from e
        except ChunkNotFound as e:
            raise HTTPException(404, f"no chunk {e}") from e
        except BadChunkId as e:
            raise HTTPException(422, "invalid chunk id") from e
        except LLMError as e:
            raise HTTPException(
                503, f"{type(e).__name__}: {e}. The thread keeps its last checkpoint."
            ) from e
        except (SqlMcpUnavailable, DocsMcpUnavailable, DocsMcpError) as e:
            raise HTTPException(503, f"data service unavailable: {e}") from e

    @app.post("/ask", dependencies=protected)
    def ask(body: AskRequest) -> dict[str, Any]:
        if len(body.question) > settings.api_max_question_chars:
            raise HTTPException(
                422, f"question is longer than {settings.api_max_question_chars} characters"
            )
        return run(svc.ask, body.question, ask=body.ask, thread_id=body.thread_id)

    @app.post("/threads/{thread_id}/resume", dependencies=protected)
    def resume(thread_id: str, body: ResumeRequest) -> dict[str, Any]:
        _check_thread_id(thread_id)
        return run(svc.resume, thread_id, body.as_graph_decision())

    @app.get("/threads", dependencies=protected)
    def threads(limit: Annotated[int, Query(ge=1, le=100)] = 20) -> dict[str, Any]:
        return {"threads": run(svc.recent_threads, limit)}

    @app.get("/threads/{thread_id}", dependencies=protected)
    def thread(thread_id: str) -> dict[str, Any]:
        _check_thread_id(thread_id)
        return run(svc.view, thread_id)

    @app.get("/chunks/{chunk_id:path}", dependencies=protected)
    def chunk(chunk_id: str) -> dict[str, Any]:
        return run(svc.chunk, chunk_id)

    return app


def _check_thread_id(thread_id: str) -> None:
    if not THREAD_ID_RE.fullmatch(thread_id):
        raise HTTPException(422, "invalid thread id")


app = create_app()
