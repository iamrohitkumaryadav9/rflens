from typing import Annotated

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.orm import Session

from rflens.db import get_session

app = FastAPI(title="RFLens")

SessionDep = Annotated[Session, Depends(get_session)]


@app.get("/health")
def health(session: SessionDep) -> dict[str, str | bool]:
    session.execute(text("SELECT 1"))
    pgvector = session.execute(
        text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
    ).scalar_one()
    return {"status": "ok", "pgvector": pgvector}
