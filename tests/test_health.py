from fastapi.testclient import TestClient
from sqlalchemy import text

from rflens.api.main import app
from rflens.db import get_engine


def test_health_reports_pgvector() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "pgvector": True}


def test_vector_extension_exists_in_database() -> None:
    with get_engine().connect() as conn:
        found = conn.execute(
            text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one()
    assert found == 1
