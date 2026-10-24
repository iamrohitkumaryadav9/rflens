"""The test database: created if missing, migrated to head, and guarded.

Tests never touch the development database. `python -m rflens.testdb` (run by `make test` and by
CI before pytest) creates the database named by TEST_DATABASE_URL in the same Postgres server, if
it does not exist, and runs `alembic upgrade head` on it. It is idempotent.

Every entry point refuses a database whose name does not end in `_test`.
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url

SUFFIX = "_test"


class DbTestSettings(BaseSettings):
    """Env: TEST_DATABASE_URL (required)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    test_database_url: str


def require_test_name(name: str | None, where: str = "TEST_DATABASE_URL") -> None:
    if not name or not name.endswith(SUFFIX):
        raise RuntimeError(
            f"refusing to run: {where} points at database {name!r}, whose name does not end "
            f"with {SUFFIX!r}. Tests create and delete data and must never run on the "
            f"development database. Point TEST_DATABASE_URL at a database such as 'rflens_test'."
        )


def get_test_database_url() -> URL:
    url = make_url(DbTestSettings().test_database_url)
    require_test_name(url.database)
    return url


def ensure_database(url: URL) -> bool:
    """Create the database `url` names if it is missing. Returns True when it was created."""
    require_test_name(url.database)
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}
            ).first()
            if exists:
                return False
            conn.execute(text(f'CREATE DATABASE "{url.database}"'))  # name checked above
            return True
    finally:
        admin.dispose()


def migrate(url: URL) -> None:
    """alembic upgrade head on `url` (migrations/env.py reads DATABASE_URL)."""
    require_test_name(url.database)
    os.environ["DATABASE_URL"] = url.render_as_string(hide_password=False)
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config("alembic.ini"), "head")


def main() -> None:
    url = get_test_database_url()
    created = ensure_database(url)
    print(f"test database {url.database!r}: {'created' if created else 'exists'}", flush=True)
    migrate(url)
    print(f"test database {url.database!r}: at head", flush=True)


if __name__ == "__main__":
    main()
