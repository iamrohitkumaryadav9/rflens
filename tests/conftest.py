"""Test session setup: tests run on the test database, never the development one.

DATABASE_URL is replaced by TEST_DATABASE_URL before any rflens module reads settings, and a
session fixture checks the database that is really connected: its name must end with _test.
`make test` creates and migrates that database first (python -m rflens.testdb).
"""

import os

import pytest

from rflens.testdb import get_test_database_url, require_test_name


def pytest_configure(config: pytest.Config) -> None:
    try:
        url = get_test_database_url()
    except Exception as e:  # missing variable or a name without the _test suffix
        raise pytest.UsageError(str(e)) from e
    os.environ["DATABASE_URL"] = url.render_as_string(hide_password=False)


@pytest.fixture(scope="session", autouse=True)
def _only_the_test_database():
    from sqlalchemy import text

    from rflens.db import get_engine

    with get_engine().connect() as conn:
        name = conn.execute(text("SELECT current_database()")).scalar_one()
    require_test_name(name, "the connected database")
    yield
