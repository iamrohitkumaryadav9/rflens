"""Setup of the container tests (`make test-analytics`): same database guard as tests/conftest.py.

These tests need a running Docker daemon and the sandbox image (`make analytics-image`); they are
not part of `make test` / `pytest` (testpaths is `tests`) and `make ci-local` cannot run them (no
Docker daemon inside its container). The GitHub workflow and `make test-analytics` run them.
"""

import os

import pytest

from rflens.testdb import get_test_database_url, require_test_name


def pytest_configure(config: pytest.Config) -> None:
    try:
        url = get_test_database_url()
    except Exception as e:
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
