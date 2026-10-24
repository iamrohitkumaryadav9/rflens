"""The test-database guard: tests must never run on the development database."""

import pytest
from sqlalchemy import text

from rflens.db import get_engine
from rflens.testdb import SUFFIX, ensure_database, get_test_database_url, require_test_name


def test_the_session_is_connected_to_the_test_database() -> None:
    with get_engine().connect() as conn:
        assert conn.execute(text("SELECT current_database()")).scalar_one().endswith(SUFFIX)
    assert get_engine().url.database == get_test_database_url().database


@pytest.mark.parametrize("name", ["rflens", "postgres", "rflens_test_old", "test", "", None])
def test_names_without_the_suffix_are_refused(name) -> None:
    with pytest.raises(RuntimeError, match="refusing to run.*development database"):
        require_test_name(name)


@pytest.mark.parametrize("name", ["rflens_test", "x_test"])
def test_names_with_the_suffix_pass(name: str) -> None:
    require_test_name(name)


def test_ensure_database_refuses_the_development_database() -> None:
    dev = get_test_database_url().set(database="rflens")
    with pytest.raises(RuntimeError, match="refusing to run"):
        ensure_database(dev)


def test_ensure_database_is_idempotent() -> None:
    assert ensure_database(get_test_database_url()) is False  # already created by make test
