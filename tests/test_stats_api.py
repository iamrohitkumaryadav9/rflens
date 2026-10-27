"""GET /artifacts/{id}: plots made by analyses, served by their analysis id (64 hex characters)."""

# ruff: noqa: F811
# (the fixtures imported from api_support are used by name)

import pytest
from fastapi.testclient import TestClient

from api_support import API_KEY, docs_conn, make_api, sql_server  # noqa: F401  (fixtures)
from rflens.ui.client import ApiClient, ApiError, source_title

ID = "ab" * 32
PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYTICS_ARTIFACTS_DIR", str(tmp_path))
    (tmp_path / f"{ID}.png").write_bytes(PNG)
    (tmp_path / f"{ID}.json").write_text('{"secret": "not an image"}')
    (tmp_path / "outside.png").write_bytes(PNG)
    return tmp_path


def test_an_artifact_is_served_as_a_png_by_its_id(make_api, artifacts) -> None:
    env = make_api([])
    response = env.client.get(f"/artifacts/{ID}")
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert response.content == PNG
    assert ApiClient(env.client).artifact(ID) == PNG


def test_it_needs_the_api_key(make_api, artifacts) -> None:
    env = make_api([])
    assert TestClient(env.app).get(f"/artifacts/{ID}").status_code in (401, 403)


@pytest.mark.parametrize(
    "bad",
    ["outside", "outside.png", "..%2F..%2F.env", "%2e%2e%2foutside", ID.upper(), ID[:-1], ID + "0"],
)
def test_only_a_64_hex_id_is_accepted_never_a_path(make_api, artifacts, bad) -> None:
    env = make_api([])
    assert env.client.get(f"/artifacts/{bad}").status_code in (404, 422)
    assert env.client.get(f"/artifacts/{bad}").content != PNG


def test_an_unknown_id_is_404_and_the_json_file_is_never_served(make_api, artifacts) -> None:
    env = make_api([])
    assert env.client.get(f"/artifacts/{'cd' * 32}").status_code == 404
    assert b"secret" not in env.client.get(f"/artifacts/{ID}").content


def test_the_ui_client_reports_a_missing_artifact(make_api, artifacts) -> None:
    env = make_api([])
    with pytest.raises(ApiError) as e:
        ApiClient(env.client).artifact("cd" * 32)
    assert e.value.status == 404


def test_a_stats_source_has_a_title() -> None:
    item = {
        "kind": "stats",
        "tag": "A1",
        "analysis": "feature_ranking",
        "call_id": 5,
        "n_runs_used": 1567,
    }
    assert source_title(item) == "[A1] Analysis feature_ranking: call #5, 1567 run(s)"
