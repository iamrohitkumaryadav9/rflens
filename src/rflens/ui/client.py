"""The UI's client for the local API (and the small pure helpers the page uses).

The UI never touches the database, the MCP servers or a model: it only calls the API with the
X-API-Key header. `ApiClient(http)` takes any `httpx.Client`, so tests hand it a TestClient.
"""

import os
from typing import Any

import httpx

from rflens.config import get_api_settings

SPEC_CHOICES = ("36.101", "38.101-1", "none")
BADGE_STYLES = {
    "verified": ("green", "verified"),
    "partial": ("orange", "partial"),
    "refused": ("gray", "refused"),
    "failed": ("red", "failed"),
    "cancelled": ("gray", "cancelled"),
    "pending approval": ("blue", "pending approval"),
    "unverified": ("red", "unverified"),
}


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class ApiClient:
    def __init__(self, http: httpx.Client):
        self.http = http

    @classmethod
    def from_env(cls) -> "ApiClient":
        settings = get_api_settings()
        base = os.environ.get("API_URL", f"http://127.0.0.1:{settings.api_port}")
        key = settings.api_key.get_secret_value() if settings.api_key else ""
        return cls(httpx.Client(base_url=base, headers={"X-API-Key": key}, timeout=600))

    def _json(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(response.status_code, str(detail))
        return response.json()

    def ask(self, question: str, *, ask: bool = False) -> dict[str, Any]:
        return self._json(self.http.post("/ask", json={"question": question, "ask": ask}))

    def resume(self, thread_id: str, decision: str, spec: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"decision": decision}
        if spec is not None:
            body["spec"] = spec
        return self._json(self.http.post(f"/threads/{thread_id}/resume", json=body))

    def thread(self, thread_id: str) -> dict[str, Any]:
        return self._json(self.http.get(f"/threads/{thread_id}"))

    def threads(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._json(self.http.get("/threads", params={"limit": limit}))["threads"]

    def chunk(self, chunk_id: str) -> dict[str, Any]:
        return self._json(self.http.get(f"/chunks/{chunk_id}"))

    def artifact(self, artifact_id: str) -> bytes:
        """The PNG of an analysis artifact."""
        response = self.http.get(f"/artifacts/{artifact_id}")
        if response.status_code >= 400:
            raise ApiError(response.status_code, response.text)
        return response.content


def badge_style(badge: str) -> tuple[str, str]:
    """(colour, label) for the verification badge."""
    return BADGE_STYLES.get(badge, ("gray", badge))


def source_title(item: dict[str, Any]) -> str:
    if item["kind"] == "stats":
        return (
            f"[{item['tag']}] Analysis {item.get('analysis')}: call #{item.get('call_id')}, "
            f"{item.get('n_runs_used')} run(s)"
        )
    if item["kind"] == "sql":
        rows = item.get("row_count")
        return f"[{item['tag']}] SQL result: query #{item.get('query_id')}, {rows} row(s)"
    title = f" - {item['clause_title']}" if item.get("clause_title") else ""
    table = f", table {item['table_number']}" if item.get("table_number") else ""
    where = f"TS {item['spec']} V{item['version']}, clause {item['clause']}"
    return f"[{item['tag']}] {where}{table}{title}"


def cost_line(cost: dict[str, Any]) -> str:
    return f"{cost.get('calls', 0)} model call(s), ${cost.get('usd', 0.0):.4f}"
