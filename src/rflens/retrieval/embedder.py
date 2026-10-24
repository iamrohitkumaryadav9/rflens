"""Embedding clients. Embeddings come from Ollama's HTTP API (nomic-embed-text), not from torch.

nomic-embed-text needs a task prefix on every input (model card: "the text prompt must include a
task instruction prefix"); Ollama adds none. The two prefixes live here and nowhere else.

The model has a 2,048-token context and Ollama truncates silently by default, so every request
sends truncate=false: an oversized input is an error that names the offending chunk, never a
quietly shortened embedding.
"""

import hashlib
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import httpx

EMBED_DIM = 768
# nomic-embed-text runs with a 2,048-token context under Ollama; 1,840 leaves a 10% margin.
MAX_SAFE_TOKENS = 1840
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def document_input(text: str) -> str:
    return DOCUMENT_PREFIX + text


def query_input(text: str) -> str:
    return QUERY_PREFIX + text


def vector_literal(vector: list[float]) -> str:
    """pgvector text form, passed as a parameter and cast with ::vector (no pgvector package)."""
    return "[" + ",".join(f"{x:.8g}" for x in vector) + "]"


class EmbedError(RuntimeError):
    pass


class ContextLengthError(EmbedError):
    """An input exceeded the model context. `index` is its position in the request."""

    def __init__(self, index: int) -> None:
        super().__init__(f"input {index} exceeds the model's context length")
        self.index = index


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    token_counts: list[int]  # the model's real token count per input, prefix included

    @property
    def tokens(self) -> int:
        return sum(self.token_counts)


class Embedder(Protocol):
    model_id: str

    def embed_documents(self, texts: list[str]) -> EmbedResult: ...

    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        transport: httpx.BaseTransport | None = None,
        max_tries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.max_tries = max_tries
        self._sleep = sleep
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self._model_id: str | None = None

    @property
    def model_id(self) -> str:
        """Tag plus digest of the model Ollama will actually run, e.g. name:tag@0a109f422b47."""
        if self._model_id is None:
            try:
                tags = self._client.get("/api/tags").json()["models"]
            except (httpx.HTTPError, KeyError, ValueError) as e:
                raise EmbedError(f"cannot read the model list from Ollama: {e}") from e
            wanted = self.model if ":" in self.model else f"{self.model}:latest"
            match = next((m for m in tags if m["name"] == wanted), None)
            if match is None:
                raise EmbedError(f"model {wanted!r} is not available in Ollama")
            self._model_id = f"{wanted}@{match['digest'][:12]}"
        return self._model_id

    def _request(self, inputs: list[str]) -> EmbedResult:
        """One embed call with up to max_tries attempts for transport and server errors."""
        last = ""
        for attempt in range(1, self.max_tries + 1):
            try:
                resp = self._client.post(
                    "/api/embed",
                    json={"model": self.model, "input": inputs, "truncate": False},
                )
            except httpx.TransportError as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 400 and "context length" in resp.text:
                    raise ContextLengthError(0)
                if resp.status_code >= 500:
                    last = f"HTTP {resp.status_code}: {resp.text[:120]}"
                elif resp.status_code != 200:
                    raise EmbedError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                else:
                    body = resp.json()
                    vectors = body["embeddings"]
                    if len(vectors) != len(inputs) or any(len(v) != EMBED_DIM for v in vectors):
                        raise EmbedError(f"expected {len(inputs)} vectors of {EMBED_DIM} floats")
                    return EmbedResult(vectors, [int(body.get("prompt_eval_count") or 0)])
            if attempt < self.max_tries:
                self._sleep(2**attempt - 1)  # 1 s, then 3 s
        raise EmbedError(f"Ollama failed {self.max_tries} times in a row; last error: {last}")

    def embed_documents(self, texts: list[str]) -> EmbedResult:
        """One request per text: Ollama reports a single token total per request, and a request
        per text also names the offending input exactly. (It is not slower on this CPU.)"""
        vectors: list[list[float]] = []
        counts: list[int] = []
        for i, t in enumerate(texts):
            try:
                one = self._request([document_input(t)])
            except ContextLengthError:
                raise ContextLengthError(i) from None
            vectors += one.vectors
            counts += one.token_counts
        return EmbedResult(vectors, counts)

    def embed_query(self, text: str) -> list[float]:
        return self._request([query_input(text)]).vectors[0]


@dataclass
class HashEmbedder:
    """Deterministic bag-of-words stand-in for tests (no Ollama): similar word sets give
    similar vectors. Records the prefixed inputs it was given."""

    model_id: str = "hash-test-embedder@0"
    inputs: list[str] = field(default_factory=list)

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * EMBED_DIM
        for word in re.findall(r"[a-z0-9_.\-]+", text.lower()):
            h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
            vec[h % EMBED_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> EmbedResult:
        self.inputs += [document_input(t) for t in texts]
        return EmbedResult([self._vector(t) for t in texts], [len(t.split()) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        self.inputs.append(query_input(text))
        return self._vector(text)
