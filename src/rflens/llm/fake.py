"""A scripted provider for tests and CI: no network, deterministic.

`script` is consumed one item per call: a string (the answer), a RawResponse, or an exception
instance (raised). When the script is empty the provider answers `default_text`. `is_paid=True`
makes the gateway apply budget accounting to it (with whatever prices the test gives), which is how
the budget, cap and loop rules are tested without a paid provider.
"""

from rflens.llm.base import Message, RawResponse, Usage


class FakeProvider:
    service_tier = "standard"

    def __init__(
        self,
        name: str = "fake",
        model: str = "fake-1",
        is_paid: bool = False,
        script: list | None = None,
        default_text: str = "ok",
        usage: Usage | None = None,
    ):
        self.name = name
        self.model = model
        self.is_paid = is_paid
        self.script = list(script or [])
        self.default_text = default_text
        self.usage = usage or Usage(10, 5, 0)
        self.calls: list[tuple[list[Message], dict | None, int]] = []

    def check_allowed(self) -> None:
        return None

    def generate(
        self, messages: list[Message], json_schema: dict | None, max_output_tokens: int
    ) -> RawResponse:
        self.calls.append((messages, json_schema, max_output_tokens))
        item = self.script.pop(0) if self.script else self.default_text
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, RawResponse):
            return item
        return RawResponse(text=item, usage=self.usage, model=self.model, finish_reason="STOP")
