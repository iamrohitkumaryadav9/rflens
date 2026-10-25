"""Which specification does a question concern? Deterministic rules first, the LLM only as a hint.

NR, 5G, FR1, n-bands (n78) and "38.101" mean TS 38.101-1; LTE, E-UTRA and "36.101" mean TS 36.101.
A question that mentions both, or neither, gets no rule-based answer ("ambiguous" / "none"); a bare
"band 5" is ambiguous on purpose (both specs have a band 5).
"""

import re
from dataclasses import dataclass

NR_SPEC = "38.101-1"
LTE_SPEC = "36.101"
SPECS = (LTE_SPEC, NR_SPEC)

_NR = (
    re.compile(r"\bNR\b"),  # upper case only: "nr" is a word in other languages
    re.compile(r"\b5G\b", re.I),
    re.compile(r"\bFR1\b", re.I),
    re.compile(r"\bn\d{1,3}\b", re.I),
    re.compile(r"38\.101"),
)
_LTE = (
    re.compile(r"\bLTE\b", re.I),
    re.compile(r"\bE-?UTRAN?\b", re.I),
    re.compile(r"36\.101"),
)


@dataclass(frozen=True)
class SpecInference:
    spec: str | None
    source: str  # "rule" | "ambiguous" | "none"
    matched: tuple[str, ...]


def infer_spec(question: str) -> SpecInference:
    nr = tuple(m.group(0) for p in _NR for m in p.finditer(question))
    lte = tuple(m.group(0) for p in _LTE for m in p.finditer(question))
    if nr and lte:
        return SpecInference(None, "ambiguous", nr + lte)
    if nr:
        return SpecInference(NR_SPEC, "rule", nr)
    if lte:
        return SpecInference(LTE_SPEC, "rule", lte)
    return SpecInference(None, "none", ())


def merge_with_hint(rules: SpecInference, hint: str | None) -> tuple[str | None, str, bool]:
    """(spec, source, conflict). A decisive rule wins; the LLM hint decides only when the rules
    have no answer. A hint that contradicts a decisive rule is a conflict (a human decides)."""
    hint = hint if hint in SPECS else None
    if rules.spec is not None:
        return rules.spec, "rule", hint is not None and hint != rules.spec
    if hint is not None:
        return hint, "llm-hint", False
    return None, rules.source, False
