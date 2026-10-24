"""CI must never be able to make a paid LLM call."""

from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def test_there_are_workflows_to_check() -> None:
    assert list(WORKFLOWS.glob("*.yml")), "no workflow files found: this guard would be empty"


def test_no_workflow_sets_llm_allow_paid() -> None:
    offenders = [
        f.name
        for f in [*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]
        if "LLM_ALLOW_PAID" in f.read_text(encoding="utf-8").upper()
    ]
    assert offenders == [], f"{offenders} mention LLM_ALLOW_PAID; CI must never enable paid calls"
