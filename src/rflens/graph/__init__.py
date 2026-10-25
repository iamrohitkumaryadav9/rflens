"""The orchestrator graph (LangGraph): planner, doc agent, SQL agent, writer, verifier.

Rules this package keeps (each has a test):
- Every LLM call goes through rflens.llm (ledger, budget cap, loop guards). The graph imports no
  chat-model wrapper and no provider SDK: only `langgraph` itself and the `langchain_core` that
  langgraph requires.
- Agents reach data only through sql-mcp and docs-mcp over MCP. The checkpointer, role
  rflens_checkpointer, is the only database connection this package opens.
- Tracing to LangSmith is off, and importing this package fails if it is switched on.
- Checkpoints are deserialized in strict msgpack mode: state holds JSON types only.

The environment is fixed before langgraph is imported, so this must stay at the top.
"""

import os

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from rflens.graph.safety import assert_tracing_off  # noqa: E402

assert_tracing_off()
