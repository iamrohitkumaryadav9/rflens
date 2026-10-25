"""RFLens UI (Streamlit) on top of the local API. Start it with `make ui`.

Ask a question; when approval is pending, a plan panel offers approve / edit / cancel; the answer
comes with a verification badge, clickable sources (the SQL text and rows for [S1], the clause and
text of a spec chunk for [D1]), calls and cost, and the sidebar lists recent threads to resume.
"""

import streamlit as st

from rflens.ui.client import (
    SPEC_CHOICES,
    ApiClient,
    ApiError,
    badge_style,
    cost_line,
    source_title,
)

EXAMPLES = [
    "How many SECOM production runs failed?",
    "What is the maximum output power and tolerance for power class 3 in band n78?",
    "Which 5 licensees hold the most active cellular licences with frequencies inside the LTE "
    "band 5 uplink range defined in TS 36.101?",
]


def get_client() -> ApiClient:
    if st.session_state.get("client") is None:
        st.session_state["client"] = ApiClient.from_env()
    return st.session_state["client"]


def open_thread(thread_id: str) -> None:
    st.session_state["thread_id"] = thread_id


def show_badge(view: dict) -> None:
    color, label = badge_style(view["badge"])
    st.badge(label, color=color)


def show_plan_panel(client: ApiClient, view: dict) -> None:
    pending = view["pending"]
    plan = view.get("plan") or {}
    st.subheader("Approval needed")
    for reason in pending.get("reasons", []):
        st.warning(reason)
    left, right = st.columns(2)
    left.markdown(f"**Route** {view.get('route')}  \n**Spec** {pending.get('proposed_spec')}")
    left.markdown(f"**Bands** {plan.get('bands')}")
    right.markdown(f"**Document query** {plan.get('doc_query')!r}")
    right.markdown(f"**Database question** {plan.get('sql_question')!r}")
    st.caption(f"Estimated worst-case cost ${plan.get('planned_worst_case_usd', 0):.4f}")
    options = list(SPEC_CHOICES)
    proposed = pending.get("proposed_spec") or "none"
    chosen = st.selectbox(
        "Spec filter for the document search",
        options,
        index=options.index(proposed) if proposed in options else 2,
        key=f"spec-{view['thread_id']}",
    )
    approve, edit, cancel = st.columns(3)
    thread_id = view["thread_id"]
    try:
        if approve.button("Approve", key="approve", type="primary"):
            client.resume(thread_id, "approve")
            st.rerun()
        if edit.button("Approve with this spec", key="edit"):
            client.resume(thread_id, "edit", chosen)
            st.rerun()
        if cancel.button("Cancel", key="cancel"):
            client.resume(thread_id, "cancel")
            st.rerun()
    except ApiError as e:
        st.error(str(e))


def chunk_text(client: ApiClient, chunk_id: str) -> dict:
    cache = st.session_state.setdefault("chunk_cache", {})
    if chunk_id not in cache:
        try:
            cache[chunk_id] = client.chunk(chunk_id)
        except ApiError as e:
            return {"text": f"(chunk not available: {e})", "clause_title": None}
    return cache[chunk_id]


def show_sources(client: ApiClient, view: dict) -> None:
    items = view.get("source_items") or []
    if not items:
        return
    st.subheader("Sources")
    for item in items:
        label = source_title(item) + ("" if item["cited"] else "  (retrieved, not cited)")
        with st.expander(label):
            if item["kind"] == "sql":
                st.code(item["sql"], language="sql")
                if item["rows"]:
                    st.dataframe(
                        [dict(zip(item["columns"], row, strict=False)) for row in item["rows"]]
                    )
                else:
                    st.caption("no rows")
                if item["truncated"]:
                    st.caption("the result was cut at the row limit")
                for note in item["notes"]:
                    st.caption(f"agent note: {note}")
            else:
                for line in item["band_lines"]:
                    st.markdown(f"**{line}**")
                st.caption(f"chunk {item['chunk_id']} (read through docs-mcp)")
                st.text(chunk_text(client, item["chunk_id"])["text"])


def show_verification(view: dict) -> None:
    v = view["verification"]
    with st.expander(f"Verification: {'passed' if v['ok'] else 'not passed'}"):
        if v["failures"]:
            for f in v["failures"]:
                st.markdown(f"- `{f['type']}`: {f['detail']}")
        else:
            st.caption("every number, quotation, citation and band edge traced to the evidence")
        if v["band_edges"]:
            st.markdown("**Band edges typed in the SQL**")
            st.dataframe(v["band_edges"])
        if v["band_joins"]:
            st.markdown(f"**Band table joins** {v['band_joins']}")
        if v["redacted"]:
            st.caption(f"redacted after retries: {', '.join(v['redacted'])}")
        st.caption(
            "The checks prove where the answer's numbers come from, not that the SQL means "
            "what you asked."
        )
    drafts = view.get("draft_history") or []
    if len(drafts) > 1 or (drafts and not drafts[0]["ok"]):
        with st.expander(f"Writer drafts ({len(drafts)})"):
            for d in drafts:
                st.markdown(f"**Draft {d['attempt']}** (verified: {d['ok']})")
                st.text(d["draft"])
                for f in d["failures"]:
                    st.caption(f"{f['type']}: {f['detail']}")


def show_cost(view: dict) -> None:
    cost = view["cost"]
    st.caption(cost_line(cost))
    with st.expander("Calls and cost per node"):
        if cost["nodes"]:
            st.dataframe(
                [
                    {
                        "node": n,
                        "runs": s.get("runs"),
                        "calls": s.get("calls"),
                        "cost_usd": s.get("cost_usd"),
                    }
                    for n, s in cost["nodes"].items()
                ]
            )
        for run in view.get("runs") or []:
            st.caption(
                f"run {run['run_id']}: "
                + ", ".join(f"{n} {s['calls']} call(s)" for n, s in run["nodes"].items())
            )


def show_thread(client: ApiClient, view: dict) -> None:
    st.divider()
    st.markdown(f"**Question** {view['question']}")
    show_badge(view)
    if view["pending"] is not None:
        show_plan_panel(client, view)
        return
    if view.get("halt"):
        st.error(view["halt"])
    if view.get("answer"):
        st.markdown(view["answer"])
    show_sources(client, view)
    show_verification(view)
    show_cost(view)


def sidebar(client: ApiClient) -> None:
    st.sidebar.header("Recent threads")
    try:
        threads = client.threads()
    except ApiError as e:
        st.sidebar.error(str(e))
        return
    for t in threads:
        _, label = badge_style(t["badge"])
        st.sidebar.button(
            f"{label}: {(t['question'] or '')[:48]}",
            key=f"thread-{t['thread_id']}",
            on_click=open_thread,
            args=(t["thread_id"],),
            help=f"{t['thread_id']}  ${t['cost_usd']:.4f}",
        )
    if not threads:
        st.sidebar.caption("none yet")


def main() -> None:
    st.set_page_config(page_title="RFLens", layout="wide")
    st.title("RFLens")
    st.caption("Ask across FCC licences, test data and 3GPP specs. Answers cite their sources.")
    client = get_client()
    question = st.text_area("Question", key="question", height=90, placeholder=EXAMPLES[0])
    ask_first = st.checkbox("Ask for approval before running", key="ask_first")
    st.caption("Examples: " + " | ".join(EXAMPLES))
    if st.button("Ask", key="ask", type="primary") and question.strip():
        try:
            with st.spinner("Planning and running..."):
                view = client.ask(question.strip(), ask=ask_first)
            open_thread(view["thread_id"])
        except ApiError as e:
            st.error(str(e))
    thread_id = st.session_state.get("thread_id")
    if thread_id:
        try:
            show_thread(client, client.thread(thread_id))
        except ApiError as e:
            st.error(str(e))
    sidebar(client)  # last, so a thread asked just now is already in the list


main()
