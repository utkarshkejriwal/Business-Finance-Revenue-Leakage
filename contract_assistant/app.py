"""Streamlit chat UI: `streamlit run app.py`."""

import uuid

import streamlit as st

from graph import ask, build_graph

st.set_page_config(page_title="Shadowfax Contract Assistant")
st.title("Shadowfax Contract Assistant")
st.caption("Answers from the merchant agreements (M01-M12) and the Schedule A rate card.")


@st.cache_resource
def load_app():
    return build_graph()


app = load_app()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.turns = []

for turn in st.session_state.turns:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                st.table(turn["sources"])

if question := st.chat_input("e.g. What is Kesari Foods' COD fee in September?"):
    st.session_state.turns.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the contracts..."):
            result = ask(app, question, st.session_state.thread_id)
        sources = [
            {
                "section": c["section_id"],
                "parent": c["parent_key"],
                "title": c["section_title"],
                "rerank score": "reference" if c["rerank_score"] is None else round(c["rerank_score"], 2),
            }
            for c in result.get("context", [])
        ]
        st.markdown(result["answer"])
        if sources:
            with st.expander("Sources"):
                st.table(sources)
    st.session_state.turns.append({"role": "assistant", "content": result["answer"], "sources": sources})
