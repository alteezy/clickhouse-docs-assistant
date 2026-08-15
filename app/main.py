import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT / "rag"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import ask  # noqa: E402
from feedback_store import log_feedback  # noqa: E402

EXAMPLE_QUESTIONS = [
    "How do asynchronous inserts work in ClickHouse?",
    "What is the MergeTree table engine used for?",
    "How do I create a materialized view?",
]

st.set_page_config(page_title="ClickHouse Docs Assistant", page_icon=":elephant:")
st.title("ClickHouse Docs Assistant")
st.caption(
    "Ask questions about ClickHouse SQL, administration, and deployment. "
    "Answers are grounded in the official ClickHouse documentation."
)

with st.sidebar:
    st.subheader("Example questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_feedback_buttons(msg):
    feedback_key = f"feedback_{msg['id']}"
    already_rated = st.session_state.get(feedback_key)
    col1, col2, _ = st.columns([1, 1, 10])
    with col1:
        if st.button("\U0001F44D", key=f"up_{msg['id']}", disabled=already_rated is not None):
            log_feedback(msg["id"], msg["question"], msg["content"], "up")
            st.session_state[feedback_key] = "up"
            st.rerun()
    with col2:
        if st.button("\U0001F44E", key=f"down_{msg['id']}", disabled=already_rated is not None):
            log_feedback(msg["id"], msg["question"], msg["content"], "down")
            st.session_state[feedback_key] = "down"
            st.rerun()
    if already_rated:
        st.caption(f"You rated this {'👍' if already_rated == 'up' else '👎'}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_feedback_buttons(msg)

pending_question = st.session_state.pop("pending_question", None)
question = st.chat_input("Ask a ClickHouse question...") or pending_question

if question:
    st.session_state.messages.append(
        {"id": str(uuid.uuid4()), "role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the docs..."):
            try:
                answer = ask(question)
            except Exception as e:
                answer = f"Sorry, something went wrong answering that: {e}"
        st.markdown(answer)
        assistant_msg = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": answer,
            "question": question,
        }
        render_feedback_buttons(assistant_msg)

    st.session_state.messages.append(assistant_msg)
