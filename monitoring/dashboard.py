import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import connect  # noqa: E402

# Validated categorical/status palette - see the dataviz skill's palette.md.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
GRIDLINE = "#e1e0d9"
MUTED = "#898781"

st.set_page_config(page_title="ClickHouse Docs Assistant - Monitoring", page_icon=":bar_chart:")
st.title("Monitoring dashboard")

RANGE_OPTIONS = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": None}
range_label = st.selectbox("Time range", list(RANGE_OPTIONS.keys()), index=3)
days = RANGE_OPTIONS[range_label]
time_filter = f"WHERE start_time >= now() - interval '{days} days'" if days else ""
time_filter_feedback = f"WHERE timestamp >= now() - interval '{days} days'" if days else ""


@st.cache_data(ttl=30)
def load_conversations(_days):
    con = connect()
    df = pd.read_sql(
        f"""
        SELECT start_time, duration_ms, question, answer, input_tokens, output_tokens
        FROM spans
        {time_filter}
        {"AND" if time_filter else "WHERE"} name LIKE 'invoke_agent%%'
        ORDER BY start_time DESC
        """,
        con,
    )
    con.close()
    return df


@st.cache_data(ttl=30)
def load_span_latency(_days):
    con = connect()
    df = pd.read_sql(f"SELECT name, duration_ms FROM spans {time_filter}", con)
    con.close()
    return df


@st.cache_data(ttl=30)
def load_feedback(_days):
    con = connect()
    df = pd.read_sql(
        f"SELECT timestamp, question, answer, rating FROM feedback {time_filter_feedback} "
        "ORDER BY timestamp DESC",
        con,
    )
    con.close()
    return df


conversations = load_conversations(days)
span_latency = load_span_latency(days)
feedback = load_feedback(days)

if conversations.empty:
    st.info("No conversations recorded yet for this time range. Ask the chat app a few questions first.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Conversations", len(conversations))
col2.metric("Avg. latency", f"{conversations['duration_ms'].mean():.0f} ms")
col3.metric("Feedback collected", len(feedback))

st.divider()

# 1. Conversations per day - single hue, magnitude over time.
st.subheader("Conversations per day")
by_day = (
    conversations.assign(day=pd.to_datetime(conversations["start_time"]).dt.date)
    .groupby("day")
    .size()
    .reset_index(name="conversations")
)
fig1 = px.bar(by_day, x="day", y="conversations", color_discrete_sequence=[BLUE])
fig1.update_layout(yaxis_title="Conversations", xaxis_title=None, plot_bgcolor="white")
fig1.update_xaxes(gridcolor=GRIDLINE)
fig1.update_yaxes(gridcolor=GRIDLINE)
st.plotly_chart(fig1, use_container_width=True)

# 2. Average latency by span type - fixed categorical order.
st.subheader("Average latency by step")
span_latency = span_latency.assign(
    step=span_latency["name"].apply(
        lambda n: "Whole request" if n.startswith("invoke_agent")
        else "LLM call" if n.startswith("chat")
        else "Doc search"
    )
)
by_step = span_latency.groupby("step")["duration_ms"].mean().reindex(
    ["Whole request", "LLM call", "Doc search"]
).reset_index()
fig2 = px.bar(
    by_step, x="step", y="duration_ms",
    color="step", color_discrete_sequence=[BLUE, ORANGE, AQUA],
)
fig2.update_layout(yaxis_title="Avg. duration (ms)", xaxis_title=None, showlegend=False, plot_bgcolor="white")
fig2.update_yaxes(gridcolor=GRIDLINE)
st.plotly_chart(fig2, use_container_width=True)

# 3. Feedback: thumbs up vs down - status colors (good/critical), never arbitrary hues.
st.subheader("Feedback")
if feedback.empty:
    st.caption("No feedback collected yet for this time range.")
else:
    counts = feedback["rating"].value_counts().reindex(["up", "down"], fill_value=0).reset_index()
    counts.columns = ["rating", "count"]
    counts["label"] = counts["rating"].map({"up": "\U0001F44D Up", "down": "\U0001F44E Down"})
    fig3 = px.bar(
        counts, x="label", y="count", color="rating",
        color_discrete_map={"up": GOOD, "down": CRITICAL},
    )
    fig3.update_layout(yaxis_title="Count", xaxis_title=None, showlegend=False, plot_bgcolor="white")
    fig3.update_yaxes(gridcolor=GRIDLINE)
    st.plotly_chart(fig3, use_container_width=True)

# 4. Token usage over time - two categorical series (input/output), one axis.
st.subheader("Token usage per day")
tokens_by_day = (
    conversations.assign(day=pd.to_datetime(conversations["start_time"]).dt.date)
    .groupby("day")[["input_tokens", "output_tokens"]]
    .sum()
    .reset_index()
    .melt(id_vars="day", value_vars=["input_tokens", "output_tokens"], var_name="type", value_name="tokens")
)
tokens_by_day["type"] = tokens_by_day["type"].map({"input_tokens": "Input", "output_tokens": "Output"})
fig4 = px.line(
    tokens_by_day, x="day", y="tokens", color="type",
    color_discrete_map={"Input": BLUE, "Output": ORANGE}, markers=True,
)
fig4.update_layout(yaxis_title="Tokens", xaxis_title=None, plot_bgcolor="white")
fig4.update_xaxes(gridcolor=GRIDLINE)
fig4.update_yaxes(gridcolor=GRIDLINE)
st.plotly_chart(fig4, use_container_width=True)

# 5. Most-asked questions - single hue, ranked magnitude.
st.subheader("Most-asked questions")
top_questions = (
    conversations["question"].value_counts().head(10).reset_index()
)
top_questions.columns = ["question", "count"]
top_questions["short_question"] = top_questions["question"].str.slice(0, 60) + top_questions["question"].str.len().gt(60).map({True: "...", False: ""})
fig5 = px.bar(
    top_questions.sort_values("count"), x="count", y="short_question",
    orientation="h", color_discrete_sequence=[BLUE],
)
fig5.update_layout(yaxis_title=None, xaxis_title="Times asked", plot_bgcolor="white")
fig5.update_xaxes(gridcolor=GRIDLINE)
st.plotly_chart(fig5, use_container_width=True)

st.divider()
st.subheader("Recent conversations")
recent = conversations.head(20)[["start_time", "question", "answer", "duration_ms"]].copy()
recent["duration_ms"] = recent["duration_ms"].round(0)
st.dataframe(recent, use_container_width=True, hide_index=True)
