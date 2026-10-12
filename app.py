"""Chat UI for the SQL analyst agent.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from agent import SQLAgent, Workspace
from agent.core import DEFAULT_MODEL

load_dotenv()
st.set_page_config(page_title="SQL Analyst Agent", page_icon="🗄️", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1100px;}
  h1 {font-family: Georgia, serif; letter-spacing: -0.5px;}
  .sub {color: #6b5e4f; margin-top: -0.6rem; margin-bottom: 1rem;}
  .step {font-size: 0.85rem; color: #6b5e4f; margin: 0.4rem 0 0.2rem;}
  .step b {color: #b5562b;}
</style>
""", unsafe_allow_html=True)

EXAMPLES = [
    "Which 5 countries bring in the most revenue? Show a chart.",
    "How did yearly revenue change from 2021 to 2025?",
    "Which sales support agent has the highest-spending customers?",
    "What are the top 3 genres by units sold, and what share of sales do they make up?",
    "Which tracks have never been sold? Just give me the count.",
]


def workspace() -> Workspace:
    # one workspace per browser session, so charts and query results don't leak between users
    if "ws" not in st.session_state:
        st.session_state.ws = Workspace()
    return st.session_state.ws


def render_chart(chart: dict):
    df = pd.DataFrame({chart["x_label"]: chart["x"], chart["y_label"]: chart["y"]})
    colors = ["#b5562b", "#d98c5f", "#e8b48f", "#8a3f1c", "#c9a27e", "#6b5e4f"]
    if chart["kind"] == "pie":
        fig = px.pie(df, names=chart["x_label"], values=chart["y_label"], color_discrete_sequence=colors)
    elif chart["kind"] == "line":
        fig = px.line(df, x=chart["x_label"], y=chart["y_label"], markers=True, color_discrete_sequence=colors)
    else:
        fig = px.bar(df, x=chart["x_label"], y=chart["y_label"], color_discrete_sequence=colors)
    fig.update_layout(title=chart.get("title") or None, height=360, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def render_steps(steps: list[dict]):
    with st.expander(f"How I got this ({len(steps)} step{'s' if len(steps) != 1 else ''})"):
        for i, s in enumerate(steps, 1):
            st.markdown(f'<div class="step"><b>{i}. {s["tool"]}</b></div>', unsafe_allow_html=True)
            if s["tool"] == "run_sql":
                st.code(s["args"].get("query", ""), language="sql")
            elif s["args"]:
                st.code(json.dumps(s["args"], ensure_ascii=False), language="json")
            r = s["result"]
            if "error" in r:
                st.error(r["error"])
            elif s["tool"] == "run_sql" and r.get("columns"):
                st.dataframe(pd.DataFrame(r["rows"], columns=r["columns"]), use_container_width=True, hide_index=True)
            elif s["tool"] in ("calculator", "make_chart"):
                st.caption(json.dumps(r))


with st.sidebar:
    st.subheader("Settings")
    api_key = st.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    model = st.selectbox("Model", [DEFAULT_MODEL, "openai/gpt-oss-20b", "qwen/qwen3.8-27b"])
    st.subheader("The database")
    st.caption("Chinook: a digital music store with 59 customers, 412 invoices, 3,503 tracks and 2021-2025 sales. "
               "The agent has read-only access.")
    tables = workspace().list_tables()["tables"]
    st.dataframe(pd.DataFrame(tables), hide_index=True, use_container_width=True, height=280)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat = []

st.title("Ask your database")
st.markdown('<div class="sub">Questions in plain English. The agent looks up the schema, writes and runs SQL, '
            'fixes its own mistakes, and shows every step.</div>', unsafe_allow_html=True)

if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("chart"):
            render_chart(m["chart"])
        if m.get("steps"):
            render_steps(m["steps"])

picked = None
if not st.session_state.chat:
    st.caption("Try one of these:")
    for ex in EXAMPLES:
        if st.button(ex, key=ex):
            picked = ex

question = st.chat_input("e.g. Which customers spent more than $45?") or picked
if question:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not api_key:
            st.info("Add a Groq API key in the sidebar to start. It's free at console.groq.com.")
            st.stop()
        os.environ["GROQ_API_KEY"] = api_key
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat[:-1]]
        with st.spinner("Working on it..."):
            try:
                result = SQLAgent(model=model, workspace=workspace()).run(question, history)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        st.markdown(result.answer)
        if result.chart:
            render_chart(result.chart)
        steps = [{"tool": s.tool, "args": s.args, "result": s.result} for s in result.steps]
        if steps:
            render_steps(steps)
    st.session_state.chat.append({"role": "assistant", "content": result.answer,
                                  "chart": result.chart, "steps": steps})
