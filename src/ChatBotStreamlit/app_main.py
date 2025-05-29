# app_main.py

import os
import time
from typing import Dict, List

import streamlit as st
from annotated_news_pipeline import run_pipeline

# Disable file‐watcher for heavy deps
os.environ["STREAMLIT_SERVER_ENABLE_WATCHDOG"] = "false"

st.set_page_config(
    page_title="Finance News Summaries",
    page_icon="📈",
    layout="centered"
)

st.markdown("""
<style>
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Yahoo Finance – News Summaries")

# Sidebar slider for number of articles
with st.sidebar:
    article_count: int = st.slider(
        "Articles to summarise", 1, 10, 5,
        help="Select how many of the most recent Yahoo Finance stories to summarise."
    )

# Chat history in session-state
if "messages" not in st.session_state:
    st.session_state.messages = []  # type: List[Dict[str, str]]

def _render_chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

_render_chat()

# User input
user_ticker = st.chat_input("Enter a stock ticker, e.g. TSLA…")
if user_ticker:
    ticker = user_ticker.strip().upper()
    st.session_state.messages.append({"role": "user", "content": ticker})

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown("🤔 **Thinking…**")

    try:
        results = run_pipeline(ticker, limit=article_count)
    except Exception as e:
        err = f"❌ Error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": err})
        thinking.markdown(err)
    else:
        thinking.empty()
        for item in results:
            text = (
                f"**{item['url']}**  \n"
                f"**Summary:** {item['summary']}  \n"
                f"[Read article]({item['url']})"
            )
            st.session_state.messages.append({"role": "assistant", "content": text})
            with st.chat_message("assistant"):
                streamer = st.empty()
                for i in range(1, len(text)+1):
                    streamer.markdown(text[:i], unsafe_allow_html=True)
                    time.sleep(0.01)

    _render_chat()
