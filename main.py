"""
Demo: word_aligner component for EN↔ZH word alignment.

Run with:
    streamlit run main.py
"""

import streamlit as st
from word_aligner import word_aligner

st.set_page_config(page_title="Word Aligner Demo", layout="wide")
st.title("EN ↔ ZH Word Aligner")

# ── Sample data ───────────────────────────────────────────────────
en_tokens = [
    "The", "gender", "inequality", "in", "shopping",
    "is", "a", "widespread", "phenomenon",
]
zh_tokens = [
    "购物", "中", "的", "性别", "不平等",
    "是", "一个", "普遍", "现象",
]

st.markdown("Click an **English word**, then a **Chinese word** to create a pair.  "
            "Press **Save Pairs** to send the data back to Python.")
st.markdown("---")

# ── Render the component ─────────────────────────────────────────
pairs = word_aligner(
    en_words=en_tokens,
    zh_words=zh_tokens,
    key="demo_aligner",
)

# ── Display returned data ────────────────────────────────────────
st.markdown("---")
if pairs:
    st.success(f"Received **{len(pairs)}** pair(s) from component")
    st.json(pairs)
else:
    st.info("No pairs saved yet — align some words and click **Save Pairs**.")
