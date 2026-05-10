"""Streamlit UI: upload databases (CSV/Excel) and chat with natural-language queries."""

import os
from pathlib import Path

import streamlit as st
import plotly.express as px

from aqie.engine import AQIEEngine

# Project root (parent of web/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEY_FILE = PROJECT_ROOT / ".openai_api_key"


def _load_saved_api_key() -> str:
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    return ""


st.set_page_config(page_title="AQIE ChatDB", page_icon="🤖", layout="wide")
st.title("🤖 AQIE ChatDB")
st.caption("Upload Excel or CSV files as your database, then ask questions in plain English.")

if "engine" not in st.session_state:
    st.session_state.engine = AQIEEngine()
if "chat" not in st.session_state:
    st.session_state.chat = []

engine: AQIEEngine = st.session_state.engine

with st.sidebar:
    st.header("Upload databases")
    files = st.file_uploader(
        "Excel (.xlsx, .xls) or CSV — multiple files supported",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="Each file becomes one or more tables (multi-sheet Excel → multiple tables).",
    )
    if st.button("Load into engine", type="primary"):
        if not files:
            st.warning("Choose at least one file.")
        else:
            with st.spinner("Profiling schema and building models..."):
                engine.load_files(files)
            st.success(f"Loaded {len(engine.tables)} table(s).")

    st.divider()
    st.subheader("LLM (optional)")
    use_llm = st.checkbox("Use GPT / compatible API for SQL planning", value=False)
    model_name = st.text_input("Model", value="gpt-4o-mini")
    base_url = st.text_input("Base URL (optional)", value=os.getenv("OPENAI_BASE_URL", ""))
    api_key = _load_saved_api_key()
    engine.configure_llm(
        enabled=use_llm,
        api_key=api_key.strip() or None,
        model=model_name.strip() or "gpt-4o-mini",
        base_url=base_url.strip() or None,
    )
    if api_key:
        st.caption("API key loaded from environment or `.openai_api_key`.")
    else:
        st.warning("No API key. Set `OPENAI_API_KEY` or create `.openai_api_key` in project root.")
    if engine.last_llm_status:
        st.info(engine.last_llm_status)

    if engine.tables:
        st.subheader("Tables")
        for t, df in engine.tables.items():
            st.write(f"- `{t}` — {df.shape[0]:,} × {df.shape[1]}")

col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("Schema")
    if not engine.schema_df.empty:
        st.dataframe(engine.schema_df, width="stretch", height=320)
    else:
        st.info("Load files to see column types and stats.")

with col_b:
    st.subheader("Semantic types")
    if not engine.feature_df.empty and "semantic_type" in engine.feature_df.columns:
        st.dataframe(
            engine.feature_df[
                ["table", "column", "semantic_type", "prop_numeric", "prop_date", "cardinality_ratio"]
            ],
            width="stretch",
            height=320,
        )
    else:
        st.info("Semantic labels appear after loading data.")

st.subheader("Chat")
user_prompt = st.chat_input("Ask anything about your data…")

if user_prompt:
    if not engine.tables:
        st.error("Load at least one database file first.")
    else:
        result = engine.ask(user_prompt)
        st.session_state.chat.append({"role": "user", "text": user_prompt})
        st.session_state.chat.append(
            {
                "role": "assistant",
                "text": result.explanation,
                "sql": result.sql,
                "warnings": result.warnings,
                "data": result.dataframe,
            }
        )

for idx, msg in enumerate(st.session_state.chat):
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        if msg["role"] == "assistant":
            st.code(msg["sql"], language="sql")
            for w in msg.get("warnings", []):
                st.warning(w)
            st.dataframe(msg["data"], width="stretch")

            df = msg["data"]
            if df is not None and not df.empty and len(df.columns) >= 2:
                num_cols = [c for c in df.columns if str(df[c].dtype).startswith(("int", "float"))]
                if num_cols:
                    x_col = df.columns[0]
                    y_col = num_cols[0]
                    fig = px.bar(df.head(20), x=x_col, y=y_col, title=f"{y_col} by {x_col} (top 20)")
                    st.plotly_chart(fig, width="stretch", key=f"chart_{idx}_{x_col}_{y_col}")
