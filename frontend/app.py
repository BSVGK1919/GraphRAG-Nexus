"""
GraphRAG-Nexus — Streamlit Frontend
AI/ML Career Intelligence Assistant
powered by GraphRAG + LangGraph + Claude
"""

import streamlit as st
import requests
import json
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG-Nexus",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── API Config ────────────────────────────────────────────────
import os
API_URL = os.getenv("API_URL", "http://api:8000")


# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .confidence-verified {
        background-color: #1a7f37;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .confidence-high {
        background-color: #0969da;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .confidence-medium {
        background-color: #bf8700;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .confidence-low {
        background-color: #cf222e;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png",
             width=60)
    st.title("GraphRAG-Nexus")
    st.caption(
        "AI/ML Career Intelligence\n"
        "powered by GraphRAG + LangGraph + Claude"
    )
    st.divider()

    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", 1, 10, 5)
    show_sources = st.toggle("Show sources", value=True)
    show_graph = st.toggle("Show graph context", value=True)
    show_metrics = st.toggle("Show metrics", value=True)

    st.divider()

    st.subheader("Query Types")
    st.markdown("""
    🔧 `skill` — libraries, tools, frameworks
    💼 `career` — roles, salaries, companies
    🚀 `project` — build, roadmap, architecture
    📄 `research` — papers, theory, algorithms
    """)

    st.divider()

    # API Health Check
    if st.button("🔍 Check API Health"):
        try:
            response = requests.get(
                f"{API_URL}/health",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                st.success("API is healthy ✅")
                st.json(data)
            else:
                st.error(f"API error: {response.status_code}")
        except Exception as e:
            st.error(f"Cannot connect to API: {e}")

    st.divider()
    st.caption(
        f"GraphRAG-Nexus v2.0.0\n"
        f"Knowledge Graph + Vector RAG"
    )


# ── Main Content ──────────────────────────────────────────────
st.title("🧠 GraphRAG-Nexus")
st.caption(
    "AI/ML Career Knowledge Assistant — "
    "powered by Knowledge Graphs + Claude + LangGraph"
)

# ── Session State ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# ── Display Chat History ──────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show metadata for assistant messages
        if (message["role"] == "assistant"
                and "metadata" in message):
            meta = message["metadata"]
            _display_metadata(meta, show_sources, show_graph,
                            show_metrics)


def _display_metadata(
    meta: dict,
    show_sources: bool,
    show_graph: bool,
    show_metrics: bool
):
    """Display response metadata."""
    if not meta:
        return

    # Confidence badge
    confidence = meta.get("confidence_band", "LOW")
    confidence_colors = {
        "VERIFIED": "confidence-verified",
        "HIGH": "confidence-high",
        "MEDIUM": "confidence-medium",
        "LOW": "confidence-low"
    }
    css_class = confidence_colors.get(
        confidence, "confidence-low"
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(
            f'<span class="{css_class}">'
            f'{confidence}</span>',
            unsafe_allow_html=True
        )
    with col2:
        st.caption(
            f"🤖 {meta.get('llm_provider', 'unknown').upper()}"
        )
    with col3:
        st.caption(
            f"⏱️ {meta.get('latency_ms', 0):.0f}ms"
        )

    if show_metrics:
        with st.expander("📊 Quality Metrics"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(
                    "Reflection",
                    f"{meta.get('reflection_score', 0):.2f}"
                )
            with c2:
                st.metric(
                    "Claim Score",
                    f"{meta.get('claim_score', 0):.2f}"
                )
            with c3:
                st.metric(
                    "Graph Coverage",
                    f"{meta.get('graph_coverage', 0):.2f}"
                )
            with c4:
                st.metric(
                    "Loops",
                    meta.get('reflection_loops', 0)
                )

    if show_sources and meta.get("sources"):
        with st.expander(
            f"📚 Sources ({len(meta['sources'])})"
        ):
            for i, source in enumerate(
                meta["sources"][:5], 1
            ):
                st.markdown(
                    f"**{i}.** `{source.get('source_type', '')}` "
                    f"— {source.get('source', '')}"
                )
                if source.get("text"):
                    st.caption(source["text"][:200])


# ── Chat Input ────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask about AI/ML skills, careers, projects or research..."
):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Query API
    with st.chat_message("assistant"):
        with st.spinner("Thinking with knowledge graph..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": prompt,
                        "session_id": "streamlit"
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer")

                    st.markdown(answer)
                    st.session_state.query_count += 1

                    meta = {
                        "confidence_band": data.get(
                            "confidence_band", "LOW"
                        ),
                        "llm_provider": data.get(
                            "llm_provider", "unknown"
                        ),
                        "latency_ms": data.get(
                            "latency_ms", 0
                        ),
                        "reflection_score": data.get(
                            "reflection_score", 0
                        ),
                        "claim_score": data.get(
                            "claim_score", 0
                        ),
                        "graph_coverage": data.get(
                            "graph_coverage", 0
                        ),
                        "reflection_loops": data.get(
                            "reflection_loops", 0
                        ),
                        "sources": data.get("sources", [])
                    }

                    _display_metadata(
                        meta,
                        show_sources,
                        show_graph,
                        show_metrics
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": meta
                    })

                else:
                    error_msg = (
                        f"❌ API error: "
                        f"{response.status_code}"
                    )
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

            except requests.exceptions.Timeout:
                msg = (
                    "❌ Request timed out. "
                    "The query is taking too long."
                )
                st.error(msg)
            except Exception as e:
                msg = f"❌ Cannot connect to API: {e}"
                st.error(msg)

# ── Footer ────────────────────────────────────────────────────
st.sidebar.markdown(
    f"Queries this session: "
    f"**{st.session_state.query_count}**"
)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.query_count = 0
    st.rerun()
