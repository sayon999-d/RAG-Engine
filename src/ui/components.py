from typing import Any

import streamlit as st


def render_sidebar(
    config,
    question_count: int,
    total_tokens: int,
    on_clear_chat,
    on_export_chat,
):
    with st.sidebar:
        st.title("📊 Dashboard")
        st.markdown("---")

        st.subheader("Usage")
        max_q = config.max_questions_per_session
        remaining = max(0, max_q - question_count)
        progress = question_count / max_q if max_q > 0 else 0
        st.progress(min(progress, 1.0))
        st.caption(
            f"Questions: {question_count}/{max_q} | ~{total_tokens} tokens used | {remaining} remaining"
        )

        st.markdown("---")
        st.subheader("⚙️ Settings")

        model_options = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ]
        current_model = getattr(config.groq, "model", "llama-3.1-8b-instant")
        selected_model = st.selectbox(
            "Groq Model",
            model_options,
            index=model_options.index(current_model)
            if current_model in model_options
            else 0,
        )

        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            getattr(config.groq, "temperature", 0.1),
            0.1,
        )

        st.markdown("---")
        st.subheader("📥 Add Knowledge")

        add_mode = st.radio(
            "Source Type:", ["Web URL", "File Upload"], label_visibility="collapsed"
        )

        if add_mode == "Web URL":
            new_url = st.text_input(
                "Paste Website Link:", placeholder="https://example.com"
            )
            if st.button("🔍 Scrape & Learn", use_container_width=True):
                return {"action": "scrape_url", "url": new_url}
        else:
            supported = [".pdf", ".docx", ".txt", ".csv", ".md"]
            uploaded_file = st.file_uploader(
                f"Upload Document (Max {config.max_file_upload_mb}MB)",
                type=[ext[1:] for ext in supported],
                help=f"Supported: {', '.join(supported)}",
            )
            if uploaded_file and st.button("📂 Process File", use_container_width=True):
                return {"action": "upload_file", "file": uploaded_file}

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                on_clear_chat()
        with col2:
            if st.button("💾 Export", use_container_width=True):
                on_export_chat()

        st.markdown("---")
        st.caption(f"Model: {selected_model} | Temp: {temperature}")

        return {"model": selected_model, "temperature": temperature}


def render_sources_expander(
    sources: list[dict[str, Any]], reranked_results: list[Any] | None = None
):
    if not sources:
        return

    with st.expander("📚 View Grounding Sources", expanded=False):
        for i, src in enumerate(sources):
            col1, col2 = st.columns([4, 1])
            with col1:
                title = src.get("title", src.get("source", f"Source {i + 1}"))
                st.markdown(f"**{title}**")
                if src.get("url"):
                    st.markdown(f"[🔗 Open Source]({src['url']})")
            with col2:
                score = src.get("score", src.get("dense_score", 0))
                st.metric("Score", f"{score:.3f}")

            snippet = src.get("snippet", src.get("content", ""))[:500]
            if snippet:
                st.code(snippet, language="markdown")

            meta_cols = st.columns(3)
            with meta_cols[0]:
                if src.get("doc_type"):
                    st.caption(f"Type: {src['doc_type']}")
            with meta_cols[1]:
                if src.get("created_at"):
                    st.caption(f"Added: {src['created_at'][:10]}")
            with meta_cols[2]:
                if src.get("namespace"):
                    st.caption(f"NS: {src['namespace']}")

            st.divider()


def render_metrics_panel(metadata: dict[str, Any]):
    if not metadata:
        return

    with st.expander("⚡ Performance Metrics", expanded=False):
        cols = st.columns(4)
        with cols[0]:
            st.metric(
                "Total Latency", f"{metadata.get('total_latency', 0) * 1000:.0f}ms"
            )
        with cols[1]:
            st.metric(
                "Retrieval", f"{metadata.get('retrieval_latency', 0) * 1000:.0f}ms"
            )
        with cols[2]:
            st.metric("Rerank", f"{metadata.get('rerank_latency', 0) * 1000:.0f}ms")
        with cols[3]:
            st.metric("Confidence", f"{metadata.get('confidence', 0):.2f}")

        if metadata.get("used_web_fallback"):
            st.warning("🌐 Web fallback was used - retrieved docs were insufficient")


def render_chat_message(
    role: str,
    content: str,
    sources: list[dict] | None = None,
    metadata: dict | None = None,
):
    with st.chat_message(role):
        st.markdown(content)
        if sources and role == "assistant":
            render_sources_expander(sources)
        if metadata and role == "assistant":
            render_metrics_panel(metadata)


def format_sources_for_display(documents: list[Any]) -> list[dict[str, Any]]:
    sources = []
    for doc in documents:
        if hasattr(doc, "metadata"):
            meta = doc.metadata
            source = {
                "source": meta.get("source", "Unknown"),
                "title": meta.get("title", meta.get("source", "Unknown")),
                "doc_type": meta.get("doc_type", "unknown"),
                "namespace": meta.get("namespace", "default"),
                "created_at": meta.get("created_at", ""),
                "url": meta.get("source", "")
                if meta.get("source", "").startswith("http")
                else "",
                "snippet": doc.page_content[:500]
                if hasattr(doc, "page_content")
                else "",
            }
            if hasattr(doc, "metadata") and "score" in doc.metadata:
                source["score"] = doc.metadata["score"]
            elif hasattr(doc, "score"):
                source["score"] = doc.score
            sources.append(source)
    return sources


def init_page_config():
    st.set_page_config(
        page_title="Advanced RAG Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
    .stApp {
        background-color: #f4f6f9;
    }
    h1, h2, h3, p, div, span, li {
        color: #2c3e50 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
    }
    [data-testid="stChatInput"] > div {
        background-color: #ffffff !important;
        border: 2px solid #b8c4d1 !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08) !important;
    }
    textarea[data-testid="stChatInputTextArea"] {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        caret-color: #1f2937 !important;
    }
    [data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] {
        background-color: #4CAF50 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_header(model_name: str):
    st.markdown(
        f"""
    <h2 style="color: #2c3e50;">🤖 Advanced RAG Assistant</h2>
    <p style="color: #555;">
        Powered by <strong>Groq ({model_name})</strong> | 
        Hybrid Search + Reranking | 
        Agentic Routing + CRAG
    </p>
    """,
        unsafe_allow_html=True,
    )
