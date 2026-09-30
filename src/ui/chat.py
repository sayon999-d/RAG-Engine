import time
from collections.abc import Generator
from datetime import datetime, timezone

import streamlit as st

from ..agent import AgenticRAGPipeline, AgentResponse
from .components import (
    format_sources_for_display,
    render_chat_message,
    render_header,
)


def init_session_state():
    defaults = {
        "messages": [],
        "question_count": 0,
        "total_tokens_used": 0,
        "pipeline": None,
        "current_model": "llama-3.1-8b-instant",
        "current_temperature": 0.1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_conversation_history(
    messages: list[dict], max_turns: int = 10
) -> list[dict[str, str]]:
    history = []
    for msg in messages[-max_turns:]:
        if msg["role"] in ["user", "assistant"]:
            history.append({"role": msg["role"], "content": msg["content"]})
    return history


def stream_response(
    pipeline: AgenticRAGPipeline, query: str, history: list[dict]
) -> Generator[str, None, AgentResponse]:
    response = pipeline.process(query, conversation_history=history)
    answer = response.answer

    for word in answer.split():
        yield word + " "
        time.sleep(0.01)

    return response


def handle_user_input(
    pipeline: AgenticRAGPipeline,
    prompt: str,
    config,
) -> AgentResponse | None:
    if len(prompt) > config.max_question_length:
        st.warning(f"Question truncated to {config.max_question_length} characters.")
        prompt = prompt[: config.max_question_length]

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.question_count += 1

    with st.chat_message("user"):
        st.markdown(prompt)

    history = format_conversation_history(st.session_state.messages)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_answer = ""

        try:
            for chunk in stream_response(pipeline, prompt, history):
                full_answer += chunk
                response_container.markdown(full_answer + "▌")

            response = pipeline.process(prompt, conversation_history=history)

            response_container.markdown(response.answer)

            sources = format_sources_for_display(response.sources)
            if sources:
                render_chat_message("assistant", "", sources, response.metadata)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": sources,
                    "metadata": response.metadata,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            return response

        except Exception as e:
            error_msg = f"Error: {e!s}"
            response_container.error(error_msg)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "metadata": {"error": str(e)},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            return None


def render_chat_interface(pipeline: AgenticRAGPipeline, config):
    render_header(config.groq.model)

    for msg in st.session_state.messages:
        render_chat_message(
            msg["role"],
            msg["content"],
            msg.get("sources"),
            msg.get("metadata"),
        )

    if prompt := st.chat_input("Ask me anything..."):
        if st.session_state.question_count >= config.max_questions_per_session:
            st.warning(
                f"Rate limit reached ({config.max_questions_per_session} questions/session). Refresh to reset."
            )
        else:
            handle_user_input(pipeline, prompt, config)
            st.rerun()


def handle_sidebar_actions(
    sidebar_result: dict, pipeline: AgenticRAGPipeline, ingestion_pipeline
):
    action = sidebar_result.get("action")
    if action == "scrape_url":
        url = sidebar_result.get("url", "").strip()
        if url:
            with st.spinner("Scraping & Indexing..."):
                try:
                    docs = ingestion_pipeline.get_pinecone_documents_from_url(url)
                    pipeline.hybrid_searcher.vectorstore.add_documents(docs)
                    pipeline.hybrid_searcher.fit_bm25(
                        [doc for doc in pipeline.hybrid_searcher.bm25._doc_map.values()]
                    )
                    st.success(f"Added {len(docs)} chunks from {url}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
    elif action == "upload_file":
        file = sidebar_result.get("file")
        if file:
            with st.spinner("Processing & Indexing..."):
                try:
                    file_bytes = file.read()
                    docs = ingestion_pipeline.get_pinecone_documents(
                        file_bytes, file.name
                    )
                    pipeline.hybrid_searcher.vectorstore.add_documents(docs)
                    pipeline.hybrid_searcher.fit_bm25(
                        [doc for doc in pipeline.hybrid_searcher.bm25._doc_map.values()]
                    )
                    st.success(f"Added {len(docs)} chunks from {file.name}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    if sidebar_result.get("model"):
        st.session_state.current_model = sidebar_result["model"]
    if sidebar_result.get("temperature") is not None:
        st.session_state.current_temperature = sidebar_result["temperature"]
