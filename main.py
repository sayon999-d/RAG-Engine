import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

from hf_embeddings import HuggingFaceAPIEmbeddings
from src.agent import create_agentic_pipeline
from src.config import get_config
from src.ingestion import create_ingestion_pipeline
from src.observability import create_langfuse_client
from src.retrieval import (
    CrossEncoderReranker,
    HybridSearcher,
    create_pinecone_client,
)
from src.ui import (
    handle_sidebar_actions,
    init_page_config,
    init_session_state,
    render_chat_interface,
    render_sidebar,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@st.cache_resource
def initialize_system():
    config = get_config()

    if not config.groq.api_key:
        st.error(
            "❌ GROQ_API_KEY not configured. Please set it in .env or Streamlit secrets."
        )
        st.stop()

    if not config.pinecone.api_key:
        st.error(
            "❌ PINECONE_API_KEY not configured. Please set it in .env or Streamlit secrets."
        )
        st.stop()

    if not config.embedding.api_key:
        st.error(
            "❌ HUGGINGFACE_API_KEY not configured. Please set it in .env or Streamlit secrets."
        )
        st.stop()

    logger.info("Initializing embeddings...")
    embeddings = HuggingFaceAPIEmbeddings(
        api_key=config.embedding.api_key,
        model_name=config.embedding.model_name,
        max_workers=config.embedding.max_workers,
        max_retries=config.embedding.max_retries,
        retry_delay=config.embedding.retry_delay,
    )

    logger.info("Connecting to Pinecone...")
    pinecone_client = create_pinecone_client(
        api_key=config.pinecone.api_key,
        index_name=config.pinecone.index_name,
        embeddings=embeddings,
        dimension=config.pinecone.dimension,
        metric=config.pinecone.metric,
        cloud=config.pinecone.cloud,
        region=config.pinecone.region,
    )

    pinecone_client.ensure_index_exists()

    logger.info("Initializing hybrid searcher...")
    hybrid_searcher = HybridSearcher(
        embeddings=embeddings,
        pinecone_index=pinecone_client.index,
        namespace=config.pinecone.namespace,
        dense_top_k=config.retrieval.dense_top_k,
        sparse_top_k=config.retrieval.sparse_top_k,
        rrf_k=config.retrieval.rrf_k,
    )

    logger.info("Initializing reranker...")
    reranker = CrossEncoderReranker(
        model_name=config.retrieval.reranker_model,
        api_key=config.embedding.api_key,
    )

    logger.info("Fitting BM25 index...")
    try:
        all_docs = []
        namespaces = pinecone_client.list_namespaces()
        for ns in namespaces:
            stats = pinecone_client.get_namespace_stats(ns)
            vec_count = stats.get("vector_count", 0)
            if vec_count > 0:
                sample_results = pinecone_client.index.query(
                    vector=[0.0] * config.pinecone.dimension,
                    top_k=min(100, vec_count),
                    namespace=ns,
                    include_metadata=True,
                )
                for match in sample_results.matches:
                    from langchain.docstore.document import Document

                    doc = Document(
                        page_content=match.metadata.get("text", ""),
                        metadata={
                            k: v for k, v in match.metadata.items() if k != "text"
                        },
                    )
                    all_docs.append(doc)
        if all_docs:
            hybrid_searcher.fit_bm25(all_docs)
            logger.info(f"BM25 fitted with {len(all_docs)} documents")
        else:
            logger.warning("No documents found in index, BM25 not fitted")
    except Exception as e:
        logger.warning(f"Could not fit BM25: {e}")

    logger.info("Creating agentic pipeline...")
    pipeline = create_agentic_pipeline(
        hybrid_searcher=hybrid_searcher,
        reranker=reranker,
        config=config,
    )

    ingestion_pipeline = create_ingestion_pipeline(
        child_chunk_size=config.chunking.child_chunk_size,
        child_chunk_overlap=config.chunking.child_chunk_overlap,
        parent_chunk_size=config.chunking.parent_chunk_size,
        parent_chunk_overlap=config.chunking.parent_chunk_overlap,
        default_namespace=config.pinecone.namespace,
    )

    langfuse_client = create_langfuse_client(
        public_key=config.observability.public_key,
        secret_key=config.observability.secret_key,
        host=config.observability.host,
        enabled=config.observability.enabled,
    )

    return {
        "config": config,
        "pipeline": pipeline,
        "ingestion_pipeline": ingestion_pipeline,
        "pinecone_client": pinecone_client,
        "embeddings": embeddings,
        "langfuse": langfuse_client,
    }


def main():
    init_page_config()
    init_session_state()

    system = initialize_system()
    config = system["config"]
    pipeline = system["pipeline"]
    ingestion_pipeline = system["ingestion_pipeline"]

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = pipeline

    sidebar_result = render_sidebar(
        config=config,
        question_count=st.session_state.question_count,
        total_tokens=st.session_state.total_tokens_used,
        on_clear_chat=lambda: st.session_state.update(
            {"messages": [], "question_count": 0, "total_tokens_used": 0}
        ),
        on_export_chat=lambda: st.download_button(
            "Download Chat",
            data=str(st.session_state.messages),
            file_name="chat_export.json",
            mime="application/json",
        ),
    )

    handle_sidebar_actions(sidebar_result, pipeline, ingestion_pipeline)

    render_chat_interface(pipeline, config)


if __name__ == "__main__":
    main()
