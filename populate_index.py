#!/usr/bin/env python3
"""
Enhanced index population script using the new ingestion pipeline.
Supports multi-format documents, web scraping, and parent-child chunking.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from hf_embeddings import HuggingFaceAPIEmbeddings
from src.config import get_config
from src.ingestion import create_ingestion_pipeline
from src.retrieval import create_pinecone_client

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


URL_LIST = [
    # AMD Radeon (Official Product & Tech Pages)
    "https://www.amd.com/en/products/graphics/desktops/radeon.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900-xt.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7800-xt.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7700-xt.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7600-xt.html",
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7600.html",
    "https://www.amd.com/en/support",
    "https://www.amd.com/en/technologies/rdna3",
    "https://pg.asrock.com/Graphics-Card/AMD/Radeon%20RX%207900%20XTX%20Phantom%20Gaming%2024GB%20OC/index.asp",
    # Microsoft DirectML
    "https://learn.microsoft.com/en-us/windows/ai/directml/dml",
    "https://learn.microsoft.com/en-us/windows/ai/directml/dml-get-started",
    "https://learn.microsoft.com/en-us/windows/ai/directml/dml-ops",
    # ONNX Runtime
    "https://onnxruntime.ai/docs/",
    "https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html",
    # ROCm
    "https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html",
    # Stable Diffusion on AMD / DirectML
    "https://github.com/microsoft/Stable-Diffusion-WebUI-DirectML",
    # Developer Docs & Benchmarks
    "https://learn.microsoft.com/en-us/windows/ai/windows-ml/",
    "https://www.club386.com/nvidia-geforce-rtx-5080-vs-amd-radeon-rx-7900-xtx/",
    "https://www.tomshardware.com/reviews/gpu-hierarchy,4388.html",
    "https://www.sapphiretech.com/en/consumer/nitro-radeon-rx-7900-xtx-vaporx-24g-gddr6",
    "https://llm-tracker.info/_TOORG/RTX-3090-vs-7900-XTX-Comparison",
    "https://www.pugetsystems.com/labs/articles/2025-consumer-gpu-content-creation-roundup/",
    "https://www.pugetsystems.com/labs/articles/amd-radeon-rx-9070-xt-content-creation-review/",
    "https://www.byteplus.com/en/topic/376338?title=radeon-rx-7900-xtx-deepseek-benchmark-comprehensive-ai-performance-analysis-for-2025",
    "https://www.gigabyte.com/Graphics-Card/GV-R79XTXAORUS-E-24GD",
    "https://pcpartpicker.com/forums/topic/443648-4080-super-vs-7900-xtx-for-editing",
]


def main():
    config = get_config()

    if not config.embedding.api_key:
        raise ValueError("HUGGINGFACE_API_KEY not set in .env")
    if not config.pinecone.api_key:
        raise ValueError("PINECONE_API_KEY not set in .env")

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

    logger.info("Creating ingestion pipeline...")
    ingestion = create_ingestion_pipeline(
        child_chunk_size=config.chunking.child_chunk_size,
        child_chunk_overlap=config.chunking.child_chunk_overlap,
        parent_chunk_size=config.chunking.parent_chunk_size,
        parent_chunk_overlap=config.chunking.parent_chunk_overlap,
        default_namespace=config.pinecone.namespace,
    )

    total_parent_chunks = 0
    total_child_chunks = 0

    logger.info(f"Processing {len(URL_LIST)} URLs...")
    for url in URL_LIST:
        try:
            logger.info(f"Scraping: {url}")
            result = ingestion.ingest_url(url, namespace=config.pinecone.namespace)

            if result.success:
                docs = ingestion.get_pinecone_documents_from_url(
                    url, namespace=config.pinecone.namespace
                )
                uploaded = pinecone_client.upsert_documents(
                    docs,
                    namespace=config.pinecone.namespace,
                    batch_size=50,
                )
                total_parent_chunks += result.parent_chunks
                total_child_chunks += result.child_chunks
                logger.info(
                    f"  ✓ Uploaded {uploaded} child chunks ({result.parent_chunks} parents)"
                )
            else:
                logger.error(f"  ✗ Failed: {result.error}")

        except Exception as e:
            logger.error(f"  ✗ Error processing {url}: {e}")

    logger.info("=" * 50)
    logger.info("Population Complete!")
    logger.info(f"  Total Parent Chunks: {total_parent_chunks}")
    logger.info(f"  Total Child Chunks: {total_child_chunks}")
    logger.info(f"  Namespace: {config.pinecone.namespace}")
    logger.info(f"  Index: {config.pinecone.index_name}")
    logger.info("=" * 50)
    logger.info("Run 'streamlit run main.py' to start the app.")


if __name__ == "__main__":
    main()
