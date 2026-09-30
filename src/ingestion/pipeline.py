import logging
from dataclasses import dataclass
from typing import Any

from .chunking import (
    ChunkConfig,
    create_parent_child_chunks,
    flatten_child_chunks,
    prepare_for_pinecone_upload,
)
from .document_parser import parse_document
from .web_scraper import scrape_url

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    source: str
    doc_type: str
    parent_chunks: int
    child_chunks: int
    namespace: str
    success: bool
    error: str | None = None


class IngestionPipeline:
    def __init__(
        self,
        chunk_config: ChunkConfig | None = None,
        default_namespace: str = "default",
    ):
        self.chunk_config = chunk_config or ChunkConfig()
        self.default_namespace = default_namespace

    def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        namespace: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        namespace = namespace or self.default_namespace
        try:
            parsed = parse_document(file_bytes, filename)
            metadata = {
                **parsed.metadata,
                **(extra_metadata or {}),
                "namespace": namespace,
            }
            parent_child_chunks = create_parent_child_chunks(
                parsed.content, metadata, self.chunk_config
            )
            child_chunks = flatten_child_chunks(parent_child_chunks)

            return IngestionResult(
                source=filename,
                doc_type=parsed.doc_type,
                parent_chunks=len(parent_child_chunks),
                child_chunks=len(child_chunks),
                namespace=namespace,
                success=True,
            )
        except Exception as e:
            logger.error(f"File ingestion failed for {filename}: {e}")
            return IngestionResult(
                source=filename,
                doc_type="unknown",
                parent_chunks=0,
                child_chunks=0,
                namespace=namespace,
                success=False,
                error=str(e),
            )

    def ingest_url(
        self,
        url: str,
        namespace: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        namespace = namespace or self.default_namespace
        try:
            scraped = scrape_url(url)
            metadata = {
                **scraped.metadata,
                **(extra_metadata or {}),
                "namespace": namespace,
            }
            parent_child_chunks = create_parent_child_chunks(
                scraped.content, metadata, self.chunk_config
            )
            child_chunks = flatten_child_chunks(parent_child_chunks)

            return IngestionResult(
                source=url,
                doc_type="web",
                parent_chunks=len(parent_child_chunks),
                child_chunks=len(child_chunks),
                namespace=namespace,
                success=True,
            )
        except Exception as e:
            logger.error(f"URL ingestion failed for {url}: {e}")
            return IngestionResult(
                source=url,
                doc_type="web",
                parent_chunks=0,
                child_chunks=0,
                namespace=namespace,
                success=False,
                error=str(e),
            )

    def ingest_multiple_urls(
        self,
        urls: list[str],
        namespace: str | None = None,
    ) -> list[IngestionResult]:
        return [self.ingest_url(url, namespace) for url in urls]

    def get_pinecone_documents(
        self,
        file_bytes: bytes,
        filename: str,
        namespace: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Any]:
        namespace = namespace or self.default_namespace
        parsed = parse_document(file_bytes, filename)
        metadata = {**parsed.metadata, **(extra_metadata or {}), "namespace": namespace}
        parent_child_chunks = create_parent_child_chunks(
            parsed.content, metadata, self.chunk_config
        )
        return prepare_for_pinecone_upload(parent_child_chunks, namespace)

    def get_pinecone_documents_from_url(
        self,
        url: str,
        namespace: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Any]:
        namespace = namespace or self.default_namespace
        scraped = scrape_url(url)
        metadata = {
            **scraped.metadata,
            **(extra_metadata or {}),
            "namespace": namespace,
        }
        parent_child_chunks = create_parent_child_chunks(
            scraped.content, metadata, self.chunk_config
        )
        return prepare_for_pinecone_upload(parent_child_chunks, namespace)


def create_ingestion_pipeline(
    child_chunk_size: int = 300,
    child_chunk_overlap: int = 50,
    parent_chunk_size: int = 1200,
    parent_chunk_overlap: int = 200,
    default_namespace: str = "default",
) -> IngestionPipeline:
    config = ChunkConfig(
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
    )
    return IngestionPipeline(chunk_config=config, default_namespace=default_namespace)
