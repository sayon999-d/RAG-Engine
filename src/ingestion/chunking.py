import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    child_chunk_size: int = 300
    child_chunk_overlap: int = 50
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 200
    separators: list[str] = field(
        default_factory=lambda: ["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )


@dataclass
class ParentChildChunk:
    child_chunks: list[Document]
    parent_chunk: Document
    parent_id: str


def create_parent_child_chunks(
    content: str,
    metadata: dict[str, Any],
    config: ChunkConfig | None = None,
) -> list[ParentChildChunk]:
    if config is None:
        config = ChunkConfig()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.parent_chunk_size,
        chunk_overlap=config.parent_chunk_overlap,
        separators=config.separators,
        length_function=len,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.child_chunk_size,
        chunk_overlap=config.child_chunk_overlap,
        separators=config.separators,
        length_function=len,
    )

    parent_docs = parent_splitter.create_documents([content], metadatas=[metadata])

    results = []
    for parent_idx, parent_doc in enumerate(parent_docs):
        parent_id = str(uuid.uuid4())
        parent_doc.metadata.update(
            {
                "parent_id": parent_id,
                "chunk_type": "parent",
                "parent_index": parent_idx,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        child_docs = child_splitter.create_documents(
            [parent_doc.page_content],
            metadatas=[
                {**parent_doc.metadata, "chunk_type": "child", "parent_id": parent_id}
            ],
        )

        for child_idx, child_doc in enumerate(child_docs):
            child_doc.metadata.update(
                {
                    "child_id": str(uuid.uuid4()),
                    "child_index": child_idx,
                    "parent_id": parent_id,
                }
            )

        results.append(
            ParentChildChunk(
                child_chunks=child_docs,
                parent_chunk=parent_doc,
                parent_id=parent_id,
            )
        )

    logger.info(
        f"Created {len(results)} parent chunks with {sum(len(r.child_chunks) for r in results)} child chunks"
    )
    return results


def flatten_child_chunks(parent_child_chunks: list[ParentChildChunk]) -> list[Document]:
    all_children = []
    for pcc in parent_child_chunks:
        all_children.extend(pcc.child_chunks)
    return all_children


def get_parent_chunks(parent_child_chunks: list[ParentChildChunk]) -> list[Document]:
    return [pcc.parent_chunk for pcc in parent_child_chunks]


def prepare_for_pinecone_upload(
    parent_child_chunks: list[ParentChildChunk],
    namespace: str = "default",
) -> list[Document]:
    all_docs = []
    for pcc in parent_child_chunks:
        for child in pcc.child_chunks:
            child.metadata["namespace"] = namespace
            all_docs.append(child)
    return all_docs
