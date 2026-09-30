import logging
from dataclasses import dataclass
from typing import Any

from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Index, Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


@dataclass
class PineconeIndexConfig:
    name: str
    dimension: int = 384
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"
    spec: ServerlessSpec | None = None

    def __post_init__(self):
        if self.spec is None:
            self.spec = ServerlessSpec(cloud=self.cloud, region=self.region)


class PineconeClient:
    def __init__(
        self,
        api_key: str,
        index_config: PineconeIndexConfig,
        embeddings: Embeddings,
    ):
        self.api_key = api_key
        self.index_config = index_config
        self.embeddings = embeddings
        self._pc = None
        self._index = None
        self._vectorstore = None

    @property
    def pc(self) -> Pinecone:
        if self._pc is None:
            self._pc = Pinecone(api_key=self.api_key)
        return self._pc

    @property
    def index(self) -> Index:
        if self._index is None:
            self._index = self.pc.Index(self.index_config.name)
        return self._index

    @property
    def vectorstore(self) -> PineconeVectorStore:
        if self._vectorstore is None:
            self._vectorstore = PineconeVectorStore(
                index_name=self.index_config.name,
                embedding=self.embeddings,
            )
        return self._vectorstore

    def ensure_index_exists(self) -> bool:
        existing = [idx.name for idx in self.pc.list_indexes()]
        if self.index_config.name not in existing:
            logger.info(f"Creating Pinecone index: {self.index_config.name}")
            self.pc.create_index(
                name=self.index_config.name,
                dimension=self.index_config.dimension,
                metric=self.index_config.metric,
                spec=self.index_config.spec,
            )
            logger.info("Index created successfully")
            return True
        logger.info(f"Index {self.index_config.name} already exists")
        return False

    def upsert_documents(
        self,
        documents: list[Document],
        namespace: str = "default",
        batch_size: int = 100,
    ) -> int:
        total = len(documents)
        uploaded = 0

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            try:
                self.vectorstore.add_documents(batch, namespace=namespace)
                uploaded += len(batch)
                logger.info(
                    f"Uploaded {uploaded}/{total} documents to namespace '{namespace}'"
                )
            except Exception as e:
                logger.error(f"Batch upload failed: {e}")
                raise

        return uploaded

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        namespace: str = "default",
        filter_dict: dict | None = None,
        include_metadata: bool = True,
    ) -> Any:
        return self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            include_metadata=include_metadata,
        )

    def query_by_text(
        self,
        text: str,
        top_k: int = 10,
        namespace: str = "default",
        filter_dict: dict | None = None,
    ) -> list[Document]:
        vector = self.embeddings.embed_query(text)
        results = self.query(vector, top_k, namespace, filter_dict)
        docs = []
        for match in results.matches:
            metadata = {k: v for k, v in match.metadata.items() if k != "text"}
            doc = Document(
                page_content=match.metadata.get("text", ""),
                metadata=metadata,
            )
            docs.append(doc)
        return docs

    def delete_namespace(self, namespace: str) -> bool:
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted namespace: {namespace}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete namespace {namespace}: {e}")
            return False

    def get_index_stats(self) -> dict[str, Any]:
        return self.index.describe_index_stats()

    def list_namespaces(self) -> list[str]:
        stats = self.get_index_stats()
        return list(stats.get("namespaces", {}).keys())

    def get_namespace_stats(self, namespace: str) -> dict[str, Any]:
        stats = self.get_index_stats()
        return stats.get("namespaces", {}).get(namespace, {})


def create_pinecone_client(
    api_key: str,
    index_name: str,
    embeddings: Embeddings,
    dimension: int = 384,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> PineconeClient:
    config = PineconeIndexConfig(
        name=index_name,
        dimension=dimension,
        metric=metric,
        cloud=cloud,
        region=region,
    )
    return PineconeClient(api_key=api_key, index_config=config, embeddings=embeddings)
