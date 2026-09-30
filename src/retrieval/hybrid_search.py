import logging
from dataclasses import dataclass

from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    document: Document
    dense_score: float
    sparse_score: float
    combined_score: float
    rank: int


class BM25SparseEncoder:
    def __init__(self):
        try:
            from rank_bm25 import BM25Okapi

            self.BM25Okapi = BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 required for sparse encoding. Install with: pip install rank-bm25"
            )

        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._doc_map = {}

    def fit(self, documents: list[Document]):
        self.corpus = [doc.page_content for doc in documents]
        self.tokenized_corpus = [self._tokenize(text) for text in self.corpus]
        self.bm25 = self.BM25Okapi(self.tokenized_corpus)
        self._doc_map = {i: doc for i, doc in enumerate(documents)}

    def _tokenize(self, text: str) -> list[str]:
        import re

        return re.findall(r"\b\w+\b", text.lower())

    def score(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if not self.bm25:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [(i, scores[i]) for i in top_indices if scores[i] > 0]


class HybridSearcher:
    def __init__(
        self,
        embeddings: Embeddings,
        pinecone_index,
        namespace: str = "default",
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        rrf_k: int = 60,
    ):
        self.embeddings = embeddings
        self.index = pinecone_index
        self.namespace = namespace
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.bm25 = BM25SparseEncoder()
        self._corpus_fitted = False

    def fit_bm25(self, documents: list[Document]):
        self.bm25.fit(documents)
        self._corpus_fitted = True

    def dense_search(
        self, query: str, top_k: int | None = None, filter_dict: dict | None = None
    ) -> list[tuple[Document, float]]:
        top_k = top_k or self.dense_top_k
        query_vector = self.embeddings.embed_query(query)

        filter_dict = filter_dict or {}
        filter_dict["namespace"] = self.namespace

        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
            namespace=self.namespace,
        )

        docs_with_scores = []
        for match in results.matches:
            doc = Document(
                page_content=match.metadata.get("text", ""),
                metadata={k: v for k, v in match.metadata.items() if k != "text"},
            )
            docs_with_scores.append((doc, match.score))
        return docs_with_scores

    def sparse_search(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[Document, float]]:
        if not self._corpus_fitted:
            logger.warning("BM25 not fitted, returning empty sparse results")
            return []
        top_k = top_k or self.sparse_top_k
        return self.bm25.score(query, top_k)

    def reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[Document, float]],
        sparse_results: list[tuple[int, float]],
    ) -> list[SearchResult]:
        doc_scores = {}

        for rank, (doc, score) in enumerate(dense_results):
            doc_id = id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "dense_rank": rank + 1,
                    "sparse_rank": None,
                }
            doc_scores[doc_id]["dense_rank"] = rank + 1

        for rank, (idx, score) in enumerate(sparse_results):
            doc_id = id(self.bm25._doc_map[idx])
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc": self.bm25._doc_map[idx],
                    "dense_rank": None,
                    "sparse_rank": rank + 1,
                }
            doc_scores[doc_id]["sparse_rank"] = rank + 1

        fused_results = []
        for doc_id, data in doc_scores.items():
            dense_rank = data["dense_rank"] or (len(dense_results) + 1)
            sparse_rank = data["sparse_rank"] or (len(sparse_results) + 1)

            rrf_score = 1.0 / (self.rrf_k + dense_rank) + 1.0 / (
                self.rrf_k + sparse_rank
            )

            dense_score = next((s for d, s in dense_results if id(d) == doc_id), 0.0)
            sparse_score = next(
                (s for i, s in sparse_results if id(self.bm25._doc_map[i]) == doc_id),
                0.0,
            )

            fused_results.append(
                SearchResult(
                    document=data["doc"],
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    combined_score=rrf_score,
                    rank=0,
                )
            )

        fused_results.sort(key=lambda x: x.combined_score, reverse=True)
        for i, result in enumerate(fused_results):
            result.rank = i + 1

        return fused_results

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_dict: dict | None = None,
    ) -> list[SearchResult]:
        dense_results = self.dense_search(query, top_k, filter_dict)
        sparse_results = self.sparse_search(query, top_k)
        fused = self.reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:top_k] if top_k else fused
