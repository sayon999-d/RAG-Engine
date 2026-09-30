import logging
import time
from dataclasses import dataclass

from langchain.docstore.document import Document

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    document: Document
    score: float
    original_rank: int
    new_rank: int


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        api_key: str | None = None,
        max_length: int = 512,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._use_api = api_key is not None

    def _load_local_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, max_length=self.max_length)
                logger.info(f"Loaded local cross-encoder: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for local reranking. Install with: pip install sentence-transformers"
                )

    def _rerank_local(self, query: str, documents: list[Document]) -> list[float]:
        self._load_local_model()
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return scores.tolist()

    def _rerank_api(self, query: str, documents: list[Document]) -> list[float]:
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "huggingface-hub required for API reranking. Install with: pip install huggingface-hub"
            )

        client = InferenceClient(token=self.api_key)
        scores = []
        for doc in documents:
            try:
                result = client.post(
                    model=self.model_name,
                    json={
                        "inputs": {
                            "source_sentence": query,
                            "sentences": [doc.page_content],
                        }
                    },
                )
                score = result[0] if isinstance(result, list) and result else 0.0
                scores.append(float(score))
            except Exception as e:
                logger.warning(f"API rerank failed for doc: {e}")
                scores.append(0.0)
        return scores

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        threshold: float = 0.0,
    ) -> list[RerankResult]:
        if not documents:
            return []

        start_time = time.time()

        if self._use_api:
            scores = self._rerank_api(query, documents)
        else:
            scores = self._rerank_local(query, documents)

        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            if score >= threshold:
                results.append(
                    RerankResult(
                        document=doc,
                        score=float(score),
                        original_rank=i + 1,
                        new_rank=0,
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(results):
            result.new_rank = i + 1

        if top_k:
            results = results[:top_k]

        elapsed = time.time() - start_time
        logger.info(
            f"Reranked {len(documents)} docs in {elapsed:.3f}s, kept {len(results)}"
        )
        return results


class FlashReranker:
    def __init__(
        self,
        model_name: str = "lytang/FlashReranker",
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key

    def rerank(
        self, query: str, documents: list[Document], top_k: int | None = None
    ) -> list[RerankResult]:
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "huggingface-hub required for FlashReranker. Install with: pip install huggingface-hub"
            )

        client = InferenceClient(token=self.api_key)
        results = []

        for i, doc in enumerate(documents):
            try:
                result = client.post(
                    model=self.model_name,
                    json={"inputs": query, "candidates": [doc.page_content]},
                )
                score = float(result[0]) if result else 0.0
                results.append(
                    RerankResult(
                        document=doc,
                        score=score,
                        original_rank=i + 1,
                        new_rank=0,
                    )
                )
            except Exception as e:
                logger.warning(f"FlashReranker failed: {e}")
                results.append(
                    RerankResult(
                        document=doc,
                        score=0.0,
                        original_rank=i + 1,
                        new_rank=0,
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(results):
            result.new_rank = i + 1

        return results[:top_k] if top_k else results


def create_reranker(
    reranker_type: str = "cross_encoder",
    model_name: str = "BAAI/bge-reranker-base",
    api_key: str | None = None,
) -> CrossEncoderReranker | FlashReranker:
    if reranker_type == "flash":
        return FlashReranker(model_name=model_name, api_key=api_key)
    return CrossEncoderReranker(model_name=model_name, api_key=api_key)
