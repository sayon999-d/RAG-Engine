from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings


class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        *,
        max_workers: int = 4,
        max_retries: int = 3,
        retry_delay: float = 1.5,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.max_workers = max(1, max_workers)
        self.max_retries = max(1, max_retries)
        self.retry_delay = max(0.0, retry_delay)
        self._thread_local = threading.local()

    def _get_client(self) -> InferenceClient:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = InferenceClient(token=self.api_key)
            self._thread_local.client = client
        return client

    def _normalize_embedding(self, result: object) -> list[float]:
        if hasattr(result, "tolist"):
            result = result.tolist()

        if not isinstance(result, list):
            raise TypeError(
                f"Unexpected embedding response type: {type(result).__name__}"
            )

        if result and isinstance(result[0], list):
            result = result[0]

        return [float(value) for value in result]

    def _embed_text(self, text: str) -> list[float]:
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._get_client().feature_extraction(
                    text, model=self.model_name
                )
                return self._normalize_embedding(result)
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"Embedding request failed after {self.max_retries} attempts"
        ) from last_error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= 1 or self.max_workers == 1:
            return [self._embed_text(text) for text in texts]

        worker_count = min(self.max_workers, len(texts))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(self._embed_text, texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)
