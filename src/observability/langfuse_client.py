import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TraceData:
    trace_id: str
    name: str
    start_time: float
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    error: str | None = None


class LangfuseClient:
    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str = "https://cloud.langfuse.com",
        enabled: bool = False,
    ):
        self.enabled = enabled
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self._client = None
        self._traces = {}

        if self.enabled and self.public_key and self.secret_key:
            self._init_client()

    def _init_client(self):
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
            logger.info("Langfuse client initialized")
        except ImportError:
            logger.warning(
                "langfuse not installed, observability disabled. Install with: pip install langfuse"
            )
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self.enabled = False

    def trace(
        self,
        name: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> TraceData:
        trace_id = str(uuid.uuid4())
        trace = TraceData(
            trace_id=trace_id,
            name=name,
            start_time=time.time(),
            metadata=metadata or {},
            tags=tags or [],
            input_data=input_data,
        )
        self._traces[trace_id] = trace
        return trace

    def end_trace(
        self,
        trace_id: str,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
    ):
        if trace_id not in self._traces:
            return

        trace = self._traces[trace_id]
        trace.end_time = time.time()
        trace.output_data = output_data
        trace.error = error

        if self.enabled and self._client:
            self._flush_trace(trace)

        del self._traces[trace_id]

    def _flush_trace(self, trace: TraceData):
        try:
            self._client.trace(
                id=trace.trace_id,
                name=trace.name,
                input=trace.input_data,
                output=trace.output_data,
                metadata={
                    **trace.metadata,
                    "duration_ms": (trace.end_time - trace.start_time) * 1000
                    if trace.end_time
                    else 0,
                },
                tags=trace.tags,
                start_time=datetime.fromtimestamp(trace.start_time, tz=timezone.utc).isoformat(),
                end_time=datetime.fromtimestamp(trace.end_time, tz=timezone.utc).isoformat()
                if trace.end_time
                else None,
            )
            self._client.flush()
        except Exception as e:
            logger.error(f"Failed to flush trace to Langfuse: {e}")

    def log_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if not self.enabled or not self._client:
            return

        try:
            self._client.generation(
                trace_id=trace_id,
                name=name,
                model=model,
                input=input_data,
                output=output_data,
                usage=usage,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to log generation: {e}")

    def log_retrieval(
        self,
        trace_id: str,
        name: str,
        query: str,
        documents: list[Any],
        scores: list[float],
        metadata: dict[str, Any] | None = None,
    ):
        if not self.enabled or not self._client:
            return

        try:
            self._client.span(
                trace_id=trace_id,
                name=name,
                input={"query": query},
                output={
                    "documents": [
                        d.page_content[:200]
                        if hasattr(d, "page_content")
                        else str(d)[:200]
                        for d in documents
                    ],
                    "scores": scores,
                },
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to log retrieval: {e}")

    @contextmanager
    def trace_context(
        self,
        name: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        trace = self.trace(name, input_data, metadata, tags)
        error = None
        try:
            yield trace
        except Exception as e:
            error = str(e)
            raise
        finally:
            self.end_trace(trace.trace_id, error=error)


_noop_client = None


def get_noop_client() -> LangfuseClient:
    global _noop_client
    if _noop_client is None:
        _noop_client = LangfuseClient(enabled=False)
    return _noop_client


def create_langfuse_client(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    enabled: bool | None = None,
) -> LangfuseClient:
    return LangfuseClient(
        public_key=public_key,
        secret_key=secret_key,
        host=host or "https://cloud.langfuse.com",
        enabled=enabled if enabled is not None else False,
    )
