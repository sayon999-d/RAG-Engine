import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from ..config import get_config
from ..retrieval import CrossEncoderReranker, HybridSearcher
from .crag import CRAGPipeline, CRAGResult
from .query_rewriter import QueryRewriter, RewrittenQuery
from .router import IntentRouter, IntentType

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    answer: str
    sources: list[Document]
    intent: IntentType
    rewritten_query: RewrittenQuery
    retrieval_results: list[Any]
    reranked_results: list[Any]
    crag_result: CRAGResult
    metadata: dict[str, Any] = field(default_factory=dict)


class AgenticRAGPipeline:
    def __init__(
        self,
        hybrid_searcher: HybridSearcher,
        reranker: CrossEncoderReranker,
        query_rewriter: QueryRewriter | None = None,
        intent_router: IntentRouter | None = None,
        crag_pipeline: CRAGPipeline | None = None,
        config=None,
    ):
        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker
        self.config = config or get_config()

        self.query_rewriter = query_rewriter or QueryRewriter(
            model=self.config.agent.query_rewriter_model,
            api_key=self.config.groq.api_key,
            temperature=self.config.agent.query_rewriter_temperature,
        )

        self.intent_router = intent_router or IntentRouter(
            model=self.config.groq.model,
            api_key=self.config.groq.api_key,
            confidence_threshold=self.config.agent.intent_confidence_threshold,
        )

        self.crag_pipeline = crag_pipeline or CRAGPipeline(
            model=self.config.groq.model,
            api_key=self.config.groq.api_key,
            confidence_threshold=self.config.agent.crag_confidence_threshold,
            web_fallback_enabled=self.config.agent.web_fallback_enabled,
        )

        self.llm = ChatGroq(
            model=self.config.groq.model,
            api_key=self.config.groq.api_key,
            temperature=self.config.groq.temperature,
            max_tokens=self.config.groq.max_tokens,
            streaming=True,
        )

    def process(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        namespace: str = "default",
        filter_dict: dict | None = None,
    ) -> AgentResponse:
        start_time = time.time()
        conversation_history = conversation_history or []

        rewritten = self.query_rewriter.rewrite(query, conversation_history)
        search_query = rewritten.rewritten_query

        intent_result = self.intent_router.classify(search_query, conversation_history)

        if not intent_result.should_retrieve:
            direct_answer = self._answer_directly(search_query, intent_result.intent)
            return AgentResponse(
                answer=direct_answer,
                sources=[],
                intent=intent_result.intent,
                rewritten_query=rewritten,
                retrieval_results=[],
                reranked_results=[],
                crag_result=CRAGResult(
                    answer=direct_answer,
                    sources=[],
                    confidence=1.0,
                    used_web_fallback=False,
                    web_sources=[],
                    reasoning="Direct answer, no retrieval needed",
                ),
                metadata={
                    "total_latency": time.time() - start_time,
                    "retrieval_latency": 0,
                    "rerank_latency": 0,
                    "generation_latency": 0,
                },
            )

        retrieval_start = time.time()
        hybrid_results = self.hybrid_searcher.search(
            search_query,
            top_k=self.config.retrieval.dense_top_k,
            filter_dict=filter_dict,
        )
        retrieval_latency = time.time() - retrieval_start

        candidate_docs = [r.document for r in hybrid_results]

        rerank_start = time.time()
        reranked = self.reranker.rerank(
            search_query,
            candidate_docs,
            top_k=self.config.retrieval.final_top_k,
            threshold=self.config.retrieval.reranker_threshold,
        )
        rerank_latency = time.time() - rerank_start

        reranked_docs = [r.document for r in reranked]

        crag_result = self.crag_pipeline.process(search_query, reranked_docs)

        total_latency = time.time() - start_time

        return AgentResponse(
            answer=crag_result.answer,
            sources=crag_result.sources,
            intent=intent_result.intent,
            rewritten_query=rewritten,
            retrieval_results=hybrid_results,
            reranked_results=reranked,
            crag_result=crag_result,
            metadata={
                "total_latency": total_latency,
                "retrieval_latency": retrieval_latency,
                "rerank_latency": rerank_latency,
                "generation_latency": crag_result.metadata.get("generation_latency", 0)
                if hasattr(crag_result, "metadata")
                else 0,
                "confidence": crag_result.confidence,
                "used_web_fallback": crag_result.used_web_fallback,
            },
        )

    def process_stream(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        namespace: str = "default",
        filter_dict: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        response = self.process(query, conversation_history, namespace, filter_dict)

        yield from self._stream_answer(response.answer)

    def _stream_answer(self, answer: str):
        words = answer.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    def _answer_directly(self, query: str, intent: IntentType) -> str:
        if intent == IntentType.GREETING:
            return "Hello! How can I help you today?"
        elif intent == IntentType.CREATIVE:
            return self._creative_response(query)
        elif intent == IntentType.CODE:
            return self._code_response(query)
        else:
            messages = [
                SystemMessage(
                    content="You are a helpful assistant. Answer the user's question directly without using external documents."
                ),
                HumanMessage(content=query),
            ]
            response = self.llm.invoke(messages)
            return response.content.strip()

    def _creative_response(self, query: str) -> str:
        messages = [
            SystemMessage(
                content="You are a creative assistant. Help with creative writing, brainstorming, and storytelling."
            ),
            HumanMessage(content=query),
        ]
        response = self.llm.invoke(messages)
        return response.content.strip()

    def _code_response(self, query: str) -> str:
        messages = [
            SystemMessage(
                content="You are a coding assistant. Help with programming questions, code generation, and debugging."
            ),
            HumanMessage(content=query),
        ]
        response = self.llm.invoke(messages)
        return response.content.strip()


def create_agentic_pipeline(
    hybrid_searcher: HybridSearcher,
    reranker: CrossEncoderReranker,
    config=None,
) -> AgenticRAGPipeline:
    return AgenticRAGPipeline(
        hybrid_searcher=hybrid_searcher,
        reranker=reranker,
        config=config,
    )
