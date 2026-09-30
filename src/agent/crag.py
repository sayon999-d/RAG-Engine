import logging
import time
from dataclasses import dataclass
from typing import Any

from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


@dataclass
class CRAGResult:
    answer: str
    sources: list[Document]
    confidence: float
    used_web_fallback: bool
    web_sources: list[dict[str, Any]]
    reasoning: str


class CRAGPipeline:
    EVALUATION_PROMPT = """You are an evaluator for a RAG system. Given a user query and retrieved documents, assess whether the documents contain sufficient information to answer the query accurately.

Rate the relevance on a scale of 0.0 to 1.0:
- 1.0: Documents directly and comprehensively answer the query
- 0.7-0.9: Documents contain most needed information with minor gaps
- 0.4-0.6: Documents have some relevant info but significant gaps
- 0.1-0.3: Documents barely relevant, mostly noise
- 0.0: Documents completely irrelevant

Consider:
1. Does the content directly address the query?
2. Is the information specific and accurate?
3. Are there critical missing pieces?

Output ONLY a JSON: {"score": 0.0-1.0, "reasoning": "brief explanation"}"""

    GENERATION_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the provided context documents. 

Rules:
1. If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer this question."
2. Cite sources using [Source: title] format
3. Be concise but comprehensive
4. Don't make up information not in the context

Context Documents:
{context}

Question: {query}

Answer:"""

    WEB_FALLBACK_PROMPT = """The provided documents were insufficient to answer the query. You may use your general knowledge to provide a helpful response, but clearly indicate when you're using general knowledge vs. retrieved information.

If you use general knowledge, start with: "Based on my general knowledge (not from retrieved documents):"

Question: {query}

Answer:"""

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        confidence_threshold: float = 0.3,
        web_fallback_enabled: bool = True,
    ):
        self.llm = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.eval_llm = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=128,
        )
        self.confidence_threshold = confidence_threshold
        self.web_fallback_enabled = web_fallback_enabled

    def evaluate_relevance(
        self, query: str, documents: list[Document]
    ) -> tuple[float, str]:
        if not documents:
            return 0.0, "No documents retrieved"

        context = "\n\n".join(
            [
                f"[Doc {i + 1}: {doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))}]\n{doc.page_content[:1000]}"
                for i, doc in enumerate(documents[:5])
            ]
        )

        messages = [
            SystemMessage(content=self.EVALUATION_PROMPT),
            HumanMessage(
                content=f"Query: {query}\n\nDocuments:\n{context}\n\nEvaluation:"
            ),
        ]

        try:
            response = self.eval_llm.invoke(messages)
            import json

            data = json.loads(response.content.strip())
            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            reasoning = data.get("reasoning", "No reasoning provided")
            return score, reasoning
        except Exception as e:
            logger.warning(f"Relevance evaluation failed: {e}")
            return 0.5, f"Evaluation error: {e}"

    def generate_answer(self, query: str, documents: list[Document]) -> str:
        if not documents:
            return "I don't have any relevant documents to answer this question."

        context = "\n\n".join(
            [
                f"[Source: {doc.metadata.get('title', doc.metadata.get('source', f'Doc {i + 1}'))}]\n{doc.page_content}"
                for i, doc in enumerate(documents)
            ]
        )

        messages = [
            HumanMessage(
                content=self.GENERATION_PROMPT.format(context=context, query=query)
            ),
        ]

        response = self.llm.invoke(messages)
        return response.content.strip()

    def generate_web_fallback(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        if not self.web_fallback_enabled:
            return "I don't have enough information to answer this question.", []

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo-search not installed, skipping web fallback")
            return "I don't have enough information to answer this question.", []

        web_results = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                for r in results:
                    web_results.append(
                        {
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": r.get("href", ""),
                        }
                    )
        except Exception as e:
            logger.error(f"Web search failed: {e}")

        if not web_results:
            return "I don't have enough information to answer this question.", []

        context = "\n\n".join(
            [f"[Web: {r['title']}] ({r['url']})\n{r['snippet']}" for r in web_results]
        )

        messages = [
            HumanMessage(
                content=self.WEB_FALLBACK_PROMPT.format(query=query)
                + f"\n\nWeb Results:\n{context}"
            ),
        ]

        response = self.llm.invoke(messages)
        return response.content.strip(), web_results

    def process(
        self,
        query: str,
        retrieved_docs: list[Document],
    ) -> CRAGResult:
        start_time = time.time()

        confidence, eval_reasoning = self.evaluate_relevance(query, retrieved_docs)

        if confidence >= self.confidence_threshold:
            answer = self.generate_answer(query, retrieved_docs)
            elapsed = time.time() - start_time
            logger.info(
                f"CRAG answered from docs (confidence: {confidence:.2f}) in {elapsed:.3f}s"
            )
            return CRAGResult(
                answer=answer,
                sources=retrieved_docs,
                confidence=confidence,
                used_web_fallback=False,
                web_sources=[],
                reasoning=f"Docs sufficient: {eval_reasoning}",
            )
        else:
            logger.info(
                f"CRAG confidence low ({confidence:.2f}), triggering web fallback"
            )
            web_answer, web_sources = self.generate_web_fallback(query)
            elapsed = time.time() - start_time
            return CRAGResult(
                answer=web_answer,
                sources=retrieved_docs,
                confidence=confidence,
                used_web_fallback=True,
                web_sources=web_sources,
                reasoning=f"Docs insufficient ({eval_reasoning}), used web fallback",
            )
