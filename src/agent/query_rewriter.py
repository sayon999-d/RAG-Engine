import logging
import time
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


@dataclass
class RewrittenQuery:
    original_query: str
    rewritten_query: str
    confidence: float
    reasoning: str


class QueryRewriter:
    SYSTEM_PROMPT = """You are a query rewriter for a RAG system. Your task is to rewrite user queries into standalone, self-contained search questions that can be used to retrieve relevant documents from a vector database.

Rules:
1. Resolve pronouns, references, and context-dependent phrases using conversation history
2. Expand abbreviations and domain-specific terms
3. Make the query specific and detailed enough for semantic search
4. Preserve the original intent and scope
5. Output ONLY the rewritten query, nothing else

Examples:
- History: "Tell me about AMD Radeon 7900 XTX" -> Current: "What are its specs?" -> Rewritten: "What are the specifications of AMD Radeon RX 7900 XTX?"
- History: "How does DirectML work?" -> Current: "Can I use it with ONNX?" -> Rewritten: "Can ONNX Runtime be used with Microsoft DirectML?"
- Current: "Compare 7900 XTX and 4080" -> Rewritten: "Compare AMD Radeon RX 7900 XTX and NVIDIA GeForce RTX 4080 specifications and performance"
"""

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.llm = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def rewrite(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
        max_history_turns: int = 5,
    ) -> RewrittenQuery:
        if not conversation_history:
            return RewrittenQuery(
                original_query=query,
                rewritten_query=query,
                confidence=1.0,
                reasoning="No conversation history to rewrite from",
            )

        history_text = self._format_history(conversation_history[-max_history_turns:])

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(
                content=f"Conversation History:\n{history_text}\n\nCurrent Query: {query}\n\nRewritten Query:"
            ),
        ]

        start_time = time.time()
        try:
            response = self.llm.invoke(messages)
            rewritten = response.content.strip()
            elapsed = time.time() - start_time

            confidence = self._calculate_confidence(query, rewritten)

            logger.info(
                f"Query rewritten in {elapsed:.3f}s: '{query}' -> '{rewritten}'"
            )
            return RewrittenQuery(
                original_query=query,
                rewritten_query=rewritten,
                confidence=confidence,
                reasoning=f"Resolved context from {len(conversation_history)} previous turns",
            )
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return RewrittenQuery(
                original_query=query,
                rewritten_query=query,
                confidence=0.5,
                reasoning=f"Rewrite failed: {e}",
            )

    def _format_history(self, history: list[dict[str, str]]) -> str:
        formatted = []
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted)

    def _calculate_confidence(self, original: str, rewritten: str) -> float:
        if original == rewritten:
            return 1.0
        orig_words = set(original.lower().split())
        rew_words = set(rewritten.lower().split())
        overlap = len(orig_words & rew_words) / max(len(orig_words), 1)
        expansion = min(len(rew_words) / max(len(orig_words), 1), 2.0)
        return min(0.5 + 0.3 * overlap + 0.2 * (expansion - 1), 1.0)
