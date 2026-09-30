import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    GREETING = "greeting"
    GENERAL = "general"
    KNOWLEDGE_QUERY = "knowledge_query"
    CREATIVE = "creative"
    CODE = "code"
    ANALYSIS = "analysis"


@dataclass
class IntentResult:
    intent: IntentType
    confidence: float
    reasoning: str
    should_retrieve: bool


class IntentRouter:
    SYSTEM_PROMPT = """You are an intent classifier for a RAG chatbot. Classify the user's input into one of these categories:

1. GREETING - Simple greetings, hellos, goodbyes, thanks
2. GENERAL - General conversation, opinions, chit-chat not requiring specific knowledge
3. KNOWLEDGE_QUERY - Questions requiring factual retrieval from documents (technical specs, documentation, facts)
4. CREATIVE - Creative writing, brainstorming, storytelling
5. CODE - Programming help, code generation, debugging
6. ANALYSIS - Analysis, comparison, reasoning tasks

Rules:
- If the query asks for specific facts, specs, or information likely in documents -> KNOWLEDGE_QUERY
- If it's a simple greeting/thanks -> GREETING
- If it's general chat without factual needs -> GENERAL
- If it asks to write code -> CODE
- If it asks for creative output -> CREATIVE
- If it asks for analysis/comparison -> ANALYSIS

Output ONLY a JSON object: {"intent": "CATEGORY", "confidence": 0.0-1.0, "reasoning": "brief explanation"}
"""

    GREETING_PATTERNS: ClassVar[list[str]] = [
        r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)",
        r"^(thanks|thank you|ty|appreciate it)",
        r"^(bye|goodbye|see you|farewell)",
        r"^(how are you|what's up|sup)",
    ]

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: str | None = None,
        temperature: float = 0.0,
        confidence_threshold: float = 0.7,
    ):
        self.llm = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=128,
        )
        self.confidence_threshold = confidence_threshold
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS
        ]

    def classify(
        self, query: str, conversation_history: list[dict[str, str]] | None = None
    ) -> IntentResult:
        quick_intent = self._quick_classify(query)
        if quick_intent:
            return quick_intent

        return self._llm_classify(query, conversation_history)

    def _quick_classify(self, query: str) -> IntentResult | None:
        query_lower = query.strip().lower()

        for pattern in self._compiled_patterns:
            if pattern.match(query_lower):
                if any(
                    g in query_lower
                    for g in [
                        "hi",
                        "hello",
                        "hey",
                        "greeting",
                        "morning",
                        "afternoon",
                        "evening",
                    ]
                ):
                    return IntentResult(
                        intent=IntentType.GREETING,
                        confidence=0.95,
                        reasoning="Matched greeting pattern",
                        should_retrieve=False,
                    )
                elif any(
                    g in query_lower
                    for g in ["thanks", "thank you", "ty", "appreciate"]
                ):
                    return IntentResult(
                        intent=IntentType.GREETING,
                        confidence=0.95,
                        reasoning="Matched thanks pattern",
                        should_retrieve=False,
                    )
                elif any(
                    g in query_lower for g in ["bye", "goodbye", "farewell", "see you"]
                ):
                    return IntentResult(
                        intent=IntentType.GREETING,
                        confidence=0.95,
                        reasoning="Matched goodbye pattern",
                        should_retrieve=False,
                    )

        return None

    def _llm_classify(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> IntentResult:
        messages = [SystemMessage(content=self.SYSTEM_PROMPT)]

        if conversation_history:
            for turn in conversation_history[-3:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))

        messages.append(HumanMessage(content=f"Query: {query}\n\nClassification:"))

        start_time = time.time()
        try:
            response = self.llm.invoke(messages)
            result = self._parse_response(response.content)
            elapsed = time.time() - start_time
            logger.info(
                f"Intent classified in {elapsed:.3f}s: {result.intent.value} (confidence: {result.confidence:.2f})"
            )
            return result
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return IntentResult(
                intent=IntentType.KNOWLEDGE_QUERY,
                confidence=0.5,
                reasoning=f"Classification failed, defaulting to knowledge query: {e}",
                should_retrieve=True,
            )

    def _parse_response(self, response: str) -> IntentResult:
        import json

        try:
            data = json.loads(response.strip())
            intent = IntentType(data.get("intent", "knowledge_query").lower())
            confidence = float(data.get("confidence", 0.7))
            reasoning = data.get("reasoning", "LLM classification")
        except (json.JSONDecodeError, ValueError, KeyError):
            intent = IntentType.KNOWLEDGE_QUERY
            confidence = 0.5
            reasoning = "Failed to parse LLM response, defaulting"

        return IntentResult(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            should_retrieve=(
                intent in [IntentType.KNOWLEDGE_QUERY, IntentType.ANALYSIS]
            ),
        )
