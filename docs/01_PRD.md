# Product Requirements Document — RAG-Engine

**Version:** 2.0 · **Status:** Implemented · **Stack:** Streamlit + Pinecone (Serverless) + Groq + Hugging Face

## 1. Executive Summary & Value Proposition

RAG-Engine v1 was a naive single-turn RAG demo: plain-text chunking (1000 chars), dense-only retrieval at $k=2$, and a blocking `RetrievalQA` chain. It hallucinated on out-of-corpus queries and could only ingest `.txt` and noisy BeautifulSoup scrapes.

RAG-Engine v2 is a **multi-format, hybrid-search, agentic retrieval engine**:

- **Ingestion:** `.pdf` (via `pymupdf4llm`, tables preserved as markdown), `.docx`, `.txt`, `.csv`, `.md`, plus noise-free web scraping via `trafilatura`.
- **Retrieval:** Hybrid dense (Hugging Face `all-MiniLM-L6-v2`) + sparse (BM25) search fused with Reciprocal Rank Fusion, retrieving $k=10$ candidates reranked by a cross-encoder (`BAAI/bge-reranker-base`) down to the top 2–3 parent passages.
- **Agentic logic:** Intent routing (retrieval skipped for greetings), conversation-aware query rewriting, and Corrective RAG — low-confidence retrievals trigger a DuckDuckGo web fallback or an explicit admission of missing context instead of hallucinating.
- **UX:** Token-by-token streaming (`st.write_stream`), expandable grounding citations with relevance scores, and optional Langfuse tracing.

**Value proposition:** zero-setup, zero-cost knowledge assistant — drop in API keys for three free tiers, ingest documents or URLs from the sidebar, and get grounded, cited answers over a technical knowledge base.

## 2. Target User Personas

| Persona                  | Needs                                                     | Zero-setup story                                                          |
| ------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Technical researcher** | Query GPU/driver/ML docs across PDFs and vendor pages     | Paste URLs in sidebar, ask spec questions, inspect citations              |
| **Support engineer**     | Grounded answers over product docs without hallucinations | Upload `.pdf`/`.docx` manuals, get cited answers with confidence gating   |
| **Hobbyist / student**   | Free-tier chatbot over personal notes                     | `.txt`/`.md` uploads, `llama-3.1-8b-instant` default keeps Groq quota low |

All personas share one constraint: **no infrastructure to manage** — Streamlit Community Cloud hosting, Pinecone serverless, Groq and Hugging Face free tiers.

## 3. Functional Requirements

### 3.1 Ingestion (`src/ingestion/`)

| ID    | Requirement                                                                                                                                         |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| ING-1 | Parse `.pdf` via `pymupdf4llm.to_markdown`, preserving tables as markdown; capture page count, title, author                                        |
| ING-2 | Parse `.docx` (paragraphs + tables flattened to `\|`-joined rows), `.csv` (via pandas → markdown table), `.md`/`.txt` (UTF-8 with latin-1 fallback) |
| ING-3 | Scrape URLs with `trafilatura` (markdown output, tables + formatting, comments excluded); reject <100-char extractions                              |
| ING-4 | SSRF guard: validate scheme, resolve hostname, reject private/loopback/link-local IPs                                                               |
| ING-5 | Hierarchical chunking: **child 200–300 tokens** (embedded, indexed) / **parent 1000–1200 tokens** (generation context), markdown-aware separators   |
| ING-6 | Tag every chunk: `source`, `doc_type`, `chunk_id`/`child_id`, `parent_id`, `created_at`, `namespace`                                                |

### 3.2 Retrieval (`src/retrieval/`)

| ID    | Requirement                                                                                              |
| ----- | -------------------------------------------------------------------------------------------------------- |
| RET-1 | Dense retrieval over Pinecone (cosine, 384-dim) at $k=10$ per namespace                                  |
| RET-2 | Sparse BM25 retrieval at $k=10$ over the indexed corpus                                                  |
| RET-3 | Merge via Reciprocal Rank Fusion (RRF, $k=60$): $\mathrm{RRF}(d)=\sum_{r}\frac{1}{k+\mathrm{rank}_r(d)}$ |
| RET-4 | Cross-encoder rerank (`BAAI/bge-reranker-base`, local or HF API) to final top $k=2$–$3$, threshold $0.3$ |
| RET-5 | Parent-passage lookup: feed parent chunks (not child fragments) to the generator                         |

### 3.3 Agentic Logic (`src/agent/`)

| ID   | Requirement                                                                                                                                                                  |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AG-1 | **Intent router:** `GREETING/GENERAL` → direct LLM answer, no retrieval; `KNOWLEDGE_QUERY`/`ANALYSIS` → full pipeline. Fast regex path for greetings, LLM fallback otherwise |
| AG-2 | **Query rewriter:** `llama-3.1-8b-instant` (temp 0.0) resolves pronouns/references against last 5 turns into a standalone search query                                       |
| AG-3 | **Corrective RAG:** LLM-judged relevance score; below `CRAG_CONFIDENCE_THRESHOLD` (default $0.3$) → DuckDuckGo fallback or explicit "insufficient information" admission     |

### 3.4 UI/UX (`src/ui/`, `main.py`)

| ID   | Requirement                                                                                       |
| ---- | ------------------------------------------------------------------------------------------------- |
| UI-1 | Streaming responses via Groq streaming + `st.write_stream`                                        |
| UI-2 | `st.expander("View Grounding Sources")` per answer: snippet, title, similarity score, source link |
| UI-3 | `st.expander("Performance Metrics")`: retrieval/rerank latency, confidence, web-fallback flag     |
| UI-4 | Sidebar: usage progress, model/temperature selectors, URL + multi-format upload, clear/export     |
| UI-5 | Missing-key error banners at startup; rate-limit toast at `MAX_QUESTIONS_PER_SESSION`             |

## 4. Non-Functional Requirements

| Category      | Requirement                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------------ |
| Latency       | Retrieval + rerank p50 < 3 s; first streamed token < 5 s on free tiers                                             |
| Cost          | $0 footprint: Groq free tier, Pinecone serverless free tier, HF Inference API                                      |
| Token budget  | Session cap (`MAX_QUESTIONS_PER_SESSION=50`), response cache by query hash, rewrite/intent calls on cheap 8B model |
| Deployability | Single `streamlit run main.py`; secrets via `.env` locally or Streamlit secrets in cloud                           |
| Observability | Langfuse tracing behind `LANGFUSE_ENABLED` toggle; no overhead when off                                            |

## 5. Success Metrics & KPIs

| Metric                 | Definition                                                      | Target                                |
| ---------------------- | --------------------------------------------------------------- | ------------------------------------- |
| Retrieval MRR          | Mean reciprocal rank of gold chunk in fused top-10              | ≥ 0.7 on eval set                     |
| Precision@3            | Relevant docs in final reranked top-3                           | ≥ 0.8                                 |
| Hallucination rate     | Ungrounded claims per 100 answers (sampled audit)               | Reduce ≥ 50% vs v1 (CRAG + citations) |
| Token efficiency       | Avg total tokens per answered query (rewrite + intent + gen)    | ≤ 2.5k                                |
| Web-fallback precision | Fallback answers rated helpful                                  | ≥ 0.6                                 |
| Uptime                 | Successful Streamlit Cloud deploys without secret/config errors | 99% of sessions init cleanly          |
