# Design System & UI Guidelines — RAG-Engine

## 1. Visual Foundation

Clean technical-dashboard aesthetic built for Streamlit:

- **Canvas:** soft neutral background (`#f4f6f9`), dark slate text (`#2c3e50`) — easy on the eyes for long reading sessions.
- **Sidebar:** dark theme (`#1e293b`) with white text — visually separates configuration from conversation.
- **Accent:** high-contrast green (`#4CAF50`, hover `#45a049`) for primary actions and chat-border affordance.
- **Cards:** assistant turns render as white cards (`12px` radius, subtle shadow, `4px` green left border) so answers scan as discrete units.

All styling lives in one `<style>` block in `src/ui/components.py::init_page_config` — contributors add tokens there, never inline.

## 2. Typography & Layout Hierarchy

| Zone            | Role                         | Elements                                                                                              |
| --------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Header**      | Identity + capability signal | `Advanced RAG Assistant` title, active Groq model, pipeline badges (Hybrid + Rerank + CRAG)           |
| **Sidebar**     | Configuration & ingestion    | Usage meter → model/temperature → Scrape & Learn (URL / file) → Clear/Export → model caption          |
| **Chat canvas** | Primary interaction          | `st.chat_message` history, `st.chat_input` composer pinned at bottom with high-contrast input styling |

Hierarchy rule: **sidebar configures, canvas converses.** Never put ingestion controls in the canvas; never put conversation in the sidebar.

## 3. UI Components

### 3.1 Chat interface (`render_chat_interface`, `handle_user_input`)

- Native `st.chat_message("user" | "assistant")` bubbles.
- Assistant tokens stream word-by-word with a `▌` caret, then settle to final markdown — implemented in `src/ui/chat.py::stream_response`.
- Cached answers carry a `Served from cache` caption.

### 3.2 Source citation cards (`render_sources_expander`)

Each assistant answer with retrieved sources gets `st.expander("View Grounding Sources")`:

```
**<title>**                              Score 0.842
[🔗 Open Source](<url>)                  (metric widget)
<code snippet: first 500 chars>          (markdown code block)
Type: pdf | Added: 2026-09-30 | NS: default   (3-col captions)
```

`format_sources_for_display` normalizes `Document` metadata into `{source, title, doc_type, namespace, created_at, url, snippet, score}` — extend that dict, not the renderer, when adding fields.

### 3.3 Telemetry & token dashboard (`render_sidebar`, `render_metrics_panel`)

- **Sidebar usage meter:** `st.progress(question_count / MAX)` + caption `Questions: n/N | ~tokens used | remaining`.
- **Model/temperature selectors:** `llama-3.1-8b-instant` (default, cheap) vs `llama-3.3-70b-versatile`; temperature slider 0.0–1.0.
- **Per-answer metrics expander:** total / retrieval / rerank latency (ms), CRAG confidence, `web fallback` warning when triggered.

## 4. State Management & Notifications

| Mechanism                                               | Usage                                                                                                                                         |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `st.session_state.messages`                             | Full chat history `{role, content, sources?, metadata?, timestamp?}`; drives `format_conversation_history` (last 10 turns to rewriter/router) |
| `st.session_state.question_count` / `total_tokens_used` | Session budget enforcement                                                                                                                    |
| `@st.cache_resource initialize_system`                  | Pipeline singletons (embeddings, Pinecone, searcher, reranker) built once                                                                     |
| Toast warnings                                          | Rate-limit reached (`MAX_QUESTIONS_PER_SESSION`), oversize question truncation                                                                |
| Progress spinners                                       | `Scraping & Indexing…`, `Processing & Indexing…` during sidebar ingestion                                                                     |
| Error banners                                           | `st.error` + `st.stop()` on missing `GROQ`/`PINECONE`/`HUGGINGFACE` keys at startup; inline `st.error` on ingestion failures                  |
| `st.rerun()`                                            | After ingestion or clear-chat so new state paints immediately                                                                                 |

**Contributor rules:** keep all Streamlit calls inside `src/ui/`; keep `src/agent/` and `src/retrieval/` UI-free and unit-testable; persist new session keys via `init_session_state` defaults.
