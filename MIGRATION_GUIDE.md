# Migration Guide: Naive RAG → Advanced RAG Engine

This guide walks you through migrating from the legacy single-file implementation to the new modular, production-grade RAG engine.

---

## New Directory Structure

```
RAG-Engine/
├── main.py                      # New entry point (replaces old main.py)
├── populate_index.py            # Legacy population script (still works)
├── requirements.txt             # Updated dependencies
├── .env.example                 # Complete configuration template
├── src/
│   ├── __init__.py
│   ├── config.py                # Centralized configuration
│   ├── hf_embeddings.py         # Unchanged (HuggingFace embeddings)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_parser.py   # Multi-format: PDF, DOCX, CSV, MD, TXT
│   │   ├── web_scraper.py       # Trafilatura-based clean scraping
│   │   ├── chunking.py          # Parent-child hierarchical chunking
│   │   └── pipeline.py          # Ingestion orchestration
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py     # Dense + Sparse (BM25) + RRF
│   │   ├── reranker.py          # Cross-encoder (BAAI/bge-reranker-base)
│   │   └── pinecone_client.py   # Pinecone wrapper with namespaces
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── query_rewriter.py    # Contextual query rewriting
│   │   ├── router.py            # Intent classification (6 types)
│   │   ├── crag.py              # Corrective RAG + Web fallback
│   │   └── pipeline.py          # Agentic orchestration
│   ├── observability/
│   │   ├── __init__.py
│   │   └── langfuse_client.py   # Langfuse tracing integration
│   └── ui/
│       ├── __init__.py
│       ├── components.py        # Streamlit UI components
│       └── chat.py              # Chat interface logic
```

---

## Step-by-Step Migration

### 1. Backup Current State

```bash
cp -r RAG-Engine RAG-Engine-backup
cd RAG-Engine
git commit -am "Backup before migration"
```

### 2. Install New Dependencies

```bash
pip install -r requirements.txt
```

**New packages added:**

- `pymupdf4llm`, `pymupdf` - PDF parsing with markdown output
- `python-docx` - DOCX parsing
- `pandas` - CSV parsing
- `trafilatura` - Clean web scraping
- `rank-bm25` - Sparse lexical search
- `sentence-transformers` - Local cross-encoder reranking
- `duckduckgo-search` - Web fallback
- `langfuse` - Observability

### 3. Update Environment Variables

Copy `.env.example` to `.env` and fill in all values:

```bash
cp .env.example .env
# Edit .env with your keys
```

**Critical new variables:**

```env
# Retrieval
DENSE_TOP_K=10
SPARSE_TOP_K=10
FINAL_TOP_K=3
RERANKER_MODEL=BAAI/bge-reranker-base
RERANKER_THRESHOLD=0.3

# Agentic
CRAG_CONFIDENCE_THRESHOLD=0.3
WEB_FALLBACK_ENABLED=true

# Observability
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

### 4. Pinecone Index Migration

**Option A: Fresh Index (Recommended for schema changes)**

```bash
# Delete old index in Pinecone console, then:
python populate_index.py  # Re-populates with new parent-child structure
```

**Option B: In-Place Upgrade (Preserve existing vectors)**
The new code uses `namespace` field in metadata. Your existing vectors will work but won't have parent-child linkage. To enable full features:

```python
# Run once to add namespace to existing vectors
from src.retrieval import create_pinecone_client
from src.hf_embeddings import HuggingFaceAPIEmbeddings
from src.config import get_config

config = get_config()
embeddings = HuggingFaceAPIEmbeddings(
    api_key=config.embedding.api_key, model_name=config.embedding.model_name
)
client = create_pinecone_client(
    config.pinecone.api_key, config.pinecone.index_name, embeddings
)

# Update metadata for existing vectors (run in batches)
# Note: Pinecone doesn't support direct metadata updates, so re-upload is needed
```

### 5. Deploy to Streamlit Community Cloud

1. Push updated repo to GitHub
2. In Streamlit Cloud dashboard:
   - Add all secrets from `.env` to **Advanced Settings → Secrets**
   - Ensure `requirements.txt` is at repo root
   - Set **Main file path** to `main.py`

**Required Secrets:**

```toml
GROQ_API_KEY = "your_key"
GROQ_MODEL = "llama-3.1-8b-instant"
HUGGINGFACE_API_KEY = "your_key"
PINECONE_API_KEY = "your_key"
PINECONE_INDEX_NAME = "rag-chatbot"
# Optional:
LANGFUSE_PUBLIC_KEY = "your_key"
LANGFUSE_SECRET_KEY = "your_key"
```

### 6. Verify Deployment

Run locally first:

```bash
streamlit run main.py
```

Check for:

- Sidebar shows "Dashboard" with usage metrics
- "Scrape & Learn" supports Web URL + File Upload (PDF, DOCX, CSV, MD, TXT)
- Chat shows streaming token-by-token response
- Expandable "View Grounding Sources" with scores & snippets
- Expandable "Performance Metrics" with latency breakdown

---

## Key Behavioral Changes

| Feature            | Legacy                  | New                                             |
| ------------------ | ----------------------- | ----------------------------------------------- |
| **Chunking**       | Single 1000-char chunks | Parent (1200) + Child (300) hierarchical        |
| **Search**         | Dense only (k=2)        | Hybrid Dense+Sparse (k=10) → RRF → Rerank (k=3) |
| **Query Handling** | Direct                  | Rewritten for context + Intent routing          |
| **Low Confidence** | Hallucinate             | CRAG evaluation → Web fallback or admit gap     |
| **File Types**     | TXT only                | PDF, DOCX, CSV, MD, TXT                         |
| **Web Scraping**   | BeautifulSoup (noisy)   | Trafilatura (clean article extraction)          |
| **Citations**      | Source list only        | Rich snippets + scores + metadata               |
| **Streaming**      | Blocking                | Token-by-token via `st.write_stream`            |
| **Observability**  | LangSmith (optional)    | Langfuse (optional, toggleable)                 |

---

## Testing the New Pipeline

```python
# Quick test script
from src.config import get_config
from src.ingestion import create_ingestion_pipeline
from src.retrieval import create_pinecone_client
from src.hf_embeddings import HuggingFaceAPIEmbeddings

config = get_config()
embeddings = HuggingFaceAPIEmbeddings(
    config.embedding.api_key, config.embedding.model_name
)
client = create_pinecone_client(
    config.pinecone.api_key, config.pinecone.index_name, embeddings
)

# Test ingestion
ingestion = create_ingestion_pipeline()
result = ingestion.ingest_url("https://example.com")
print(f"Ingestion: {result}")

# Test retrieval
from src.retrieval import HybridSearcher

searcher = HybridSearcher(embeddings, client.index, config.pinecone.namespace)
results = searcher.search("test query", top_k=5)
print(f"Found {len(results)} results")
```

---

## Troubleshooting

### "Module not found: src.xxx"

```bash
# Ensure you're running from project root
cd /path/to/RAG-Engine
streamlit run main.py
```

### "Pinecone dimension mismatch"

- Check `PINECONE_DIMENSION` matches your embedding model (384 for all-MiniLM-L6-v2)
- Recreate index if changed: `python populate_index.py`

### "BM25 not fitted"

- First query triggers lazy fitting from Pinecone
- Or manually call `hybrid_searcher.fit_bm25(docs)`

### "Reranker fails"

- Local: `pip install sentence-transformers`
- API: Requires `HUGGINGFACE_API_KEY` with inference permissions

### "Web fallback not working"

- Install: `pip install duckduckgo-search`
- Check `WEB_FALLBACK_ENABLED=true` in `.env`

### Streamlit Cloud deployment fails

- Verify all secrets in **Settings → Secrets** (TOML format)
- Check Python version (3.10+ recommended)
- View logs in Streamlit dashboard for specific errors

---

## Rollback Procedure

If issues arise:

```bash
git checkout HEAD~1 -- main.py populate_index.py requirements.txt .env.example
pip install -r requirements.txt
streamlit run main.py
```

Or restore from backup:

```bash
rm -rf RAG-Engine
mv RAG-Engine-backup RAG-Engine
```

---

## Support

- Check logs in `streamlit run main.py` output
- Enable `LANGFUSE_ENABLED=true` for detailed tracing
- Review `pinecone_client.get_index_stats()` for index health
- Verify all API keys have correct permissions

---

**Migration complete!** Your RAG engine now features hierarchical chunking, hybrid search, reranking, agentic routing, CRAG, and full observability.
