# RAG Pipeline Architecture

This document describes the end-to-end architecture of the RAG Expert Assistant: document ingestion, hybrid retrieval with fusion and reranking, security controls, generation, and evaluation.

## Pipeline overview

High-level flow from documents through retrieval, security, generation, and evaluation:

```
Documents (TXT / MD)
    |
    v
+----------------------------------------------+
|  Ingestion Pipeline                           |
|  Load -> Chunk (512 chars, 50 overlap)        |
|  -> Embed (see base.yaml) -> FAISS           |
|  -> BM25 Index (Keyword Search)               |
+----------------------------------------------+
    |
    v
+--------------------------+  +---------------------+
|  Retrieval + Fusion       |  |  Security Layer      |
|  Dense @k=50 + BM25 @k=50 |  |  PII Detection       |
|  -> RRF -> pool of 20     |  |  Injection Defense   |
|  -> FlashRank -> Top-5    |  |  Output Filtering    |
+--------------------------+  +---------------------+
    |                                   |
    v                                   |
+-----------------------------------+   |
|  Unified Pipeline Core Generator  |<--+
|  (Grounded prompt, hybrid memory, |
|   citation extraction)            |
+-----------------------------------+
    |                   |
    v                   v
+---------------+  +-------------+
| Streamlit UI  |  | Terminal UI |
| (app.py)      |  | (cli.py)    |
+---------------+  +-------------+

## Request path (runtime)

Detailed stages from user query to final answer. The Streamlit UI actively listens to these stages and surfaces them in real-time under a "🔍 Pipeline Stages" expander, detailing sanitization blocks, routing decisions, multi-query expansions, and exact chunk counts retrieved by FAISS vs BM25.

```
User Query
    │
    ▼
┌──────────────────┐
│ Input Sanitizer  │  Prompt injection defense: strips phrases like
└────────┬─────────┘  "ignore previous instructions", "you are now …",
         │            fake `system:` roles, and `<|…|>`-style markers
         ▼            (regex removal before the LLM sees the query).
┌──────────────────┐
│ Query Router     │  Steers search toward hr, support, technical,
└────────┬─────────┘  or product via chunk metadata. Greetings get a
         │            short direct reply without document retrieval.
         │            Other questions use the full index when needed.
         ▼
┌──────────────────┐
│ Multi-Query      │  When `use_multi_query` is on: N paraphrases of the
│ Expansion        │  question so dense and BM25 see multiple surface
└────────┬─────────┘  forms (default config: single-query path).
         ▼
┌──────────────────────────────────────┐
│         Hybrid Retrieval             │
│  ┌────────────┐    ┌──────────────┐  │
│  │   Dense    │    │    BM25      │  │
│  │  k=50      │    │    k=50      │  │
│  └──────┬─────┘    └──────┬───────┘  │
│         └────────┬────────┘          │
│                  ▼                   │
│         ┌────────────────┐           │
│         │  RRF Fusion    │  merge ranks → take top 20 for reranking
│         └────────┬───────┘           │
│                  ▼                   │
│         ┌────────────────┐           │
│         │ FlashRank      │  cross-encoder rerank on those 20
│         │ (top 5 out)    │           │
│         └────────────────┘           │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────┐
│ Confidence Gate  │  Refuses if retrieval score < threshold. Dual thresholds
└────────┬─────────┘  exist: `confidence_threshold` (0.3) for reranking and
         ▼            `rrf_confidence_threshold` (0.01) for RRF.
┌──────────────────┐
│ RAG Generation   │  Grounded answer + citations; prior turns from
└────────┬─────────┘  HybridMemory (managed via `memory.token_budget`).
         ▼
┌──────────────────┐
│ Self-Check       │  If enabled: 2nd LLM call (YES/NO) whether the
└────────┬─────────┘  answer states anything not supported by context;
         │            on YES, appends a warning (see stage details).
         ▼
┌──────────────────┐
│ PII Filter       │  Redacts emails, phones, SSNs, cards (if security=ON).
└────────┬─────────┘  Note: Streams are buffered when this is active.
         ▼
    Final Answer + Source Citations
```

### Retrieval widths (configured in `configs/base.yaml`)

| Stage | Parameter | Default | Role |
|-------|-----------|---------|------|
| Dense (FAISS) | `retrieval.top_k` | 50 | Candidates per query variant from vector search. |
| BM25 | same `top_k` | 50 | Candidates per query variant from keyword search. |
| RRF merge | `hybrid_search.rrf_k` | 60 | Smoothing constant in `1 / (rrf_k + rank)` (not a hit count). |
| After fusion | `retrieval.fusion_top_n` | 20 | Top fused documents passed to the reranker. |
| After rerank | `retrieval.top_n` | 5 | Chunks sent to the generator. |

RRF scores every distinct chunk that appeared in any dense or BM25 list; the reranker only scores the top `fusion_top_n` by that fused ordering so cross-encoder cost stays bounded while still combining both retrieval paths.

## Stage details

### 1. Ingest

Documents are loaded from `data/sample_docs/` using LangChain's `DirectoryLoader`. The loader supports `.txt` and `.md` files.

- **Entry point**: `src/ingestion.py :: load_documents()`
- **Input**: Raw documents (TXT, MD)
- **Output**: List of LangChain `Document` objects with metadata

### 2. Chunk

Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`. The splitter respects semantic boundaries (paragraphs, sentences) to preserve context.

- **Entry point**: `src/ingestion.py :: chunk_documents()`
- **Configuration** (from `configs/base.yaml`):
  - `chunk_size`: 512 characters
  - `chunk_overlap`: 50 characters
  - `separators`: `["\n\n", "\n", ". ", " "]`
- **Output**: List of chunked `Document` objects

### 3. Embed and index

Each chunk is embedded using Google's embedding model from config. Embeddings are stored in FAISS for dense search, while a parallel BM25 index is built for keyword search.

- **Entry point**: `src/ingestion.py :: build_vectorstore()`
- **Model**: `configs/base.yaml` → `embedding.model` (e.g. `models/gemini-embedding-001`)
- **Vector store**: FAISS (local filesystem) and BM25 (in-memory)
- **Output**: Populated FAISS and BM25 indexes

### 4. Retrieve, fuse, and rerank (hybrid search)

For each query (original plus any multi-query variants), the pipeline runs dense and BM25 search at `top_k`, merges all ranked lists with reciprocal rank fusion, keeps the top `fusion_top_n` fused documents, reranks them with FlashRank when enabled, and returns the top `top_n` for generation.

- **Entry point**: `src/retrieval.py :: hybrid_retrieve()`
- **Search**: Dense similarity (FAISS) + BM25
- **Fusion**: RRF across all per-query/per-modality lists
- **Reranker**: FlashRank (`ms-marco-MiniLM-L-12-v2`) on the fused pool
- **Output**: Top-`top_n` documents for the prompt

Multi-query is controlled by `features.use_multi_query` in `configs/base.yaml`. When enabled, the LLM produces additional phrasings (`multi_query.num_variants`) so retrieval is less sensitive to a single wording.

### 5. Generate

The LLM generates a grounded response using the retrieved context. The system prompt enforces citation rules and context-only answering to reduce hallucination.

- **Entry point**: `src/pipeline.py :: stream_query_pipeline()`
- **Model**: from `configs/base.yaml` → `llm.model`
- **Prompt strategy**: Grounded system prompt with `[Source N]` citation format
- **Conversation memory**: `HybridMemory` (`src/memory.py`) keeps recent user and assistant messages and injects them into the system prompt’s `{history}` slot so follow-ups stay coherent. `memory.token_budget` in `configs/base.yaml` caps how many tokens the history can consume. When a new turn would exceed that cap, the **oldest** user+assistant pair is automatically dropped from the verbatim history and dynamically summarized by an LLM into a running summary. This bounds prompt growth intelligently while preserving critical context indefinitely.
- **Security**: Input sanitization before query, output PII filtering after generation
- **Output**: Streamed answer with citations

### 5b. Self-check (optional guard)

When `features.use_self_check` is true in `configs/base.yaml`, after the main answer is produced `run_self_check` in `src/pipeline.py` issues a **second** call to the same chat LLM. The prompt is a short binary check: does the answer contain any claim **not** supported by the retrieved context string? The model must reply with YES or NO only. If the response contains `YES`, a fixed warning line is appended to the answer so the user knows some statements might not be fully grounded. This is a lightweight heuristic, not a full claim-by-claim audit.

### 6. Evaluate

The RAGAS framework evaluates pipeline quality across four metrics. An A/B comparison framework measures naive vs optimized settings.

- **Entry points**: `src/evaluate.py`, `src/ab_comparison.py`
- **Metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall

## Security layer

The security module (`src/security/sanitizer.py`) provides defense in depth:

1. **Input sanitization** (`sanitize_input`): Removes known prompt-injection patterns (instruction overrides, role hijacks, delimiter tricks) before the LLM runs.
2. **Output filtering** (`filter_output_pii`): Redacts PII categories detected via regex (emails, phones, SSNs, credit-card-like spans). Note: Name detection is explicitly ignored to prevent false positives in the output filter.

## Domain routing

Document chunks carry a `domain` tag set at ingest time (`src/ingestion.py` → `DOMAIN_MAP`, defaulting to **none**). Those tags should align with router labels **hr**, **support**, **technical**, and **product** in `configs/base.yaml` so filters return results. When the router assigns one of these labels, **FAISS** uses LangChain's metadata filter on that field; **BM25** ranks a wider candidate set then keeps up to `top_k` hits whose `domain` metadata matches, so both branches respect the same slice before RRF.

**Greetings** (hello, small talk, etc.) are detected upstream of retrieval: the pipeline answers with a brief, polite reply **without** querying the vector index.

For other user messages—when the question does not map cleanly to a single slice—the pipeline runs retrieval **without** a domain filter so the whole indexed corpus can surface. The model is still constrained to the retrieved passages; if they do not support an answer, it is instructed to respond along the lines of *not having enough information in the provided documents* rather than inventing facts.


## Project layout

```
rag-expert-assistant/
├── configs/base.yaml            # Pipeline configuration
├── data/sample_docs/            # Source documents
├── src/
│   ├── ingestion.py             # Load -> chunk -> embed -> index
│   ├── retrieval.py             # Hybrid search, routing, RRF, multi-query, reranking
│   ├── pipeline.py              # Full streaming RAG pipeline generator
│   ├── memory.py                # Conversation history buffer
│   ├── evaluate.py              # RAGAS evaluation metrics
│   ├── ab_comparison.py         # Naive vs Optimized config tester
│   ├── cli.py                   # Interactive terminal interface
│   ├── __main__.py              # Entry point
│   └── security/
│       └── sanitizer.py         # PII detection & prompt injection defense
├── app.py                       # Streamlit Application Entry Point
├── Dockerfile                   # Docker image configuration for Streamlit app
├── docker-compose.yml           # Compose stack with persistent FAISS volume
├── .dockerignore                # Excluded files for docker build context
├── .gitignore                   # Git ignore file
├── docs/architecture.md         # This file
├── requirements.txt             # Dependencies
└── README.md                    # Project overview
```
