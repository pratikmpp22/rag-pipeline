# Architecture

## System Overview

The RAG Expert Assistant implements a multi-stage retrieval and generation pipeline designed for grounded, citation-backed answers.

## Pipeline Flow

```
User Query
    │
    ▼
┌─────────────────┐
│ Input Sanitizer  │  Prompt injection defense
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Router     │  Classifies into hr/technical/product domain
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Multi-Query      │  Generates N query variants (optional)
│ Expansion        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         Hybrid Retrieval            │
│  ┌──────────┐    ┌──────────────┐   │
│  │  Dense    │    │   BM25       │   │
│  │  Vector   │    │   Keyword    │   │
│  │  Search   │    │   Search     │   │
│  └─────┬─────┘    └──────┬──────┘   │
│        │                 │          │
│        └────────┬────────┘          │
│                 ▼                   │
│        ┌──────────────┐             │
│        │  RRF Fusion  │             │
│        └──────────────┘             │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────┐
│ Confidence Gate  │  Refuses if retrieval score < threshold
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RAG Generation   │  LLM generates grounded answer with citations
│ + Memory         │  Conversation history injected into prompt
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Self-Check       │  Verifies no unsupported claims
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PII Filter       │  Redacts emails, phones, SSNs, credit cards
└────────┬────────┘
         │
         ▼
    Final Answer
    + Source Citations
# RAG Pipeline Architecture

This document describes the end-to-end architecture of the RAG Expert Assistant, covering each stage from document ingestion through evaluation.

## Pipeline Overview

```
Documents (PDF/MD/TXT)
    |
    v
+----------------------------------------------+
|  Ingestion Pipeline                           |
|  Load -> Chunk (512 tokens, 50 overlap)       |
|  -> Embed (gemini-embedding-exp-03-07)->FAISS |
|  -> BM25 Index (Keyword Search)               |
+----------------------------------------------+
    |
    v
+--------------------------+  +---------------------+
|  Retrieval + Fusion       |  |  Security Layer      |
|  Dense Search (FAISS)     |  |  PII Detection       |
|  Keyword Search (BM25)    |  |  Injection Defense   |
|  -> RRF Fusion -> Top-20  |  |  Output Filtering    |
|  -> FlashRank -> Top-5    |  +---------------------+
+--------------------------+
    |                                   |
    v                                   |
+--------------------------+            |
|  Generation (Gemini 3.1   |<----------+
|  Flash)                   |
|  Grounded prompt          |
|  + Citation extraction    |
+--------------------------+
    |
    v
+--------------------------+
|  Evaluation (RAGAS)       |
|  Faithfulness | Relevancy |
|  Precision | Recall       |
|  A/B: Naive vs Optimized  |
+--------------------------+
```

## Stage Details

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

### 3. Embed & Index

Each chunk is embedded into vectors using Google's `gemini-embedding-exp-03-07` model. Embeddings are stored in FAISS for dense vector search, while a parallel BM25 index is built for keyword search.

- **Entry point**: `src/ingestion.py :: build_vectorstore()`
- **Model**: `gemini-embedding-exp-03-07`
- **Vector store**: FAISS (local filesystem) and BM25 (in-memory)
- **Output**: Populated FAISS and BM25 indexes

### 4. Retrieve & Fuse & Rerank (Hybrid Search + Cross-Encoder)

Given a user query, the system generates multi-query variants and executes parallel searches against FAISS (semantic) and BM25 (keyword). The results are fused using Reciprocal Rank Fusion (RRF) to get the best 20 candidates.
Finally, a local Cross-Encoder model (`FlashRank`) evaluates the exact text of the query against the text of the 20 candidates to re-score and select the top 5 most relevant documents.

- **Entry point**: `src/retrieval.py :: hybrid_retrieve()`
- **Search type**: Dense Cosine Similarity + BM25 Keyword Search
- **Fusion**: RRF (Reciprocal Rank Fusion)
- **Reranker**: FlashRank (`ms-marco-MiniLM-L-12-v2`)
- **Output**: Top-5 reranked documents

### 5. Generate

The LLM generates a grounded response using the retrieved context. The system prompt enforces citation rules, and context-only answering to minimize hallucination.

- **Entry point**: `src/pipeline.py :: stream_query_pipeline()`
- **Model**: `gemini-3.1-flash`
- **Prompt strategy**: Grounded system prompt with `[Source N]` citation format
- **Security**: Input sanitization before query, output PII filtering after generation
- **Output**: Streamed answer with citations

### 6. Evaluate

The RAGAS framework evaluates pipeline quality across four independent metrics, each isolating a different failure mode. An A/B comparison framework measures the impact of each optimization.

- **Entry points**:
  - `src/evaluate.py` - RAGAS metrics
  - `src/ab_comparison.py` - Naive vs Optimized
- **Metrics**:
  - **Faithfulness**: Does the answer stick to the retrieved context?
  - **Answer Relevancy**: Does the answer address the question?
  - **Context Precision**: Are retrieved chunks relevant to the question?
  - **Context Recall**: Did retrieval find all relevant information?

## Security Layer

The security module (`src/security/sanitizer.py`) provides defense-in-depth across the pipeline:

1. **Input sanitization** (`sanitize_input`): Blocks known prompt injection patterns before they reach the LLM.
2. **Output filtering** (`filter_output_pii`): Redacts any PII (Emails, Names, SSNs, Phones) that appears in LLM responses before returning to the user.

## Domain Routing

The system includes an intelligent query router that classifies incoming questions into specific domains (`hr`, `technical`, `support`, `product`, `greeting`) to apply targeted metadata filters during retrieval.

## Project Layout

```
rag-expert-assistant/
├── configs/base.yaml            # Pipeline configuration
├── data/sample_docs/            # Source documents
├── src/
│   ├── ingestion.py             # Load -> chunk -> embed -> index
│   ├── retrieval.py             # Hybrid search, Multi-query, RRF fusion, Routing
│   ├── pipeline.py              # Full streaming RAG pipeline generator
│   ├── memory.py                # Conversation history buffer
│   ├── evaluate.py              # RAGAS evaluation metrics
│   ├── ab_comparison.py         # Naive vs Optimized config tester
│   ├── cli.py                   # Interactive terminal interface
│   ├── __main__.py              # Entry point
│   └── security/
│       └── sanitizer.py         # PII detection & prompt injection defense
├── docs/architecture.md         # This file
├── .env.example                 # API key template
├── requirements.txt             # Dependencies
└── README.md                    # Project overview
```
