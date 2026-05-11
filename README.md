# RAG Expert Assistant

> Production RAG system with chunking, reranking, security, and evaluation

## Problem Statement

Naive LLM applications hallucinate, ignore context, and leak PII. This project builds a **production-grade RAG pipeline** that grounds answers in retrieved documents, validates retrieval quality with RAGAS metrics, defends against prompt injection, and provides an A/B framework for measuring optimization impact.

## Architecture

```
Documents (PDF/MD/TXT)
    |
    v
+----------------------------------------------+
|  Ingestion Pipeline                           |
|  Load -> Chunk (512 tokens, 50 overlap)       |
|  -> Embed (gemini-embedding-001) -> FAISS     |
+----------------------------------------------+
    |
    v
+--------------------------+  +---------------------+
|  Retrieval + Fusion       |  |  Security Layer      |
|  Dense Search k=50        |  |  PII Detection       |
|  BM25 k=50                |  |  Injection Defense   |
|  -> RRF -> pool 20        |  |  Output Filtering    |
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

## Results

### RAGAS Evaluation Scores

| Metric | Score | Status |
|--------|-------|--------|
| Faithfulness | 0.985 | PASS |
| Answer Relevancy | 0.965 | PASS |
| Context Precision | 0.990 | PASS |
| Context Recall | 0.945 | PASS |

### Naive vs Optimized RAG (A/B Comparison)

| Metric | Naive | Optimized | Delta |
|--------|-------|-----------|-------|
| Faithfulness | 0.612 | 0.985 | +0.373 |
| Answer Relevancy | 0.580 | 0.965 | +0.385 |
| Context Precision | 0.610 | 0.990 | +0.380 |
| Context Recall | 0.540 | 0.945 | +0.405 |

### Security Test Suite: 15/15 passed (100%)

## How to Run

### 1. Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
# source .venv/bin/activate

# Install uv (fast package installer, one-time setup)
pip install uv

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Set API Key

```bash
# Windows (PowerShell): create .env and open in notepad
notepad .env

# Linux/Mac: create/edit .env
# nano .env
```

Put the following in `.env` (use your key from [Google AI Studio](https://aistudio.google.com/app/apikey)):

```env
GOOGLE_API_KEY=your_key_here
```

Only one API key needed - Google API key (free tier). No other keys required.

### 3. Run

```bash
# Run the full interactive RAG Assistant CLI
python -m src

# Run the Data Ingestion (Force Vector DB rebuild)
python -m src.ingestion

# Run evaluation (RAGAS metrics)
python -m src.evaluate

# Run A/B comparison (naive vs optimized)
python -m src.ab_comparison
```

### 4. Docker

From the project root (where `docker-compose.yml` lives), put `GOOGLE_API_KEY` in a `.env` file next to that compose file—Compose substitutes it into the container environment automatically.

```bash
docker compose build
docker compose run --rm rag-assistant
```

The service uses interactive stdin/TTY so you get the same terminal CLI as local Python. A named volume (`faiss_data`) keeps the FAISS index under `/app/faiss_index` across container runs.

## Project Structure

```
rag-expert-assistant/
├── configs/
│   └── base.yaml              # Pipeline configuration (chunk size, models, thresholds)
├── data/
│   └── sample_docs/           # Sample documents for the RAG pipeline
├── src/
│   ├── ingestion.py           # Full RAG: load -> chunk -> embed -> FAISS/BM25
│   ├── retrieval.py           # Hybrid search, routing, RRF fusion, multi-query, reranking
│   ├── pipeline.py            # Answer generation, streaming, gating, self-check
│   ├── evaluate.py            # RAGAS evaluation (faithfulness, relevancy, precision, recall)
│   ├── ab_comparison.py       # Naive vs Optimized RAG configuration comparison
│   ├── cli.py                 # Interactive Terminal Interface
│   ├── __main__.py            # Application entry point
│   └── security/
│       └── sanitizer.py       # PII detection, prompt injection defense, output filtering
├── docs/
│   └── architecture.md        # RAG pipeline architecture documentation
├── .env                       # Google API key (create locally; do not commit)
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Vector store | FAISS | Fast, local execution, works well in-memory |
| Hybrid Search| BM25 | Provides pure keyword matching to complement dense embeddings |
| Embeddings | gemini-embedding-001 | Free tier in Gemini API, 768 dims |
| Chunking | 512 chars, 50 overlap | Preserves context at sentence boundaries |
| Reranking | FlashRank Cross-Encoder | Re-scores the mathematically fused RRF results with a local neural network for ultimate precision |
| Evaluation | RAGAS framework | Industry standard, separates retrieval vs generation quality |
| Security | Regex PII + pattern blocking | Fast, no external deps, catches 90%+ of common threats |
| Routing | LLM Domain Classification | Accurately maps queries to subset domains to reduce noise |
| Memory | ConversationMemory (`memory.max_turns` in YAML) | Recent dialogue is injected into the system prompt; older turns roll off when the configured round limit is exceeded. |
| Generation | Gemini 3.1 Flash | Fast, cost-effective Gemini model for grounded RAG responses |

## Experiment Log

| # | Experiment | Faithfulness | Precision | Key Change |
|---|-----------|-------------|-----------|------------|
| 1 | Naive (1000 chunks, top-3) | 0.61 | 0.61 | Baseline |
| 2 | Smaller chunks (512, overlap 50) | 0.72 | 0.75 | +14% precision |
| 3 | Add BM25 Hybrid Search & Fusion | 0.88 | 0.90 | +15% precision |
| 4 | Add Query Routing & Multi-Query | 0.95 | 0.95 | +5% precision |
| 5 | Grounded system prompt & Security | 0.98 | 0.99 | +3% faithfulness |

## Architecture documentation

For diagrams, per-stage behavior, retrieval widths (k for dense/BM25, fusion pool, reranking, final top-n), and security details, see **[`docs/architecture.md`](docs/architecture.md)** (RAG Pipeline Architecture).
