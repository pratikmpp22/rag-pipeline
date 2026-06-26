# RAG Expert Assistant

> Production RAG system with chunking, reranking, security, and evaluation

## Problem Statement

Naive LLM applications hallucinate, ignore context, and leak PII. This project builds a **production-grade RAG pipeline** that grounds answers in retrieved documents, validates retrieval quality with RAGAS metrics, defends against prompt injection, and provides an A/B framework for measuring optimization impact.

## Architecture

```
Documents (TXT / MD)
    |
    v
+----------------------------------------------+
|  Ingestion Pipeline                           |
|  Load -> Chunk (512 chars, 50 overlap)        |
|  -> Embed (see base.yaml) -> FAISS            |
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
```

## Results

### RAGAS Evaluation Scores

| Metric | Score | Status |
|--------|-------|--------|
| Faithfulness | 0.917 | PASS |
| Answer Relevancy | 0.903 | PASS |
| Context Precision | 0.881 | PASS |
| Context Recall | 0.862 | PASS |

### Naive vs Optimized RAG (A/B Comparison)

| Metric | Naive | Optimized | Delta |
|--------|-------|-----------|-------|
| Faithfulness | 0.581 | 0.917 | +0.336 |
| Answer Relevancy | 0.623 | 0.903 | +0.280 |
| Context Precision | 0.594 | 0.881 | +0.287 |
| Context Recall | 0.518 | 0.862 | +0.344 |

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
# Run the Interactive Streamlit Web UI (Primary)
streamlit run app.py

# Or run via Docker Compose
docker-compose up --build

# Or run the Terminal CLI
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
docker compose up --build
```

The service launches the Streamlit UI on port `8501`. Open `http://localhost:8501` in your browser. A named volume (`faiss_data`) keeps the FAISS index under `/app/faiss_index` across container restarts.

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
│   ├── pipeline.py            # Unified answer generation, streaming, gating, self-check
│   ├── memory.py              # HybridMemory with token budget and LLM summarization
│   ├── config.py              # Safe, reloadable, path-keyed configuration caching
│   ├── evaluate.py            # RAGAS evaluation (faithfulness, relevancy, precision, recall)
│   ├── ab_comparison.py       # Naive vs Optimized RAG configuration comparison
│   ├── cli.py                 # Interactive Terminal Interface
│   ├── __main__.py            # Application entry point
│   └── security/
│       └── sanitizer.py       # PII detection, prompt injection defense, output filtering
├── docs/
│   └── architecture.md        # RAG pipeline architecture documentation
├── .streamlit/
│   └── config.toml            # Streamlit configurations (e.g. file watcher overrides)
├── app.py                     # Streamlit Application Entry Point
├── Dockerfile                 # Docker image configuration for Streamlit app
├── docker-compose.yml         # Compose stack with persistent FAISS volume
├── .dockerignore              # Excluded files for docker build context
├── .gitignore                 # Git ignore file
├── .env                       # Google API key (create locally; do not commit)
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Vector store | FAISS | Fast, local execution, works well in-memory |
| UI Framework | Streamlit | Rapid reactive frontend with dynamic sidebar feature flags and a native Pipeline Stages transparency expander |
| Hybrid Search| BM25 | Provides pure keyword matching to complement dense embeddings |
| Embeddings | gemini-embedding-001 | Free tier in Gemini API, 768 dims |
| Chunking | 512 chars, 50 overlap | Preserves context at sentence boundaries |
| Reranking | FlashRank Cross-Encoder | Re-scores the mathematically fused RRF results with a local neural network for ultimate precision |
| Evaluation | RAGAS framework | Industry standard, separates retrieval vs generation quality |
| Security | Regex PII + pattern blocking | Fast, no external deps, catches 90%+ of common threats |
| Routing | LLM Domain Classification | Accurately maps queries to subset domains to reduce noise |
| Memory | HybridMemory (`memory.token_budget`) | Tracks token usage; summarizes older turns dynamically to prevent context overflow while preserving fidelity of recent exchanges |
| Generation | Gemini 3.1 Flash | Fast, cost-effective Gemini model for grounded RAG responses |

## Experiment Log

| # | Experiment | Faithfulness | Precision | Key Change |
|---|-----------|-------------|-----------|------------|
| 1 | Naive (1000 chunks, top-3) | 0.581 | 0.594 | Baseline |
| 2 | Smaller chunks (512, overlap 50) | 0.664 | 0.713 | +12% precision |
| 3 | Add BM25 Hybrid Search & Fusion | 0.762 | 0.812 | +10% precision |
| 4 | Add Query Routing & Multi-Query | 0.834 | 0.857 | +5% precision |
| 5 | Grounded system prompt & Security | 0.917 | 0.881 | +9% faithfulness |

## Architecture documentation

For diagrams, per-stage behavior, retrieval widths (k for dense/BM25, fusion pool, reranking, final top-n), and security details, see **[`docs/architecture.md`](docs/architecture.md)** (RAG Pipeline Architecture).
