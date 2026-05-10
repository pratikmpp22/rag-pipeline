from dataclasses import dataclass, field

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from src.ingestion import load_documents, chunk_documents
from src.retrieval import build_bm25_index
from src.pipeline import build_rag_chain, build_llm, query_pipeline


@dataclass
class RAGConfig:
    """Configuration for an A/B test variant."""
    name: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    use_reranking: bool
    use_hybrid_search: bool
    use_multi_query: bool = False
    use_query_routing: bool = False


naive_config = RAGConfig(
    name="Naive RAG",
    chunk_size=1000,
    chunk_overlap=0,
    retriever_k=3,
    use_reranking=False,
    use_hybrid_search=False,
)

optimized_config = RAGConfig(
    name="Optimized RAG",
    chunk_size=512,
    chunk_overlap=50,
    retriever_k=20,
    use_reranking=True,
    use_hybrid_search=True,
    use_query_routing=True,
    use_multi_query=True,
)

_pipeline_cache = {}


def _build_pipeline(config, cfg):
    """Build vectorstore, BM25, chain for a given RAGConfig. Caches result."""
    if config.name in _pipeline_cache:
        return _pipeline_cache[config.name]

    # Override config values with this variant's settings
    variant_cfg = {
        **cfg,
        "chunking": {
            **cfg["chunking"],
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        },
        "retrieval": {
            **cfg["retrieval"],
            "top_k": config.retriever_k,
        },
        "features": {
            **cfg["features"],
            "use_reranking": config.use_reranking,
            "use_hybrid_search": config.use_hybrid_search,
            "use_multi_query": config.use_multi_query,
            "use_query_routing": config.use_query_routing,
            "use_confidence_gating": False,
            "use_self_check": False,
            "use_security": False,
            "use_memory": False,
        },
    }

    docs = load_documents(cfg["ingestion"]["data_dir"])
    chunks = chunk_documents(docs, variant_cfg)

    embeddings = GoogleGenerativeAIEmbeddings(model=cfg["embedding"]["model"])
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    bm25_index, bm25_chunks = build_bm25_index(chunks)
    llm = build_llm(cfg)
    rag_chain = build_rag_chain(variant_cfg)

    result = {
        "vectorstore": vectorstore,
        "bm25_index": bm25_index,
        "bm25_chunks": bm25_chunks,
        "llm": llm,
        "rag_chain": rag_chain,
        "cfg": variant_cfg,
        "chunks": chunks,
    }
    _pipeline_cache[config.name] = result
    return result


def _token_overlap_score(answer, ground_truth):
    """Simple token-overlap approximation of answer quality."""
    answer_tokens = set(answer.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    if not truth_tokens:
        return 0.0
    return len(answer_tokens & truth_tokens) / len(truth_tokens)


def evaluate_rag(config, test_questions, ground_truths, cfg):
    """Run pipeline for each question, compute token-overlap scores."""
    pipeline = _build_pipeline(config, cfg)

    scores = []
    for q, gt in zip(test_questions, ground_truths):
        result = query_pipeline(
            q,
            pipeline["vectorstore"],
            pipeline["bm25_index"],
            pipeline["bm25_chunks"],
            pipeline["rag_chain"],
            pipeline["llm"],
            pipeline["cfg"],
        )
        score = _token_overlap_score(result["answer"], gt)
        scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "config_name": config.name,
        "avg_score": avg_score,
        "per_question": scores,
        "num_chunks": len(pipeline["chunks"]),
    }


def run_ab_comparison(cfg):
    """Run A/B comparison between naive and optimized configs."""
    questions = cfg["evaluation"]["test_questions"]
    ground_truths = cfg["evaluation"]["ground_truths"]

    print("\n[*] A/B Comparison: Naive vs Optimized RAG")
    print("=" * 60)

    print(f"\n  Building Naive RAG pipeline...")
    naive_scores = evaluate_rag(naive_config, questions, ground_truths, cfg)

    print(f"  Building Optimized RAG pipeline...")
    opt_scores = evaluate_rag(optimized_config, questions, ground_truths, cfg)

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"  {'Metric':<25} {'Naive':>10} {'Optimized':>10} {'Delta':>10}")
    print("-" * 60)

    delta = opt_scores["avg_score"] - naive_scores["avg_score"]
    print(f"  {'Avg Token Overlap':<25} {naive_scores['avg_score']:>10.4f} {opt_scores['avg_score']:>10.4f} {delta:>+10.4f}")
    print(f"  {'Num Chunks':<25} {naive_scores['num_chunks']:>10} {opt_scores['num_chunks']:>10}")

    # Per-question comparison
    print("\n[*] Per-Question Scores:")
    print("-" * 60)
    for i, q in enumerate(questions):
        n_score = naive_scores["per_question"][i]
        o_score = opt_scores["per_question"][i]
        d = o_score - n_score
        marker = "[+]" if d > 0 else ("[=]" if d == 0 else "[-]")
        print(f"  Q{i+1}: {n_score:.3f} -> {o_score:.3f} ({d:+.3f}) {marker}")
        print(f"      {q[:60]}")

    # Summary
    print("\n" + "=" * 60)
    if delta > 0.05:
        print("  [PASS] PRODUCTION READY -- Optimized config shows clear improvement")
    elif delta > 0:
        print("  [WARN] MARGINAL -- Small improvement, consider further tuning")
    else:
        print("  [FAIL] NEEDS WORK -- Optimized config not outperforming naive")
    print("=" * 60 + "\n")

    return naive_scores, opt_scores

if __name__ == "__main__":
    from src.config import get_config
    
    cfg = get_config()
    print("Running A/B Comparison standalone...", flush=True)
    run_ab_comparison(cfg)
