from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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
    retriever_k=20,
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


import time

def evaluate_rag(config, test_questions, ground_truths, cfg):
    """Run pipeline for each question, compute multiple metrics."""
    pipeline = _build_pipeline(config, cfg)

    overlaps = []
    latencies = []
    retrieved_counts = []
    sent_counts = []
    llm_tokens = []

    for q, gt in zip(test_questions, ground_truths):
        start_time = time.time()
        result = query_pipeline(
            q,
            pipeline["vectorstore"],
            pipeline["bm25_index"],
            pipeline["bm25_chunks"],
            pipeline["rag_chain"],
            pipeline["llm"],
            pipeline["cfg"],
        )
        end_time = time.time()
        
        score = _token_overlap_score(result["answer"], gt)
        overlaps.append(score)
        latencies.append(end_time - start_time)
        
        # Candidate pool size (fused count or top_k fallback)
        pool_size = result.get("telemetry", {}).get("fused_count")
        if not pool_size:
            pool_size = config.retriever_k
        retrieved_counts.append(pool_size)
        
        # Final chunks passed to LLM
        sent_counts.append(len(result["docs"]))
        
        # Estimate context tokens passed to LLM (4 chars per token roughly)
        token_count = sum(len(doc.page_content) // 4 for doc in result["docs"])
        llm_tokens.append(token_count)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "config_name": config.name,
        "avg_overlap": avg(overlaps),
        "avg_latency": avg(latencies),
        "avg_retrieved": avg(retrieved_counts),
        "avg_sent": avg(sent_counts),
        "avg_tokens": avg(llm_tokens),
        "per_question_overlap": overlaps,
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
    print("\n" + "=" * 90)
    print(f"  {'Metric':<45} {'Naive':>10} {'Optimized':>10} {'Delta':>10}")
    print("-" * 90)

    def print_row(name, val_n, val_o, is_float=False, is_time=False, is_approx=False):
        d = val_o - val_n
        if is_float:
            fmt_n, fmt_o, fmt_d = f"{val_n:.2f}", f"{val_o:.2f}", f"{d:+.2f}"
        elif is_time:
            fmt_n, fmt_o, fmt_d = f"{val_n:.1f}s", f"{val_o:.1f}s", f"{d:+.1f}s"
        else:
            fmt_n, fmt_o = f"{int(val_n)}", f"{int(val_o)}"
            fmt_d = f"{int(d):+d}"
            if is_approx:
                fmt_n = f"~{fmt_n}"
                fmt_o = f"~{fmt_o}"
        
        print(f"  {name:<45} {fmt_n:>10} {fmt_o:>10} {fmt_d:>10}")

    print_row("Num Chunks Retrieved (candidate pool)", naive_scores["avg_retrieved"], opt_scores["avg_retrieved"])
    print_row("Num Chunks Sent to LLM (post-rerank)", naive_scores["avg_sent"], opt_scores["avg_sent"])
    print_row("Avg Token Overlap (context, diagnostic only)", naive_scores["avg_overlap"], opt_scores["avg_overlap"], is_float=True)
    print_row("Avg Latency / query", naive_scores["avg_latency"], opt_scores["avg_latency"], is_time=True)
    print_row("Avg Tokens to LLM / query", naive_scores["avg_tokens"], opt_scores["avg_tokens"], is_approx=True)

    # Per-question comparison
    print("\n[*] Per-Question Scores (Overlap):")
    print("-" * 90)
    for i, q in enumerate(questions):
        n_score = naive_scores["per_question_overlap"][i]
        o_score = opt_scores["per_question_overlap"][i]
        d = o_score - n_score
        marker = "[+]" if d > 0 else ("[=]" if d == 0 else "[-]")
        print(f"  Q{i+1}: {n_score:.3f} -> {o_score:.3f} ({d:+.3f}) {marker}")
        print(f"      {q[:80]}")

    # Summary
    delta = opt_scores["avg_overlap"] - naive_scores["avg_overlap"]
    print("\n" + "=" * 90)
    if delta > 0.05:
        print("  [PASS] PRODUCTION READY -- Optimized config shows clear improvement")
    elif delta > 0:
        print("  [WARN] MARGINAL -- Small improvement, consider further tuning")
    else:
        print("  [FAIL] NEEDS WORK -- Optimized config not outperforming naive")
    print("=" * 90 + "\n")

    return naive_scores, opt_scores

if __name__ == "__main__":
    from src.config import get_config
    from dotenv import load_dotenv
    
    load_dotenv()
    cfg = get_config()
    print("Running A/B Comparison standalone...", flush=True)
    run_ab_comparison(cfg)
