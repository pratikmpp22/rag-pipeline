import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.pipeline import query_pipeline


def build_eval_dataset(rag_chain, vectorstore, bm25_index, bm25_chunks, llm, cfg):
    """Run pipeline for each test question, collect real answers and contexts."""
    questions = cfg["evaluation"]["test_questions"]
    ground_truths = cfg["evaluation"]["ground_truths"]

    answers = []
    contexts = []

    for q in questions:
        result = query_pipeline(
            q, vectorstore, bm25_index, bm25_chunks, rag_chain, llm, cfg
        )
        answers.append(result["answer"])
        # Extract page content from retrieved docs
        ctx = [doc.page_content for doc in result.get("docs", [])]
        contexts.append(ctx)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def run_evaluation(rag_chain, vectorstore, bm25_index, bm25_chunks, llm, cfg):
    """Run full RAGAS evaluation and print formatted results."""
    print("\n[*] Running RAGAS evaluation...")
    print("   Generating answers for test questions...\n")

    eval_data = build_eval_dataset(
        rag_chain, vectorstore, bm25_index, bm25_chunks, llm, cfg
    )

    dataset = Dataset.from_dict(eval_data)

    # Wrap LLM and embeddings for RAGAS
    ragas_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=cfg["llm"]["model"],
        temperature=0,
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model=cfg["embedding"]["model"],
    ))

    # Set llm/embeddings on pre-instantiated metric singletons
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    # Print metrics table
    pass_threshold = cfg["evaluation"]["pass_threshold"]
    needs_work_threshold = cfg["evaluation"]["needs_work_threshold"]

    print("\n" + "=" * 60)
    print("  RAGAS Evaluation Results")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Score':>8}  {'Status'}")
    print("-" * 60)

    # Extract aggregate scores from repr string (e.g. "{'faithfulness': 0.85}")
    # Use results.scores (list of per-question dicts) to compute averages
    metric_names = [m.name for m in metrics]
    avg_scores = {}
    for name in metric_names:
        values = [s.get(name, 0.0) for s in results.scores if name in s]
        avg_scores[name] = sum(values) / len(values) if values else 0.0

    for metric_name, score in avg_scores.items():
        if score >= pass_threshold:
            status = "[PASS]"
        elif score >= needs_work_threshold:
            status = "[NEEDS WORK]"
        else:
            status = "[FAILING]"
        print(f"  {metric_name:<25} {score:>8.4f}  {status}")

    print("=" * 60)

    # Per-question breakdown
    df = results.to_pandas()
    print("\n[*] Per-Question Breakdown:")
    print("-" * 60)

    questions = cfg["evaluation"]["test_questions"]
    for i, q in enumerate(questions):
        if i < len(df):
            row = df.iloc[i]
            print(f"\n  Q{i+1}: {q}")
            for name in metric_names:
                if name in df.columns:
                    val = row[name]
                    if isinstance(val, float):
                        flag = " [!]" if val < needs_work_threshold else ""
                        print(f"      {name}: {val:.4f}{flag}")

    print("\n" + "=" * 60)

    return results


if __name__ == "__main__":
    from src.config import get_config
    from src.ingestion import setup_pipeline_data
    from src.retrieval import build_bm25_index
    from src.pipeline import build_llm, build_rag_chain

    print("Setting up pipeline for evaluation...", flush=True)
    cfg = get_config()
    vectorstore, chunks = setup_pipeline_data(cfg)
    bm25_index, bm25_chunks = build_bm25_index(chunks)
    llm = build_llm(cfg)
    rag_chain = build_rag_chain(cfg)

    run_evaluation(rag_chain, vectorstore, bm25_index, bm25_chunks, llm, cfg)
