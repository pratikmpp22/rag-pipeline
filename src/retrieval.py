from collections import defaultdict

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from pydantic import BaseModel, Field

try:
    from flashrank import Ranker, RerankRequest
    # Initialize singleton ranker to avoid reloading the model
    _RANKER = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
except ImportError:
    _RANKER = None


def build_bm25_index(chunks):
    """Tokenize chunks and build BM25Okapi index. Return (index, chunks)."""
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    index = BM25Okapi(tokenized)
    return index, chunks


def bm25_search(index, chunks, query, k=5):
    """Search BM25 index, return top-k Document list."""
    tokenized_query = query.lower().split()
    results = index.get_top_n(tokenized_query, chunks, n=k)
    return results


def generate_query_variants(question, llm, n=2):
    """Generate n rephrasings of the question via LLM. Return list including original."""
    prompt = (
        f"Generate {n} alternative phrasings of this question. "
        f"Return only the rephrased questions, one per line.\n\n"
        f"Question: {question}"
    )
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join(str(c) if not isinstance(c, dict) else c.get("text", "") for c in content)
        
    variants = [line.strip() for line in content.strip().split("\n") if line.strip()]
    # Prepend original question
    return [question] + variants[:n]


def reciprocal_rank_fusion(ranked_lists, rrf_k=60):
    """Fuse multiple ranked lists using RRF scoring. Return sorted Document list."""
    scores = defaultdict(float)
    doc_map = {}

    for ranked_list in ranked_lists:
        for i, doc in enumerate(ranked_list):
            # Deduplicate key: source + first 50 chars of content
            source = doc.metadata.get("source", "unknown")
            key = f"{source}:{doc.page_content[:50]}"
            scores[key] += 1.0 / (rrf_k + i + 1)
            doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    
    # Attach RRF score to metadata
    final_docs = []
    for k in sorted_keys:
        doc = doc_map[k]
        doc.metadata["relevance_score"] = scores[k]
        final_docs.append(doc)
        
    return final_docs


def classify_domain(question, llm, cfg):
    """Classify question into a domain via LLM. Return domain string or None."""
    domains = cfg["query_routing"]["domains"]
    domain_desc = "\n".join([f"- {d['name']}: {d['description']}" for d in domains])
    domain_names = [d["name"] for d in domains]

    prompt = (
        f"Classify this question into exactly one domain. "
        f"Reply with the domain name only, one word.\n\n"
        f"Domains:\n{domain_desc}\n\n"
        f"Question: {question}\n\n"
        f"Domain:"
    )

    try:
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = "".join(str(c) if not isinstance(c, dict) else c.get("text", "") for c in content)
            
        domain = content.strip().lower()
        if domain in domain_names and domain != "none":
            return domain
    except Exception:
        pass

    return None  # Search all domains on failure or if domain is 'none'


def hybrid_retrieve(question, vectorstore, bm25_index, bm25_chunks, cfg, llm=None, domain_filter=None, queries=None):
    """Orchestrate retrieval: dense+BM25 and RRF fusion using pre-computed routing/variants."""
    top_k = cfg["retrieval"]["top_k"]
    top_n = cfg["retrieval"]["top_n"]
    rrf_k = cfg["hybrid_search"]["rrf_k"]

    # 1. Fallback if not pre-computed
    if not domain_filter and cfg["features"]["use_query_routing"] and llm:
        domain = classify_domain(question, llm, cfg)
        if domain:
            domain_filter = {"domain": domain}

    if not queries:
        if cfg["features"]["use_multi_query"] and llm:
            queries = generate_query_variants(question, llm, n=cfg["multi_query"]["num_variants"])
        else:
            queries = [question]

    # 3. Retrieve from both indexes for each query
    all_ranked_lists = []

    for q in queries:
        # Dense vector search
        search_kwargs = {"k": top_k}
        if domain_filter:
            search_kwargs["filter"] = domain_filter

        try:
            dense_results = vectorstore.similarity_search(q, **search_kwargs)
        except Exception:
            dense_results = vectorstore.similarity_search(q, k=top_k)

        all_ranked_lists.append(dense_results)

        # BM25 search
        if cfg["features"]["use_hybrid_search"]:
            bm25_results = bm25_search(bm25_index, bm25_chunks, q, k=top_k)
            all_ranked_lists.append(bm25_results)

    # 4. RRF fusion
    fused = reciprocal_rank_fusion(all_ranked_lists, rrf_k=rrf_k)

    # 5. Cross-Encoder Reranking
    if cfg["features"].get("use_reranking") and _RANKER:
        # Prepare docs for FlashRank
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(fused[:20]) # Rerank top 20 fused candidates
        ]
        
        rerank_request = RerankRequest(query=question, passages=passages)
        reranked_results = _RANKER.rerank(rerank_request)
        
        # Convert back to Langchain Documents
        final_docs = []
        for res in reranked_results:
            doc = Document(
                page_content=res["text"],
                metadata={**res["meta"], "relevance_score": res["score"]}
            )
            final_docs.append(doc)
            
        return final_docs[:top_n]

    # Trim to top_n if no reranking
    return fused[:top_n]
