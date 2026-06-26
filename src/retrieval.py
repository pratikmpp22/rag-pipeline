from collections import defaultdict

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


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


def hybrid_retrieve(question, vectorstore, bm25_index, bm25_chunks, cfg, domain_filter=None, queries=None, stats_dict=None):
    """Orchestrate retrieval: dense+BM25 and RRF fusion using pre-computed routing/variants."""
    top_k = cfg["retrieval"]["top_k"]
    top_n = cfg["retrieval"]["top_n"]
    fusion_top_n = cfg["retrieval"].get("fusion_top_n", 20)
    rrf_k = cfg["hybrid_search"]["rrf_k"]

    if not queries:
        queries = [question]

    # 3. Retrieve from both indexes for each query
    all_ranked_lists = []
    total_dense = 0
    total_bm25 = 0

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
        total_dense += len(dense_results)

        # BM25 search (over-fetch then filter by domain so hybrid matches FAISS scope)
        if cfg["features"]["use_hybrid_search"]:
            bm25_cap = min(len(bm25_chunks), top_k * 20 if domain_filter else top_k)
            bm25_results = bm25_search(bm25_index, bm25_chunks, q, k=bm25_cap)
            if domain_filter:
                dom = domain_filter.get("domain")
                bm25_results = [
                    d for d in bm25_results if d.metadata.get("domain") == dom
                ][:top_k]
            else:
                bm25_results = bm25_results[:top_k]
            all_ranked_lists.append(bm25_results)
            total_bm25 += len(bm25_results)

    # 4. RRF fusion
    fused = reciprocal_rank_fusion(all_ranked_lists, rrf_k=rrf_k)
    
    if stats_dict is not None:
        stats_dict["dense_count"] = total_dense
        stats_dict["bm25_count"] = total_bm25
        stats_dict["fused_count"] = len(fused)

    # 5. Cross-Encoder Reranking
    if cfg["features"].get("use_reranking") and _RANKER:
        # Prepare docs for FlashRank
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(fused[:fusion_top_n])
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
