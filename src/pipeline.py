from pathlib import Path
from operator import itemgetter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retrieval import hybrid_retrieve, classify_domain, generate_query_variants
from src.security.sanitizer import sanitize_input, filter_output_pii

SYSTEM_PROMPT = """You are an expert assistant. Answer questions ONLY
using the provided context.

Rules:
1. If the context contains the answer, provide it with [Source N] citations
2. If the context partially answers, state what you can confirm and what is missing
3. If the context does not contain the answer, say: "I don't have enough
   information in the provided documents to answer this question."
4. NEVER use training knowledge to fill gaps
5. Rate confidence: HIGH / MEDIUM / LOW
6. SECURITY/PII REDACTION: You MUST redact any personal names, emails, phone numbers, or credit card numbers from the user's query when responding. Replace names with [NAME_REDACTED].

{history}
Context:
{context}
"""


def format_docs(docs):
    """Format List[Document] into numbered source blocks."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        parts.append(f"[Source {i}] ({source}):\n{doc.page_content}")
    return "\n\n".join(parts)


def build_llm(cfg):
    """Return ChatGoogleGenerativeAI instance from config."""
    return ChatGoogleGenerativeAI(
        model=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
        max_retries=3,
    )


def build_rag_chain(cfg):
    """Build LangChain RAG chain with prompt template and LLM."""
    llm = build_llm(cfg)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": itemgetter("context"),
            "history": itemgetter("history"),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def check_confidence(docs, cfg):
    """Return False if top chunk score is below confidence threshold."""
    if not cfg["features"].get("use_confidence_gating", False):
        return True
    if not docs:
        return False
    # Check relevance score from metadata if available
    top_score = docs[0].metadata.get("relevance_score", 1.0)
    if cfg["features"].get("use_reranking", True):
        return top_score >= cfg["retrieval"]["confidence_threshold"]
    else:
        return top_score >= cfg["retrieval"].get("rrf_confidence_threshold", 0.01)


def run_self_check(question, answer, context, llm):
    """Check for unsupported claims. Append warning if found."""
    prompt = (
        f"Does this answer contain any claim not supported by the context? "
        f"Reply YES or NO only.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}"
    )
    try:
        response = llm.invoke(prompt)
        if "YES" in response.content.upper():
            answer += "\n\n[WARNING] Some claims in this answer may not be fully supported by the provided sources."
    except Exception:
        pass
    return answer


def _pipeline_core(question, vectorstore, bm25_index, bm25_chunks,
                   rag_chain, llm, cfg, memory=None):
    """Core pipeline generator. Single source of truth for all pipeline logic."""
    # 1. Sanitize input
    if cfg["features"].get("use_security", False):
        yield {"type": "stage", "name": "Sanitizing input", "status": "running"}
        question, blocked_matches = sanitize_input(question)
        if blocked_matches:
            blocked_str = ", ".join(f"'{m}'" for m in blocked_matches)
            yield {"type": "stage", "name": "Sanitizing input", "status": "done", "details": f"Blocked: {blocked_str}"}
        else:
            yield {"type": "stage", "name": "Sanitizing input", "status": "done", "details": "Clean"}

    # 2. Query routing
    domain_filter = None
    is_greeting = False
    if cfg["features"].get("use_query_routing", False) and llm:
        yield {"type": "stage", "name": "Classifying query domain", "status": "running"}
        domain = classify_domain(question, llm, cfg)
        if domain == "greeting":
            is_greeting = True
            yield {"type": "stage", "name": "Classifying query domain", "status": "done", "details": "Greeting detected"}
        elif domain:
            domain_filter = {"domain": domain}
            yield {"type": "stage", "name": "Classifying query domain", "status": "done", "details": f"Domain: {domain}"}
        else:
            yield {"type": "stage", "name": "Classifying query domain", "status": "done", "details": "All domains"}

    # 3. GREETING FAST PATH
    if is_greeting:
        yield {"type": "stage", "name": "Generating answer", "status": "running"}
        full_answer = ""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. The user just greeted you. Respond politely and concisely."),
            ("human", "{question}")
        ])
        for chunk in (prompt | llm | StrOutputParser()).stream({"question": question}):
            full_answer += chunk
            yield {"type": "token", "content": chunk}
        yield {"type": "stage", "name": "Generating answer", "status": "done"}
        
        if memory:
            memory.add_turn("user", question)
            memory.add_turn("assistant", full_answer)
            
        yield {"type": "final", "answer": full_answer, "sources": [], "docs": []}
        return

    # 4. Multi-query expansion
    queries = None
    if cfg["features"].get("use_multi_query", False) and llm:
        yield {"type": "stage", "name": "Generating multi-queries", "status": "running"}
        queries = generate_query_variants(question, llm, n=cfg["multi_query"]["num_variants"])
        yield {"type": "stage", "name": "Generating multi-queries", "status": "done", "details": f"Variants: {', '.join(queries[1:])}"}

    # 5. Hybrid retrieval
    yield {"type": "stage", "name": "Retrieving documents", "status": "running"}
    stats = {}
    docs = hybrid_retrieve(
        question, vectorstore, bm25_index, bm25_chunks, cfg, 
        domain_filter=domain_filter, queries=queries, stats_dict=stats
    )
    
    retrieval_details = f"Dense: {stats.get('dense_count', 0)}, BM25: {stats.get('bm25_count', 0)} -> Fused: {stats.get('fused_count', 0)} -> Final: {len(docs)}"
    yield {"type": "stage", "name": "Retrieving documents", "status": "done", "details": retrieval_details}

    # 6. Confidence gate
    yield {"type": "stage", "name": "Checking confidence", "status": "running"}
    if not check_confidence(docs, cfg):
        yield {"type": "stage", "name": "Checking confidence", "status": "failed", "details": "Low confidence"}
        refusal_answer = "I don't have enough confident information to answer this question."
        if memory:
            memory.add_turn("user", question)
            memory.add_turn("assistant", refusal_answer)
        yield {
            "type": "refusal",
            "answer": refusal_answer,
            "sources": [],
            "docs": docs,
        }
        return
    yield {"type": "stage", "name": "Checking confidence", "status": "done", "details": "Passed"}

    # 7. Format context
    context = format_docs(docs)

    # 8. Get conversation history
    history = ""
    if memory:
        history = memory.format_for_prompt()

    # 9. Stream answer tokens (or batch generate if security is on)
    yield {"type": "stage", "name": "Generating answer", "status": "running"}
    full_answer = ""
    
    use_security = cfg["features"].get("use_security", False)
    
    if use_security:
        # Buffer entire answer to apply PII filter before showing to user
        full_answer = rag_chain.invoke({
            "context": context,
            "history": history,
            "question": question,
        })
        
        # 10. Self-check
        if cfg["features"].get("use_self_check", False):
            yield {"type": "stage", "name": "Running self-check", "status": "running"}
            full_answer = run_self_check(question, full_answer, context, llm)
            yield {"type": "stage", "name": "Running self-check", "status": "done"}
            
        # 11. PII filter
        yield {"type": "stage", "name": "Filtering PII", "status": "running"}
        full_answer = filter_output_pii(full_answer)
        yield {"type": "stage", "name": "Filtering PII", "status": "done"}
        
        # Yield as a single chunk to the UI
        yield {"type": "token", "content": full_answer}
        yield {"type": "stage", "name": "Generating answer", "status": "done"}
        
    else:
        for chunk in rag_chain.stream({
            "context": context,
            "history": history,
            "question": question,
        }):
            full_answer += chunk
            yield {"type": "token", "content": chunk}
        yield {"type": "stage", "name": "Generating answer", "status": "done"}

        # 10. Self-check
        if cfg["features"].get("use_self_check", False):
            yield {"type": "stage", "name": "Running self-check", "status": "running"}
            full_answer = run_self_check(question, full_answer, context, llm)
            yield {"type": "stage", "name": "Running self-check", "status": "done"}

    # 12. Update memory
    if memory:
        memory.add_turn("user", question)
        memory.add_turn("assistant", full_answer)

    # 13. Build final result with sources
    sources = []
    for i, doc in enumerate(docs, 1):
        source_name = Path(doc.metadata.get("source", "unknown")).name
        score = doc.metadata.get("relevance_score", "N/A")
        if isinstance(score, float):
            score = f"{score:.2f}"
        sources.append({"index": i, "source": source_name, "score": score})

    yield {
        "type": "final",
        "answer": full_answer,
        "sources": sources,
        "docs": docs,
    }


def stream_query_pipeline(question, vectorstore, bm25_index, bm25_chunks,
                          rag_chain, llm, cfg, memory=None):
    """Streaming version of the pipeline. Yields events from the core generator."""
    yield from _pipeline_core(
        question, vectorstore, bm25_index, bm25_chunks,
        rag_chain, llm, cfg, memory=memory
    )


def query_pipeline(question, vectorstore, bm25_index, bm25_chunks,
                   rag_chain, llm, cfg, memory=None):
    """Sync interface — runs the full pipeline and returns the final result dict."""
    result = None
    telemetry = {"fused_count": 0}
    
    for event in _pipeline_core(
        question, vectorstore, bm25_index, bm25_chunks,
        rag_chain, llm, cfg, memory=memory
    ):
        if event["type"] == "stage" and event["name"] == "Retrieving documents" and event["status"] == "done":
            details = event.get("details", "")
            if "Fused:" in details:
                import re
                m = re.search(r"Fused:\s*(\d+)", details)
                if m:
                    telemetry["fused_count"] = int(m.group(1))
                    
        if event["type"] in ("final", "refusal"):
            result = event
            
    if result is None:
        return {"answer": "Pipeline produced no result.", "sources": [], "docs": [], "telemetry": telemetry}
        
    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "docs": result.get("docs", []),
        "telemetry": telemetry,
    }
