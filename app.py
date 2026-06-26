import streamlit as st
from dotenv import load_dotenv

from src.config import get_config
from src.ingestion import setup_pipeline_data
from src.retrieval import build_bm25_index
from src.pipeline import build_llm, build_rag_chain, stream_query_pipeline
from src.memory import HybridMemory

# Load env variables at the entry point
load_dotenv()


def initialize_state():
    """Initialize or completely reset the session state."""
    cfg = get_config()
    vectorstore, chunks = setup_pipeline_data(cfg)
    bm25_index, bm25_chunks = build_bm25_index(chunks)
    llm = build_llm(cfg)
    rag_chain = build_rag_chain(cfg)
    memory = HybridMemory(
        token_budget=cfg["memory"]["token_budget"],
        summary_llm=llm,
    )

    st.session_state.cfg = cfg
    st.session_state.vectorstore = vectorstore
    st.session_state.bm25_index = bm25_index
    st.session_state.bm25_chunks = bm25_chunks
    st.session_state.llm = llm
    st.session_state.rag_chain = rag_chain
    st.session_state.memory = memory
    st.session_state.messages = []
    st.session_state.initialized = True


if "initialized" not in st.session_state:
    initialize_state()

# Configuration and sidebar
st.sidebar.title("⚙️ RAG Assistant")
st.sidebar.markdown(f"**Model:** {st.session_state.cfg['llm']['model']}")
st.sidebar.markdown(f"**Chunks:** {len(st.session_state.bm25_chunks)}")
st.sidebar.markdown("---")

st.sidebar.subheader("Features")
flags = st.session_state.cfg["features"]

# Reactive checkboxes that update the config in session_state
flags["use_hybrid_search"] = st.sidebar.checkbox("Hybrid Search", value=flags.get("use_hybrid_search", True))
flags["use_reranking"] = st.sidebar.checkbox("Reranking", value=flags.get("use_reranking", True))
flags["use_multi_query"] = st.sidebar.checkbox("Multi-Query", value=flags.get("use_multi_query", False))
flags["use_query_routing"] = st.sidebar.checkbox("Query Routing", value=flags.get("use_query_routing", True))
flags["use_confidence_gating"] = st.sidebar.checkbox("Confidence Gating", value=flags.get("use_confidence_gating", False))
flags["use_self_check"] = st.sidebar.checkbox("Self-Check", value=flags.get("use_self_check", True))
flags["use_security"] = st.sidebar.checkbox("Security", value=flags.get("use_security", True))

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Re-index Docs"):
    with st.spinner("Rebuilding indexes..."):
        initialize_state()
    st.sidebar.success("Indexes rebuilt successfully!")

if st.sidebar.button("🗑️ Clear Memory"):
    st.session_state.memory.clear()
    st.session_state.messages = []
    st.sidebar.success("Memory cleared!")

# Main chat interface
st.title("RAG Expert Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "stages" in msg and msg["stages"]:
            with st.expander("🔍 Pipeline Stages", expanded=False):
                for stage in msg["stages"]:
                    icon = "✅" if stage["status"] == "done" else "❌" if stage["status"] == "failed" else "⏳"
                    details = f" - {stage['details']}" if stage.get("details") else ""
                    st.caption(f"{icon} **{stage['name']}**{details}")
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    st.caption(f"[{src['index']}] {src['source']}  (score: {src['score']})")

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        token_container = st.empty()
        full_answer = ""
        sources = []
        stages = []

        # Stream response
        for event in stream_query_pipeline(
            prompt, st.session_state.vectorstore,
            st.session_state.bm25_index, st.session_state.bm25_chunks,
            st.session_state.rag_chain, st.session_state.llm,
            st.session_state.cfg, memory=st.session_state.memory
        ):
            if event["type"] == "token":
                full_answer += event["content"]
                token_container.markdown(full_answer + "▌")
            elif event["type"] == "final":
                sources = event.get("sources", [])
                full_answer = event["answer"]
            elif event["type"] == "refusal":
                full_answer = event["answer"]
            elif event["type"] == "stage":
                stages.append(event)

        token_container.markdown(full_answer)
        
        if stages:
            with st.expander("🔍 Pipeline Stages", expanded=False):
                for stage in stages:
                    icon = "✅" if stage["status"] == "done" else "❌" if stage["status"] == "failed" else "⏳"
                    details = f" - {stage['details']}" if stage.get("details") else ""
                    st.caption(f"{icon} **{stage['name']}**{details}")

        if sources:
            with st.expander("📄 Sources", expanded=False):
                for src in sources:
                    st.caption(f"[{src['index']}] {src['source']}  (score: {src['score']})")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_answer,
        "sources": sources,
        "stages": stages
    })
