import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Filename stem → domain mapping
DOMAIN_MAP = {
    "refund_policy": "hr",
    "billing_plans": "hr",
    "hr_onboarding": "hr",
    "api_reference": "technical",
    "security_compliance": "technical",
    "product_features": "product",
    "product_pricing": "product",
}


def load_documents(data_dir):
    """Load .txt and .md files, assign domain metadata, return List[Document]."""
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    # Also load markdown files if any
    md_loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs.extend(md_loader.load())

    # Assign domain metadata based on filename
    for doc in docs:
        stem = Path(doc.metadata.get("source", "")).stem
        doc.metadata["domain"] = DOMAIN_MAP.get(stem, "general")

    return docs


def chunk_documents(docs, cfg):
    """Split documents using RecursiveCharacterTextSplitter, preserve metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
        separators=cfg["chunking"]["separators"],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_vectorstore(chunks, cfg):
    """Create FAISS vectorstore from chunks, persist to disk, return vectorstore."""
    embeddings = GoogleGenerativeAIEmbeddings(model=cfg["embedding"]["model"])
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(cfg["vectorstore"]["persist_directory"])
    return vectorstore


def load_vectorstore(cfg):
    """Load existing FAISS vectorstore from disk, return vectorstore."""
    embeddings = GoogleGenerativeAIEmbeddings(model=cfg["embedding"]["model"])
    vectorstore = FAISS.load_local(
        cfg["vectorstore"]["persist_directory"],
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def setup_pipeline_data(cfg):
    """Load or build vectorstore and chunks. Return (vectorstore, chunks)."""
    persist_dir = cfg["vectorstore"]["persist_directory"]
    index_file = os.path.join(persist_dir, "index.faiss")

    if os.path.exists(index_file):
        vectorstore = load_vectorstore(cfg)
        # Reload docs and chunk for BM25 index
        docs = load_documents(cfg["ingestion"]["data_dir"])
        chunks = chunk_documents(docs, cfg)
        return vectorstore, chunks

    docs = load_documents(cfg["ingestion"]["data_dir"])
    chunks = chunk_documents(docs, cfg)
    vectorstore = build_vectorstore(chunks, cfg)
    return vectorstore, chunks


if __name__ == "__main__":
    from src.config import get_config
    import shutil
    
    cfg = get_config()
    persist_dir = cfg["vectorstore"]["persist_directory"]
    
    print(f"Force rebuilding vector database from {cfg['ingestion']['data_dir']}...", flush=True)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        
    vs, ch = setup_pipeline_data(cfg)
    print(f"Successfully ingested {len(set(d.metadata.get('source','') for d in ch))} documents into {len(ch)} chunks.", flush=True)
