import sys

from src.config import get_config
from src.ingestion import setup_pipeline_data
from src.retrieval import build_bm25_index
from src.pipeline import build_llm, build_rag_chain, stream_query_pipeline
from src.memory import ConversationMemory
from src.evaluate import run_evaluation
from src.ab_comparison import run_ab_comparison

BANNER = """
============================================================
  RAG Expert Assistant
  Type your question or /help for commands
============================================================"""

HELP_TEXT = """
  Commands:
  /help     Show this help message
  /status   Show model info, chunk count, and feature flags
  /clear    Clear conversation memory
  /ingest   Re-index documents
  /eval     Run RAGAS evaluation
  /ab       Run A/B comparison
  /quit     Exit the assistant
  /exit     Exit the assistant
"""


def print_feature_status(cfg):
    """Print feature flag status line."""
    flags = cfg["features"]
    parts = []
    flag_names = {
        "use_query_routing": "routing",
        "use_hybrid_search": "hybrid",
        "use_reranking": "reranking",
        "use_multi_query": "multi-query",
        "use_confidence_gating": "gating",
        "use_self_check": "self-check",
        "use_security": "security",
        "use_streaming": "streaming",
    }
    for key, label in flag_names.items():
        status = "ON" if flags.get(key, False) else "OFF"
        parts.append(f"{label}={status}")
    parts.append("memory=ON")
    print(f"Features: {'  '.join(parts)}")


def handle_command(command, memory, vectorstore, bm25_index, bm25_chunks,
                   rag_chain, llm, cfg, state):
    """Handle slash commands. Return updated state dict."""
    cmd = command.strip().lower()

    if cmd == "/help":
        print(HELP_TEXT)

    elif cmd == "/status":
        print(f"\n  Model: {cfg['llm']['model']}")
        print(f"  Chunks: {len(bm25_chunks)}")
        print_feature_status(cfg)
        print()

    elif cmd == "/clear":
        memory.clear()
        print("Memory cleared.")

    elif cmd == "/ingest":
        print("Re-indexing documents...")
        vs, chunks = setup_pipeline_data(cfg)
        idx, bm25_c = build_bm25_index(chunks)
        state["vectorstore"] = vs
        state["bm25_index"] = idx
        state["bm25_chunks"] = bm25_c
        state["rag_chain"] = build_rag_chain(cfg)
        print(f"Done. {len(chunks)} chunks indexed.")

    elif cmd == "/eval":
        run_evaluation(
            state["rag_chain"], state["vectorstore"],
            state["bm25_index"], state["bm25_chunks"],
            llm, cfg
        )

    elif cmd == "/ab":
        run_ab_comparison(cfg)

    elif cmd in ("/quit", "/exit"):
        print("Goodbye.")
        sys.exit(0)

    else:
        print(f"Unknown command: {cmd}. Type /help for available commands.")

    return state


def customize_features(cfg):
    """Interactively toggle feature flags."""
    print("\n" + "-" * 60)
    print("  Feature Configuration")
    print("-" * 60)
    
    flags = cfg["features"]
    
    for key in flags.keys():
        current = "y" if flags[key] else "n"
        choice = input(f"Enable {key}? [y/n] (default: {current}): ").strip().lower()
        if choice == 'y':
            flags[key] = True
        elif choice == 'n':
            flags[key] = False
            
    print("-" * 60 + "\n")


def main():
    """Entry point for the RAG Expert Assistant CLI."""
    cfg = get_config()

    print(BANNER)
    
    # Prompt user to customize settings
    customize = input("\nWould you like to customize feature flags before starting? [y/N]: ").strip().lower()
    if customize == 'y':
        customize_features(cfg)

    # Setup pipeline
    print("Indexing documents...", end=" ", flush=True)
    vectorstore, chunks = setup_pipeline_data(cfg)
    print(f"{len(set(d.metadata.get('source','') for d in chunks))} docs | {len(chunks)} chunks")

    # Build BM25 index
    bm25_index, bm25_chunks = build_bm25_index(chunks)

    # Build LLM and chain
    llm = build_llm(cfg)
    rag_chain = build_rag_chain(cfg)

    # Initialize memory
    memory = ConversationMemory(cfg["memory"]["max_turns"])

    # Print status
    print_feature_status(cfg)
    print("Ready.\n")

    # Mutable state for command handlers
    state = {
        "vectorstore": vectorstore,
        "bm25_index": bm25_index,
        "bm25_chunks": bm25_chunks,
        "rag_chain": rag_chain,
    }

    # REPL loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.startswith("/"):
            state = handle_command(
                user_input, memory, state["vectorstore"],
                state["bm25_index"], state["bm25_chunks"],
                state["rag_chain"], llm, cfg, state
            )
            continue

        # Stream query
        print()

        final_result = None
        started_streaming = False

        for event in stream_query_pipeline(
            user_input, state["vectorstore"],
            state["bm25_index"], state["bm25_chunks"],
            state["rag_chain"], llm, cfg, memory=memory
        ):
            if event["type"] == "stage":
                if event["status"] == "running":
                    print(f"  [~] {event['name']}...", end=" ", flush=True)
                elif event["status"] == "done":
                    details = f" ({event.get('details')})" if "details" in event else ""
                    print(f"Done{details}")
                elif event["status"] == "failed":
                    details = f" ({event.get('details')})" if "details" in event else ""
                    print(f"Failed{details}")

            elif event["type"] == "token":
                if not started_streaming:
                    print("\nAssistant: ", end="", flush=True)
                    started_streaming = True
                print(event["content"], end="", flush=True)

            elif event["type"] == "refusal":
                print(f"\nAssistant: {event['answer']}")
                final_result = event

            elif event["type"] == "final":
                final_result = event

        # Print sources
        if final_result and final_result.get("sources"):
            print("\n\nSources:")
            for src in final_result["sources"]:
                score = src["score"]
                print(f"  [{src['index']}] {src['source']}  (score: {score})")

        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
