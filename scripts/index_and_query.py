#!/usr/bin/env python3
"""
CLI tool for the Legal RAG Assistant.

Usage examples:
  # Index documents from a directory
  python scripts/index_and_query.py index ./data/sample_docs

  # Ask a question
  python scripts/index_and_query.py ask "What is the termination notice period?"

  # Semantic search
  python scripts/index_and_query.py search "intellectual property ownership"

  # Show index stats
  python scripts/index_and_query.py stats
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_pipeline(args):
    from app.rag_pipeline import LegalRAGPipeline
    return LegalRAGPipeline(
        endee_url=args.endee_url,
        endee_auth_token=args.auth_token or None,
        index_name=args.index_name,
        top_k=args.top_k,
    )


def cmd_index(args):
    pipeline = get_pipeline(args)
    print(f"\n🗄️  Connecting to Endee at {args.endee_url} …")

    if not pipeline.endee.health_check():
        print("❌ Cannot connect to Endee. Start it with: docker compose up")
        sys.exit(1)

    print(f"📄 Indexing documents from: {args.source}")
    result = pipeline.index_documents(args.source, recreate=args.recreate)
    print(f"\n✅ Done!")
    print(f"   Documents indexed : {result['documents_indexed']}")
    print(f"   Chunks indexed    : {result['chunks_indexed']}")
    print(f"   Index stats       : {json.dumps(result.get('index_stats', {}), indent=2)}\n")


def cmd_ask(args):
    pipeline = get_pipeline(args)

    if not pipeline.endee.health_check():
        print("❌ Cannot connect to Endee. Start it with: docker compose up")
        sys.exit(1)

    question = " ".join(args.question)
    print(f"\n❓ Question: {question}\n")

    response = pipeline.ask(question, doc_type_filter=args.filter)

    print("💡 Answer:")
    print("-" * 60)
    print(response.answer)
    print("-" * 60)
    print(f"\n📊 Metadata:")
    print(f"   Contexts retrieved : {len(response.contexts)}")
    print(f"   Latency            : {response.latency_ms:.0f}ms")
    print(f"   Embedding model    : {response.model_used}")

    if response.contexts:
        print("\n📎 Source Contexts:")
        for i, ctx in enumerate(response.contexts, 1):
            print(f"   [{i}] {ctx.filename} | Score: {ctx.score:.4f} | Section: {ctx.section[:50]}")
    print()


def cmd_search(args):
    pipeline = get_pipeline(args)

    if not pipeline.endee.health_check():
        print("❌ Cannot connect to Endee. Start it with: docker compose up")
        sys.exit(1)

    query = " ".join(args.query)
    print(f"\n🔍 Semantic Search: '{query}'\n")

    results = pipeline.semantic_search(query, top_k=args.top_k)

    if not results:
        print("No results found. Have you indexed any documents?")
        return

    print(f"Found {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.filename}")
        print(f"    Section : {r.section[:60]}")
        print(f"    Score   : {r.score:.4f}")
        print(f"    Excerpt : {r.text[:200].strip()}…")
        print()


def cmd_stats(args):
    pipeline = get_pipeline(args)

    if not pipeline.endee.health_check():
        print("❌ Cannot connect to Endee.")
        sys.exit(1)

    stats = pipeline.get_index_stats()
    print(f"\n📊 Index '{args.index_name}' Stats:")
    print(json.dumps(stats, indent=2))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Legal RAG Assistant CLI — powered by Endee vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--endee-url", default="http://localhost:8080", help="Endee server URL")
    parser.add_argument("--auth-token", default="", help="Endee auth token (optional)")
    parser.add_argument("--index-name", default="legal_docs", help="Endee index name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")

    sub = parser.add_subparsers(dest="command")

    # index
    p_index = sub.add_parser("index", help="Index documents from a directory or file")
    p_index.add_argument("source", help="Path to directory or document file")
    p_index.add_argument("--recreate", action="store_true", help="Recreate index from scratch")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question (RAG)")
    p_ask.add_argument("question", nargs="+", help="The question to ask")
    p_ask.add_argument("--filter", default=None, help="Filter by doc type (contract|policy|court_order)")

    # search
    p_search = sub.add_parser("search", help="Semantic search over indexed documents")
    p_search.add_argument("query", nargs="+", help="The search query")

    # stats
    sub.add_parser("stats", help="Show Endee index statistics")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
