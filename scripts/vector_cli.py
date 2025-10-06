#!/usr/bin/env python3
"""
Synndicate Vector Store CLI
Copyright (c) 2025 Himokai. All Rights Reserved.

CLI tool for interacting with Synndicate vector store.

A command-line interface for managing documents in the Synndicate vector store.
Supports adding, querying, and deleting documents with automatic embedding generation.

Usage:
    # Health check
    python scripts/vector_cli.py health

    # Add documents with metadata
    python scripts/vector_cli.py add --text "Hello world" --id "doc1" --metadata '{"topic":"greeting"}'

    # Search by semantic similarity
    python scripts/vector_cli.py query --text "Hello" --limit 5

    # Delete documents
    python scripts/vector_cli.py delete --ids "doc1,doc2"

Environment Variables:
    SYN_RAG_VECTOR_API: Vector store URL (default: http://localhost:8080)
    SYN_RAG_VECTOR_API_KEY: API key for authentication (optional)

Platform Support:
    - Arch Linux (primary development platform)
    - Ubuntu/Debian (CI/CD and production)
    - macOS (development)
    - Windows (via WSL2 recommended)

Prerequisites:
    # Arch Linux
    sudo pacman -S python python-pip

    # Ubuntu/Debian
    sudo apt install python3 python3-pip python3-venv

    # macOS
    brew install python

    # Windows (WSL2)
    sudo apt install python3 python3-pip python3-venv

Setup:
    1. Install Synndicate: pip install -e .
    2. Start vector store server (optional for distributed RAG)
    3. Set environment variables for authentication
    4. Use CLI commands to manage documents

Examples:
    # Start authenticated vector store
    export SYN_VECTORSTORE_API_KEY="$(openssl rand -hex 32)"
    uvicorn --app-dir scripts vectorstore_server:app --port 8080 &

    # Configure client
    export SYN_RAG_VECTOR_API="http://localhost:8080"
    export SYN_RAG_VECTOR_API_KEY="$SYN_VECTORSTORE_API_KEY"

    # Add and search documents
    python scripts/vector_cli.py add --text "Machine learning tutorial" --id "ml1"
    python scripts/vector_cli.py query --text "AI tutorial" --limit 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# Use editable install: pip install -e .
import httpx

from synndicate.rag.retriever import RAGRetriever


async def health_check(base_url: str) -> None:
    """Check vector store health."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/health")
            resp.raise_for_status()
            data = resp.json()
            print(f"✓ Vector store healthy: {data}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        sys.exit(1)


async def add_document(
    base_url: str, api_key: str | None, text: str, doc_id: str, metadata: dict
) -> None:
    """Add a document to the vector store."""
    try:
        # Use retriever to generate embedding
        retriever = RAGRetriever()
        await retriever.initialize()

        embedding = await retriever._get_or_create_embedding(text)
        if embedding is None:
            print("✗ Failed to generate embedding")
            sys.exit(1)

        # Add to vector store
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        payload = {
            "items": [
                {
                    "id": doc_id,
                    "embedding": embedding.tolist(),
                    "document": text,
                    "metadata": metadata,
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{base_url}/vectors/add", json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()
            print(f"✓ Added document: {result}")

    except Exception as e:
        print(f"✗ Failed to add document: {e}")
        sys.exit(1)


async def query_documents(base_url: str, api_key: str | None, text: str, limit: int) -> None:
    """Query documents from the vector store."""
    try:
        # Use retriever to generate query embedding
        retriever = RAGRetriever()
        await retriever.initialize()

        embedding = await retriever._get_or_create_embedding(text)
        if embedding is None:
            print("✗ Failed to generate query embedding")
            sys.exit(1)

        # Query vector store
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        payload = {"embedding": embedding.tolist(), "n_results": limit}

        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{base_url}/vectors/query", json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()

            print(f"✓ Found {len(result['results'])} results:")
            for i, item in enumerate(result["results"], 1):
                print(f"  {i}. ID: {item['id']}")
                print(f"     Score: {item['score']:.4f}")
                print(f"     Text: {item['document'][:100]}...")
                if item.get("metadata"):
                    print(f"     Metadata: {item['metadata']}")
                print()

    except Exception as e:
        print(f"✗ Failed to query documents: {e}")
        sys.exit(1)


async def delete_documents(base_url: str, api_key: str | None, doc_ids: list[str]) -> None:
    """Delete documents from the vector store."""
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        payload = {"ids": doc_ids}

        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{base_url}/vectors/delete", json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()
            print(f"✓ Deleted documents: {result}")

    except Exception as e:
        print(f"✗ Failed to delete documents: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Synndicate Vector Store CLI")
    parser.add_argument(
        "--url",
        default=os.getenv("SYN_RAG_VECTOR_API", "http://localhost:8080"),
        help="Vector store URL",
    )
    parser.add_argument(
        "--api-key", default=os.getenv("SYN_RAG_VECTOR_API_KEY"), help="API key for authentication"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Health command
    subparsers.add_parser("health", help="Check vector store health")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add document to vector store")
    add_parser.add_argument("--text", required=True, help="Document text")
    add_parser.add_argument("--id", required=True, help="Document ID")
    add_parser.add_argument("--metadata", default="{}", help="JSON metadata")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query vector store")
    query_parser.add_argument("--text", required=True, help="Query text")
    query_parser.add_argument("--limit", type=int, default=5, help="Max results")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete documents")
    delete_parser.add_argument("--ids", required=True, help="Comma-separated document IDs")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "health":
        asyncio.run(health_check(args.url))
    elif args.command == "add":
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("✗ Invalid JSON metadata")
            sys.exit(1)
        asyncio.run(add_document(args.url, args.api_key, args.text, args.id, metadata))
    elif args.command == "query":
        asyncio.run(query_documents(args.url, args.api_key, args.text, args.limit))
    elif args.command == "delete":
        doc_ids = [id.strip() for id in args.ids.split(",")]
        asyncio.run(delete_documents(args.url, args.api_key, doc_ids))


if __name__ == "__main__":
    main()
