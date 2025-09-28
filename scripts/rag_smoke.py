#!/usr/bin/env python3
from __future__ import annotations

import asyncio
from pathlib import Path

from synndicate.rag.retriever import RAGRetriever, QueryContext
from synndicate.rag.chunking import Chunk, ChunkType


async def main() -> None:
    retriever = RAGRetriever(
        embedding_cache_path=str(Path.home() / ".synndicate/emb_cache.json")
    )
    await retriever.initialize()

    # Index a couple of chunks
    chunks = [
        Chunk(
            content="hello world from synndicate",
            chunk_type=ChunkType.TEXT,
            start_index=0,
            end_index=27,
            metadata={"id": "c1"},
        ),
        Chunk(
            content="another document about vectors",
            chunk_type=ChunkType.TEXT,
            start_index=0,
            end_index=29,
            metadata={"id": "c2"},
        ),
    ]
    await retriever.add_chunks(chunks)

    # Query
    results = await retriever.retrieve(QueryContext(query="hello world"))
    print("Top results:")
    for r in results[:2]:
        print(f"- id? {r.chunk.metadata.get('id')} score={r.score:.3f} mode={r.search_mode.value}")

    print("Stats:", retriever.get_stats())


if __name__ == "__main__":
    asyncio.run(main())
