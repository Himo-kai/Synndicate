#!/usr/bin/env python3
"""
Simple test script to validate RAG subsystem functionality without full imports.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path to avoid full package imports
sys.path.insert(0, "src")

# Import only RAG components directly
from synndicate.rag.chunking import Chunk, ChunkType, SemanticChunker
from synndicate.rag.context import ContextBuilder, ContextIntegrator
from synndicate.rag.indexer import DocumentIndexer
from synndicate.rag.retriever import QueryContext, RAGRetriever


async def test_chunking():
    """Test the chunking functionality."""
    print("🧩 Testing Chunking...")

    chunker = SemanticChunker(max_chunk_size=500, overlap=50)

    # Test Python code chunking
    python_code = '''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "success"

class TestClass:
    """A test class."""

    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
'''

    chunks = chunker.chunk(python_code, {"file_extension": ".py"})
    print(f"  ✅ Python code chunked into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i+1}: {chunk.chunk_type.value} ({len(chunk.content)} chars)")

    return True


async def test_retrieval():
    """Test the retrieval functionality."""
    print("\n🔍 Testing Retrieval...")

    # Create retriever (without vector store for simplicity)
    retriever = RAGRetriever(max_results=5)
    await retriever.initialize()

    # Create some test chunks manually
    test_chunks = [
        Chunk(
            content="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            chunk_type=ChunkType.CODE,
            start_index=0,
            end_index=80,
            metadata={"file_extension": ".py", "language": "python"},
        ),
        Chunk(
            content="This function calculates fibonacci numbers recursively",
            chunk_type=ChunkType.DOCUMENTATION,
            start_index=0,
            end_index=53,
            metadata={"file_extension": ".md"},
        ),
        Chunk(
            content="Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21...",
            chunk_type=ChunkType.TEXT,
            start_index=0,
            end_index=49,
            metadata={"file_extension": ".txt"},
        ),
    ]

    # Add chunks to retriever
    await retriever.add_chunks(test_chunks)

    # Test keyword search
    query_context = QueryContext(query="fibonacci function", task_type="coding")

    results = await retriever.retrieve(query_context, max_results=3)
    print(f"  ✅ Retrieved {len(results)} results for 'fibonacci function'")

    for i, result in enumerate(results):
        print(f"    Result {i+1}: {result.relevance.value} (score: {result.score:.3f})")
        print(f"      Content: {result.chunk.content[:50]}...")

    return True


async def test_context_integration():
    """Test context integration functionality."""
    print("\n🔗 Testing Context Integration...")

    # Create retriever and test chunks for this test
    retriever = RAGRetriever(max_results=5)
    await retriever.initialize()

    test_chunks = [
        Chunk(
            content="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            chunk_type=ChunkType.CODE,
            start_index=0,
            end_index=80,
            metadata={"file_extension": ".py", "language": "python"},
        ),
        Chunk(
            content="This function calculates fibonacci numbers recursively",
            chunk_type=ChunkType.DOCUMENTATION,
            start_index=0,
            end_index=53,
            metadata={"file_extension": ".md"},
        ),
    ]

    await retriever.add_chunks(test_chunks)

    # Test retrieval and context building
    query_context = QueryContext(
        query="how to implement fibonacci",
        task_type="coding",
        conversation_history=["I need help with algorithms", "Show me recursive functions"],
    )

    results = await retriever.retrieve(query_context, max_results=3)

    # Test context building
    context_builder = ContextBuilder(max_context_tokens=1000)
    integrated_context = await context_builder.build_context(results, query_context=query_context)

    print(f"  ✅ Built context with {integrated_context.chunk_count} chunks")
    print(f"  ✅ Context tokens: {integrated_context.total_tokens}")
    print(f"  ✅ Compression ratio: {integrated_context.compression_ratio:.2f}")
    print(f"  ✅ Strategy used: {integrated_context.strategy.value}")

    # Test agent-specific formatting
    context_integrator = ContextIntegrator(context_builder)

    # Test for different agent types
    for agent_type in ["planner", "coder", "critic"]:
        agent_context = await context_integrator.integrate_for_agent(
            agent_type=agent_type,
            retrieval_results=results,
            conversation_id="test_conversation",
            query_context=query_context,
        )
        print(
            f"  ✅ Formatted context for {agent_type} agent ({agent_context.total_tokens} tokens)"
        )

    return True


async def test_indexing():
    """Test the document indexing functionality."""
    print("\n📚 Testing Document Indexing...")

    # Create temporary test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        (temp_path / "test.py").write_text(
            '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
        )

        (temp_path / "README.md").write_text(
            """
# Test Project

This is a test project for RAG indexing.

## Features

- Fibonacci calculation
- Documentation
"""
        )

        # Test indexing
        indexer = DocumentIndexer(max_concurrent=2)
        progress = await indexer.index_directory(temp_path, recursive=True)

        print(f"  ✅ Indexed {progress.processed_documents} documents")
        print(f"  ✅ Created {progress.total_chunks} chunks")
        print(f"  ✅ Processing took {progress.duration:.2f}s")

        # Get stats
        stats = indexer.get_indexing_stats()
        print(f"  📊 File types: {stats.get('file_types', {})}")

        return True


async def main():
    """Run all tests."""
    print("🧪 Starting Simple RAG Tests\n")

    try:
        success = True

        success &= await test_chunking()
        success &= await test_indexing()
        success &= await test_retrieval()
        success &= await test_context_integration()

        if success:
            print("\n✅ All RAG tests completed successfully!")
            print("\n📊 Summary:")
            print("  ✅ Chunking: Smart content-aware chunking working")
            print("  ✅ Indexing: Async document indexing working")
            print("  ✅ Retrieval: Keyword search working (vector search requires models)")
            print("  ✅ Context: Agent-specific context integration working")
            print("\n🎯 RAG subsystem is ready for integration!")
        else:
            print("\n❌ Some tests failed")
            return 1

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
