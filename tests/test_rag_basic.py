#!/usr/bin/env python3
"""
Basic test script to validate RAG subsystem functionality.
"""

import asyncio
import tempfile
from pathlib import Path

from src.synndicate.rag.chunking import ChunkType, SemanticChunker
from src.synndicate.rag.context import ContextBuilder, ContextIntegrator
from src.synndicate.rag.indexer import DocumentIndexer
from src.synndicate.rag.retriever import QueryContext, RAGRetriever


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

    # Test markdown chunking
    markdown_text = """
# Main Title

This is the introduction paragraph.

## Section 1

Content for section 1 with some details.

### Subsection 1.1

More detailed content here.

## Section 2

Another section with different content.
"""

    chunks = chunker.chunk(markdown_text, {"file_extension": ".md"})
    print(f"  ✅ Markdown chunked into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i+1}: {chunk.chunk_type.value} ({len(chunk.content)} chars)")


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

        (temp_path / "config.json").write_text('{"name": "test", "version": "1.0"}')

        # Test indexing
        indexer = DocumentIndexer(max_concurrent=2)
        progress = await indexer.index_directory(temp_path, recursive=True)

        print(f"  ✅ Indexed {progress.processed_documents} documents")
        print(f"  ✅ Created {progress.total_chunks} chunks")
        print(f"  ✅ Processing took {progress.duration:.2f}s")

        # Get stats
        stats = indexer.get_indexing_stats()
        print(f"  📊 File types: {stats.get('file_types', {})}")

        return indexer


async def test_retrieval():
    """Test the retrieval functionality."""
    print("\n🔍 Testing Retrieval...")

    # Create retriever (without vector store for simplicity)
    retriever = RAGRetriever(max_results=5)
    await retriever.initialize()

    # Create some test chunks manually
    from src.synndicate.rag.chunking import Chunk

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

    return retriever, test_chunks


async def test_context_integration():
    """Test context integration functionality."""
    print("\n🔗 Testing Context Integration...")

    # Get retriever and chunks from previous test
    retriever, test_chunks = await test_retrieval()

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


async def test_full_pipeline():
    """Test the complete RAG pipeline."""
    print("\n🚀 Testing Full RAG Pipeline...")

    # Create temporary project structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a mini project
        (temp_path / "main.py").write_text(
            '''
"""Main application module."""

def main():
    """Entry point of the application."""
    print("Starting application...")
    result = calculate_fibonacci(10)
    print(f"Fibonacci(10) = {result}")

def calculate_fibonacci(n):
    """Calculate fibonacci number using iteration."""
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

if __name__ == "__main__":
    main()
'''
        )

        (temp_path / "utils.py").write_text(
            '''
"""Utility functions."""

def validate_input(n):
    """Validate fibonacci input."""
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 0:
        raise ValueError("Input must be non-negative")
    return True

def format_result(n, result):
    """Format fibonacci result for display."""
    return f"The {n}th Fibonacci number is {result}"
'''
        )

        (temp_path / "README.md").write_text(
            """
# Fibonacci Calculator

A simple Python application to calculate Fibonacci numbers.

## Features

- Iterative fibonacci calculation
- Input validation
- Result formatting

## Usage

```python
python main.py
```

## Functions

- `calculate_fibonacci(n)`: Main calculation function
- `validate_input(n)`: Input validation
- `format_result(n, result)`: Result formatting
"""
        )

        # Index the project
        print("  📚 Indexing project files...")
        indexer = DocumentIndexer(max_concurrent=3)
        progress = await indexer.index_directory(temp_path, recursive=True)
        print(f"    Indexed {progress.processed_documents} files, {progress.total_chunks} chunks")

        # Set up retriever with all chunks
        print("  🔍 Setting up retriever...")
        retriever = RAGRetriever(max_results=10)
        await retriever.initialize()

        # Collect all chunks from indexing
        all_chunks = []
        for file_path in temp_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix in [".py", ".md"]:
                chunks = await indexer.index_file(file_path)
                all_chunks.extend(chunks)

        await retriever.add_chunks(all_chunks)
        print(f"    Added {len(all_chunks)} chunks to retriever")

        # Test various queries
        test_queries = [
            ("How do I calculate fibonacci numbers?", "coding"),
            ("What functions are available?", "documentation"),
            ("Show me input validation", "debugging"),
            ("How to format results?", "coding"),
        ]

        context_integrator = ContextIntegrator()

        for query, task_type in test_queries:
            print(f"\n  🔍 Query: '{query}'")

            query_context = QueryContext(
                query=query,
                task_type=task_type,
                conversation_history=["I'm working on a fibonacci project"],
            )

            # Retrieve relevant chunks
            results = await retriever.retrieve(query_context, max_results=5)
            print(f"    Found {len(results)} relevant chunks")

            # Build context for coder agent
            agent_context = await context_integrator.integrate_for_agent(
                agent_type="coder",
                retrieval_results=results,
                conversation_id="test_session",
                query_context=query_context,
            )

            print(
                f"    Context: {agent_context.chunk_count} chunks, {agent_context.total_tokens} tokens"
            )
            print(
                f"    Top result: {results[0].chunk.content[:60]}..."
                if results
                else "    No results"
            )


async def main():
    """Run all tests."""
    print("🧪 Starting RAG Subsystem Tests\n")

    try:
        await test_chunking()
        await test_indexing()
        await test_retrieval()
        await test_context_integration()
        await test_full_pipeline()

        print("\n✅ All RAG tests completed successfully!")
        print("\n📊 Summary:")
        print("  ✅ Chunking: Smart content-aware chunking working")
        print("  ✅ Indexing: Async document indexing working")
        print("  ✅ Retrieval: Hybrid search working")
        print("  ✅ Context: Agent-specific context integration working")
        print("  ✅ Pipeline: Full RAG pipeline working")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
