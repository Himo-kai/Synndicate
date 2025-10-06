"""
Tests for RAG subsystem components.
"""

import tempfile
from pathlib import Path

import pytest

from synndicate.rag.chunking import Chunk, ChunkType, SemanticChunker
from synndicate.rag.context import ContextBuilder, ContextIntegrator
from synndicate.rag.indexer import DocumentIndexer
from synndicate.rag.retriever import QueryContext, RAGRetriever


class TestChunking:
    """Test chunking functionality."""

    def test_semantic_chunker_python_code(self):
        """Test chunking Python code."""
        chunker = SemanticChunker(max_chunk_size=500, overlap=50)

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

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.chunk_type == ChunkType.CODE for chunk in chunks)

    def test_semantic_chunker_markdown(self):
        """Test chunking Markdown content."""
        chunker = SemanticChunker(max_chunk_size=300, overlap=30)

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

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.chunk_type == ChunkType.MARKDOWN for chunk in chunks)


class TestIndexing:
    """Test document indexing functionality."""

    @pytest.mark.asyncio
    async def test_index_directory(self):
        """Test indexing a directory of files."""
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

            assert progress.processed_documents == 2
            assert progress.total_chunks > 0
            assert progress.failed_documents == 0

            # Check stats
            stats = indexer.get_indexing_stats()
            assert stats["total_files"] == 2
            assert ".py" in stats["file_types"]
            assert ".md" in stats["file_types"]

    @pytest.mark.asyncio
    async def test_index_single_file(self):
        """Test indexing a single file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.py"

            test_file.write_text(
                """
def hello():
    print("Hello, World!")
"""
            )

            indexer = DocumentIndexer()
            chunks = await indexer.index_file(test_file)

            assert len(chunks) > 0
            assert all(isinstance(chunk, Chunk) for chunk in chunks)


class TestRetrieval:
    """Test retrieval functionality."""

    @pytest.mark.asyncio
    async def test_retriever_initialization(self):
        """Test retriever initialization."""
        retriever = RAGRetriever(max_results=5)
        await retriever.initialize()

        stats = retriever.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["total_queries"] == 0
        assert stats["embedding_model"] == "all-MiniLM-L6-v2"

    @pytest.mark.asyncio
    async def test_add_and_retrieve_chunks(self):
        """Test adding chunks and retrieving them."""
        retriever = RAGRetriever(max_results=5)
        await retriever.initialize()

        # Create test chunks
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

        # Add chunks
        await retriever.add_chunks(test_chunks)

        # Test retrieval
        query_context = QueryContext(query="fibonacci function", task_type="coding")

        results = await retriever.retrieve(query_context, max_results=3)

        assert len(results) > 0
        assert all(hasattr(result, "chunk") for result in results)
        assert all(hasattr(result, "score") for result in results)
        assert all(0 <= result.score <= 1 for result in results)


class TestContextIntegration:
    """Test context integration functionality."""

    @pytest.mark.asyncio
    async def test_context_builder(self):
        """Test context building from retrieval results."""
        # Create mock retrieval results
        from synndicate.rag.retriever import RetrievalResult, SearchMode

        test_chunk = Chunk(
            content="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            chunk_type=ChunkType.CODE,
            start_index=0,
            end_index=80,
            metadata={"file_extension": ".py"},
        )

        retrieval_result = RetrievalResult.from_chunk(
            chunk=test_chunk, score=0.9, search_mode=SearchMode.HYBRID
        )

        context_builder = ContextBuilder(max_context_tokens=1000)
        integrated_context = await context_builder.build_context([retrieval_result])

        assert integrated_context.chunk_count == 1
        assert integrated_context.total_tokens > 0
        assert integrated_context.compression_ratio > 0
        assert len(integrated_context.content) > 0

    @pytest.mark.asyncio
    async def test_context_integrator_agent_formatting(self):
        """Test agent-specific context formatting."""
        from synndicate.rag.retriever import RetrievalResult, SearchMode

        test_chunk = Chunk(
            content="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            chunk_type=ChunkType.CODE,
            start_index=0,
            end_index=80,
            metadata={"file_extension": ".py"},
        )

        retrieval_result = RetrievalResult.from_chunk(
            chunk=test_chunk, score=0.9, search_mode=SearchMode.HYBRID
        )

        context_integrator = ContextIntegrator()

        # Test different agent types
        for agent_type in ["planner", "coder", "critic"]:
            agent_context = await context_integrator.integrate_for_agent(
                agent_type=agent_type,
                retrieval_results=[retrieval_result],
                conversation_id="test_conversation",
            )

            assert agent_context.chunk_count == 1
            assert agent_context.total_tokens > 0
            assert agent_context.metadata["agent_type"] == agent_type
            assert agent_context.metadata["formatted"] is True
