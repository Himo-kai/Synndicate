"""
Modern RAG (Retrieval-Augmented Generation) system with hybrid search and smart chunking.

Improvements over original:
- Hybrid retrieval (vector + keyword + graph-based)
- Semantic chunking instead of fixed-size
- Async indexing with progress tracking
- Retrieval evaluation and feedback loops
- Context integration with agent workflows
"""

from .chunking import ChunkingStrategy, FixedSizeChunker, SemanticChunker
from .context import ContextBuilder, ContextIntegrator
from .indexer import DocumentIndexer, IndexingProgress
from .retriever import RAGRetriever, RetrievalResult

__all__ = [
    "RAGRetriever",
    "RetrievalResult",
    "ChunkingStrategy",
    "SemanticChunker",
    "FixedSizeChunker",
    "DocumentIndexer",
    "IndexingProgress",
    "ContextBuilder",
    "ContextIntegrator",
]
