"""
Advanced RAG retriever with hybrid search capabilities.

Improvements over original:
- Hybrid search combining vector, keyword, and semantic similarity
- Query expansion and reformulation
- Result ranking and relevance scoring
- Context-aware retrieval with conversation history
- Retrieval evaluation and feedback loops
"""

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..observability.logging import get_logger
from .chunking import Chunk, ChunkType

logger = get_logger(__name__)


class SearchMode(Enum):
    """Search modes for retrieval."""

    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"


class RelevanceScore(Enum):
    """Relevance score levels."""

    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"  # 0.7 - 0.9
    MEDIUM = "medium"  # 0.5 - 0.7
    LOW = "low"  # 0.3 - 0.5
    VERY_LOW = "very_low"  # < 0.3


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    chunk: Chunk
    score: float
    relevance: RelevanceScore
    search_mode: SearchMode
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        score: float,
        search_mode: SearchMode,
        metadata: dict[str, Any] | None = None,
    ) -> "RetrievalResult":
        """Create result from chunk and score."""
        # Determine relevance level
        if score >= 0.9:
            relevance = RelevanceScore.VERY_HIGH
        elif score >= 0.7:
            relevance = RelevanceScore.HIGH
        elif score >= 0.5:
            relevance = RelevanceScore.MEDIUM
        elif score >= 0.3:
            relevance = RelevanceScore.LOW
        else:
            relevance = RelevanceScore.VERY_LOW

        return cls(
            chunk=chunk,
            score=score,
            relevance=relevance,
            search_mode=search_mode,
            metadata=metadata or {},
        )


@dataclass
class QueryContext:
    """Context for retrieval queries."""

    query: str
    conversation_history: list[str] = field(default_factory=list)
    user_preferences: dict[str, Any] = field(default_factory=dict)
    domain_context: str | None = None
    task_type: str | None = None

    def get_expanded_query(self) -> str:
        """Get query expanded with context."""
        expanded_parts = [self.query]

        if self.domain_context:
            expanded_parts.append(f"Domain: {self.domain_context}")

        if self.task_type:
            expanded_parts.append(f"Task: {self.task_type}")

        # Add relevant conversation context
        if self.conversation_history:
            recent_context = " ".join(self.conversation_history[-3:])  # Last 3 messages
            expanded_parts.append(f"Context: {recent_context}")

        return " ".join(expanded_parts)


class RAGRetriever:
    """
    Advanced RAG retriever with hybrid search capabilities.

    Features:
    - Multiple search modes (vector, keyword, hybrid, semantic)
    - Query expansion and reformulation
    - Result ranking and relevance scoring
    - Context-aware retrieval
    - Retrieval evaluation and feedback
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_store_path: str | None = None,
        max_results: int = 10,
        min_relevance_score: float = 0.3,
    ):
        self.embedding_model_name = embedding_model
        self.vector_store_path = vector_store_path
        self.max_results = max_results
        self.min_relevance_score = min_relevance_score

        # Initialize components
        self._embedding_model = None
        self._vector_store = None
        self._keyword_index: dict[str, list[tuple[str, Chunk]]] = {}
        self._chunks: dict[str, Chunk] = {}

        # Performance tracking
        self._query_count = 0
        self._total_retrieval_time = 0.0

    async def initialize(self) -> None:
        """Initialize the retriever components."""
        logger.info("Initializing RAG retriever...")

        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        else:
            logger.warning("sentence-transformers not available, vector search disabled")

        # Initialize vector store
        if CHROMADB_AVAILABLE and self.vector_store_path:
            try:
                self._vector_store = chromadb.PersistentClient(
                    path=self.vector_store_path, settings=Settings(anonymized_telemetry=False)
                )
                logger.info(f"Initialized ChromaDB at: {self.vector_store_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize ChromaDB: {e}")
        else:
            logger.warning("ChromaDB not available or no path specified, using in-memory storage")

    async def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the retrieval index."""
        logger.info(f"Adding {len(chunks)} chunks to index")

        for chunk in chunks:
            chunk_id = self._generate_chunk_id(chunk)
            self._chunks[chunk_id] = chunk

            # Add to keyword index
            await self._add_to_keyword_index(chunk_id, chunk)

            # Add to vector store if available
            if self._vector_store and self._embedding_model:
                await self._add_to_vector_store(chunk_id, chunk)

        logger.info(f"Successfully indexed {len(chunks)} chunks")

    async def retrieve(
        self,
        query_context: QueryContext,
        search_mode: SearchMode = SearchMode.HYBRID,
        max_results: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query."""
        start_time = asyncio.get_event_loop().time()
        max_results = max_results or self.max_results

        try:
            # Expand query with context
            expanded_query = query_context.get_expanded_query()

            # Perform search based on mode
            if search_mode == SearchMode.VECTOR_ONLY:
                results = await self._vector_search(expanded_query, max_results)
            elif search_mode == SearchMode.KEYWORD_ONLY:
                results = await self._keyword_search(expanded_query, max_results)
            elif search_mode == SearchMode.SEMANTIC:
                results = await self._semantic_search(query_context, max_results)
            else:  # HYBRID
                results = await self._hybrid_search(expanded_query, max_results)

            # Filter by minimum relevance score
            filtered_results = [
                result for result in results if result.score >= self.min_relevance_score
            ]

            # Re-rank results with context
            reranked_results = await self._rerank_with_context(filtered_results, query_context)

            # Update metrics
            self._query_count += 1
            retrieval_time = asyncio.get_event_loop().time() - start_time
            self._total_retrieval_time += retrieval_time

            logger.info(
                f"Retrieved {len(reranked_results)} results for query "
                f"in {retrieval_time:.3f}s (mode: {search_mode.value})"
            )

            return reranked_results[:max_results]

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    async def _vector_search(self, query: str, max_results: int) -> list[RetrievalResult]:
        """Perform vector similarity search."""
        if not self._embedding_model:
            logger.warning("Vector search requested but embedding model not available")
            return []

        try:
            # Generate query embedding
            query_embedding = self._embedding_model.encode([query])[0]

            if self._vector_store:
                # Use ChromaDB for search
                return await self._chromadb_search(query_embedding, max_results)
            else:
                # Use in-memory similarity search
                return await self._memory_vector_search(query_embedding, max_results)

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _keyword_search(self, query: str, max_results: int) -> list[RetrievalResult]:
        """Perform keyword-based search."""
        results = []
        query_terms = self._extract_keywords(query.lower())

        # Score chunks based on keyword matches
        chunk_scores: dict[str, float] = {}

        for term in query_terms:
            if term in self._keyword_index:
                for chunk_id, chunk in self._keyword_index[term]:
                    if chunk_id not in chunk_scores:
                        chunk_scores[chunk_id] = 0.0

                    # Simple TF-IDF-like scoring
                    term_frequency = chunk.content.lower().count(term)
                    doc_frequency = len(self._keyword_index[term])
                    total_docs = len(self._chunks)

                    idf = np.log(total_docs / (doc_frequency + 1))
                    score = term_frequency * idf
                    chunk_scores[chunk_id] += score

        # Normalize scores and create results
        if chunk_scores:
            max_score = max(chunk_scores.values())
            for chunk_id, score in chunk_scores.items():
                normalized_score = score / max_score if max_score > 0 else 0.0
                chunk = self._chunks[chunk_id]

                result = RetrievalResult.from_chunk(
                    chunk=chunk,
                    score=normalized_score,
                    search_mode=SearchMode.KEYWORD_ONLY,
                    metadata={
                        "keyword_matches": len(
                            [t for t in query_terms if t in chunk.content.lower()]
                        )
                    },
                )
                results.append(result)

        # Sort by score and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    async def _hybrid_search(self, query: str, max_results: int) -> list[RetrievalResult]:
        """Perform hybrid search combining vector and keyword search."""
        # Get results from both methods
        vector_results = await self._vector_search(query, max_results * 2)
        keyword_results = await self._keyword_search(query, max_results * 2)

        # Combine and deduplicate results
        combined_results: dict[str, RetrievalResult] = {}

        # Add vector results
        for result in vector_results:
            chunk_id = self._generate_chunk_id(result.chunk)
            combined_results[chunk_id] = result
            combined_results[chunk_id].search_mode = SearchMode.HYBRID

        # Merge keyword results
        for result in keyword_results:
            chunk_id = self._generate_chunk_id(result.chunk)
            if chunk_id in combined_results:
                # Combine scores (weighted average)
                existing_result = combined_results[chunk_id]
                combined_score = (existing_result.score * 0.6) + (result.score * 0.4)
                existing_result.score = combined_score
                existing_result.metadata.update(result.metadata)
            else:
                result.search_mode = SearchMode.HYBRID
                combined_results[chunk_id] = result

        # Sort by combined score
        results = list(combined_results.values())
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:max_results]

    async def _semantic_search(
        self, query_context: QueryContext, max_results: int
    ) -> list[RetrievalResult]:
        """Perform semantic search with context understanding."""
        # Start with hybrid search
        results = await self._hybrid_search(query_context.query, max_results * 2)

        # Apply semantic filtering and boosting
        semantic_results = []

        for result in results:
            # Boost score based on chunk type relevance
            type_boost = self._get_type_relevance_boost(
                result.chunk.chunk_type, query_context.task_type
            )

            # Boost score based on domain relevance
            domain_boost = self._get_domain_relevance_boost(
                result.chunk, query_context.domain_context
            )

            # Apply conversation context boost
            context_boost = self._get_context_relevance_boost(
                result.chunk, query_context.conversation_history
            )

            # Calculate final semantic score
            semantic_score = result.score * (1 + type_boost + domain_boost + context_boost)
            semantic_score = min(semantic_score, 1.0)  # Cap at 1.0

            semantic_result = RetrievalResult.from_chunk(
                chunk=result.chunk,
                score=semantic_score,
                search_mode=SearchMode.SEMANTIC,
                metadata={
                    **result.metadata,
                    "type_boost": type_boost,
                    "domain_boost": domain_boost,
                    "context_boost": context_boost,
                    "original_score": result.score,
                },
            )
            semantic_results.append(semantic_result)

        # Sort by semantic score
        semantic_results.sort(key=lambda x: x.score, reverse=True)
        return semantic_results[:max_results]

    async def _rerank_with_context(
        self, results: list[RetrievalResult], query_context: QueryContext
    ) -> list[RetrievalResult]:
        """Re-rank results based on additional context."""
        if not results:
            return results

        # Apply diversity penalty to avoid too similar results
        diverse_results = self._apply_diversity_penalty(results)

        # Apply user preference boosting
        preference_boosted = self._apply_preference_boosting(
            diverse_results, query_context.user_preferences
        )

        return preference_boosted

    def _generate_chunk_id(self, chunk: Chunk) -> str:
        """Generate unique ID for a chunk."""
        content_hash = hash(chunk.content)
        return f"{chunk.chunk_type.value}_{chunk.start_index}_{content_hash}"

    async def _add_to_keyword_index(self, chunk_id: str, chunk: Chunk) -> None:
        """Add chunk to keyword index."""
        keywords = self._extract_keywords(chunk.content.lower())

        for keyword in keywords:
            if keyword not in self._keyword_index:
                self._keyword_index[keyword] = []
            self._keyword_index[keyword].append((chunk_id, chunk))

    async def _add_to_vector_store(self, chunk_id: str, chunk: Chunk) -> None:
        """Add chunk to vector store."""
        if not self._embedding_model:
            return

        try:
            # Generate embedding
            embedding = self._embedding_model.encode([chunk.content])[0]

            if self._vector_store:
                # Add to ChromaDB
                collection = self._vector_store.get_or_create_collection("chunks")
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding.tolist()],
                    documents=[chunk.content],
                    metadatas=[chunk.metadata],
                )
        except Exception as e:
            logger.error(f"Failed to add chunk to vector store: {e}")

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        # Simple keyword extraction (can be enhanced with NLP libraries)
        # Remove punctuation and split into words
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
        }

        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return list(set(keywords))  # Remove duplicates

    def _get_type_relevance_boost(self, chunk_type: ChunkType, task_type: str | None) -> float:
        """Get relevance boost based on chunk type and task type."""
        if not task_type:
            return 0.0

        # Define type relevance mappings
        type_mappings = {
            "coding": {ChunkType.CODE: 0.3, ChunkType.DOCUMENTATION: 0.1},
            "documentation": {ChunkType.DOCUMENTATION: 0.3, ChunkType.MARKDOWN: 0.2},
            "analysis": {ChunkType.TEXT: 0.2, ChunkType.DOCUMENTATION: 0.1},
            "debugging": {ChunkType.CODE: 0.3, ChunkType.COMMENT: 0.1},
        }

        task_lower = task_type.lower()
        for task_key, boosts in type_mappings.items():
            if task_key in task_lower:
                return boosts.get(chunk_type, 0.0)

        return 0.0

    def _get_domain_relevance_boost(self, chunk: Chunk, domain_context: str | None) -> float:
        """Get relevance boost based on domain context."""
        if not domain_context:
            return 0.0

        domain_lower = domain_context.lower()
        content_lower = chunk.content.lower()

        # Simple domain keyword matching
        domain_keywords = self._extract_keywords(domain_lower)
        content_keywords = self._extract_keywords(content_lower)

        matches = len(set(domain_keywords) & set(content_keywords))
        total_domain_keywords = len(domain_keywords)

        if total_domain_keywords == 0:
            return 0.0

        return min(0.2, (matches / total_domain_keywords) * 0.2)

    def _get_context_relevance_boost(self, chunk: Chunk, conversation_history: list[str]) -> float:
        """Get relevance boost based on conversation context."""
        if not conversation_history:
            return 0.0

        # Extract keywords from recent conversation
        recent_context = " ".join(conversation_history[-3:])
        context_keywords = set(self._extract_keywords(recent_context.lower()))
        chunk_keywords = set(self._extract_keywords(chunk.content.lower()))

        if not context_keywords:
            return 0.0

        matches = len(context_keywords & chunk_keywords)
        return min(0.15, (matches / len(context_keywords)) * 0.15)

    def _apply_diversity_penalty(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Apply diversity penalty to avoid too similar results."""
        if len(results) <= 1:
            return results

        diverse_results = [results[0]]  # Always include top result

        for result in results[1:]:
            # Check similarity with already selected results
            max_similarity = 0.0

            for selected in diverse_results:
                similarity = self._calculate_content_similarity(
                    result.chunk.content, selected.chunk.content
                )
                max_similarity = max(max_similarity, similarity)

            # Apply penalty based on similarity
            diversity_penalty = max_similarity * 0.3
            result.score = result.score * (1 - diversity_penalty)
            diverse_results.append(result)

        # Re-sort after applying penalties
        diverse_results.sort(key=lambda x: x.score, reverse=True)
        return diverse_results

    def _apply_preference_boosting(
        self, results: list[RetrievalResult], user_preferences: dict[str, Any]
    ) -> list[RetrievalResult]:
        """Apply user preference boosting to results."""
        if not user_preferences:
            return results

        for result in results:
            # Boost based on preferred file types
            preferred_types = user_preferences.get("file_types", [])
            if preferred_types:
                file_ext = result.chunk.metadata.get("file_extension", "")
                if file_ext in preferred_types:
                    result.score = min(1.0, result.score * 1.1)

            # Boost based on preferred content types
            preferred_content = user_preferences.get("content_types", [])
            if preferred_content and result.chunk.chunk_type.value in preferred_content:
                result.score = min(1.0, result.score * 1.05)

        return results

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity between two texts."""
        words1 = set(self._extract_keywords(content1.lower()))
        words2 = set(self._extract_keywords(content2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    async def _chromadb_search(
        self, query_embedding: np.ndarray, max_results: int
    ) -> list[RetrievalResult]:
        """Search using ChromaDB."""
        try:
            collection = self._vector_store.get_or_create_collection("chunks")
            results = collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=max_results
            )

            retrieval_results = []
            for i, (doc_id, document, metadata, distance) in enumerate(
                zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0], strict=False,
                )
            ):
                # Convert distance to similarity score
                score = 1.0 / (1.0 + distance)

                # Reconstruct chunk (simplified)
                chunk = Chunk(
                    content=document,
                    chunk_type=ChunkType.TEXT,  # Would need to store and retrieve actual type
                    start_index=0,
                    end_index=len(document),
                    metadata=metadata,
                )

                result = RetrievalResult.from_chunk(
                    chunk=chunk,
                    score=score,
                    search_mode=SearchMode.VECTOR_ONLY,
                    metadata={"chromadb_distance": distance},
                )
                retrieval_results.append(result)

            return retrieval_results

        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

    async def _memory_vector_search(
        self, query_embedding: np.ndarray, max_results: int
    ) -> list[RetrievalResult]:
        """Perform in-memory vector search."""
        if not self._chunks:
            return []

        results = []

        for chunk_id, chunk in self._chunks.items():
            try:
                # Generate embedding for chunk (this would be cached in practice)
                chunk_embedding = self._embedding_model.encode([chunk.content])[0]

                # Calculate cosine similarity
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )

                result = RetrievalResult.from_chunk(
                    chunk=chunk,
                    score=float(similarity),
                    search_mode=SearchMode.VECTOR_ONLY,
                    metadata={"cosine_similarity": float(similarity)},
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to compute similarity for chunk {chunk_id}: {e}")
                continue

        # Sort by score and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    def get_stats(self) -> dict[str, Any]:
        """Get retriever statistics."""
        avg_retrieval_time = (
            self._total_retrieval_time / self._query_count if self._query_count > 0 else 0.0
        )

        return {
            "total_chunks": len(self._chunks),
            "total_queries": self._query_count,
            "avg_retrieval_time": avg_retrieval_time,
            "keyword_index_size": len(self._keyword_index),
            "embedding_model": self.embedding_model_name,
            "vector_store_available": self._vector_store is not None,
            "embedding_model_available": self._embedding_model is not None,
        }
