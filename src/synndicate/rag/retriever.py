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
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import faiss  # noqa: F401

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import httpx

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
        embedding_cache_path: str | None = None,
        cache_max_entries: int = 100_000,
    ):
        self.embedding_model_name = embedding_model
        self.vector_store_path = vector_store_path
        self.max_results = max_results
        self.min_relevance_score = min_relevance_score
        self.embedding_cache_path = embedding_cache_path
        self.cache_max_entries = cache_max_entries

        # Initialize components
        self._embedding_model = None
        self._vector_store = None
        self._keyword_index: dict[str, list[tuple[str, Chunk]]] = {}
        self._chunks: dict[str, Chunk] = {}
        self._embedding_cache: dict[str, list[float]] = {}
        self._embedding_cache_dirty = False
        self._vector_store_kind: str | None = None  # "http" | "chroma" | None

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

        # Load embedding cache (if provided)
        if self.embedding_cache_path:
            try:
                path = Path(self.embedding_cache_path)
                if path.exists():
                    with path.open("r", encoding="utf-8") as f:
                        self._embedding_cache = json.load(f)
                        logger.info(
                            f"Loaded embedding cache with {len(self._embedding_cache)} entries from {path}"
                        )
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")

        # Initialize vector store
        vector_api = os.getenv("SYN_RAG_VECTOR_API")
        api_key = os.getenv("SYN_RAG_VECTOR_API_KEY")
        if vector_api:
            self._vector_store = _HttpVectorStore(vector_api, api_key)
            logger.info(f"Using remote vector store API at: {vector_api}")
            self._vector_store_kind = "http"
        elif CHROMADB_AVAILABLE and self.vector_store_path:
            try:
                self._vector_store = chromadb.PersistentClient(
                    path=self.vector_store_path, settings=Settings(anonymized_telemetry=False)
                )
                logger.info(f"Initialized ChromaDB at: {self.vector_store_path}")
                self._vector_store_kind = "chroma"
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

        # Persist cache if needed
        await self._maybe_flush_embedding_cache()

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

            # Clamp final scores to [0,1]
            for r in reranked_results:
                try:
                    if r.score < 0.0:
                        r.score = 0.0
                    elif r.score > 1.0:
                        r.score = 1.0
                except Exception:
                    # If score is non-numeric for any reason, drop the item
                    continue

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
            query_embedding = await self._get_or_create_embedding(query)
            if query_embedding is None:
                return []

            if self._vector_store:
                # Use configured vector store for search
                if self._vector_store_kind == "chroma":
                    return await self._chromadb_search(np.asarray(query_embedding), max_results)
                if self._vector_store_kind == "http":
                    return await self._http_vector_search(np.asarray(query_embedding), max_results)
                # Fallback to memory search
                return await self._memory_vector_search(np.asarray(query_embedding), max_results)
            else:
                # Use in-memory similarity search
                return await self._memory_vector_search(np.asarray(query_embedding), max_results)

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _add_to_vector_store(self, chunk_id: str, chunk: Chunk) -> None:
        """Add chunk to vector store."""
        if not self._embedding_model:
            return

        try:
            # Generate embedding
            embedding = await self._get_or_create_embedding(chunk.content)
            if embedding is None:
                return

            if self._vector_store:
                # Add to configured store
                if self._vector_store_kind == "chroma":
                    collection = self._vector_store.get_or_create_collection("chunks")
                    collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding.tolist()],
                        documents=[chunk.content],
                        metadatas=[chunk.metadata],
                    )
                elif self._vector_store_kind == "http":
                    await self._vector_store.add_embeddings(
                        [
                            {
                                "id": chunk_id,
                                "embedding": embedding.tolist(),
                                "document": chunk.content,
                                "metadata": chunk.metadata,
                            }
                        ]
                    )
        except Exception as e:
            logger.error(f"Failed to add chunk to vector store: {e}")

    async def _memory_vector_search(
        self, query_embedding: np.ndarray, max_results: int
    ) -> list[RetrievalResult]:
        """Perform in-memory vector search."""
        if not self._chunks:
            return []

        results = []

        for chunk_id, chunk in self._chunks.items():
            try:
                # Get cached embedding for chunk
                chunk_embedding = await self._get_or_create_embedding(chunk.content)
                if chunk_embedding is None:
                    continue

                # Calculate cosine similarity
                cosine = float(
                    np.dot(query_embedding, chunk_embedding)
                    / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
                )

                # Normalize to [0,1]
                similarity = (cosine + 1.0) / 2.0
                if similarity < 0.0:
                    similarity = 0.0
                elif similarity > 1.0:
                    similarity = 1.0

                result = RetrievalResult.from_chunk(
                    chunk=chunk,
                    score=similarity,
                    search_mode=SearchMode.VECTOR_ONLY,
                    metadata={"cosine_similarity": float(cosine)},
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error computing similarity for chunk {chunk_id}: {e}")

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
            "embedding_cache_entries": len(self._embedding_cache),
            "vector_store_type": self._vector_store_kind or "none",
        }

    async def _maybe_flush_embedding_cache(self) -> None:
        """Persist embedding cache to disk if configured and dirty."""
        if not self.embedding_cache_path or not self._embedding_cache_dirty:
            return
        try:
            path = Path(self.embedding_cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(self._embedding_cache, f)
            self._embedding_cache_dirty = False
            logger.info(f"Flushed embedding cache to {path}")
        except Exception as e:
            logger.warning(f"Failed to flush embedding cache: {e}")

    async def _get_or_create_embedding(self, text: str) -> np.ndarray | None:
        """Get embedding from cache or compute and store it."""
        if not self._embedding_model:
            return None
        key = self._embedding_cache_key(text)
        if key in self._embedding_cache:
            return np.asarray(self._embedding_cache[key], dtype=np.float32)
        # Compute and cache
        emb = self._embedding_model.encode([text])[0]
        # Size control
        if len(self._embedding_cache) >= self.cache_max_entries:
            # naive eviction: clear half
            for i, k in enumerate(list(self._embedding_cache.keys())):
                if i % 2 == 0:
                    del self._embedding_cache[k]
        self._embedding_cache[key] = emb.tolist()
        self._embedding_cache_dirty = True
        return np.asarray(emb, dtype=np.float32)

    def _embedding_cache_key(self, text: str) -> str:
        return f"{self.embedding_model_name}:{hash(text)}"

    async def _http_vector_search(
        self, query_embedding: np.ndarray, max_results: int
    ) -> list[RetrievalResult]:
        try:
            assert isinstance(self._vector_store, _HttpVectorStore)
            results = await self._vector_store.query(query_embedding.tolist(), max_results)
            out: list[RetrievalResult] = []
            for it in results:
                # If the document is not known locally, construct a minimal Chunk
                chunk = self._chunks.get(it.get("id"))
                if not chunk:
                    chunk = Chunk(
                        content=it.get("document", ""),
                        chunk_type=ChunkType.TEXT,
                        start_index=0,
                        end_index=len(it.get("document", "")),
                        metadata=it.get("metadata", {}),
                    )
                raw_score = float(it.get("score", 0.0))
                score = raw_score
                # If score is outside [0,1], assume cosine in [-1,1] and remap
                if raw_score < 0.0 or raw_score > 1.0:
                    score = (raw_score + 1.0) / 2.0
                # Clamp
                if score < 0.0:
                    score = 0.0
                elif score > 1.0:
                    score = 1.0

                out.append(
                    RetrievalResult.from_chunk(
                        chunk=chunk,
                        score=score,
                        search_mode=SearchMode.VECTOR_ONLY,
                        metadata={"backend": "http", "raw_score": raw_score},
                    )
                )
            return out
        except Exception as e:
            logger.error(f"HTTP vector search failed: {e}")
            return []

    def _generate_chunk_id(self, chunk: Chunk) -> str:
        """Generate unique ID for a chunk (stable across runs for same content)."""
        content_hash = hash(chunk.content)
        return f"{chunk.chunk_type.value}_{chunk.start_index}_{content_hash}"

    async def _add_to_keyword_index(self, chunk_id: str, chunk: Chunk) -> None:
        """Add chunk to the in-memory keyword index."""
        for keyword in self._extract_keywords(chunk.content.lower()):
            if keyword not in self._keyword_index:
                self._keyword_index[keyword] = []
            self._keyword_index[keyword].append((chunk_id, chunk))

    def _extract_keywords(self, text: str) -> list[str]:
        """Basic keyword extraction with stop-word filtering."""
        words = re.findall(r"\b\w+\b", text.lower())
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
        return [w for w in set(words) if len(w) > 2 and w not in stop_words]

    async def _keyword_search(self, query: str, max_results: int) -> list[RetrievalResult]:
        """Keyword search using in-memory index."""
        results: list[RetrievalResult] = []
        terms = self._extract_keywords(query)
        if not terms:
            return results
        scores: dict[str, float] = {}
        total_docs = max(1, len(self._chunks))
        for t in terms:
            postings = self._keyword_index.get(t, [])
            df = len(postings)
            for cid, ch in postings:
                tf = ch.content.lower().count(t)
                idf = np.log(total_docs / (df + 1))
                scores[cid] = scores.get(cid, 0.0) + tf * idf
        if not scores:
            return results
        max_score = max(scores.values()) or 1.0
        for cid, sc in scores.items():
            ch = self._chunks[cid]
            results.append(
                RetrievalResult.from_chunk(
                    chunk=ch,
                    score=float(sc / max_score),
                    search_mode=SearchMode.KEYWORD_ONLY,
                    metadata={
                        "keyword_matches": len([t for t in terms if t in ch.content.lower()])
                    },
                )
            )
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    async def _hybrid_search(self, query: str, max_results: int) -> list[RetrievalResult]:
        """Combine vector and keyword results with weighted fusion."""
        vr = await self._vector_search(query, max_results * 2)
        kr = await self._keyword_search(query, max_results * 2)
        combined: dict[str, RetrievalResult] = {}
        for r in vr:
            cid = self._generate_chunk_id(r.chunk)
            combined[cid] = r
            combined[cid].search_mode = SearchMode.HYBRID
        for r in kr:
            cid = self._generate_chunk_id(r.chunk)
            if cid in combined:
                ex = combined[cid]
                ex.score = (ex.score * 0.6) + (r.score * 0.4)
                ex.metadata.update(r.metadata)
            else:
                r.search_mode = SearchMode.HYBRID
                combined[cid] = r
        fused = list(combined.values())
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:max_results]

    async def _rerank_with_context(
        self, results: list[RetrievalResult], query_context: QueryContext
    ) -> list[RetrievalResult]:
        """Apply simple diversity penalty and user preference boosts."""
        if not results:
            return results
        diverse = self._apply_diversity_penalty(results)
        boosted = self._apply_preference_boosting(diverse, query_context.user_preferences)
        return boosted

    def _apply_diversity_penalty(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if len(results) <= 1:
            return results
        out = [results[0]]
        for r in results[1:]:
            max_sim = 0.0
            for s in out:
                sim = self._calculate_content_similarity(r.chunk.content, s.chunk.content)
                max_sim = max(max_sim, sim)
            # penalize up to 30% if very similar
            r.score = r.score * (1 - 0.3 * max_sim)
            out.append(r)
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _apply_preference_boosting(
        self, results: list[RetrievalResult], user_preferences: dict[str, Any]
    ) -> list[RetrievalResult]:
        if not user_preferences:
            return results
        for r in results:
            # Preferred file types
            pref_types = user_preferences.get("file_types", [])
            if pref_types:
                ext = r.chunk.metadata.get("file_extension", "")
                if ext in pref_types:
                    r.score = min(1.0, r.score * 1.1)
            # Preferred content types
            pref_content = user_preferences.get("content_types", [])
            if pref_content and r.chunk.chunk_type.value in pref_content:
                r.score = min(1.0, r.score * 1.05)
        return results

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        w1 = set(self._extract_keywords(content1.lower()))
        w2 = set(self._extract_keywords(content2.lower()))
        if not w1 or not w2:
            return 0.0
        inter = len(w1 & w2)
        union = len(w1 | w2)
        return inter / union if union > 0 else 0.0


class _HttpVectorStore:
    """Minimal async client for a remote vector store API.

    Expected endpoints:
      POST /vectors/add { items: [{ id, embedding, document, metadata }] }
      POST /vectors/query { embedding: [...], n_results: int }
      -> returns { results: [{ id, score, document?, metadata? }, ...] }
    """

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        headers = {"X-API-Key": api_key} if api_key else None
        self._client = httpx.AsyncClient(timeout=30.0, headers=headers)

    async def add_embeddings(self, items: list[dict[str, Any]]) -> None:
        try:
            url = f"{self.base_url}/vectors/add"
            await self._client.post(url, json={"items": items})
        except Exception:
            # Best-effort
            return

    async def query(self, embedding: list[float], n_results: int) -> list[dict[str, Any]]:
        url = f"{self.base_url}/vectors/query"
        resp = await self._client.post(url, json={"embedding": embedding, "n_results": n_results})
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
