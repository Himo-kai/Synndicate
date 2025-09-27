"""
Context integration for RAG system with agent workflows.

Improvements over original:
- Context-aware chunk selection and ranking
- Integration with agent conversation flows
- Dynamic context window management
- Context compression and summarization
- Multi-turn conversation context preservation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..observability.logging import get_logger
from .chunking import Chunk, ChunkType
from .retriever import QueryContext, RetrievalResult

logger = get_logger(__name__)


class ContextStrategy(Enum):
    """Strategies for context integration."""

    CONCATENATE = "concatenate"
    SUMMARIZE = "summarize"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class ContextPriority(Enum):
    """Priority levels for context chunks."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ContextChunk:
    """A chunk with context-specific metadata."""

    chunk: Chunk
    priority: ContextPriority
    relevance_score: float
    context_type: str
    position_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        """Get score weighted by position and priority."""
        priority_weights = {
            ContextPriority.CRITICAL: 2.0,
            ContextPriority.HIGH: 1.5,
            ContextPriority.MEDIUM: 1.0,
            ContextPriority.LOW: 0.7,
        }

        priority_weight = priority_weights[self.priority]
        return self.relevance_score * self.position_weight * priority_weight


@dataclass
class IntegratedContext:
    """Integrated context for agent consumption."""

    content: str
    chunks: list[ContextChunk]
    strategy: ContextStrategy
    total_tokens: int
    compression_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        """Get number of chunks in context."""
        return len(self.chunks)

    @property
    def avg_relevance(self) -> float:
        """Get average relevance score."""
        if not self.chunks:
            return 0.0
        return sum(chunk.relevance_score for chunk in self.chunks) / len(self.chunks)


class ContextBuilder:
    """
    Builds integrated context from retrieval results.

    Features:
    - Multiple context integration strategies
    - Dynamic context window management
    - Priority-based chunk selection
    - Context compression and summarization
    """

    def __init__(
        self,
        max_context_tokens: int = 4000,
        min_context_tokens: int = 500,
        overlap_tokens: int = 100,
    ):
        self.max_context_tokens = max_context_tokens
        self.min_context_tokens = min_context_tokens
        self.overlap_tokens = overlap_tokens

    async def build_context(
        self,
        retrieval_results: list[RetrievalResult],
        strategy: ContextStrategy = ContextStrategy.ADAPTIVE,
        query_context: QueryContext | None = None,
    ) -> IntegratedContext:
        """Build integrated context from retrieval results."""
        if not retrieval_results:
            return IntegratedContext(
                content="", chunks=[], strategy=strategy, total_tokens=0, compression_ratio=1.0
            )

        # Convert retrieval results to context chunks
        context_chunks = await self._create_context_chunks(retrieval_results, query_context)

        # Select and prioritize chunks based on strategy
        if strategy == ContextStrategy.CONCATENATE:
            integrated = await self._concatenate_strategy(context_chunks)
        elif strategy == ContextStrategy.SUMMARIZE:
            integrated = await self._summarize_strategy(context_chunks)
        elif strategy == ContextStrategy.HIERARCHICAL:
            integrated = await self._hierarchical_strategy(context_chunks)
        else:  # ADAPTIVE
            integrated = await self._adaptive_strategy(context_chunks, query_context)

        integrated.strategy = strategy
        return integrated

    async def _create_context_chunks(
        self, retrieval_results: list[RetrievalResult], query_context: QueryContext | None
    ) -> list[ContextChunk]:
        """Convert retrieval results to context chunks with priorities."""
        context_chunks = []

        for i, result in enumerate(retrieval_results):
            # Determine priority based on score and position
            priority = self._determine_priority(result.score, i)

            # Calculate position weight (earlier results get higher weight)
            position_weight = 1.0 / (1.0 + i * 0.1)

            # Determine context type
            context_type = self._determine_context_type(result.chunk, query_context)

            context_chunk = ContextChunk(
                chunk=result.chunk,
                priority=priority,
                relevance_score=result.score,
                context_type=context_type,
                position_weight=position_weight,
                metadata={
                    "original_rank": i,
                    "search_mode": result.search_mode.value,
                    "relevance_level": result.relevance.value,
                },
            )
            context_chunks.append(context_chunk)

        return context_chunks

    async def _concatenate_strategy(self, context_chunks: list[ContextChunk]) -> IntegratedContext:
        """Simple concatenation strategy with token limit."""
        selected_chunks = []
        current_tokens = 0
        content_parts = []

        # Sort by weighted score
        sorted_chunks = sorted(context_chunks, key=lambda x: x.weighted_score, reverse=True)

        for chunk in sorted_chunks:
            chunk_tokens = self._estimate_tokens(chunk.chunk.content)

            if current_tokens + chunk_tokens <= self.max_context_tokens:
                selected_chunks.append(chunk)
                content_parts.append(f"## {chunk.context_type.title()}\n{chunk.chunk.content}")
                current_tokens += chunk_tokens
            elif current_tokens < self.min_context_tokens:
                # Force include if we haven't met minimum
                selected_chunks.append(chunk)
                content_parts.append(f"## {chunk.context_type.title()}\n{chunk.chunk.content}")
                current_tokens += chunk_tokens
                break

        content = "\n\n".join(content_parts)
        original_tokens = sum(
            self._estimate_tokens(chunk.chunk.content) for chunk in context_chunks
        )
        compression_ratio = current_tokens / original_tokens if original_tokens > 0 else 1.0

        return IntegratedContext(
            content=content,
            chunks=selected_chunks,
            strategy=ContextStrategy.CONCATENATE,
            total_tokens=current_tokens,
            compression_ratio=compression_ratio,
        )

    async def _summarize_strategy(self, context_chunks: list[ContextChunk]) -> IntegratedContext:
        """Summarization strategy for large contexts."""
        # Group chunks by type and priority
        grouped_chunks = self._group_chunks_by_type(context_chunks)

        content_parts = []
        selected_chunks = []
        current_tokens = 0

        for context_type, chunks in grouped_chunks.items():
            # Sort chunks by priority and score
            sorted_chunks = sorted(chunks, key=lambda x: x.weighted_score, reverse=True)

            # Take top chunks for this type
            type_content = []
            type_chunks = []
            type_tokens = 0
            max_type_tokens = self.max_context_tokens // len(grouped_chunks)

            for chunk in sorted_chunks:
                chunk_tokens = self._estimate_tokens(chunk.chunk.content)

                if type_tokens + chunk_tokens <= max_type_tokens:
                    type_content.append(chunk.chunk.content)
                    type_chunks.append(chunk)
                    type_tokens += chunk_tokens
                else:
                    break

            if type_content:
                # Create summary for this type
                type_summary = await self._create_type_summary(context_type, type_content)
                content_parts.append(f"## {context_type.title()}\n{type_summary}")
                selected_chunks.extend(type_chunks)
                current_tokens += self._estimate_tokens(type_summary)

        content = "\n\n".join(content_parts)
        original_tokens = sum(
            self._estimate_tokens(chunk.chunk.content) for chunk in context_chunks
        )
        compression_ratio = current_tokens / original_tokens if original_tokens > 0 else 1.0

        return IntegratedContext(
            content=content,
            chunks=selected_chunks,
            strategy=ContextStrategy.SUMMARIZE,
            total_tokens=current_tokens,
            compression_ratio=compression_ratio,
        )

    async def _hierarchical_strategy(self, context_chunks: list[ContextChunk]) -> IntegratedContext:
        """Hierarchical strategy organizing by priority and type."""
        # Group by priority first, then by type
        priority_groups = {}
        for chunk in context_chunks:
            if chunk.priority not in priority_groups:
                priority_groups[chunk.priority] = []
            priority_groups[chunk.priority].append(chunk)

        content_parts = []
        selected_chunks = []
        current_tokens = 0

        # Process in priority order
        priority_order = [
            ContextPriority.CRITICAL,
            ContextPriority.HIGH,
            ContextPriority.MEDIUM,
            ContextPriority.LOW,
        ]

        for priority in priority_order:
            if priority not in priority_groups:
                continue

            chunks = priority_groups[priority]
            type_groups = self._group_chunks_by_type(chunks)

            priority_content = []
            for context_type, type_chunks in type_groups.items():
                # Sort by score within type
                sorted_chunks = sorted(type_chunks, key=lambda x: x.relevance_score, reverse=True)

                for chunk in sorted_chunks:
                    chunk_tokens = self._estimate_tokens(chunk.chunk.content)

                    if current_tokens + chunk_tokens <= self.max_context_tokens:
                        priority_content.append(
                            f"### {context_type.title()}\n{chunk.chunk.content}"
                        )
                        selected_chunks.append(chunk)
                        current_tokens += chunk_tokens
                    else:
                        break

                if current_tokens >= self.max_context_tokens:
                    break

            if priority_content:
                content_parts.append(
                    f"# {priority.value.title()} Priority\n" + "\n\n".join(priority_content)
                )

            if current_tokens >= self.max_context_tokens:
                break

        content = "\n\n".join(content_parts)
        original_tokens = sum(
            self._estimate_tokens(chunk.chunk.content) for chunk in context_chunks
        )
        compression_ratio = current_tokens / original_tokens if original_tokens > 0 else 1.0

        return IntegratedContext(
            content=content,
            chunks=selected_chunks,
            strategy=ContextStrategy.HIERARCHICAL,
            total_tokens=current_tokens,
            compression_ratio=compression_ratio,
        )

    async def _adaptive_strategy(
        self, context_chunks: list[ContextChunk], query_context: QueryContext | None
    ) -> IntegratedContext:
        """Adaptive strategy that chooses best approach based on context."""
        # Analyze context characteristics
        total_chunks = len(context_chunks)
        avg_chunk_size = sum(len(chunk.chunk.content) for chunk in context_chunks) / total_chunks
        total_estimated_tokens = sum(
            self._estimate_tokens(chunk.chunk.content) for chunk in context_chunks
        )

        # Choose strategy based on characteristics
        if total_estimated_tokens <= self.max_context_tokens:
            # Small enough for concatenation
            return await self._concatenate_strategy(context_chunks)
        elif avg_chunk_size > 500 and total_chunks > 10:
            # Large chunks, many results - use summarization
            return await self._summarize_strategy(context_chunks)
        else:
            # Use hierarchical for better organization
            return await self._hierarchical_strategy(context_chunks)

    def _determine_priority(self, score: float, position: int) -> ContextPriority:
        """Determine priority based on score and position."""
        if score >= 0.9 or position == 0:
            return ContextPriority.CRITICAL
        elif score >= 0.7 or position <= 2:
            return ContextPriority.HIGH
        elif score >= 0.5 or position <= 5:
            return ContextPriority.MEDIUM
        else:
            return ContextPriority.LOW

    def _determine_context_type(self, chunk: Chunk, query_context: QueryContext | None) -> str:
        """Determine context type for a chunk."""
        # Base type on chunk type
        type_mapping = {
            ChunkType.CODE: "code",
            ChunkType.DOCUMENTATION: "documentation",
            ChunkType.MARKDOWN: "documentation",
            ChunkType.TEXT: "reference",
            ChunkType.COMMENT: "explanation",
        }

        base_type = type_mapping.get(chunk.chunk_type, "reference")

        # Enhance with file extension if available
        if "file_extension" in chunk.metadata:
            ext = chunk.metadata["file_extension"].lower()
            if ext in [".py", ".js", ".ts", ".rs", ".go"]:
                return f"{ext[1:]} code"
            elif ext in [".md", ".txt"]:
                return "documentation"

        return base_type

    def _group_chunks_by_type(self, chunks: list[ContextChunk]) -> dict[str, list[ContextChunk]]:
        """Group chunks by context type."""
        groups = {}
        for chunk in chunks:
            context_type = chunk.context_type
            if context_type not in groups:
                groups[context_type] = []
            groups[context_type].append(chunk)
        return groups

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4

    async def _create_type_summary(self, context_type: str, contents: list[str]) -> str:
        """Create a summary for a specific context type."""
        # Simple summarization - in practice, this could use an LLM
        combined_content = "\n\n".join(contents)

        if len(combined_content) <= 500:
            return combined_content

        # Extract key sentences (simple heuristic)
        sentences = combined_content.split(".")
        important_sentences = []

        for sentence in sentences[:10]:  # Take first 10 sentences
            sentence = sentence.strip()
            if len(sentence) > 20:  # Skip very short sentences
                important_sentences.append(sentence)

        summary = ". ".join(important_sentences)
        if len(summary) > 1000:
            summary = summary[:1000] + "..."

        return summary


class ContextIntegrator:
    """
    Integrates RAG context with agent workflows.

    Features:
    - Multi-turn conversation context management
    - Agent-specific context formatting
    - Context window optimization
    - Dynamic context updates
    """

    def __init__(self, context_builder: ContextBuilder | None = None):
        self.context_builder = context_builder or ContextBuilder()
        self._conversation_contexts: dict[str, list[IntegratedContext]] = {}
        self._agent_preferences: dict[str, dict[str, Any]] = {}

    async def integrate_for_agent(
        self,
        agent_type: str,
        retrieval_results: list[RetrievalResult],
        conversation_id: str,
        query_context: QueryContext | None = None,
    ) -> IntegratedContext:
        """Integrate context specifically for an agent type."""
        # Get agent preferences
        preferences = self._agent_preferences.get(agent_type, {})

        # Choose strategy based on agent type
        strategy = self._get_agent_strategy(agent_type, preferences)

        # Build context
        integrated_context = await self.context_builder.build_context(
            retrieval_results, strategy, query_context
        )

        # Format for agent
        formatted_context = await self._format_for_agent(
            agent_type, integrated_context, preferences
        )

        # Store in conversation history
        self._store_conversation_context(conversation_id, formatted_context)

        return formatted_context

    async def get_conversation_context(
        self, conversation_id: str, max_contexts: int = 5
    ) -> list[IntegratedContext]:
        """Get conversation context history."""
        contexts = self._conversation_contexts.get(conversation_id, [])
        return contexts[-max_contexts:]

    async def update_agent_preferences(self, agent_type: str, preferences: dict[str, Any]) -> None:
        """Update preferences for an agent type."""
        self._agent_preferences[agent_type] = preferences
        logger.info(f"Updated preferences for agent {agent_type}")

    def _get_agent_strategy(self, agent_type: str, preferences: dict[str, Any]) -> ContextStrategy:
        """Get appropriate context strategy for agent type."""
        # Default strategies by agent type
        default_strategies = {
            "planner": ContextStrategy.HIERARCHICAL,
            "coder": ContextStrategy.CONCATENATE,
            "critic": ContextStrategy.SUMMARIZE,
            "executor": ContextStrategy.ADAPTIVE,
        }

        # Check preferences override
        preferred_strategy = preferences.get("context_strategy")
        if preferred_strategy:
            try:
                return ContextStrategy(preferred_strategy)
            except ValueError:
                pass

        return default_strategies.get(agent_type, ContextStrategy.ADAPTIVE)

    async def _format_for_agent(
        self, agent_type: str, context: IntegratedContext, preferences: dict[str, Any]
    ) -> IntegratedContext:
        """Format context specifically for an agent type."""
        # Agent-specific formatting
        if agent_type == "coder":
            formatted_content = await self._format_for_coder(context)
        elif agent_type == "planner":
            formatted_content = await self._format_for_planner(context)
        elif agent_type == "critic":
            formatted_content = await self._format_for_critic(context)
        else:
            formatted_content = context.content

        # Create new context with formatted content
        formatted_context = IntegratedContext(
            content=formatted_content,
            chunks=context.chunks,
            strategy=context.strategy,
            total_tokens=self.context_builder._estimate_tokens(formatted_content),
            compression_ratio=context.compression_ratio,
            metadata={**context.metadata, "agent_type": agent_type, "formatted": True},
        )

        return formatted_context

    async def _format_for_coder(self, context: IntegratedContext) -> str:
        """Format context for coder agent."""
        parts = ["# Code Context\n"]

        # Separate code and documentation
        code_chunks = [c for c in context.chunks if c.chunk.chunk_type == ChunkType.CODE]
        doc_chunks = [
            c
            for c in context.chunks
            if c.chunk.chunk_type in [ChunkType.DOCUMENTATION, ChunkType.MARKDOWN]
        ]

        if code_chunks:
            parts.append("## Relevant Code\n")
            for chunk in code_chunks:
                lang = chunk.metadata.get("language", "text")
                parts.append(f"```{lang}\n{chunk.chunk.content}\n```\n")

        if doc_chunks:
            parts.append("## Documentation\n")
            for chunk in doc_chunks:
                parts.append(f"{chunk.chunk.content}\n")

        return "\n".join(parts)

    async def _format_for_planner(self, context: IntegratedContext) -> str:
        """Format context for planner agent."""
        parts = ["# Planning Context\n"]

        # Group by priority for planning
        priority_groups = {}
        for chunk in context.chunks:
            priority = chunk.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(chunk)

        for priority in [ContextPriority.CRITICAL, ContextPriority.HIGH, ContextPriority.MEDIUM]:
            if priority in priority_groups:
                parts.append(f"## {priority.value.title()} Information\n")
                for chunk in priority_groups[priority]:
                    parts.append(f"- {chunk.context_type}: {chunk.chunk.content[:200]}...\n")

        return "\n".join(parts)

    async def _format_for_critic(self, context: IntegratedContext) -> str:
        """Format context for critic agent."""
        parts = ["# Review Context\n"]

        # Focus on code quality and documentation
        parts.append("## Items to Review\n")
        for chunk in context.chunks:
            if chunk.chunk.chunk_type == ChunkType.CODE:
                parts.append(f"### Code Block ({chunk.context_type})\n")
                parts.append(f"```\n{chunk.chunk.content}\n```\n")
            else:
                parts.append(f"### {chunk.context_type.title()}\n")
                parts.append(f"{chunk.chunk.content}\n")

        return "\n".join(parts)

    def _store_conversation_context(self, conversation_id: str, context: IntegratedContext) -> None:
        """Store context in conversation history."""
        if conversation_id not in self._conversation_contexts:
            self._conversation_contexts[conversation_id] = []

        self._conversation_contexts[conversation_id].append(context)

        # Keep only recent contexts (memory management)
        max_contexts = 10
        if len(self._conversation_contexts[conversation_id]) > max_contexts:
            self._conversation_contexts[conversation_id] = self._conversation_contexts[
                conversation_id
            ][-max_contexts:]
