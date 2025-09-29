"""
Modern agent base with protocol-based design and improved architecture.

Key improvements:
- Protocol-based interfaces for type safety
- Async context managers for resource lifecycle
- Multi-factor confidence scoring
- Streaming response support
- Circuit breaker pattern for reliability
- Proper error handling and retries
"""

import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config.settings import AgentConfig, ModelEndpoint
from ..observability.logging import get_logger, get_trace_id
from ..observability.metrics import counter, timer
from ..observability.probe import probe
from ..observability.tracing import trace_span

logger = get_logger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for agent responses."""

    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class AgentResponse:
    """Enhanced agent response with detailed metadata."""

    reasoning: str
    response: str
    confidence: float
    confidence_factors: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time: float | None = None
    token_usage: dict[str, int] | None = None

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level enum."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.55:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.35:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class AgentProtocol(Protocol):
    """Protocol defining the agent interface."""

    async def process(self, query: str, context: dict[str, Any] | None = None) -> AgentResponse:
        """Process a query and return a response."""
        ...

    async def stream_process(
        self, query: str, context: dict[str, Any] | None = None
    ) -> AsyncIterator[str]:
        """Stream process a query."""
        ...

    async def health_check(self) -> bool:
        """Check if the agent is healthy."""
        ...


class Agent(ABC):
    """
    Modern agent base class with improved architecture.

    Key improvements over original:
    - Proper async lifecycle management
    - Multi-factor confidence scoring
    - Circuit breaker for reliability
    - Streaming support
    - Better error handling
    """

    # Class-level hook for DI/mocking in tests (e.g., patched by unittest.mock)
    # Expected to expose an async `generate_response(prompt, **kwargs)` API
    model_manager: Any | None = None

    def __init__(
        self,
        endpoint: ModelEndpoint,
        config: AgentConfig,
        http_client: httpx.AsyncClient | None = None,
        model_manager: Any | None = None,
    ):
        self.endpoint = endpoint
        self.config = config
        self._http_client = http_client
        self._owned_client = http_client is None
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = 0
        # Instance-level model manager falls back to class-level hook
        self.model_manager = model_manager or self.__class__.model_manager

    async def __aenter__(self):
        """Async context manager entry."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.endpoint.timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=2),
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._owned_client and self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    @abstractmethod
    def system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        ...

    @abstractmethod
    def _calculate_confidence_factors(self, response: str) -> dict[str, float]:
        """Calculate confidence factors specific to this agent type."""
        ...

    @trace_span("agent.call_model")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def _call_model(self, prompt: str) -> tuple[str, str]:
        """
        Call the model with retry logic and circuit breaker.
        Returns (reasoning, final_response).
        """
        # Prefer the injected model_manager path for tests/mocks
        if self.model_manager is not None:
            try:
                result = await self.model_manager.generate_response(prompt)
                if isinstance(result, tuple) and len(result) >= 2:
                    reasoning, final = result[0], result[1]
                else:
                    reasoning, final = "", str(result)
                # Reset circuit breaker on success
                self._circuit_breaker_failures = 0
                counter("agent.model_calls_total").inc()
                return reasoning, final
            except Exception as e:
                self._circuit_breaker_failures += 1
                self._circuit_breaker_last_failure = asyncio.get_event_loop().time()
                counter("agent.model_calls_failed_total").inc()
                logger.error(f"Model manager call failed: {e}")
                raise

        if not self._http_client:
            raise RuntimeError("Agent not properly initialized. Use async context manager.")

        # Circuit breaker check
        if self._is_circuit_open():
            raise RuntimeError("Circuit breaker is open")

        try:
            with timer("agent.model_call_duration"):
                response = await self._http_client.post(
                    f"{self.endpoint.base_url}/api/generate",
                    json={
                        "model": self.endpoint.name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                        },
                    },
                    headers=self._get_headers(),
                )
                response.raise_for_status()

                result = response.json()
                text = result.get("response", "")

                counter("agent.model_calls_total").inc()

                # Reset circuit breaker on success
                self._circuit_breaker_failures = 0

                return self._extract_sections(text)

        except Exception as e:
            self._circuit_breaker_failures += 1
            self._circuit_breaker_last_failure = asyncio.get_event_loop().time()
            counter("agent.model_calls_failed_total").inc()
            logger.error(f"Model call failed: {e}")
            raise

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for model requests."""
        headers = {"Content-Type": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        return headers

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_breaker_failures < 5:
            return False

        # Circuit breaker timeout (60 seconds)
        current_time = asyncio.get_event_loop().time()
        return (current_time - self._circuit_breaker_last_failure) < 60

    def _extract_sections(self, text: str) -> tuple[str, str]:
        """Extract ANALYSIS and FINAL sections from response."""
        analysis_match = re.search(r"ANALYSIS:\s*(.*?)(?:\nFINAL:|$)", text, re.DOTALL)
        final_match = re.search(r"FINAL:\s*(.*)$", text, re.DOTALL)

        analysis = analysis_match.group(1).strip() if analysis_match else ""
        final = final_match.group(1).strip() if final_match else text.strip()

        return analysis, final

    def _calculate_base_confidence(self, response: str) -> float:
        """Calculate base confidence score."""
        score = 0.4  # Base score

        # Length factor
        if len(response) > 100:
            score += 0.1
        if len(response) > 500:
            score += 0.1

        # Structure factor
        if "```" in response:
            score += 0.2

        # Uncertainty indicators (negative factors)
        uncertainty_words = ["maybe", "perhaps", "might", "could be", "not sure", "unclear"]
        uncertainty_count = sum(1 for word in uncertainty_words if word in response.lower())
        score -= uncertainty_count * 0.05

        return min(0.95, max(0.1, score))

    def _combine_confidence_factors(self, factors: dict[str, float]) -> float:
        """Combine multiple confidence factors into final score."""
        if not factors:
            return 0.5

        # Weighted average with base confidence having higher weight
        weights = {"base": 0.4, "structure": 0.3, "content": 0.3}

        total_weight = 0
        weighted_sum = 0

        for factor_name, factor_value in factors.items():
            weight = weights.get(factor_name, 0.1)
            weighted_sum += factor_value * weight
            total_weight += weight

        return min(0.95, max(0.1, weighted_sum / total_weight if total_weight > 0 else 0.5))

    @trace_span("agent.process")
    async def process(self, query: str, context: dict[str, Any] | None = None) -> AgentResponse:
        """Process a query and return a structured response."""
        start_time = asyncio.get_event_loop().time()

        # Get trace ID from context or current trace
        ctx = context or {}
        trace_id = ctx.get("trace_id") or get_trace_id()
        agent_name = self.__class__.__name__.lower().replace("agent", "")

        # Ensure class-level patched model_manager is honored even if set after __init__
        if self.model_manager is None and getattr(self.__class__, "model_manager", None) is not None:
            self.model_manager = self.__class__.model_manager

        with probe(f"agent.{agent_name}.process", trace_id):
            prompt = self._build_prompt(query, ctx)

            logger.info(
                f"Processing query with {agent_name} agent",
                agent=agent_name,
                query_length=len(query),
                trace_id=trace_id,
            )

        try:
            reasoning, final_response = await self._call_model(prompt)

            # Calculate confidence factors
            base_confidence = self._calculate_base_confidence(final_response)
            agent_factors = self._calculate_confidence_factors(final_response)

            all_factors = {"base": base_confidence, **agent_factors}
            final_confidence = self._combine_confidence_factors(all_factors)

            execution_time = asyncio.get_event_loop().time() - start_time

            return AgentResponse(
                reasoning=reasoning,
                response=final_response,
                confidence=final_confidence,
                confidence_factors=all_factors,
                execution_time=execution_time,
                metadata={
                    "agent_type": self.__class__.__name__,
                    "model": self.endpoint.name,
                    "query_length": len(query),
                    "response_length": len(final_response),
                },
            )

        except Exception as e:
            logger.error(f"Agent processing failed: {e}")
            # If neither HTTP nor model_manager is available, synthesize a minimal response
            if self.model_manager is None and self._http_client is None:
                execution_time = asyncio.get_event_loop().time() - start_time
                synthetic = "Response: " + (query[:200] if query else "OK")
                base_confidence = self._calculate_base_confidence(synthetic)
                return AgentResponse(
                    reasoning="Synthetic response (no model available)",
                    response=synthetic,
                    confidence=base_confidence,
                    confidence_factors={"base": base_confidence},
                    execution_time=execution_time,
                    metadata={"synthetic": True},
                )
            return AgentResponse(
                reasoning=f"Error: {str(e)}",
                response="",
                confidence=0.0,
                execution_time=asyncio.get_event_loop().time() - start_time,
                metadata={"error": str(e)},
            )

    async def stream_process(
        self, query: str, context: dict[str, Any] | None = None
    ) -> AsyncIterator[str]:
        """Stream process a query (placeholder for streaming implementation)."""
        # For now, yield the full response
        response = await self.process(query, context)
        yield response.response

    async def health_check(self) -> bool:
        """Check if the agent is healthy."""
        try:
            if not self._http_client:
                return False

            # Simple health check - try to get model info
            response = await self._http_client.get(
                f"{self.endpoint.base_url}/api/tags", timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False

    def _build_prompt(self, query: str, context: dict[str, Any]) -> str:
        """Build the full prompt for the model."""
        context_str = ""
        if context:
            context_items = []
            for key, value in context.items():
                if isinstance(value, str) and value.strip():
                    context_items.append(f"{key.title()}: {value}")
            if context_items:
                context_str = "\nContext:\n" + "\n".join(context_items) + "\n"

        return f"""{self.system_prompt()}{context_str}

Task: {query}

Please respond with:
ANALYSIS:
[Your reasoning and analysis]

FINAL:
[Your final answer or solution]
"""
