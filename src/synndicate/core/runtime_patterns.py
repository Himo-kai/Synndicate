"""
Phase 3 Runtime Patterns - Production-grade orchestration primitives.

This module implements the core runtime patterns for bulletproof orchestration:
- Deadline propagation and budget management
- Bounded retries with jitter
- Circuit breaker pattern
- Single-flight deduplication
- Cancellation scopes

These patterns ensure the orchestrator doesn't fold under real-world chaos.
"""

import asyncio
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from synndicate.observability.logging import get_logger
from synndicate.observability.metrics import counter

logger = get_logger(__name__)

T = TypeVar("T")

# Type alias for in-flight task registry
# Tasks can return any type since we handle heterogeneous operations
InFlightTask = asyncio.Task[Any]


def remaining_budget(deadline: float) -> float:
    """Calculate remaining time budget from absolute deadline."""
    return max(0.0, deadline - time.time())


async def with_timeout(coro: Awaitable[T], deadline: float) -> T:
    """Execute coroutine with deadline-based timeout."""
    budget = remaining_budget(deadline)
    if budget <= 0:
        raise TimeoutError("Deadline exceeded before execution")

    return await asyncio.wait_for(coro, timeout=budget)


@asynccontextmanager
async def cancellation_scope():
    """Ensure proper cancellation propagation to child tasks."""
    task = asyncio.current_task()
    try:
        yield
    except asyncio.CancelledError:
        # Cancel all child tasks spawned in this scope
        for t in asyncio.all_tasks():
            if t is not task and not t.done():
                t.cancel()
        raise


async def retry(
    op: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base: float = 0.2,
    max_backoff: float = 2.0,
    is_retryable: Callable[[Exception], bool] = lambda e: False,
    deadline: float | None = None,
) -> T:
    """Bounded retries with exponential backoff and jitter."""
    last_exception = None

    for i in range(attempts):
        # Check deadline before each attempt
        if deadline and remaining_budget(deadline) <= 0:
            raise TimeoutError("Deadline exceeded during retry")

        try:
            return await op()
        except Exception as e:
            last_exception = e
            counter("runtime.retry_attempts_total").inc()

            if i == attempts - 1 or not is_retryable(e):
                counter("runtime.retry_exhausted_total").inc()
                raise e from None

            # Exponential backoff with jitter
            backoff = min(max_backoff, base * (2**i)) * (0.5 + random.random())

            # Respect deadline in backoff
            if deadline:
                backoff = min(backoff, remaining_budget(deadline))
                if backoff <= 0:
                    raise TimeoutError("Deadline exceeded during backoff") from None

            logger.debug(f"Retrying after {backoff:.2f}s (attempt {i + 1}/{attempts}): {e}")
            await asyncio.sleep(backoff)

    # Should never reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry loop completed without result")


@dataclass
class CircuitBreaker:
    """Simple, effective circuit breaker for external service calls."""

    threshold: int = 5
    cooldown: float = 10.0
    failures: int = field(default=0, init=False)
    open_until: float = field(default=0.0, init=False)

    def allow(self) -> bool:
        """Check if requests are allowed through the breaker."""
        if time.time() < self.open_until:
            counter("runtime.circuit_breaker_blocked_total").inc()
            raise RuntimeError("circuit_breaker_open")
        return True

    def success(self) -> None:
        """Record successful operation."""
        self.failures = 0

    def failure(self) -> None:
        """Record failed operation."""
        self.failures += 1
        counter("runtime.circuit_breaker_failures_total").inc()

        if self.failures >= self.threshold:
            self.open_until = time.time() + self.cooldown
            self.failures = 0
            counter("runtime.circuit_breaker_opened_total").inc()
            logger.warning(f"Circuit breaker opened for {self.cooldown}s")


# Global in-flight registry for single-flight operations
_inflight: dict[str, InFlightTask] = {}


async def run_once(key: str, coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Single-flight pattern: deduplicate concurrent operations by key."""
    existing_task = _inflight.get(key)

    if existing_task is None or existing_task.done():
        # Start new operation
        task: asyncio.Task[T] = asyncio.create_task(coro_factory())
        _inflight[key] = task
        counter("runtime.single_flight_started_total").inc()
        result = await task
    else:
        # Join existing operation - cast to proper type
        counter("runtime.single_flight_deduplicated_total").inc()
        result = await cast("asyncio.Task[T]", existing_task)

    # Clean up completed tasks
    if key in _inflight and _inflight[key].done():
        _inflight.pop(key, None)

    return result


@dataclass(frozen=True)
class RequestContext:
    """Runtime context for request processing."""

    correlation_id: str
    trace_id: str
    deadline: float
    max_retries: int = 3

    @classmethod
    def create(
        cls,
        timeout_ms: int | None = None,
        correlation_id: str | None = None,
        trace_id: str | None = None,
    ) -> "RequestContext":
        """Create request context with defaults."""
        now = time.time()

        return cls(
            correlation_id=correlation_id or str(uuid.uuid4()),
            trace_id=trace_id or f"{int(now * 1000):x}",
            deadline=now + (timeout_ms or 30000) / 1000.0,
        )

    def remaining_budget(self) -> float:
        """Get remaining time budget."""
        return remaining_budget(self.deadline)

    def is_expired(self) -> bool:
        """Check if deadline has passed."""
        return self.remaining_budget() <= 0


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable."""
    # Network/timeout errors are retryable
    if isinstance(error, (asyncio.TimeoutError, ConnectionError, OSError)):
        return True

    # HTTP errors: 408, 429, 5xx are retryable
    if hasattr(error, "status_code"):
        code = error.status_code
        return code in (408, 429) or 500 <= code < 600

    # Circuit breaker open is not retryable (fail fast)
    if "circuit_breaker_open" in str(error):
        return False

    return False


async def with_circuit_breaker(
    breaker: CircuitBreaker,
    op: Callable[[], Awaitable[T]],
    deadline: float | None = None,
) -> T:
    """Execute operation with circuit breaker protection."""
    breaker.allow()

    try:
        if deadline:
            result = await with_timeout(op(), deadline)
        else:
            result = await op()

        breaker.success()
        return result
    except Exception:
        breaker.failure()
        raise


# Global circuit breakers for common services
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(service: str) -> CircuitBreaker:
    """Get or create circuit breaker for a service."""
    if service not in _circuit_breakers:
        _circuit_breakers[service] = CircuitBreaker()
    return _circuit_breakers[service]
