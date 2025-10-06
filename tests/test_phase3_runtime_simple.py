"""
Simplified Phase 3 Runtime Chaos Test Suite.

This test suite validates the core runtime patterns without external dependencies:
- Deadline enforcement
- Circuit breaker behavior
- Single-flight deduplication
- Atomic state transitions
- Idempotent operations
"""

import asyncio
import time

import pytest

from synndicate.core.enhanced_state_machine import EnhancedStateMachine, StateType, StateContext, TransitionResult
from synndicate.core.runtime_patterns import (
    CircuitBreaker,
    RequestContext,
    retry,
    run_once,
    with_timeout,
)


class TestDeadlineEnforcement:
    """Test deadline enforcement and timeout handling."""

    @pytest.mark.asyncio
    async def test_deadline_enforcement(self):
        """Test that operations respect deadlines."""
        ctx = RequestContext.create(timeout_ms=100)

        async def slow_operation():
            await asyncio.sleep(0.5)  # Longer than deadline
            return "completed"

        start_time = time.time()
        with pytest.raises(asyncio.TimeoutError):
            await with_timeout(slow_operation(), ctx.deadline)

        # Should timeout quickly
        duration = time.time() - start_time
        assert duration < 0.2, f"Request took too long: {duration}s"

    @pytest.mark.asyncio
    async def test_deadline_budget_tracking(self):
        """Test that deadline budgets are properly tracked."""
        ctx = RequestContext.create(timeout_ms=1000)

        # Initial budget should be close to 1 second
        initial_budget = ctx.remaining_budget()
        assert 0.9 < initial_budget <= 1.0

        # Wait a bit and check budget decreased
        await asyncio.sleep(0.1)
        remaining_budget = ctx.remaining_budget()
        assert remaining_budget < initial_budget
        assert remaining_budget > 0.8


class TestCircuitBreaker:
    """Test circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker(threshold=2, cooldown=0.1)

        # Initially closed
        assert breaker.allow()

        # Record failures
        breaker.failure()
        assert breaker.allow()  # Still closed after 1 failure

        breaker.failure()
        # Should raise RuntimeError when open (circuit breaker pattern)
        with pytest.raises(RuntimeError, match="circuit_breaker_open"):
            breaker.allow()

    @pytest.mark.asyncio
    async def test_circuit_breaker_auto_recovery(self):
        """Test circuit breaker auto-recovery after cooldown."""
        breaker = CircuitBreaker(threshold=1, cooldown=0.1)

        # Trigger failure to open breaker
        breaker.failure()
        with pytest.raises(RuntimeError, match="circuit_breaker_open"):
            breaker.allow()

        # Wait for cooldown
        await asyncio.sleep(0.15)

        # Should allow requests again
        assert breaker.allow()


class TestSingleFlightDeduplication:
    """Test single-flight deduplication pattern."""

    @pytest.mark.asyncio
    async def test_single_flight_deduplication(self):
        """Test that concurrent operations are deduplicated."""
        call_count = 0

        async def expensive_operation():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return f"result_{call_count}"

        # Submit same operation multiple times concurrently
        tasks = [
            run_once("test_key", expensive_operation)
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should return the same result (single execution)
        assert all(r == results[0] for r in results)
        assert call_count == 1, f"Expected 1 call, got {call_count}"


class TestRetryLogic:
    """Test retry logic with jitter and deadlines."""

    @pytest.mark.asyncio
    async def test_retry_with_jitter_and_deadline(self):
        """Test retry logic respects deadlines and uses jitter."""
        attempt_count = 0

        async def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        # Should succeed after retries
        result = await retry(
            failing_operation,
            attempts=3,
            base=0.01,  # Very short backoff for testing
            is_retryable=lambda e: isinstance(e, ConnectionError)
        )

        assert result == "success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_retry_respects_deadline(self):
        """Test retry logic respects deadlines."""
        ctx = RequestContext.create(timeout_ms=100)

        async def always_failing():
            await asyncio.sleep(0.05)  # Small delay
            raise ConnectionError("Always fails")

        start_time = time.time()
        with pytest.raises((asyncio.TimeoutError, ConnectionError)):
            await retry(
                always_failing,
                attempts=10,  # Many attempts
                base=0.02,
                deadline=ctx.deadline,
                is_retryable=lambda e: isinstance(e, ConnectionError)
            )

        # Should timeout quickly due to deadline
        duration = time.time() - start_time
        assert duration < 0.2, f"Retry took too long: {duration}s"


class TestAtomicTransitions:
    """Test atomic state machine transitions."""

    @pytest.mark.asyncio
    async def test_atomic_transitions(self):
        """Test that state transitions are atomic."""
        sm = EnhancedStateMachine("test_sm", "initial")

        # Define states and transitions
        sm.add_state("initial", StateType.INITIAL)
        sm.add_state("processing", StateType.INTERMEDIATE)
        sm.add_state("completed", StateType.TERMINAL)
        sm.add_state("error", StateType.TERMINAL)

        # Add transitions with proper TransitionResult objects
        sm.add_transition("initial", "start", lambda ctx: TransitionResult(new_context=ctx.with_state("processing")))
        sm.add_transition("processing", "complete", lambda ctx: TransitionResult(new_context=ctx.with_state("completed")))
        sm.add_transition("processing", "error", lambda ctx: TransitionResult(new_context=ctx.with_state("error")))

        # Start state machine
        request_ctx = RequestContext.create()
        context = await sm.start(request_ctx)
        assert context.state == "initial"

        # Successful transition
        context = await sm.transition(request_ctx.correlation_id, "start", request_ctx)
        assert context.state == "processing"

        # Complete transition
        context = await sm.transition(request_ctx.correlation_id, "complete", request_ctx)
        assert context.state == "completed"

    @pytest.mark.asyncio
    async def test_transition_rollback_on_exception(self):
        """Test that transitions roll back on exceptions."""
        sm = EnhancedStateMachine("test_sm", "initial")

        # Define states
        sm.add_state("initial", StateType.INITIAL)
        sm.add_state("processing", StateType.INTERMEDIATE)
        sm.add_state("error", StateType.TERMINAL)

        # Add transition that will fail
        def failing_reducer(ctx):
            raise RuntimeError("Simulated failure")

        sm.add_transition("initial", "fail", failing_reducer)

        # Start state machine
        request_ctx = RequestContext.create()
        context = await sm.start(request_ctx)
        assert context.state == "initial"

        # Attempt transition that will fail
        with pytest.raises(RuntimeError):
            await sm.transition(request_ctx.correlation_id, "start", request_ctx)

        # State should remain unchanged (or move to error if defined)
        final_context = sm.get_context(request_ctx.correlation_id)
        assert final_context is not None
        assert final_context.state in ("initial", "error")


class TestIdempotentOperations:
    """Test idempotent operation behavior."""

    @pytest.mark.asyncio
    async def test_idempotent_state_transitions(self):
        """Test that state transitions are idempotent."""
        sm = EnhancedStateMachine("test_sm", "initial")

        # Define states
        sm.add_state("initial", StateType.INITIAL)
        sm.add_state("completed", StateType.TERMINAL)
        sm.add_transition("initial", "complete", lambda ctx: TransitionResult(new_context=ctx.with_state("completed")))

        # Start state machine
        request_ctx = RequestContext.create()
        context = await sm.start(request_ctx)

        # Transition to completed
        context = await sm.transition(request_ctx.correlation_id, "complete", request_ctx)
        assert context.state == "completed"

        # Applying same transition again should be no-op
        context2 = await sm.transition(request_ctx.correlation_id, "complete", request_ctx)
        assert context2.state == "completed"
        assert context2.version == context.version  # No state change


class TestPerformanceBenchmarks:
    """Performance benchmarks for runtime patterns."""

    @pytest.mark.asyncio
    async def test_state_machine_transition_performance(self):
        """Benchmark state machine transition performance."""
        sm = EnhancedStateMachine("perf_test", "initial")

        # Setup simple state machine
        sm.add_state("initial", StateType.INITIAL)
        sm.add_state("final", StateType.TERMINAL)
        sm.add_transition("initial", "go", lambda ctx: TransitionResult(new_context=ctx.with_state("final")))

        # Benchmark transitions
        start_time = time.time()

        for i in range(100):
            request_ctx = RequestContext.create(correlation_id=f"test_{i}")
            await sm.start(request_ctx)
            await sm.transition(request_ctx.correlation_id, "go", request_ctx)

        duration = time.time() - start_time
        transitions_per_second = 200 / duration  # 2 transitions per iteration

        # Should handle at least 1000 transitions per second
        assert transitions_per_second > 1000, f"Too slow: {transitions_per_second:.1f} transitions/sec"

    @pytest.mark.asyncio
    async def test_circuit_breaker_performance(self):
        """Benchmark circuit breaker performance."""
        breaker = CircuitBreaker(threshold=5, cooldown=1.0)

        start_time = time.time()

        # Test 10000 allow() calls
        for _ in range(10000):
            breaker.allow()

        duration = time.time() - start_time
        calls_per_second = 10000 / duration

        # Should handle at least 100k calls per second
        assert calls_per_second > 100000, f"Too slow: {calls_per_second:.1f} calls/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
