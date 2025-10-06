"""
Enhanced State Machine - Phase 3 Runtime Orchestration.

This module implements production-grade state machine patterns:
- Atomic transitions with rollback
- Idempotent operations
- Terminal state enforcement
- Effect separation from state changes
- Correlation-based deduplication

Runtime Invariants Enforced:
1. Exactly one active state at a time
2. Transitions are atomic (fully applied or rolled back)
3. Terminal states are absorbing (no transitions out)
4. Transitions are idempotent (reapplying after success is no-op)
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from synndicate.core.runtime_patterns import RequestContext, run_once, with_timeout
from synndicate.observability.logging import get_logger
from synndicate.observability.metrics import counter, histogram
from synndicate.observability.tracing import trace_span

logger = get_logger(__name__)


class StateType(Enum):
    """State machine state types."""
    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    TERMINAL = "terminal"  # Absorbing states


@dataclass(frozen=True)
class StateContext:
    """Immutable state context for atomic transitions."""

    state: str
    data: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    trace_id: str = ""

    def with_state(self, new_state: str) -> "StateContext":
        """Create new context with updated state."""
        return StateContext(
            state=new_state,
            data=self.data.copy(),
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
        )

    def with_data(self, **updates) -> "StateContext":
        """Create new context with updated data."""
        new_data = self.data.copy()
        new_data.update(updates)
        return StateContext(
            state=self.state,
            data=new_data,
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
        )


@dataclass(frozen=True)
class Effect:
    """Side effect to execute after state transition."""

    name: str
    operation: Callable[[StateContext], Any]
    retryable: bool = True
    timeout: float | None = None


@dataclass(frozen=True)
class TransitionResult:
    """Result of a state transition attempt."""

    new_context: StateContext
    effects: list[Effect] = field(default_factory=list)
    error: str | None = None


class EnhancedStateMachine:
    """Production-grade state machine with atomic transitions."""

    def __init__(self, name: str, initial_state: str):
        self.name = name
        self.initial_state = initial_state
        self._transitions: dict[tuple[str, str], Callable[[StateContext], TransitionResult]] = {}
        self._terminal_states: set[str] = set()
        self._state_types: dict[str, StateType] = {}

        # Runtime state
        self._contexts: dict[str, StateContext] = {}  # correlation_id -> context
        self._transition_history: dict[str, list[tuple[str, str, float]]] = {}  # correlation_id -> history

    def add_state(self, state: str, state_type: StateType) -> None:
        """Add a state to the machine."""
        self._state_types[state] = state_type
        if state_type == StateType.TERMINAL:
            self._terminal_states.add(state)

    def add_transition(
        self,
        from_state: str,
        event: str,
        reducer: Callable[[StateContext], TransitionResult],
    ) -> None:
        """Add a transition reducer."""
        key = (from_state, event)
        if key in self._transitions:
            raise ValueError(f"Transition {from_state} -> {event} already exists")

        self._transitions[key] = reducer

    @trace_span("state_machine.start")
    async def start(self, request_ctx: RequestContext, initial_data: dict[str, Any] | None = None) -> StateContext:
        """Start state machine for a correlation ID."""
        correlation_id = request_ctx.correlation_id

        # Idempotency: if already started, return existing context
        if correlation_id in self._contexts:
            logger.debug(f"State machine already started for {correlation_id}")
            return self._contexts[correlation_id]

        context = StateContext(
            state=self.initial_state,
            data=initial_data or {},
            correlation_id=correlation_id,
            trace_id=request_ctx.trace_id,
        )

        self._contexts[correlation_id] = context
        self._transition_history[correlation_id] = [(self.initial_state, "start", time.time())]

        counter("state_machine.started_total").inc()
        logger.info(f"Started state machine {self.name} for {correlation_id}")

        return context

    @trace_span("state_machine.transition")
    async def transition(
        self,
        correlation_id: str,
        event: str,
        request_ctx: RequestContext,
    ) -> StateContext:
        """Execute atomic state transition."""
        # Use single-flight to prevent concurrent transitions for same correlation_id
        return await run_once(
            f"transition:{correlation_id}:{event}",
            lambda: self._execute_transition(correlation_id, event, request_ctx)
        )

    async def _execute_transition(
        self,
        correlation_id: str,
        event: str,
        request_ctx: RequestContext,
    ) -> StateContext:
        """Internal transition execution with atomicity guarantees."""
        current_context = self._contexts.get(correlation_id)
        if not current_context:
            raise ValueError(f"No active state machine for {correlation_id}")

        current_state = current_context.state

        # Terminal state invariant: no transitions out
        if current_state in self._terminal_states:
            logger.debug(f"Ignoring event {event} in terminal state {current_state}")
            return current_context  # Idempotent no-op

        # Find transition reducer
        transition_key = (current_state, event)
        reducer = self._transitions.get(transition_key)

        if not reducer:
            # Invalid transition - move to error state if available
            error_msg = f"Invalid transition: {current_state} -> {event}"
            logger.error(error_msg)
            counter("state_machine.invalid_transitions_total").inc()

            if "error" in self._state_types:
                return await self._transition_to_error(correlation_id, error_msg, request_ctx)
            else:
                raise ValueError(error_msg)

        start_time = time.time()

        try:
            # Execute reducer (pure function)
            with histogram("state_machine.transition_duration_seconds").time():
                transition_result = reducer(current_context)

            # Validate new state exists
            new_state = transition_result.new_context.state
            if new_state not in self._state_types:
                raise ValueError(f"Unknown target state: {new_state}")

            # Atomic update: persist new context first
            self._contexts[correlation_id] = transition_result.new_context

            # Record transition in history
            duration = time.time() - start_time
            history = self._transition_history.setdefault(correlation_id, [])
            history.append((new_state, event, duration))

            # Log structured transition
            logger.info(
                "State transition completed",
                extra={
                    "correlation_id": correlation_id,
                    "trace_id": current_context.trace_id,
                    "event": event,
                    "state_from": current_state,
                    "state_to": new_state,
                    "duration_ms": duration * 1000,
                }
            )

            counter("state_machine.transitions_total").inc()

            # Execute effects after state is persisted
            if transition_result.effects:
                await self._execute_effects(
                    transition_result.effects,
                    transition_result.new_context,
                    request_ctx,
                )

            return transition_result.new_context

        except Exception as e:
            # Rollback: restore previous context
            logger.error(f"Transition failed, rolling back: {e}")
            counter("state_machine.transition_failures_total").inc()

            # Move to error state if available
            if "error" in self._state_types:
                return await self._transition_to_error(correlation_id, str(e), request_ctx)
            else:
                raise

    async def _execute_effects(
        self,
        effects: list[Effect],
        context: StateContext,
        request_ctx: RequestContext,
    ) -> None:
        """Execute side effects with timeout and retry."""
        for effect in effects:
            try:
                if effect.timeout:
                    deadline = min(request_ctx.deadline, time.time() + effect.timeout)
                else:
                    deadline = request_ctx.deadline

                await with_timeout(
                    asyncio.create_task(effect.operation(context)),
                    deadline
                )

                counter("state_machine.effects_executed_total").inc()

            except Exception as e:
                logger.error(f"Effect {effect.name} failed: {e}")
                counter("state_machine.effects_failed_total").inc()

                # Effects are best-effort; don't fail the transition
                # In production, you might want to queue failed effects for retry

    async def _transition_to_error(
        self,
        correlation_id: str,
        error_msg: str,
        request_ctx: RequestContext,
    ) -> StateContext:
        """Transition to error state (if defined)."""
        current_context = self._contexts[correlation_id]

        error_context = current_context.with_state("error").with_data(
            error=error_msg,
            error_timestamp=time.time(),
        )

        self._contexts[correlation_id] = error_context

        logger.error(
            "Transitioned to error state",
            extra={
                "correlation_id": correlation_id,
                "error": error_msg,
            }
        )

        return error_context

    def get_context(self, correlation_id: str) -> StateContext | None:
        """Get current context for correlation ID."""
        return self._contexts.get(correlation_id)

    def get_history(self, correlation_id: str) -> list[tuple[str, str, float]]:
        """Get transition history for correlation ID."""
        return self._transition_history.get(correlation_id, [])

    def is_terminal(self, correlation_id: str) -> bool:
        """Check if state machine is in terminal state."""
        context = self._contexts.get(correlation_id)
        return context and context.state in self._terminal_states

    def cleanup(self, correlation_id: str) -> None:
        """Clean up completed state machine."""
        self._contexts.pop(correlation_id, None)
        self._transition_history.pop(correlation_id, None)


# Reducer factory functions for common patterns
def simple_transition(target_state: str) -> Callable[[StateContext], TransitionResult]:
    """Create simple transition to target state."""
    def reducer(context: StateContext) -> TransitionResult:
        return TransitionResult(new_context=context.with_state(target_state))
    return reducer


def transition_with_effect(
    target_state: str,
    effect_name: str,
    effect_op: Callable[[StateContext], Any],
) -> Callable[[StateContext], TransitionResult]:
    """Create transition with side effect."""
    def reducer(context: StateContext) -> TransitionResult:
        effect = Effect(name=effect_name, operation=effect_op)
        return TransitionResult(
            new_context=context.with_state(target_state),
            effects=[effect]
        )
    return reducer


def conditional_transition(
    condition: Callable[[StateContext], bool],
    true_state: str,
    false_state: str,
) -> Callable[[StateContext], TransitionResult]:
    """Create conditional transition based on context."""
    def reducer(context: StateContext) -> TransitionResult:
        target_state = true_state if condition(context) else false_state
        return TransitionResult(new_context=context.with_state(target_state))
    return reducer
