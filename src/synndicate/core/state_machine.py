"""
State machine for orchestrator execution flow with transitions and guards.

Improvements over original:
- Explicit state management for complex workflows
- Conditional transitions with guard functions
- State history and rollback capabilities
- Event-driven state changes
- Validation and error handling
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..observability.logging import get_logger

logger = get_logger(__name__)


class StateType(Enum):
    """Types of states in the state machine."""

    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    FINAL = "final"
    ERROR = "error"


@dataclass
class StateContext:
    """Context data passed between states."""

    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get data value with default."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set data value."""
        self.data[key] = value

    def update(self, data: dict[str, Any]) -> None:
        """Update context data."""
        self.data.update(data)


class State(ABC):
    """Abstract base class for state machine states."""

    def __init__(
        self,
        name: str,
        state_type: StateType = StateType.INTERMEDIATE,
        timeout: float | None = None,
    ):
        self.name = name
        self.state_type = state_type
        self.timeout = timeout
        self.entry_time: float | None = None

    @abstractmethod
    async def execute(self, context: StateContext) -> str:
        """
        Execute the state logic.
        Returns the name of the next state to transition to.
        """
        ...

    async def on_entry(self, context: StateContext) -> None:
        """Called when entering this state."""
        self.entry_time = time.time()
        logger.debug(f"Entering state: {self.name}")

    async def on_exit(self, context: StateContext) -> None:
        """Called when exiting this state."""
        if self.entry_time:
            duration = time.time() - self.entry_time
            logger.debug(f"Exiting state: {self.name} (duration: {duration:.2f}s)")

    def is_timeout_exceeded(self) -> bool:
        """Check if state timeout has been exceeded."""
        if not self.timeout or not self.entry_time:
            return False
        return (time.time() - self.entry_time) > self.timeout

    def __str__(self) -> str:
        return f"State({self.name})"


@dataclass
class Transition:
    """State transition with optional guard condition."""

    from_state: str
    to_state: str
    event: str | None = None
    guard: Callable[[StateContext], bool] | None = None
    action: Callable[[StateContext], None] | None = None

    def can_transition(self, context: StateContext) -> bool:
        """Check if transition is allowed based on guard condition."""
        if self.guard:
            return self.guard(context)
        return True

    async def execute_action(self, context: StateContext) -> None:
        """Execute transition action if present."""
        if self.action:
            if hasattr(self.action, "__call__"):
                result = self.action(context)
                if hasattr(result, "__await__"):
                    await result


class StateMachine:
    """
    Finite state machine for orchestrator workflow management.

    Features:
    - State transitions with guards and actions
    - Event-driven state changes
    - State history and rollback
    - Timeout handling
    - Error state management
    """

    def __init__(self, name: str, initial_state: str):
        self.name = name
        self.initial_state = initial_state
        self.current_state: str | None = None
        self.states: dict[str, State] = {}
        self.transitions: list[Transition] = []
        self.state_history: list[str] = []
        self.context = StateContext()
        self.is_running = False

    def add_state(self, state: State) -> None:
        """Add a state to the state machine."""
        self.states[state.name] = state
        logger.debug(f"Added state: {state.name}")

    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the state machine."""
        self.transitions.append(transition)
        logger.debug(f"Added transition: {transition.from_state} -> {transition.to_state}")

    def get_valid_transitions(self, from_state: str) -> list[Transition]:
        """Get all valid transitions from a given state."""
        return [
            t
            for t in self.transitions
            if t.from_state == from_state and t.can_transition(self.context)
        ]

    async def start(self, initial_context: dict[str, Any] | None = None) -> None:
        """Start the state machine."""
        if self.is_running:
            raise RuntimeError("State machine is already running")

        if self.initial_state not in self.states:
            raise ValueError(f"Initial state '{self.initial_state}' not found")

        self.is_running = True
        self.current_state = self.initial_state
        self.state_history = [self.initial_state]

        if initial_context:
            self.context.update(initial_context)

        logger.info(f"Starting state machine '{self.name}' in state '{self.initial_state}'")

        # Enter initial state
        await self.states[self.current_state].on_entry(self.context)

    async def step(self) -> bool:
        """
        Execute one step of the state machine.
        Returns True if the machine should continue, False if finished.
        """
        if not self.is_running or not self.current_state:
            return False

        current_state_obj = self.states[self.current_state]

        # Check for timeout
        if current_state_obj.is_timeout_exceeded():
            logger.warning(f"State '{self.current_state}' timed out")
            await self._transition_to_error("State timeout exceeded")
            return False

        try:
            # Execute current state
            next_state_name = await current_state_obj.execute(self.context)

            # Check if we should transition
            if next_state_name and next_state_name != self.current_state:
                await self.transition_to(next_state_name)

            # Check if we've reached a final state
            if current_state_obj.state_type in [StateType.FINAL, StateType.ERROR]:
                await self.stop()
                return False

            return True

        except Exception as e:
            logger.error(f"Error executing state '{self.current_state}': {e}")
            await self._transition_to_error(str(e))
            return False

    async def transition_to(self, state_name: str, event: str | None = None) -> bool:
        """Transition to a specific state."""
        if not self.current_state:
            raise RuntimeError("State machine not started")

        if state_name not in self.states:
            raise ValueError(f"State '{state_name}' not found")

        # Find valid transition
        valid_transitions = [
            t
            for t in self.get_valid_transitions(self.current_state)
            if t.to_state == state_name and (not event or t.event == event)
        ]

        if not valid_transitions:
            logger.warning(f"No valid transition from '{self.current_state}' to '{state_name}'")
            return False

        transition = valid_transitions[0]  # Use first valid transition

        # Execute transition
        await self._execute_transition(transition)
        return True

    async def _execute_transition(self, transition: Transition) -> None:
        """Execute a state transition."""
        old_state = self.current_state
        new_state = transition.to_state

        logger.debug(f"Transitioning from '{old_state}' to '{new_state}'")

        # Exit current state
        if old_state and old_state in self.states:
            await self.states[old_state].on_exit(self.context)

        # Execute transition action
        await transition.execute_action(self.context)

        # Update current state
        self.current_state = new_state
        self.state_history.append(new_state)

        # Enter new state
        await self.states[new_state].on_entry(self.context)

    async def _transition_to_error(self, error_message: str) -> None:
        """Transition to error state."""
        self.context.set("error_message", error_message)

        # Look for error state
        error_states = [
            name for name, state in self.states.items() if state.state_type == StateType.ERROR
        ]

        if error_states:
            error_transition = Transition(
                from_state=self.current_state or "unknown", to_state=error_states[0]
            )
            await self._execute_transition(error_transition)
        else:
            # No error state defined, just stop
            logger.error(f"No error state defined, stopping state machine: {error_message}")
            await self.stop()

    async def run_to_completion(self, max_steps: int = 100) -> StateContext:
        """Run the state machine until completion or max steps."""
        step_count = 0

        while self.is_running and step_count < max_steps:
            should_continue = await self.step()
            if not should_continue:
                break
            step_count += 1

        if step_count >= max_steps:
            logger.warning(f"State machine '{self.name}' stopped after {max_steps} steps")
            await self.stop()

        return self.context

    async def stop(self) -> None:
        """Stop the state machine."""
        if not self.is_running:
            return

        logger.info(f"Stopping state machine '{self.name}' in state '{self.current_state}'")

        # Exit current state
        if self.current_state and self.current_state in self.states:
            await self.states[self.current_state].on_exit(self.context)

        self.is_running = False

    def get_state_history(self) -> list[str]:
        """Get the history of states visited."""
        return self.state_history.copy()

    def can_rollback(self) -> bool:
        """Check if rollback is possible."""
        return len(self.state_history) > 1

    async def rollback(self) -> bool:
        """Rollback to the previous state."""
        if not self.can_rollback():
            return False

        # Remove current state from history
        self.state_history.pop()
        previous_state = self.state_history[-1]

        # Transition back
        rollback_transition = Transition(
            from_state=self.current_state or "unknown", to_state=previous_state
        )

        await self._execute_transition(rollback_transition)
        logger.info(f"Rolled back to state '{previous_state}'")
        return True

    def get_current_state_info(self) -> dict[str, Any]:
        """Get information about the current state."""
        if not self.current_state:
            return {}

        state = self.states[self.current_state]
        return {
            "name": state.name,
            "type": state.state_type.value,
            "entry_time": state.entry_time,
            "timeout": state.timeout,
            "is_timeout_exceeded": state.is_timeout_exceeded(),
        }
