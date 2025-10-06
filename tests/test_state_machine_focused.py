"""
Focused tests for state machine functionality based on actual implementation.

Tests the core StateMachine, State, Transition, and StateContext classes.
"""

import pytest

from synndicate.core.state_machine import State, StateContext, StateMachine, StateType, Transition


class MockState(State):
    """Concrete State implementation for testing."""

    def __init__(
        self,
        name: str,
        state_type: StateType = StateType.INTERMEDIATE,
        timeout: float | None = None,
        next_state: str | None = None,
    ):
        super().__init__(name, state_type, timeout)
        self.next_state = next_state or name  # Default to staying in same state
        self.execute_called = False

    async def execute(self, context: StateContext) -> str:
        """Execute the state logic."""
        self.execute_called = True
        return self.next_state


class TestStateContext:
    """Test StateContext functionality."""

    def test_state_context_initialization(self):
        """Test StateContext initialization."""
        context = StateContext()

        assert isinstance(context.data, dict)
        assert isinstance(context.metadata, dict)
        assert len(context.data) == 0
        assert len(context.metadata) == 0

    def test_state_context_with_data(self):
        """Test StateContext with initial data."""
        initial_data = {"key1": "value1", "key2": 42}
        initial_metadata = {"meta1": "info"}

        context = StateContext(data=initial_data, metadata=initial_metadata)

        assert context.data == initial_data
        assert context.metadata == initial_metadata

    def test_get_set_operations(self):
        """Test get and set operations."""
        context = StateContext()

        # Test set and get
        context.set("test_key", "test_value")
        assert context.get("test_key") == "test_value"

        # Test get with default
        assert context.get("nonexistent", "default") == "default"
        assert context.get("nonexistent") is None

    def test_update_operation(self):
        """Test update operation."""
        context = StateContext()
        context.set("existing", "old_value")

        update_data = {"existing": "new_value", "new_key": "new_value"}
        context.update(update_data)

        assert context.get("existing") == "new_value"
        assert context.get("new_key") == "new_value"


class TestState:
    """Test State functionality."""

    def test_state_initialization(self):
        """Test State initialization."""
        state = MockState("test_state")

        assert state.name == "test_state"
        assert state.state_type == StateType.INTERMEDIATE
        assert state.timeout is None
        assert state.entry_time is None

    def test_state_with_parameters(self):
        """Test State with custom parameters."""
        state = MockState("custom_state", state_type=StateType.FINAL, timeout=30.0)

        assert state.name == "custom_state"
        assert state.state_type == StateType.FINAL
        assert state.timeout == 30.0

    async def test_state_entry_exit_methods(self):
        """Test state entry and exit methods."""
        state = MockState("test_state")
        context = StateContext()

        # Test entry
        await state.on_entry(context)
        assert state.entry_time is not None

        # Test exit
        await state.on_exit(context)
        # Entry time should still be set after exit

    def test_state_timeout_check(self):
        """Test timeout checking."""
        state = MockState("test_state", timeout=1.0)

        # Initially no timeout
        assert not state.is_timeout_exceeded()

        # Set entry time to simulate timeout
        import time

        state.entry_time = time.time() - 2.0  # 2 seconds ago
        assert state.is_timeout_exceeded()

    def test_state_string_representation(self):
        """Test state string representation."""
        state = MockState("test_state", StateType.INITIAL)
        # State.__str__ returns "State(name)" format, not including state_type
        assert "test_state" in str(state)
        assert state.state_type == StateType.INITIAL


class TestTransition:
    """Test Transition functionality."""

    def test_transition_initialization(self):
        """Test Transition initialization."""
        transition = Transition("from_state", "to_state", "event")

        assert transition.from_state == "from_state"
        assert transition.to_state == "to_state"
        assert transition.event == "event"
        assert transition.guard is None
        assert transition.action is None

    def test_transition_with_guard(self):
        """Test Transition with guard condition."""

        def guard_func(context):
            return context.get("allowed", False)

        transition = Transition("from", "to", "event", guard=guard_func)

        # Test guard evaluation
        context = StateContext()
        context.set("allowed", False)
        assert not transition.can_transition(context)

        context.set("allowed", True)
        assert transition.can_transition(context)

    async def test_transition_with_action(self):
        """Test transition with action callback."""
        action_called = 0

        def test_action(context):
            nonlocal action_called
            action_called += 1

        transition = Transition(
            from_state="start", to_state="end", event="trigger", action=test_action
        )

        context = StateContext()
        await transition.execute_action(context)
        assert action_called == 1

    def test_transition_without_guard(self):
        """Test Transition without guard (should always allow)."""
        transition = Transition("from", "to", "event")
        context = StateContext()

        assert transition.can_transition(context)

    async def test_transition_without_action(self):
        """Test transition without action callback."""
        transition = Transition(from_state="start", to_state="end", event="trigger")

        context = StateContext()
        # Should not raise exception
        await transition.execute_action(context)


class TestStateMachine:
    """Test StateMachine functionality."""

    def test_state_machine_initialization(self):
        """Test StateMachine initialization."""
        sm = StateMachine("test_machine", "initial")

        assert sm.name == "test_machine"
        assert sm.initial_state == "initial"
        assert sm.current_state is None
        assert isinstance(sm.states, dict)
        assert isinstance(sm.transitions, list)
        assert isinstance(sm.state_history, list)
        assert isinstance(sm.context, StateContext)
        assert not sm.is_running

    def test_add_state(self):
        """Test adding states to state machine."""
        sm = StateMachine("test_sm", "initial")
        state = MockState("test_state", StateType.INTERMEDIATE)

        sm.add_state(state)

        assert "test_state" in sm.states
        assert sm.states["test_state"] == state

    def test_add_transition(self):
        """Test adding transitions to state machine."""
        sm = StateMachine("test_sm", "initial")
        transition = Transition("from", "to", "event")

        sm.add_transition(transition)

        assert transition in sm.transitions

    def test_get_valid_transitions(self):
        """Test getting valid transitions from a state."""
        sm = StateMachine("test_sm", "initial")

        # Add transitions
        t1 = Transition("state1", "state2", "event1")
        t2 = Transition("state1", "state3", "event2")
        t3 = Transition("state2", "state3", "event3")

        sm.add_transition(t1)
        sm.add_transition(t2)
        sm.add_transition(t3)

        # Get valid transitions from state1
        valid = sm.get_valid_transitions("state1")

        assert len(valid) == 2
        assert t1 in valid
        assert t2 in valid
        assert t3 not in valid

    async def test_start_state_machine(self):
        """Test starting the state machine."""
        sm = StateMachine("test_sm", "initial")
        initial_state = MockState("initial", StateType.INITIAL)
        sm.add_state(initial_state)

        await sm.start()

        assert sm.is_running
        assert sm.current_state == "initial"
        assert len(sm.state_history) == 1
        assert sm.state_history[0] == "initial"

    async def test_start_with_context(self):
        """Test starting state machine with initial context."""
        sm = StateMachine("test", "initial")
        initial_state = MockState("initial")
        sm.add_state(initial_state)

        initial_context = {"key": "value"}
        await sm.start(initial_context)

        assert sm.is_running
        assert sm.current_state == "initial"
        assert sm.context.get("key") == "value"

    async def test_transition_to_state(self):
        """Test transitioning to a specific state."""
        sm = StateMachine("test", "initial")
        initial_state = MockState("initial")
        next_state = MockState("next")
        sm.add_state(initial_state)
        sm.add_state(next_state)

        transition = Transition(from_state="initial", to_state="next", event="go")
        sm.add_transition(transition)

        await sm.start()
        result = await sm.transition_to("next", "go")

        assert result is True
        assert sm.current_state == "next"
        assert len(sm.state_history) == 2
        assert sm.state_history[-1] == "next"

    async def test_get_state_history(self):
        """Test getting state history."""
        sm = StateMachine("test", "initial")
        initial_state = MockState("initial")
        next_state = MockState("next")
        sm.add_state(initial_state)
        sm.add_state(next_state)

        transition = Transition(from_state="initial", to_state="next")
        sm.add_transition(transition)

        await sm.start()
        await sm.transition_to("next")

        history = sm.get_state_history()
        assert "initial" in history
        assert "next" in history

    async def test_stop_state_machine(self):
        """Test stopping the state machine."""
        sm = StateMachine("test", "initial")
        initial_state = MockState("initial")
        sm.add_state(initial_state)

        await sm.start()
        assert sm.is_running

        await sm.stop()
        assert not sm.is_running

    async def test_get_current_state_info(self):
        """Test getting current state information."""
        sm = StateMachine("test", "initial")
        initial_state = MockState("initial")
        sm.add_state(initial_state)

        await sm.start()

        info = sm.get_current_state_info()
        # Check for 'name' key instead of 'state' based on actual implementation
        assert "name" in info
        assert info["name"] == "initial"

    async def test_invalid_transition(self):
        """Test handling of invalid transitions."""
        sm = StateMachine("test", "initial")
        initial_state = MockState("initial")
        sm.add_state(initial_state)

        await sm.start()

        # Try invalid transition to non-existent state (should raise ValueError)
        try:
            await sm.transition_to("next", "invalid_event")
            raise AssertionError("Expected ValueError for non-existent state")
        except ValueError as e:
            assert "not found" in str(e)

        # Try invalid transition with existing state but no valid transition (should return False)
        next_state = MockState("next")
        sm.add_state(next_state)
        result = await sm.transition_to("next", "invalid_event")
        assert result is False


class TestStateMachineIntegration:
    """Test state machine integration scenarios."""

    async def test_complete_workflow(self):
        """Test complete state machine workflow."""
        sm = StateMachine("workflow", "start")

        # Add states
        start_state = MockState("start")
        processing_state = MockState("processing")
        end_state = MockState("end", StateType.FINAL)

        sm.add_state(start_state)
        sm.add_state(processing_state)
        sm.add_state(end_state)

        # Add transitions
        sm.add_transition(Transition("start", "processing", "begin"))
        sm.add_transition(Transition("processing", "end", "complete"))

        # Execute workflow
        await sm.start()
        await sm.transition_to("processing", "begin")
        await sm.transition_to("end", "complete")

        assert sm.current_state == "end"
        history = sm.get_state_history()
        assert len(history) == 3
        assert history == ["start", "processing", "end"]

        # Check history
        history = sm.get_state_history()
        assert "start" in history
        assert "processing" in history
        assert "end" in history

    async def test_conditional_transitions(self):
        """Test transitions with guard conditions."""

        def check_permission(context):
            return context.get("has_permission", False)

        sm = StateMachine("conditional", "waiting")

        # Add states
        sm.add_state(MockState("waiting"))
        sm.add_state(MockState("authorized", StateType.FINAL))
        sm.add_state(MockState("denied", StateType.FINAL))

        # Add conditional transitions
        sm.add_transition(Transition("waiting", "authorized", "check", guard=check_permission))
        sm.add_transition(Transition("waiting", "denied", "check"))

        # Test with permission
        await sm.start({"has_permission": True})
        await sm.transition_to("authorized", "check")

        # Should have transitioned successfully
        assert sm.current_state == "authorized"

        # Test without permission
        sm2 = StateMachine("conditional2", "waiting")
        sm2.add_state(MockState("waiting"))
        sm2.add_state(MockState("authorized", StateType.FINAL))
        sm2.add_state(MockState("denied", StateType.FINAL))
        sm2.add_transition(Transition("waiting", "authorized", "check", guard=check_permission))
        sm2.add_transition(
            Transition("waiting", "denied", "check", guard=lambda ctx: not check_permission(ctx))
        )

        await sm2.start({"has_permission": False})
        await sm2.transition_to("denied", "check")
        assert sm2.current_state == "denied"

    async def test_state_machine_with_actions(self):
        """Test state machine with transition actions."""
        sm = StateMachine("actions", "start")

        # Track action execution
        actions_executed = []

        def log_action(name):
            def action(context):
                actions_executed.append(name)

            return action

        # Add states
        start_state = MockState("start")
        middle_state = MockState("middle")
        end_state = MockState("end")

        sm.add_state(start_state)
        sm.add_state(middle_state)
        sm.add_state(end_state)

        # Add transitions with actions
        sm.add_transition(Transition("start", "middle", action=log_action("start_to_middle")))
        sm.add_transition(Transition("middle", "end", action=log_action("middle_to_end")))

        # Execute workflow
        await sm.start()
        await sm.transition_to("middle")
        await sm.transition_to("end")

        assert "start_to_middle" in actions_executed
        assert "middle_to_end" in actions_executed
        assert len(actions_executed) == 2


if __name__ == "__main__":
    pytest.main([__file__])
