"""
Focused tests for state machine functionality based on actual implementation.

Tests the core StateMachine, State, Transition, and StateContext classes.
"""

import pytest
from unittest.mock import MagicMock

from synndicate.core.state_machine import (
    StateMachine, State, Transition, StateContext, StateType
)


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
        state = State("test_state")
        
        assert state.name == "test_state"
        assert state.state_type == StateType.INTERMEDIATE
        assert state.timeout is None
        assert state.entry_time is None
    
    def test_state_with_parameters(self):
        """Test State with custom parameters."""
        state = State("initial_state", StateType.INITIAL, timeout=30.0)
        
        assert state.name == "initial_state"
        assert state.state_type == StateType.INITIAL
        assert state.timeout == 30.0
    
    def test_state_entry_exit_methods(self):
        """Test state entry and exit methods."""
        state = State("test_state")
        context = StateContext()
        
        # These should not raise exceptions (base implementation)
        state.on_entry(context)
        state.on_exit(context)
    
    def test_state_timeout_check(self):
        """Test timeout checking."""
        state = State("test_state", timeout=1.0)
        
        # Initially no timeout
        assert not state.is_timeout_exceeded()
        
        # Set entry time to simulate timeout
        import time
        state.entry_time = time.time() - 2.0  # 2 seconds ago
        assert state.is_timeout_exceeded()
    
    def test_state_string_representation(self):
        """Test state string representation."""
        state = State("test_state", StateType.INITIAL)
        
        str_repr = str(state)
        assert "test_state" in str_repr
        assert "INITIAL" in str_repr


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
    
    def test_transition_with_action(self):
        """Test Transition with action."""
        action_called = []
        
        def action_func(context):
            action_called.append(True)
            context.set("action_executed", True)
        
        transition = Transition("from", "to", "event", action=action_func)
        context = StateContext()
        
        transition.execute_action(context)
        
        assert len(action_called) == 1
        assert context.get("action_executed") is True
    
    def test_transition_without_guard(self):
        """Test Transition without guard (should always allow)."""
        transition = Transition("from", "to", "event")
        context = StateContext()
        
        assert transition.can_transition(context)
    
    def test_transition_without_action(self):
        """Test Transition without action (should not error)."""
        transition = Transition("from", "to", "event")
        context = StateContext()
        
        # Should not raise exception
        transition.execute_action(context)


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
        sm = StateMachine("test", "initial")
        state = State("test_state", StateType.INTERMEDIATE)
        
        sm.add_state(state)
        
        assert "test_state" in sm.states
        assert sm.states["test_state"] == state
    
    def test_add_transition(self):
        """Test adding transitions to state machine."""
        sm = StateMachine("test", "initial")
        transition = Transition("from", "to", "event")
        
        sm.add_transition(transition)
        
        assert transition in sm.transitions
    
    def test_get_valid_transitions(self):
        """Test getting valid transitions from a state."""
        sm = StateMachine("test", "initial")
        
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
    
    def test_start_state_machine(self):
        """Test starting the state machine."""
        sm = StateMachine("test", "initial")
        
        # Add initial state
        initial_state = State("initial", StateType.INITIAL)
        sm.add_state(initial_state)
        
        # Start the machine
        sm.start()
        
        assert sm.is_running
        assert sm.current_state == "initial"
        assert len(sm.state_history) == 1
        assert sm.state_history[0] == "initial"
    
    def test_start_with_context(self):
        """Test starting with initial context."""
        sm = StateMachine("test", "initial")
        initial_state = State("initial", StateType.INITIAL)
        sm.add_state(initial_state)
        
        initial_context = {"key": "value", "number": 42}
        sm.start(initial_context)
        
        assert sm.context.get("key") == "value"
        assert sm.context.get("number") == 42
    
    def test_transition_to_state(self):
        """Test transitioning to a specific state."""
        sm = StateMachine("test", "initial")
        
        # Add states
        sm.add_state(State("initial", StateType.INITIAL))
        sm.add_state(State("next", StateType.INTERMEDIATE))
        
        # Add transition
        sm.add_transition(Transition("initial", "next", "go"))
        
        # Start and transition
        sm.start()
        sm.transition_to("next", "go")
        
        assert sm.current_state == "next"
        assert len(sm.state_history) == 2
        assert sm.state_history[-1] == "next"
    
    def test_get_state_history(self):
        """Test getting state history."""
        sm = StateMachine("test", "initial")
        sm.add_state(State("initial", StateType.INITIAL))
        sm.add_state(State("next", StateType.INTERMEDIATE))
        sm.add_transition(Transition("initial", "next", "go"))
        
        sm.start()
        sm.transition_to("next", "go")
        
        history = sm.get_state_history()
        assert history == ["initial", "next"]
    
    def test_stop_state_machine(self):
        """Test stopping the state machine."""
        sm = StateMachine("test", "initial")
        sm.add_state(State("initial", StateType.INITIAL))
        
        sm.start()
        assert sm.is_running
        
        sm.stop()
        assert not sm.is_running
    
    def test_get_current_state_info(self):
        """Test getting current state information."""
        sm = StateMachine("test", "initial")
        initial_state = State("initial", StateType.INITIAL)
        sm.add_state(initial_state)
        
        sm.start()
        
        info = sm.get_current_state_info()
        assert info is not None
        assert "state" in info
        assert "type" in info
        assert info["state"] == "initial"
    
    def test_invalid_transition(self):
        """Test invalid state transition."""
        sm = StateMachine("test", "initial")
        sm.add_state(State("initial", StateType.INITIAL))
        sm.add_state(State("next", StateType.INTERMEDIATE))
        
        # No transition defined
        sm.start()
        
        # This should handle gracefully or raise appropriate error
        try:
            sm.transition_to("next", "invalid_event")
        except Exception as e:
            # Should be a meaningful error
            assert "transition" in str(e).lower() or "invalid" in str(e).lower()


class TestStateMachineIntegration:
    """Test state machine integration scenarios."""
    
    def test_complete_workflow(self):
        """Test a complete workflow through the state machine."""
        sm = StateMachine("workflow", "start")
        
        # Add states
        sm.add_state(State("start", StateType.INITIAL))
        sm.add_state(State("processing", StateType.INTERMEDIATE))
        sm.add_state(State("completed", StateType.FINAL))
        
        # Add transitions
        sm.add_transition(Transition("start", "processing", "begin"))
        sm.add_transition(Transition("processing", "completed", "finish"))
        
        # Execute workflow
        sm.start({"task": "test_task"})
        assert sm.current_state == "start"
        
        sm.transition_to("processing", "begin")
        assert sm.current_state == "processing"
        
        sm.transition_to("completed", "finish")
        assert sm.current_state == "completed"
        
        # Check final state
        history = sm.get_state_history()
        assert history == ["start", "processing", "completed"]
    
    def test_conditional_transitions(self):
        """Test transitions with guard conditions."""
        sm = StateMachine("conditional", "start")
        
        # Add states
        sm.add_state(State("start", StateType.INITIAL))
        sm.add_state(State("allowed", StateType.FINAL))
        sm.add_state(State("denied", StateType.FINAL))
        
        # Add conditional transitions
        def check_permission(context):
            return context.get("has_permission", False)
        
        sm.add_transition(Transition("start", "allowed", "proceed", guard=check_permission))
        sm.add_transition(Transition("start", "denied", "proceed", guard=lambda ctx: not check_permission(ctx)))
        
        # Test with permission
        sm.start({"has_permission": True})
        sm.transition_to("allowed", "proceed")
        assert sm.current_state == "allowed"
        
        # Reset and test without permission
        sm2 = StateMachine("conditional2", "start")
        sm2.add_state(State("start", StateType.INITIAL))
        sm2.add_state(State("allowed", StateType.FINAL))
        sm2.add_state(State("denied", StateType.FINAL))
        sm2.add_transition(Transition("start", "allowed", "proceed", guard=check_permission))
        sm2.add_transition(Transition("start", "denied", "proceed", guard=lambda ctx: not check_permission(ctx)))
        
        sm2.start({"has_permission": False})
        sm2.transition_to("denied", "proceed")
        assert sm2.current_state == "denied"
    
    def test_state_machine_with_actions(self):
        """Test state machine with transition actions."""
        sm = StateMachine("actions", "start")
        
        # Track action execution
        actions_executed = []
        
        def log_action(name):
            def action(context):
                actions_executed.append(name)
                context.set(f"{name}_executed", True)
            return action
        
        # Add states
        sm.add_state(State("start", StateType.INITIAL))
        sm.add_state(State("middle", StateType.INTERMEDIATE))
        sm.add_state(State("end", StateType.FINAL))
        
        # Add transitions with actions
        sm.add_transition(Transition("start", "middle", "go", action=log_action("first")))
        sm.add_transition(Transition("middle", "end", "finish", action=log_action("second")))
        
        # Execute workflow
        sm.start()
        sm.transition_to("middle", "go")
        sm.transition_to("end", "finish")
        
        # Check actions were executed
        assert actions_executed == ["first", "second"]
        assert sm.context.get("first_executed") is True
        assert sm.context.get("second_executed") is True


if __name__ == "__main__":
    pytest.main([__file__])
