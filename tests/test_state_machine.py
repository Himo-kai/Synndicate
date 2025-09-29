"""
Comprehensive tests for state machine functionality.

Tests cover:
- State transitions and validation
- State persistence and recovery
- Event handling and processing
- Error states and recovery
- Concurrent state access
- State machine lifecycle
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from enum import Enum

import pytest

from synndicate.core.state_machine import (
    StateMachine, State, Transition, StateContext, StateType
)
from synndicate.core.orchestrator import ExecutionState, TaskState, AgentStatus as AgentState


class TestState(Enum):
    """Test states for state machine testing."""
    INITIAL = "initial"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@pytest.fixture
def simple_state_machine():
    """Create a simple state machine for testing."""
    sm = StateMachine(name="test_sm", initial_state="initial")
    
    # Add states
    sm.add_state(State("initial", StateType.INITIAL))
    sm.add_state(State("processing", StateType.INTERMEDIATE))
    sm.add_state(State("completed", StateType.FINAL))
    sm.add_state(State("error", StateType.ERROR))
    sm.add_state(State("cancelled", StateType.FINAL))
    
    # Add transitions
    sm.add_transition(Transition("initial", "processing", "start"))
    sm.add_transition(Transition("processing", "completed", "complete"))
    sm.add_transition(Transition("processing", "error", "error"))
    sm.add_transition(Transition("error", "processing", "retry"))
    sm.add_transition(Transition("processing", "cancelled", "cancel"))
    
    return sm


@pytest.fixture
def complex_state_machine():
    """Create a complex state machine with multiple paths."""
    sm = StateMachine(initial_state=ExecutionState.PENDING)
    
    # Add comprehensive transitions
    sm.add_transition(ExecutionState.PENDING, ExecutionState.RUNNING, "start")
    sm.add_transition(ExecutionState.RUNNING, ExecutionState.PAUSED, "pause")
    sm.add_transition(ExecutionState.PAUSED, ExecutionState.RUNNING, "resume")
    sm.add_transition(ExecutionState.RUNNING, ExecutionState.COMPLETED, "complete")
    sm.add_transition(ExecutionState.RUNNING, ExecutionState.FAILED, "fail")
    sm.add_transition(ExecutionState.FAILED, ExecutionState.RUNNING, "retry")
    sm.add_transition(ExecutionState.PAUSED, ExecutionState.CANCELLED, "cancel")
    sm.add_transition(ExecutionState.RUNNING, ExecutionState.CANCELLED, "cancel")
    
    return sm


class TestStateMachineBasics:
    """Test basic state machine functionality."""
    
    def test_state_machine_initialization(self):
        """Test state machine initialization."""
        sm = StateMachine(initial_state=TestState.INITIAL)
        
        assert sm.current_state == TestState.INITIAL
        assert sm.previous_state is None
        assert len(sm.transitions) == 0
        assert len(sm.state_history) == 1
        assert sm.state_history[0].state == TestState.INITIAL
    
    def test_add_transition(self, simple_state_machine):
        """Test adding transitions to state machine."""
        sm = simple_state_machine
        
        # Add new transition
        sm.add_transition(TestState.COMPLETED, TestState.INITIAL, "reset")
        
        # Verify transition exists
        assert sm.can_transition(TestState.COMPLETED, "reset")
        assert sm.get_next_state(TestState.COMPLETED, "reset") == TestState.INITIAL
    
    def test_add_duplicate_transition(self, simple_state_machine):
        """Test adding duplicate transition overwrites existing."""
        sm = simple_state_machine
        
        # Add duplicate transition with different target
        sm.add_transition(TestState.INITIAL, TestState.ERROR, "start")
        
        # Should overwrite previous transition
        assert sm.get_next_state(TestState.INITIAL, "start") == TestState.ERROR
    
    def test_remove_transition(self, simple_state_machine):
        """Test removing transitions."""
        sm = simple_state_machine
        
        # Remove existing transition
        sm.remove_transition(TestState.INITIAL, "start")
        
        # Should no longer be able to transition
        assert not sm.can_transition(TestState.INITIAL, "start")
        
        # Removing non-existent transition should not error
        sm.remove_transition(TestState.INITIAL, "nonexistent")
    
    def test_can_transition(self, simple_state_machine):
        """Test checking if transitions are possible."""
        sm = simple_state_machine
        
        # Valid transitions
        assert sm.can_transition(TestState.INITIAL, "start")
        assert sm.can_transition(TestState.PROCESSING, "complete")
        assert sm.can_transition(TestState.PROCESSING, "error")
        
        # Invalid transitions
        assert not sm.can_transition(TestState.INITIAL, "complete")
        assert not sm.can_transition(TestState.COMPLETED, "start")
        assert not sm.can_transition(TestState.INITIAL, "nonexistent")
    
    def test_get_next_state(self, simple_state_machine):
        """Test getting next state for transition."""
        sm = simple_state_machine
        
        # Valid transitions
        assert sm.get_next_state(TestState.INITIAL, "start") == TestState.PROCESSING
        assert sm.get_next_state(TestState.PROCESSING, "complete") == TestState.COMPLETED
        
        # Invalid transitions
        assert sm.get_next_state(TestState.INITIAL, "complete") is None
        assert sm.get_next_state(TestState.COMPLETED, "nonexistent") is None


class TestStateTransitions:
    """Test state transitions and validation."""
    
    def test_valid_transition(self, simple_state_machine):
        """Test valid state transition."""
        sm = simple_state_machine
        
        # Perform valid transition
        result = sm.transition("start")
        
        assert result is True
        assert sm.current_state == TestState.PROCESSING
        assert sm.previous_state == TestState.INITIAL
        assert len(sm.state_history) == 2
    
    def test_invalid_transition(self, simple_state_machine):
        """Test invalid state transition."""
        sm = simple_state_machine
        
        # Attempt invalid transition
        with pytest.raises(StateMachineError, match="Invalid transition"):
            sm.transition("complete")  # Can't complete from INITIAL
    
    def test_transition_with_data(self, simple_state_machine):
        """Test transition with associated data."""
        sm = simple_state_machine
        
        transition_data = {"reason": "user_initiated", "timestamp": datetime.now()}
        
        result = sm.transition("start", data=transition_data)
        
        assert result is True
        assert sm.current_state == TestState.PROCESSING
        
        # Check that data is stored in history
        latest_entry = sm.state_history[-1]
        assert latest_entry.data == transition_data
    
    def test_multiple_transitions(self, simple_state_machine):
        """Test multiple consecutive transitions."""
        sm = simple_state_machine
        
        # Chain of transitions
        sm.transition("start")  # INITIAL -> PROCESSING
        assert sm.current_state == TestState.PROCESSING
        
        sm.transition("complete")  # PROCESSING -> COMPLETED
        assert sm.current_state == TestState.COMPLETED
        assert sm.previous_state == TestState.PROCESSING
        
        # History should contain all states
        assert len(sm.state_history) == 3
        states = [entry.state for entry in sm.state_history]
        assert states == [TestState.INITIAL, TestState.PROCESSING, TestState.COMPLETED]
    
    def test_transition_to_error_state(self, simple_state_machine):
        """Test transition to error state."""
        sm = simple_state_machine
        
        sm.transition("start")  # INITIAL -> PROCESSING
        
        error_data = {"error": "Processing failed", "code": 500}
        sm.transition("error", data=error_data)
        
        assert sm.current_state == TestState.ERROR
        assert sm.state_history[-1].data == error_data
    
    def test_transition_retry_from_error(self, simple_state_machine):
        """Test retry transition from error state."""
        sm = simple_state_machine
        
        # Go to error state
        sm.transition("start")
        sm.transition("error")
        assert sm.current_state == TestState.ERROR
        
        # Retry from error
        sm.transition("retry")
        assert sm.current_state == TestState.PROCESSING
        assert sm.previous_state == TestState.ERROR


class TestStateHistory:
    """Test state history tracking."""
    
    def test_state_history_tracking(self, simple_state_machine):
        """Test that state history is properly tracked."""
        sm = simple_state_machine
        
        initial_time = datetime.now()
        
        # Perform transitions
        sm.transition("start")
        sm.transition("complete")
        
        # Check history
        assert len(sm.state_history) == 3
        
        # All entries should have timestamps
        for entry in sm.state_history:
            assert entry.timestamp >= initial_time
            assert isinstance(entry.timestamp, datetime)
    
    def test_get_state_duration(self, simple_state_machine):
        """Test getting duration in each state."""
        sm = simple_state_machine
        
        sm.transition("start")
        
        # Get duration in INITIAL state
        duration = sm.get_state_duration(TestState.INITIAL)
        assert duration > 0
        
        # Current state should have duration from entry to now
        current_duration = sm.get_current_state_duration()
        assert current_duration > 0
    
    def test_state_history_limit(self):
        """Test state history size limiting."""
        sm = StateMachine(initial_state=TestState.INITIAL, max_history_size=3)
        
        # Add transitions to exceed history limit
        sm.add_transition(TestState.INITIAL, TestState.PROCESSING, "start")
        sm.add_transition(TestState.PROCESSING, TestState.COMPLETED, "complete")
        sm.add_transition(TestState.COMPLETED, TestState.INITIAL, "reset")
        
        # Perform multiple transitions
        sm.transition("start")
        sm.transition("complete")
        sm.transition("reset")
        sm.transition("start")
        
        # History should be limited
        assert len(sm.state_history) <= 3
    
    def test_clear_history(self, simple_state_machine):
        """Test clearing state history."""
        sm = simple_state_machine
        
        sm.transition("start")
        sm.transition("complete")
        
        # Clear history
        sm.clear_history()
        
        # Should only have current state
        assert len(sm.state_history) == 1
        assert sm.state_history[0].state == sm.current_state


class TestStateValidation:
    """Test state validation and constraints."""
    
    def test_state_validation_callback(self):
        """Test state validation with callback."""
        def validate_processing_state(state, data):
            if state == TestState.PROCESSING and data and data.get("invalid"):
                return False, "Invalid processing data"
            return True, None
        
        sm = StateMachine(
            initial_state=TestState.INITIAL,
            state_validator=validate_processing_state
        )
        sm.add_transition(TestState.INITIAL, TestState.PROCESSING, "start")
        
        # Valid transition
        result = sm.transition("start", data={"valid": True})
        assert result is True
        
        # Reset for next test
        sm.current_state = TestState.INITIAL
        
        # Invalid transition
        with pytest.raises(StateMachineError, match="Invalid processing data"):
            sm.transition("start", data={"invalid": True})
    
    def test_state_entry_exit_callbacks(self):
        """Test state entry and exit callbacks."""
        entry_calls = []
        exit_calls = []
        
        def on_entry(state, data):
            entry_calls.append((state, data))
        
        def on_exit(state, data):
            exit_calls.append((state, data))
        
        sm = StateMachine(
            initial_state=TestState.INITIAL,
            on_state_entry=on_entry,
            on_state_exit=on_exit
        )
        sm.add_transition(TestState.INITIAL, TestState.PROCESSING, "start")
        
        # Perform transition
        sm.transition("start", data={"test": True})
        
        # Check callbacks were called
        assert len(exit_calls) == 1
        assert exit_calls[0][0] == TestState.INITIAL
        
        assert len(entry_calls) == 1
        assert entry_calls[0][0] == TestState.PROCESSING
        assert entry_calls[0][1] == {"test": True}
    
    def test_transition_guard_conditions(self):
        """Test transition guard conditions."""
        def can_start(current_state, event, data):
            return data and data.get("authorized", False)
        
        sm = StateMachine(initial_state=TestState.INITIAL)
        sm.add_transition(
            TestState.INITIAL, 
            TestState.PROCESSING, 
            "start",
            guard=can_start
        )
        
        # Transition without authorization should fail
        with pytest.raises(StateMachineError, match="Guard condition failed"):
            sm.transition("start", data={"authorized": False})
        
        # Transition with authorization should succeed
        result = sm.transition("start", data={"authorized": True})
        assert result is True
        assert sm.current_state == TestState.PROCESSING


class TestConcurrentAccess:
    """Test concurrent access to state machine."""
    
    @pytest.mark.asyncio
    async def test_concurrent_transitions(self):
        """Test concurrent state transitions."""
        sm = StateMachine(initial_state=TestState.INITIAL)
        sm.add_transition(TestState.INITIAL, TestState.PROCESSING, "start")
        sm.add_transition(TestState.PROCESSING, TestState.COMPLETED, "complete")
        
        async def perform_transition(event):
            try:
                return sm.transition(event)
            except StateMachineError:
                return False
        
        # Try concurrent transitions
        results = await asyncio.gather(
            perform_transition("start"),
            perform_transition("start"),
            return_exceptions=True
        )
        
        # Only one should succeed (depending on implementation)
        successful = sum(1 for r in results if r is True)
        assert successful <= 1
    
    @pytest.mark.asyncio
    async def test_thread_safe_state_access(self):
        """Test thread-safe access to state machine."""
        sm = StateMachine(initial_state=TestState.INITIAL)
        
        async def read_state():
            return sm.current_state
        
        async def get_history():
            return len(sm.state_history)
        
        # Concurrent reads should be safe
        results = await asyncio.gather(
            read_state(),
            read_state(),
            get_history(),
            get_history()
        )
        
        # All reads should return consistent values
        assert all(r == TestState.INITIAL for r in results[:2])
        assert all(r == 1 for r in results[2:])


class TestStatePersistence:
    """Test state persistence and recovery."""
    
    def test_get_state_snapshot(self, simple_state_machine):
        """Test getting state machine snapshot."""
        sm = simple_state_machine
        
        sm.transition("start")
        sm.transition("complete")
        
        snapshot = sm.get_state_snapshot()
        
        assert "current_state" in snapshot
        assert "previous_state" in snapshot
        assert "state_history" in snapshot
        assert "transitions" in snapshot
        
        assert snapshot["current_state"] == TestState.COMPLETED.value
        assert snapshot["previous_state"] == TestState.PROCESSING.value
        assert len(snapshot["state_history"]) == 3
    
    def test_restore_from_snapshot(self, simple_state_machine):
        """Test restoring state machine from snapshot."""
        sm = simple_state_machine
        
        # Create snapshot
        sm.transition("start")
        snapshot = sm.get_state_snapshot()
        
        # Create new state machine and restore
        new_sm = StateMachine(initial_state=TestState.INITIAL)
        new_sm.restore_from_snapshot(snapshot)
        
        assert new_sm.current_state == sm.current_state
        assert new_sm.previous_state == sm.previous_state
        assert len(new_sm.state_history) == len(sm.state_history)
    
    def test_invalid_snapshot_restoration(self):
        """Test restoration from invalid snapshot."""
        sm = StateMachine(initial_state=TestState.INITIAL)
        
        # Invalid snapshot
        invalid_snapshot = {"invalid": "data"}
        
        with pytest.raises(StateMachineError, match="Invalid snapshot"):
            sm.restore_from_snapshot(invalid_snapshot)


class TestExecutionStateIntegration:
    """Test integration with ExecutionState enum."""
    
    def test_execution_state_machine(self, complex_state_machine):
        """Test state machine with ExecutionState."""
        sm = complex_state_machine
        
        # Test execution flow
        sm.transition("start")
        assert sm.current_state == ExecutionState.RUNNING
        
        sm.transition("pause")
        assert sm.current_state == ExecutionState.PAUSED
        
        sm.transition("resume")
        assert sm.current_state == ExecutionState.RUNNING
        
        sm.transition("complete")
        assert sm.current_state == ExecutionState.COMPLETED
    
    def test_execution_error_recovery(self, complex_state_machine):
        """Test error recovery in execution state machine."""
        sm = complex_state_machine
        
        sm.transition("start")
        sm.transition("fail")
        assert sm.current_state == ExecutionState.FAILED
        
        # Retry from failure
        sm.transition("retry")
        assert sm.current_state == ExecutionState.RUNNING
    
    def test_execution_cancellation(self, complex_state_machine):
        """Test cancellation from various states."""
        sm = complex_state_machine
        
        # Cancel from running
        sm.transition("start")
        sm.transition("cancel")
        assert sm.current_state == ExecutionState.CANCELLED
        
        # Reset and test cancel from paused
        sm.current_state = ExecutionState.PENDING
        sm.transition("start")
        sm.transition("pause")
        sm.transition("cancel")
        assert sm.current_state == ExecutionState.CANCELLED


class TestTaskStateIntegration:
    """Test integration with TaskState enum."""
    
    def test_task_state_machine(self):
        """Test state machine with TaskState."""
        sm = StateMachine(initial_state=TaskState.CREATED)
        
        # Add task-specific transitions
        sm.add_transition(TaskState.CREATED, TaskState.QUEUED, "queue")
        sm.add_transition(TaskState.QUEUED, TaskState.RUNNING, "start")
        sm.add_transition(TaskState.RUNNING, TaskState.COMPLETED, "complete")
        sm.add_transition(TaskState.RUNNING, TaskState.FAILED, "fail")
        
        # Test task lifecycle
        sm.transition("queue")
        assert sm.current_state == TaskState.QUEUED
        
        sm.transition("start")
        assert sm.current_state == TaskState.RUNNING
        
        sm.transition("complete")
        assert sm.current_state == TaskState.COMPLETED


class TestAgentStateIntegration:
    """Test integration with AgentState enum."""
    
    def test_agent_state_machine(self):
        """Test state machine with AgentState."""
        sm = StateMachine(initial_state=AgentState.IDLE)
        
        # Add agent-specific transitions
        sm.add_transition(AgentState.IDLE, AgentState.ACTIVE, "activate")
        sm.add_transition(AgentState.ACTIVE, AgentState.BUSY, "start_task")
        sm.add_transition(AgentState.BUSY, AgentState.ACTIVE, "finish_task")
        sm.add_transition(AgentState.ACTIVE, AgentState.IDLE, "deactivate")
        
        # Test agent lifecycle
        sm.transition("activate")
        assert sm.current_state == AgentState.ACTIVE
        
        sm.transition("start_task")
        assert sm.current_state == AgentState.BUSY
        
        sm.transition("finish_task")
        assert sm.current_state == AgentState.ACTIVE
        
        sm.transition("deactivate")
        assert sm.current_state == AgentState.IDLE


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_initial_state(self):
        """Test invalid initial state."""
        with pytest.raises(ValueError, match="Initial state cannot be None"):
            StateMachine(initial_state=None)
    
    def test_transition_from_invalid_state(self, simple_state_machine):
        """Test transition from invalid current state."""
        sm = simple_state_machine
        
        # Manually set invalid state
        sm.current_state = "invalid_state"
        
        with pytest.raises(StateMachineError, match="Invalid transition"):
            sm.transition("start")
    
    def test_empty_event_name(self, simple_state_machine):
        """Test transition with empty event name."""
        sm = simple_state_machine
        
        with pytest.raises(ValueError, match="Event name cannot be empty"):
            sm.transition("")
    
    def test_none_event_name(self, simple_state_machine):
        """Test transition with None event name."""
        sm = simple_state_machine
        
        with pytest.raises(ValueError, match="Event name cannot be None"):
            sm.transition(None)
    
    def test_circular_transitions(self):
        """Test handling of circular transitions."""
        sm = StateMachine(initial_state=TestState.INITIAL)
        
        # Create circular transitions
        sm.add_transition(TestState.INITIAL, TestState.PROCESSING, "start")
        sm.add_transition(TestState.PROCESSING, TestState.INITIAL, "reset")
        
        # Should handle circular transitions without issues
        sm.transition("start")
        sm.transition("reset")
        sm.transition("start")
        sm.transition("reset")
        
        assert sm.current_state == TestState.INITIAL
        assert len(sm.state_history) == 5  # Initial + 4 transitions


class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_state_history_performance(self):
        """Test performance with large state history."""
        sm = StateMachine(initial_state=TestState.INITIAL)
        sm.add_transition(TestState.INITIAL, TestState.PROCESSING, "start")
        sm.add_transition(TestState.PROCESSING, TestState.INITIAL, "reset")
        
        # Perform many transitions
        for i in range(1000):
            if i % 2 == 0:
                sm.transition("start")
            else:
                sm.transition("reset")
        
        # Should handle large history efficiently
        assert len(sm.state_history) == 1001  # Initial + 1000 transitions
        
        # Operations should still be fast
        duration = sm.get_current_state_duration()
        assert duration >= 0
    
    def test_many_transitions_performance(self):
        """Test performance with many possible transitions."""
        sm = StateMachine(initial_state=TestState.INITIAL)
        
        # Add many transitions
        for i in range(100):
            sm.add_transition(TestState.INITIAL, TestState.PROCESSING, f"event_{i}")
        
        # Should handle many transitions efficiently
        assert len(sm.transitions[TestState.INITIAL]) == 100
        
        # Transition lookup should be fast
        assert sm.can_transition(TestState.INITIAL, "event_50")
        assert sm.get_next_state(TestState.INITIAL, "event_50") == TestState.PROCESSING


if __name__ == "__main__":
    pytest.main([__file__])
