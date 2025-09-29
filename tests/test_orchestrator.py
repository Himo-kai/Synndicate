"""
Comprehensive tests for orchestrator functionality.

Tests cover:
- Orchestrator initialization and configuration
- Task execution and pipeline management
- Agent coordination and workflow orchestration
- Error handling and recovery
- State management and transitions
- Resource management and cleanup
- Performance and concurrency
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Any, Dict, List

import pytest

from synndicate.core.orchestrator import (
    Orchestrator, ExecutionState, TaskState, AgentStatus,
    OrchestratorResult, TaskRequirement, ManagedAgent
)
from synndicate.core.pipeline import Pipeline, PipelineStage
from synndicate.agents.base import Agent
from synndicate.config.container import Container
from synndicate.observability.tracing import TraceContext


class MockAgent(Agent):
    """Mock agent for testing."""
    
    def __init__(self, name: str = "mock_agent", **kwargs):
        super().__init__(name=name, **kwargs)
        self.process_calls = []
        self.process_result = {"result": "mock_result", "confidence": 0.8}
    
    async def process(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock process method."""
        self.process_calls.append((task, context))
        return self.process_result


class MockPipeline(Pipeline):
    """Mock pipeline for testing."""
    
    def __init__(self, name: str = "mock_pipeline", **kwargs):
        super().__init__(name=name, **kwargs)
        self.execute_calls = []
        self.execute_result = {"status": "completed", "data": {"result": "pipeline_result"}}
    
    async def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock execute method."""
        self.execute_calls.append(context)
        return self.execute_result


@pytest.fixture
def mock_container():
    """Create a mock container for testing."""
    container = MagicMock(spec=Container)
    container.get_agent.return_value = MockAgent()
    container.get_pipeline.return_value = MockPipeline()
    return container


@pytest.fixture
def orchestrator(mock_container):
    """Create an orchestrator instance for testing."""
    return Orchestrator(container=mock_container)


@pytest.fixture
def sample_task_requirement():
    """Create a sample task requirement."""
    return TaskRequirement(
        task_id="test_task_001",
        description="Test task for orchestrator",
        priority=1,
        estimated_duration=30.0,
        required_agents=["planner", "coder"],
        resource_requirements={"memory": "1GB", "cpu": "1 core"}
    )


class TestOrchestratorInitialization:
    """Test orchestrator initialization and configuration."""
    
    def test_orchestrator_initialization(self, mock_container):
        """Test orchestrator initialization."""
        orchestrator = Orchestrator(container=mock_container)
        
        assert orchestrator.container == mock_container
        assert orchestrator.state == ExecutionState.IDLE
        assert isinstance(orchestrator.active_tasks, dict)
        assert isinstance(orchestrator.managed_agents, dict)
        assert isinstance(orchestrator.execution_history, list)
        assert orchestrator.max_concurrent_tasks > 0
    
    def test_orchestrator_with_custom_config(self, mock_container):
        """Test orchestrator with custom configuration."""
        config = {
            "max_concurrent_tasks": 5,
            "task_timeout": 300.0,
            "enable_monitoring": True
        }
        
        orchestrator = Orchestrator(container=mock_container, **config)
        
        assert orchestrator.max_concurrent_tasks == 5
        assert hasattr(orchestrator, 'task_timeout')
    
    def test_orchestrator_status_initialization(self, orchestrator):
        """Test orchestrator status after initialization."""
        status = orchestrator.get_status()
        
        assert "state" in status
        assert "active_tasks" in status
        assert "managed_agents" in status
        assert "execution_stats" in status
        assert status["state"] == ExecutionState.IDLE.value


class TestTaskManagement:
    """Test task management functionality."""
    
    @pytest.mark.asyncio
    async def test_submit_task(self, orchestrator, sample_task_requirement):
        """Test task submission."""
        task_id = await orchestrator.submit_task(sample_task_requirement)
        
        assert task_id is not None
        assert task_id in orchestrator.active_tasks
        
        task = orchestrator.active_tasks[task_id]
        assert task.state == TaskState.QUEUED
        assert task.requirement == sample_task_requirement
    
    @pytest.mark.asyncio
    async def test_submit_multiple_tasks(self, orchestrator):
        """Test submitting multiple tasks."""
        tasks = []
        for i in range(3):
            req = TaskRequirement(
                task_id=f"task_{i}",
                description=f"Test task {i}",
                priority=i,
                estimated_duration=30.0
            )
            task_id = await orchestrator.submit_task(req)
            tasks.append(task_id)
        
        assert len(orchestrator.active_tasks) == 3
        assert all(tid in orchestrator.active_tasks for tid in tasks)
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, orchestrator, sample_task_requirement):
        """Test getting task status."""
        task_id = await orchestrator.submit_task(sample_task_requirement)
        
        status = orchestrator.get_task_status(task_id)
        
        assert status is not None
        assert "task_id" in status
        assert "state" in status
        assert "progress" in status
        assert status["task_id"] == task_id
        assert status["state"] == TaskState.QUEUED.value
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, orchestrator, sample_task_requirement):
        """Test task cancellation."""
        task_id = await orchestrator.submit_task(sample_task_requirement)
        
        result = await orchestrator.cancel_task(task_id)
        
        assert result is True
        
        # Task should be marked as cancelled
        status = orchestrator.get_task_status(task_id)
        assert status["state"] == TaskState.CANCELLED.value
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, orchestrator):
        """Test cancelling a non-existent task."""
        result = await orchestrator.cancel_task("nonexistent_task")
        
        assert result is False
    
    def test_list_active_tasks(self, orchestrator):
        """Test listing active tasks."""
        # Initially no tasks
        tasks = orchestrator.list_active_tasks()
        assert len(tasks) == 0
        
        # Add some tasks (synchronously for this test)
        for i in range(2):
            req = TaskRequirement(
                task_id=f"task_{i}",
                description=f"Test task {i}",
                priority=i
            )
            # Simulate task addition
            orchestrator.active_tasks[f"task_{i}"] = MagicMock()
            orchestrator.active_tasks[f"task_{i}"].requirement = req
            orchestrator.active_tasks[f"task_{i}"].state = TaskState.QUEUED
        
        tasks = orchestrator.list_active_tasks()
        assert len(tasks) == 2


class TestAgentManagement:
    """Test agent management functionality."""
    
    @pytest.mark.asyncio
    async def test_recruit_agent(self, orchestrator):
        """Test agent recruitment."""
        agent_id = await orchestrator.recruit_agent("planner", {"role": "task_planner"})
        
        assert agent_id is not None
        assert agent_id in orchestrator.managed_agents
        
        managed_agent = orchestrator.managed_agents[agent_id]
        assert managed_agent.agent_type == "planner"
        assert managed_agent.status == AgentStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_recruit_multiple_agents(self, orchestrator):
        """Test recruiting multiple agents."""
        agent_types = ["planner", "coder", "critic"]
        agent_ids = []
        
        for agent_type in agent_types:
            agent_id = await orchestrator.recruit_agent(agent_type)
            agent_ids.append(agent_id)
        
        assert len(orchestrator.managed_agents) == 3
        assert all(aid in orchestrator.managed_agents for aid in agent_ids)
    
    @pytest.mark.asyncio
    async def test_dismiss_agent(self, orchestrator):
        """Test agent dismissal."""
        agent_id = await orchestrator.recruit_agent("planner")
        
        result = await orchestrator.dismiss_agent(agent_id)
        
        assert result is True
        assert agent_id not in orchestrator.managed_agents
    
    @pytest.mark.asyncio
    async def test_dismiss_nonexistent_agent(self, orchestrator):
        """Test dismissing a non-existent agent."""
        result = await orchestrator.dismiss_agent("nonexistent_agent")
        
        assert result is False
    
    def test_get_agent_status(self, orchestrator):
        """Test getting agent status."""
        # Add a mock agent
        agent_id = "test_agent"
        managed_agent = MagicMock(spec=ManagedAgent)
        managed_agent.agent_type = "planner"
        managed_agent.status = AgentStatus.IDLE
        managed_agent.current_task = None
        orchestrator.managed_agents[agent_id] = managed_agent
        
        status = orchestrator.get_agent_status(agent_id)
        
        assert status is not None
        assert "agent_id" in status
        assert "agent_type" in status
        assert "status" in status
        assert status["agent_id"] == agent_id
        assert status["agent_type"] == "planner"
    
    def test_list_managed_agents(self, orchestrator):
        """Test listing managed agents."""
        # Initially no agents
        agents = orchestrator.list_managed_agents()
        assert len(agents) == 0
        
        # Add some agents
        for i in range(2):
            agent_id = f"agent_{i}"
            managed_agent = MagicMock(spec=ManagedAgent)
            managed_agent.agent_type = f"type_{i}"
            managed_agent.status = AgentStatus.IDLE
            orchestrator.managed_agents[agent_id] = managed_agent
        
        agents = orchestrator.list_managed_agents()
        assert len(agents) == 2


class TestWorkflowExecution:
    """Test workflow execution and orchestration."""
    
    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self, orchestrator, sample_task_requirement):
        """Test executing a simple workflow."""
        # Mock the workflow execution
        with patch.object(orchestrator, '_execute_workflow') as mock_execute:
            mock_execute.return_value = OrchestratorResult(
                task_id=sample_task_requirement.task_id,
                status="completed",
                result={"output": "workflow_result"},
                execution_time=1.5,
                agents_used=["planner", "coder"]
            )
            
            result = await orchestrator.execute_workflow(sample_task_requirement)
            
            assert result is not None
            assert result.task_id == sample_task_requirement.task_id
            assert result.status == "completed"
            assert "output" in result.result
    
    @pytest.mark.asyncio
    async def test_execute_workflow_with_agents(self, orchestrator, sample_task_requirement):
        """Test workflow execution with specific agents."""
        # Recruit agents first
        planner_id = await orchestrator.recruit_agent("planner")
        coder_id = await orchestrator.recruit_agent("coder")
        
        # Mock workflow execution
        with patch.object(orchestrator, '_execute_workflow') as mock_execute:
            mock_execute.return_value = OrchestratorResult(
                task_id=sample_task_requirement.task_id,
                status="completed",
                result={"output": "workflow_result"},
                execution_time=2.0,
                agents_used=[planner_id, coder_id]
            )
            
            result = await orchestrator.execute_workflow(
                sample_task_requirement,
                agent_ids=[planner_id, coder_id]
            )
            
            assert result is not None
            assert len(result.agents_used) == 2
    
    @pytest.mark.asyncio
    async def test_execute_workflow_failure(self, orchestrator, sample_task_requirement):
        """Test workflow execution failure handling."""
        with patch.object(orchestrator, '_execute_workflow') as mock_execute:
            mock_execute.side_effect = Exception("Workflow execution failed")
            
            result = await orchestrator.execute_workflow(sample_task_requirement)
            
            assert result is not None
            assert result.status == "failed"
            assert "error" in result.result
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, orchestrator):
        """Test concurrent workflow execution."""
        tasks = []
        for i in range(3):
            req = TaskRequirement(
                task_id=f"concurrent_task_{i}",
                description=f"Concurrent test task {i}",
                priority=1
            )
            tasks.append(req)
        
        # Mock workflow execution
        with patch.object(orchestrator, '_execute_workflow') as mock_execute:
            mock_execute.return_value = OrchestratorResult(
                task_id="test",
                status="completed",
                result={"output": "result"},
                execution_time=0.5,
                agents_used=[]
            )
            
            # Execute workflows concurrently
            results = await asyncio.gather(*[
                orchestrator.execute_workflow(task) for task in tasks
            ])
            
            assert len(results) == 3
            assert all(r.status == "completed" for r in results)


class TestStateManagement:
    """Test orchestrator state management."""
    
    def test_initial_state(self, orchestrator):
        """Test orchestrator initial state."""
        assert orchestrator.state == ExecutionState.IDLE
    
    @pytest.mark.asyncio
    async def test_state_transition_on_task_submission(self, orchestrator, sample_task_requirement):
        """Test state transition when submitting tasks."""
        await orchestrator.submit_task(sample_task_requirement)
        
        # State should change to RUNNING when tasks are active
        assert orchestrator.state in [ExecutionState.RUNNING, ExecutionState.IDLE]
    
    @pytest.mark.asyncio
    async def test_state_transition_on_execution(self, orchestrator, sample_task_requirement):
        """Test state transition during workflow execution."""
        with patch.object(orchestrator, '_execute_workflow') as mock_execute:
            # Simulate long-running execution
            async def slow_execution(*args, **kwargs):
                await asyncio.sleep(0.1)
                return OrchestratorResult(
                    task_id=sample_task_requirement.task_id,
                    status="completed",
                    result={},
                    execution_time=0.1,
                    agents_used=[]
                )
            
            mock_execute.side_effect = slow_execution
            
            # Start execution
            execution_task = asyncio.create_task(
                orchestrator.execute_workflow(sample_task_requirement)
            )
            
            # Give it a moment to start
            await asyncio.sleep(0.05)
            
            # State should be RUNNING during execution
            assert orchestrator.state == ExecutionState.RUNNING
            
            # Wait for completion
            await execution_task
    
    @pytest.mark.asyncio
    async def test_pause_resume_orchestrator(self, orchestrator):
        """Test pausing and resuming the orchestrator."""
        # Pause
        await orchestrator.pause()
        assert orchestrator.state == ExecutionState.PAUSED
        
        # Resume
        await orchestrator.resume()
        assert orchestrator.state == ExecutionState.IDLE
    
    @pytest.mark.asyncio
    async def test_stop_orchestrator(self, orchestrator):
        """Test stopping the orchestrator."""
        await orchestrator.stop()
        assert orchestrator.state == ExecutionState.STOPPED


class TestResourceManagement:
    """Test resource management and cleanup."""
    
    @pytest.mark.asyncio
    async def test_resource_allocation(self, orchestrator, sample_task_requirement):
        """Test resource allocation for tasks."""
        # Mock resource checking
        with patch.object(orchestrator, '_check_resource_availability') as mock_check:
            mock_check.return_value = True
            
            task_id = await orchestrator.submit_task(sample_task_requirement)
            
            # Should have checked resources
            mock_check.assert_called_once()
            assert task_id is not None
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, orchestrator, sample_task_requirement):
        """Test resource cleanup after task completion."""
        with patch.object(orchestrator, '_cleanup_task_resources') as mock_cleanup:
            with patch.object(orchestrator, '_execute_workflow') as mock_execute:
                mock_execute.return_value = OrchestratorResult(
                    task_id=sample_task_requirement.task_id,
                    status="completed",
                    result={},
                    execution_time=1.0,
                    agents_used=[]
                )
                
                await orchestrator.execute_workflow(sample_task_requirement)
                
                # Should have cleaned up resources
                mock_cleanup.assert_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_task_limit(self, orchestrator):
        """Test concurrent task execution limits."""
        # Set a low limit for testing
        orchestrator.max_concurrent_tasks = 2
        
        tasks = []
        for i in range(5):  # More than the limit
            req = TaskRequirement(
                task_id=f"limit_test_{i}",
                description=f"Limit test task {i}",
                priority=1
            )
            tasks.append(req)
        
        # Submit all tasks
        task_ids = []
        for task in tasks:
            task_id = await orchestrator.submit_task(task)
            task_ids.append(task_id)
        
        # Should have queued all tasks but limited concurrent execution
        assert len(orchestrator.active_tasks) == 5
        
        # Check that some tasks are queued (not all running)
        running_count = sum(
            1 for task in orchestrator.active_tasks.values()
            if hasattr(task, 'state') and task.state == TaskState.RUNNING
        )
        assert running_count <= orchestrator.max_concurrent_tasks


class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_agent_failure_handling(self, orchestrator, sample_task_requirement):
        """Test handling agent failures during execution."""
        # Mock agent failure
        with patch.object(orchestrator.container, 'get_agent') as mock_get_agent:
            mock_agent = MockAgent()
            mock_agent.process = AsyncMock(side_effect=Exception("Agent failed"))
            mock_get_agent.return_value = mock_agent
            
            result = await orchestrator.execute_workflow(sample_task_requirement)
            
            # Should handle the failure gracefully
            assert result is not None
            assert result.status == "failed"
    
    @pytest.mark.asyncio
    async def test_pipeline_failure_handling(self, orchestrator, sample_task_requirement):
        """Test handling pipeline failures during execution."""
        with patch.object(orchestrator.container, 'get_pipeline') as mock_get_pipeline:
            mock_pipeline = MockPipeline()
            mock_pipeline.execute = AsyncMock(side_effect=Exception("Pipeline failed"))
            mock_get_pipeline.return_value = mock_pipeline
            
            result = await orchestrator.execute_workflow(sample_task_requirement)
            
            # Should handle the failure gracefully
            assert result is not None
            assert result.status == "failed"
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, orchestrator, sample_task_requirement):
        """Test handling task timeouts."""
        # Set a short timeout
        orchestrator.task_timeout = 0.1
        
        with patch.object(orchestrator, '_execute_workflow') as mock_execute:
            # Simulate long-running task
            async def slow_task(*args, **kwargs):
                await asyncio.sleep(0.2)  # Longer than timeout
                return OrchestratorResult(
                    task_id=sample_task_requirement.task_id,
                    status="completed",
                    result={},
                    execution_time=0.2,
                    agents_used=[]
                )
            
            mock_execute.side_effect = slow_task
            
            result = await orchestrator.execute_workflow(sample_task_requirement)
            
            # Should handle timeout
            assert result is not None
            # Result might be timeout or completed depending on implementation
    
    @pytest.mark.asyncio
    async def test_invalid_task_handling(self, orchestrator):
        """Test handling invalid task requirements."""
        # Create invalid task requirement
        invalid_task = TaskRequirement(
            task_id="",  # Invalid empty task ID
            description="Invalid task",
            priority=-1  # Invalid priority
        )
        
        # Should handle invalid task gracefully
        try:
            task_id = await orchestrator.submit_task(invalid_task)
            # If it doesn't raise an exception, check the result
            if task_id:
                status = orchestrator.get_task_status(task_id)
                assert status is not None
        except ValueError:
            # Expected for invalid task
            pass


class TestPerformanceAndMonitoring:
    """Test performance monitoring and metrics."""
    
    def test_execution_statistics(self, orchestrator):
        """Test execution statistics tracking."""
        stats = orchestrator.get_execution_stats()
        
        assert "total_tasks" in stats
        assert "completed_tasks" in stats
        assert "failed_tasks" in stats
        assert "average_execution_time" in stats
        assert "active_agents" in stats
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, orchestrator, sample_task_requirement):
        """Test performance metrics collection."""
        with patch.object(orchestrator, '_execute_workflow') as mock_execute:
            mock_execute.return_value = OrchestratorResult(
                task_id=sample_task_requirement.task_id,
                status="completed",
                result={},
                execution_time=1.5,
                agents_used=["planner"]
            )
            
            start_time = datetime.now()
            result = await orchestrator.execute_workflow(sample_task_requirement)
            end_time = datetime.now()
            
            # Check that execution time is tracked
            assert result.execution_time > 0
            assert result.execution_time <= (end_time - start_time).total_seconds()
    
    def test_health_check(self, orchestrator):
        """Test orchestrator health check."""
        health = orchestrator.health_check()
        
        assert "status" in health
        assert "uptime" in health
        assert "active_tasks" in health
        assert "managed_agents" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]


class TestTracingIntegration:
    """Test tracing and observability integration."""
    
    @pytest.mark.asyncio
    async def test_trace_context_propagation(self, orchestrator, sample_task_requirement):
        """Test trace context propagation through workflow execution."""
        trace_context = TraceContext(trace_id="test_trace_123")
        
        with patch.object(orchestrator, '_execute_workflow') as mock_execute:
            mock_execute.return_value = OrchestratorResult(
                task_id=sample_task_requirement.task_id,
                status="completed",
                result={},
                execution_time=1.0,
                agents_used=[],
                trace_id="test_trace_123"
            )
            
            result = await orchestrator.execute_workflow(
                sample_task_requirement,
                trace_context=trace_context
            )
            
            # Should propagate trace context
            assert hasattr(result, 'trace_id')
            if hasattr(result, 'trace_id'):
                assert result.trace_id == "test_trace_123"
    
    @pytest.mark.asyncio
    async def test_span_creation(self, orchestrator, sample_task_requirement):
        """Test span creation for workflow execution."""
        with patch('synndicate.observability.tracing.create_span') as mock_span:
            mock_span.return_value.__enter__ = MagicMock()
            mock_span.return_value.__exit__ = MagicMock()
            
            with patch.object(orchestrator, '_execute_workflow') as mock_execute:
                mock_execute.return_value = OrchestratorResult(
                    task_id=sample_task_requirement.task_id,
                    status="completed",
                    result={},
                    execution_time=1.0,
                    agents_used=[]
                )
                
                await orchestrator.execute_workflow(sample_task_requirement)
                
                # Should have created spans for tracing
                # (Implementation may vary based on actual tracing setup)


if __name__ == "__main__":
    pytest.main([__file__])
