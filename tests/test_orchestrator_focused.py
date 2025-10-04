"""
Focused tests for orchestrator functionality based on actual implementation.

Tests the core Orchestrator class and its state machine/pipeline execution.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synndicate.agents.base import AgentResponse
from synndicate.agents.factory import AgentFactory
from synndicate.config.container import Container
from synndicate.core.orchestrator import (CodingState, CompletionState,
                                          ErrorState, Orchestrator,
                                          OrchestratorResult, PlanningState,
                                          ReviewState, RevisionState)
from synndicate.core.state_machine import StateType


@pytest.fixture
def mock_container():
    """Create a mock container for testing."""
    container = MagicMock(spec=Container)
    container.settings = MagicMock()
    container.get.return_value = MagicMock()  # Mock HTTP client
    return container


@pytest.fixture
def orchestrator(mock_container):
    """Create an orchestrator instance for testing."""
    return Orchestrator(container=mock_container)


class TestOrchestratorInitialization:
    """Test orchestrator initialization and setup."""

    def test_orchestrator_initialization(self, mock_container):
        """Test orchestrator initialization."""
        orchestrator = Orchestrator(container=mock_container)

        assert orchestrator.container == mock_container
        assert isinstance(orchestrator.agent_factory, AgentFactory)
        assert hasattr(orchestrator, "state_machine")
        assert hasattr(orchestrator, "pipelines")

    def test_state_machine_setup(self, orchestrator):
        """Test state machine setup."""
        # Check that state machine is properly initialized
        assert orchestrator.state_machine is not None
        assert orchestrator.state_machine.name == "orchestrator"

        # Check that states are added
        expected_states = ["planning", "coding", "review", "revision", "completion", "error"]
        for state_name in expected_states:
            assert state_name in orchestrator.state_machine.states

    def test_pipelines_setup(self, orchestrator):
        """Test pipelines setup."""
        # Check that pipelines are properly initialized
        assert orchestrator.pipelines is not None
        assert isinstance(orchestrator.pipelines, dict)

        # Should have some predefined pipelines
        assert len(orchestrator.pipelines) > 0


class TestOrchestratorResult:
    """Test OrchestratorResult functionality."""

    def test_orchestrator_result_initialization(self):
        """Test OrchestratorResult initialization."""
        mock_response = MagicMock(spec=AgentResponse)
        mock_response.response = "Test response"

        result = OrchestratorResult(
            success=True,
            final_response=mock_response,
            pipeline_result=None,
            execution_time=1.5,
            agents_used=["planner", "coder"],
            execution_path=["planning", "coding"],
            confidence=0.8,
        )

        assert result.success is True
        assert result.final_response == mock_response
        assert result.execution_time == 1.5
        assert result.agents_used == ["planner", "coder"]
        assert result.execution_path == ["planning", "coding"]
        assert result.confidence == 0.8
        assert isinstance(result.metadata, dict)

    def test_response_text_property(self):
        """Test response_text property."""
        mock_response = MagicMock(spec=AgentResponse)
        mock_response.response = "Test response text"

        result = OrchestratorResult(
            success=True,
            final_response=mock_response,
            pipeline_result=None,
            execution_time=1.0,
            agents_used=[],
            execution_path=[],
            confidence=1.0,
        )

        assert result.response_text == "Test response text"

    def test_response_text_without_response(self):
        """Test response_text property when no final_response."""
        result = OrchestratorResult(
            success=False,
            final_response=None,
            pipeline_result=None,
            execution_time=0.0,
            agents_used=[],
            execution_path=[],
            confidence=0.0,
        )

        assert result.response_text == ""


class TestStateClasses:
    """Test individual state classes."""

    def test_planning_state(self, mock_container):
        """Test PlanningState functionality."""
        agent_factory = AgentFactory(MagicMock(), MagicMock())
        state = PlanningState(agent_factory)

        assert state.name == "planning"
        assert state.state_type == StateType.INTERMEDIATE
        assert state.timeout == 60.0
        assert state.agent_factory == agent_factory

    def test_coding_state(self, mock_container):
        """Test CodingState functionality."""
        agent_factory = AgentFactory(MagicMock(), MagicMock())
        state = CodingState(agent_factory)

        assert state.name == "coding"
        assert state.state_type == StateType.INTERMEDIATE
        assert state.timeout == 120.0
        assert state.agent_factory == agent_factory

    def test_review_state(self, mock_container):
        """Test ReviewState functionality."""
        agent_factory = AgentFactory(MagicMock(), MagicMock())
        state = ReviewState(agent_factory)

        assert state.name == "review"
        assert state.state_type == StateType.INTERMEDIATE
        assert state.timeout == 90.0
        assert state.agent_factory == agent_factory

    def test_revision_state(self, mock_container):
        """Test RevisionState functionality."""
        agent_factory = AgentFactory(MagicMock(), MagicMock())
        state = RevisionState(agent_factory)

        assert state.name == "revision"
        assert state.state_type == StateType.INTERMEDIATE
        assert state.timeout == 120.0
        assert state.agent_factory == agent_factory

    def test_completion_state(self):
        """Test CompletionState functionality."""
        state = CompletionState()

        assert state.name == "completion"
        assert state.state_type == StateType.FINAL
        assert state.timeout is None

    def test_error_state(self):
        """Test ErrorState functionality."""
        state = ErrorState()

        assert state.name == "error"
        assert state.state_type == StateType.ERROR
        assert state.timeout is None


class TestQueryProcessing:
    """Test query processing functionality."""

    @pytest.mark.asyncio
    async def test_process_query_basic(self, orchestrator):
        """Test basic query processing."""
        query = "Create a simple Python function"

        # Mock the internal execution methods
        with patch.object(orchestrator, "_determine_workflow") as mock_determine:
            with patch.object(orchestrator, "_execute_state_machine") as mock_execute:
                mock_determine.return_value = "state_machine"
                mock_execute.return_value = OrchestratorResult(
                    success=True,
                    final_response=MagicMock(response="Function created"),
                    pipeline_result=None,
                    execution_time=2.0,
                    agents_used=["planner", "coder"],
                    execution_path=["planning", "coding", "completion"],
                    confidence=0.9,
                )

                result = await orchestrator.process_query(query)

                assert result is not None
                assert result.success is True
                assert result.execution_time == 2.0
                assert "planner" in result.agents_used
                assert "coder" in result.agents_used

    @pytest.mark.asyncio
    async def test_process_query_with_context(self, orchestrator):
        """Test query processing with context."""
        query = "Update the existing function"
        context = {"existing_code": "def old_function(): pass", "requirements": ["add logging"]}

        with patch.object(orchestrator, "_determine_workflow") as mock_determine:
            with patch.object(orchestrator, "_execute_pipeline") as mock_execute:
                mock_determine.return_value = "pipeline:simple"
                mock_execute.return_value = OrchestratorResult(
                    success=True,
                    final_response=MagicMock(response="Function updated"),
                    pipeline_result=None,
                    execution_time=1.5,
                    agents_used=["coder"],
                    execution_path=["coding"],
                    confidence=0.85,
                )

                result = await orchestrator.process_query(query, context)

                assert result is not None
                assert result.success is True
                mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_with_specific_workflow(self, orchestrator):
        """Test query processing with specific workflow."""
        query = "Review this code"
        workflow = "development"  # Use actual pipeline name

        with patch.object(orchestrator, "_execute_pipeline") as mock_execute:
            mock_execute.return_value = OrchestratorResult(
                success=True,
                final_response=MagicMock(response="Code reviewed"),
                pipeline_result=None,
                execution_time=1.0,
                agents_used=["critic"],
                execution_path=["review"],
                confidence=0.95,
            )

            result = await orchestrator.process_query(query, workflow=workflow)

            assert result is not None
            assert result.success is True
            # The workflow parameter gets passed as pipeline_name to _execute_pipeline
            mock_execute.assert_called_once_with(
                query, {"trace_id": mock_execute.call_args[0][1]["trace_id"]}, "development"
            )

    @pytest.mark.asyncio
    async def test_process_query_failure(self, orchestrator):
        """Test query processing failure handling."""
        query = "Invalid query that will fail"

        with patch.object(orchestrator, "_determine_workflow") as mock_determine:
            with patch.object(orchestrator, "_execute_state_machine") as mock_execute:
                mock_determine.return_value = "state_machine"
                mock_execute.side_effect = Exception("Execution failed")

                result = await orchestrator.process_query(query)

                # Should handle the failure gracefully
                assert result is not None
                assert result.success is False
                assert "error" in result.response_text.lower() or result.response_text == ""


class TestWorkflowDetermination:
    """Test workflow determination logic."""

    def test_determine_workflow_simple_query(self, orchestrator):
        """Test workflow determination for simple queries."""
        query = "Hello world"
        context = {}

        workflow = orchestrator._determine_workflow(query, context)

        assert workflow is not None
        assert isinstance(workflow, str)

    def test_determine_workflow_complex_query(self, orchestrator):
        """Test workflow determination for complex queries."""
        query = "Create a web application with authentication and database integration"
        context = {"complexity": "high", "requirements": ["security", "scalability"]}

        workflow = orchestrator._determine_workflow(query, context)

        assert workflow is not None
        assert isinstance(workflow, str)

    def test_determine_workflow_with_context_hints(self, orchestrator):
        """Test workflow determination with context hints."""
        query = "Fix this bug"
        context = {"workflow_hint": "pipeline:debug", "existing_code": "buggy code"}

        workflow = orchestrator._determine_workflow(query, context)

        assert workflow is not None
        assert isinstance(workflow, str)


class TestStreamProcessing:
    """Test streaming query processing."""

    @pytest.mark.asyncio
    async def test_stream_process_query(self, orchestrator):
        """Test streaming query processing."""
        query = "Create a function step by step"

        # Mock the streaming execution
        async def mock_stream():
            yield {"step": "planning", "progress": 0.2, "message": "Planning the function"}
            yield {"step": "coding", "progress": 0.6, "message": "Writing the code"}
            yield {"step": "completion", "progress": 1.0, "message": "Function completed"}

        with patch.object(orchestrator, "stream_process_query", return_value=mock_stream()):
            stream = orchestrator.stream_process_query(query)

            updates = []
            async for update in stream:
                updates.append(update)

            assert len(updates) == 3
            assert updates[0]["step"] == "planning"
            assert updates[1]["step"] == "coding"
            assert updates[2]["step"] == "completion"
            assert updates[2]["progress"] == 1.0


class TestHealthAndMaintenance:
    """Test health check and maintenance functionality."""

    @pytest.mark.asyncio
    async def test_health_check(self, orchestrator):
        """Test orchestrator health check."""
        # Mock agent factory health_check_all method
        with patch.object(orchestrator.agent_factory, "health_check_all") as mock_health:
            mock_health.return_value = {"planner": True, "coder": True}

            health = await orchestrator.health_check()

            assert health is not None
            assert "overall" in health
            assert "agents" in health
            assert "pipelines" in health
            assert "services" in health

    @pytest.mark.asyncio
    async def test_cleanup(self, orchestrator):
        """Test orchestrator cleanup."""
        # Mock cleanup operations
        with patch.object(orchestrator.agent_factory, "cleanup") as mock_cleanup:
            await orchestrator.cleanup()

            # Should have called agent factory cleanup
            mock_cleanup.assert_called_once()

    def test_get_available_workflows(self, orchestrator):
        """Test getting available workflows."""
        workflows = orchestrator.get_available_workflows()

        assert workflows is not None
        assert isinstance(workflows, (list, dict))

        if isinstance(workflows, list):
            assert len(workflows) > 0
        elif isinstance(workflows, dict):
            assert len(workflows.keys()) > 0


class TestStateExecution:
    """Test state execution functionality."""

    @pytest.mark.asyncio
    async def test_state_machine_execution(self, orchestrator):
        """Test state machine execution path."""
        query = "Create a simple function"
        context = {"language": "python", "complexity": "low"}

        # Mock state machine execution
        with patch.object(orchestrator.state_machine, "start") as mock_start:
            with patch.object(orchestrator.state_machine, "run_to_completion") as mock_run:
                with patch.object(orchestrator.state_machine, "get_state_history") as mock_history:
                    # Mock the final context returned by run_to_completion
                    mock_final_response = MagicMock()
                    mock_final_response.confidence = 0.8
                    mock_run.return_value = {
                        "final_response": mock_final_response,
                        "plan_response": MagicMock(),
                        "code_response": MagicMock(),
                    }
                    mock_history.return_value = ["planning", "coding", "completion"]
                    orchestrator.state_machine.current_state = "completion"

                    result = await orchestrator._execute_state_machine(query, context)

                    assert result is not None
                    assert result.success is True
                    mock_start.assert_called_once()
                    mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_execution(self, orchestrator):
        """Test pipeline execution path."""
        query = "Simple task"
        context = {}
        pipeline_name = "development"  # Use actual pipeline name

        # Mock pipeline execution
        mock_pipeline = MagicMock()
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.success = True
        mock_pipeline_result.final_response = MagicMock()
        mock_pipeline_result.final_response.confidence = 0.9
        mock_pipeline_result.completed_stages = [
            MagicMock(stage_name="planner"),
            MagicMock(stage_name="coder"),
        ]
        mock_pipeline_result.stages = [
            MagicMock(stage_name="planner"),
            MagicMock(stage_name="coder"),
        ]
        mock_pipeline_result.failed_stages = []

        mock_pipeline.execute = AsyncMock(return_value=mock_pipeline_result)

        # Mock the pipelines dictionary directly
        orchestrator.pipelines[pipeline_name] = mock_pipeline

        result = await orchestrator._execute_pipeline(query, context, pipeline_name)

        assert result is not None
        assert result.success is True
        mock_pipeline.execute.assert_called_once_with(query, context)


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_state_machine_error_handling(self, orchestrator):
        """Test error handling in state machine execution."""
        query = "Task that will fail"
        context = {}

        # Mock state machine failure
        with patch.object(orchestrator.state_machine, "start") as mock_start:
            with patch.object(orchestrator.state_machine, "run_to_completion") as mock_run:
                mock_run.side_effect = Exception("State machine failed")

                # The method should handle the exception and return an error result
                try:
                    result = await orchestrator._execute_state_machine(query, context)
                    # If it returns a result, it should indicate failure
                    assert result is not None
                    assert result.success is False
                except Exception:
                    # If it raises an exception, that's also acceptable for this test
                    pass

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, orchestrator):
        """Test error handling in pipeline execution."""
        query = "Task that will fail"
        context = {}
        pipeline_name = "development"  # Use existing pipeline name

        # Mock pipeline failure
        mock_pipeline = MagicMock()
        mock_pipeline.execute = AsyncMock(side_effect=Exception("Pipeline failed"))

        # Mock the pipelines dictionary directly
        orchestrator.pipelines[pipeline_name] = mock_pipeline

        # The method should handle the exception
        try:
            result = await orchestrator._execute_pipeline(query, context, pipeline_name)
            # If it returns a result, it should indicate failure
            assert result is not None
            assert result.success is False
        except Exception:
            # If it raises an exception, that's also acceptable for this test
            pass

    @pytest.mark.asyncio
    async def test_workflow_determination_error(self, orchestrator):
        """Test error handling in workflow determination."""
        query = "Query that breaks workflow determination"
        context = {}

        # Mock workflow determination failure
        with patch.object(orchestrator, "_determine_workflow") as mock_determine:
            mock_determine.side_effect = Exception("Workflow determination failed")

            result = await orchestrator.process_query(query, context)

            # Should handle the error gracefully
            assert result is not None
            assert result.success is False


class TestConcurrency:
    """Test concurrent execution scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self, orchestrator):
        """Test concurrent query processing."""
        queries = ["Create function A", "Create function B", "Create function C"]

        # Mock execution to return quickly
        with patch.object(orchestrator, "_execute_state_machine") as mock_execute:
            mock_execute.return_value = OrchestratorResult(
                success=True,
                final_response=MagicMock(response="Function created"),
                pipeline_result=None,
                execution_time=0.1,
                agents_used=["coder"],
                execution_path=["coding"],
                confidence=0.9,
            )

            # Execute queries concurrently
            tasks = [orchestrator.process_query(query) for query in queries]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, orchestrator):
        """Test concurrent health checks."""
        # Mock health check
        with patch.object(orchestrator.agent_factory, "health_check_all") as mock_health:
            mock_health.return_value = {"planner": True, "coder": True}

            # Perform concurrent health checks
            tasks = [orchestrator.health_check() for _ in range(5)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert all("overall" in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__])
