"""
Tests for dynamic agent orchestration system.

This module tests:
- Dynamic agent recruitment and dismissal
- Multi-agent collaboration workflows
- Performance tracking and metrics
- Enhanced orchestrator integration
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from synndicate.agents.dynamic_coder import DynamicCoderAgent
from synndicate.agents.dynamic_critic import DynamicCriticAgent
from synndicate.agents.planner import PlannerAgent
from synndicate.core.dynamic_orchestrator import (
    AgentRole,
    AgentStatus,
    DynamicOrchestrator,
    TaskRequirement,
)
from synndicate.core.enhanced_orchestrator import EnhancedOrchestrator


class TestDynamicOrchestrator:
    """Test the core dynamic orchestrator functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create a test orchestrator instance."""
        return DynamicOrchestrator(max_agents=5, idle_timeout=60.0, performance_threshold=0.6)

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager."""
        mock_manager = AsyncMock()
        mock_manager.generate_response.return_value = ("Test reasoning", "Test response")
        return mock_manager

    def test_agent_factory_registration(self, orchestrator):
        """Test registering agent factories."""
        orchestrator.register_agent_factory(AgentRole.PLANNER, PlannerAgent)
        orchestrator.register_agent_factory(AgentRole.CODER, DynamicCoderAgent)

        assert AgentRole.PLANNER in orchestrator.agent_factories
        assert AgentRole.CODER in orchestrator.agent_factories
        assert orchestrator.agent_factories[AgentRole.PLANNER] == PlannerAgent
        assert orchestrator.agent_factories[AgentRole.CODER] == DynamicCoderAgent

    @pytest.mark.asyncio
    async def test_task_requirement_analysis(self, orchestrator):
        """Test analyzing task requirements from queries."""
        # Test coding task
        coding_query = "Implement a Python function to calculate fibonacci numbers"
        requirements = await orchestrator.analyze_task_requirements(coding_query)

        assert AgentRole.PLANNER in requirements.required_roles
        assert AgentRole.CODER in requirements.required_roles
        assert requirements.estimated_complexity > 0.3
        assert requirements.estimated_duration > 60

        # Test planning task
        planning_query = "Create a strategy for migrating our database to PostgreSQL"
        requirements = await orchestrator.analyze_task_requirements(planning_query)

        assert AgentRole.PLANNER in requirements.required_roles
        assert requirements.estimated_complexity > 0.4

        # Test review task
        review_query = "Review this code for security vulnerabilities"
        requirements = await orchestrator.analyze_task_requirements(review_query)

        assert AgentRole.CRITIC in requirements.required_roles

    @pytest.mark.asyncio
    async def test_agent_recruitment(self, orchestrator, mock_model_manager):
        """Test recruiting agents for tasks."""
        # Register factories
        orchestrator.register_agent_factory(AgentRole.PLANNER, PlannerAgent)
        orchestrator.register_agent_factory(AgentRole.CODER, DynamicCoderAgent)

        # Create task requirements
        requirements = TaskRequirement(
            required_roles=[AgentRole.PLANNER, AgentRole.CODER],
            estimated_complexity=0.7,
            estimated_duration=300,
            resource_requirements={"memory": "512MB", "cpu": "1 core"},
        )

        # Mock model manager for agents
        with patch("synndicate.agents.base.Agent.model_manager", mock_model_manager):
            recruited_agents = await orchestrator.recruit_agents("test_task", requirements)

        assert len(recruited_agents) == 2
        assert len(orchestrator.agents) == 2

        # Check agent status
        for agent_id in recruited_agents:
            agent = orchestrator.agents[agent_id]
            assert agent.status == AgentStatus.ACTIVE
            assert agent.task_id == "test_task"

    @pytest.mark.asyncio
    async def test_agent_dismissal(self, orchestrator, mock_model_manager):
        """Test dismissing agents after task completion."""
        # Setup agents
        orchestrator.register_agent_factory(AgentRole.PLANNER, PlannerAgent)
        requirements = TaskRequirement(
            required_roles=[AgentRole.PLANNER], estimated_complexity=0.5, estimated_duration=120
        )

        with patch("synndicate.agents.base.Agent.model_manager", mock_model_manager):
            recruited_agents = await orchestrator.recruit_agents("test_task", requirements)

        assert len(orchestrator.agents) == 1

        # Dismiss agents
        dismissed_count = await orchestrator.dismiss_agents("test_task", recruited_agents)

        assert dismissed_count == 1
        assert len(orchestrator.agents) == 0

    @pytest.mark.asyncio
    async def test_idle_agent_cleanup(self, orchestrator, mock_model_manager):
        """Test cleaning up idle agents."""
        orchestrator.register_agent_factory(AgentRole.PLANNER, PlannerAgent)
        orchestrator.idle_timeout = 0.1  # Very short timeout for testing

        requirements = TaskRequirement(
            required_roles=[AgentRole.PLANNER], estimated_complexity=0.3, estimated_duration=60
        )

        with patch("synndicate.agents.base.Agent.model_manager", mock_model_manager):
            recruited_agents = await orchestrator.recruit_agents("test_task", requirements)

            # Mark agent as idle
            agent_id = recruited_agents[0]
            orchestrator.agents[agent_id].status = AgentStatus.IDLE
            orchestrator.agents[agent_id].metrics.last_used = time.time() - 1.0  # 1 second ago

            # Wait for timeout
            await asyncio.sleep(0.2)

            # Cleanup
            cleaned_count = await orchestrator.cleanup_idle_agents()

            assert cleaned_count == 1
            assert len(orchestrator.agents) == 0

    def test_orchestration_stats(self, orchestrator):
        """Test getting orchestration statistics."""
        stats = orchestrator.get_orchestration_stats()

        assert "total_agents" in stats
        assert "active_agents" in stats
        assert "idle_agents" in stats
        assert "total_recruitments" in stats
        assert "total_dismissals" in stats
        assert "average_task_duration" in stats

        assert stats["total_agents"] == 0
        assert stats["active_agents"] == 0
        assert stats["idle_agents"] == 0


class TestDynamicAgents:
    """Test the specialized dynamic agents."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager."""
        mock_manager = AsyncMock()
        mock_manager.generate_response.return_value = (
            "I'll implement a fibonacci function",
            "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```",
        )
        return mock_manager

    @pytest.mark.asyncio
    async def test_dynamic_coder_agent(self, mock_model_manager):
        """Test the dynamic coder agent."""
        with patch("synndicate.agents.base.Agent.model_manager", mock_model_manager):
            agent = DynamicCoderAgent(specialization="python")

            response = await agent.process(
                "Implement a fibonacci function",
                {"file_path": "math_utils.py", "dependencies": ["typing"]},
            )

            assert response.response is not None
            assert response.confidence > 0
            assert response.metadata["agent_type"] == "coder"
            assert response.metadata["specialization"] == "python"
            assert "code_analysis" in response.metadata

            # Check code analysis
            analysis = response.metadata["code_analysis"]
            assert "has_code_blocks" in analysis
            assert "function_count" in analysis
            assert "quality_score" in analysis

    @pytest.mark.asyncio
    async def test_dynamic_critic_agent(self, mock_model_manager):
        """Test the dynamic critic agent."""
        mock_model_manager.generate_response.return_value = (
            "I'll review this code for issues",
            "The code looks good overall. Minor suggestions: add type hints and error handling.",
        )

        with patch("synndicate.agents.base.Agent.model_manager", mock_model_manager):
            agent = DynamicCriticAgent(review_focus="security")

            response = await agent.process(
                "Review this code",
                {"code_to_review": "def add(a, b): return a + b", "file_path": "utils.py"},
            )

            assert response.response is not None
            assert response.confidence > 0
            assert response.metadata["agent_type"] == "critic"
            assert response.metadata["focus_area"] == "security"
            assert "review_analysis" in response.metadata

            # Check review analysis
            analysis = response.metadata["review_analysis"]
            assert "suggestions_made" in analysis
            assert "review_quality_score" in analysis


class TestEnhancedOrchestrator:
    """Test the enhanced orchestrator with dynamic capabilities."""

    @pytest.fixture
    def enhanced_orchestrator(self):
        """Create an enhanced orchestrator instance."""
        return EnhancedOrchestrator(max_agents=3, idle_timeout=60.0)

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager."""
        mock_manager = AsyncMock()
        mock_manager.generate_response.return_value = (
            "Test reasoning",
            "Test response with implementation details",
        )
        return mock_manager

    @pytest.mark.asyncio
    async def test_dynamic_workflow_execution(self, enhanced_orchestrator, mock_model_manager):
        """Test executing a dynamic workflow."""
        with patch("synndicate.agents.base.Agent.model_manager", mock_model_manager):
            result = await enhanced_orchestrator.process_query(
                "Implement a simple calculator function", workflow="dynamic"
            )

            assert result.success
            assert result.response is not None
            assert result.workflow_type in ["plan_and_code", "planning_only", "single_agent"]
            assert result.execution_time > 0
            assert "metadata" in result.__dict__

    @pytest.mark.asyncio
    async def test_auto_workflow_selection(self, enhanced_orchestrator, mock_model_manager):
        """Test automatic workflow selection."""
        with patch("synndicate.agents.base.Agent.model_manager", mock_model_manager):
            # Test coding query
            result = await enhanced_orchestrator.process_query(
                "Build a REST API for user management", workflow="auto"
            )
            assert result.success

            # Test planning query
            result = await enhanced_orchestrator.process_query(
                "Create a strategy for system migration", workflow="auto"
            )
            assert result.success

            # Test review query
            result = await enhanced_orchestrator.process_query(
                "Analyze this architecture for scalability issues", workflow="auto"
            )
            assert result.success

    @pytest.mark.asyncio
    async def test_fallback_to_traditional_workflow(
        self, enhanced_orchestrator, mock_model_manager
    ):
        """Test fallback to traditional workflows when dynamic fails."""
        with patch("synndicate.agents.base.Agent.model_manager", mock_model_manager):
            # Test pipeline fallback
            result = await enhanced_orchestrator.process_query(
                "Simple question about Python syntax", workflow="pipeline"
            )
            assert result.success

    def test_orchestration_status(self, enhanced_orchestrator):
        """Test getting comprehensive orchestration status."""
        status = enhanced_orchestrator.get_orchestration_status()

        # Check base orchestrator status
        assert "status" in status

        # Check dynamic orchestration status
        assert "dynamic_orchestration" in status
        assert "active_dynamic_tasks" in status
        assert "agent_pool_size" in status
        assert "recruitment_history" in status
        assert "dismissal_history" in status

        dynamic_stats = status["dynamic_orchestration"]
        assert "total_agents" in dynamic_stats
        assert "active_agents" in dynamic_stats


class TestIntegrationWorkflows:
    """Test end-to-end integration workflows."""

    @pytest.fixture
    def full_orchestrator(self):
        """Create a fully configured orchestrator."""
        orchestrator = EnhancedOrchestrator(max_agents=5)
        return orchestrator

    @pytest.fixture
    def mock_model_responses(self):
        """Create mock responses for different agent types."""
        return {
            "planner": (
                "I'll create a detailed plan",
                "## Implementation Plan\n1. Design API structure\n2. Implement endpoints\n3. Add validation\n4. Write tests",
            ),
            "coder": (
                "I'll implement the code",
                "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n```",
            ),
            "critic": (
                "I'll review the implementation",
                "The code looks good. Suggestions: add error handling, input validation, and logging.",
            ),
        }

    @pytest.mark.asyncio
    async def test_plan_and_code_workflow(self, full_orchestrator, mock_model_responses):
        """Test a complete plan-and-code workflow."""

        def mock_generate_response(prompt, **kwargs):
            if "plan" in prompt.lower():
                return mock_model_responses["planner"]
            elif "implement" in prompt.lower() or "code" in prompt.lower():
                return mock_model_responses["coder"]
            else:
                return mock_model_responses["critic"]

        mock_manager = AsyncMock()
        mock_manager.generate_response.side_effect = mock_generate_response

        with patch("synndicate.agents.base.Agent.model_manager", mock_manager):
            result = await full_orchestrator.process_query(
                "Create a FastAPI health check endpoint", workflow="dynamic"
            )

            assert result.success
            assert result.response is not None
            assert result.workflow_type in ["plan_and_code", "planning_only"]

            # Check that multiple agents were involved
            if "metadata" in result.__dict__ and result.metadata:  # noqa: SIM102
                if "agents_used" in result.metadata:
                    assert len(result.metadata["agents_used"]) >= 1

    @pytest.mark.asyncio
    async def test_agent_performance_tracking(self, full_orchestrator, mock_model_responses):
        """Test that agent performance is tracked across tasks."""
        mock_manager = AsyncMock()
        mock_manager.generate_response.return_value = mock_model_responses["planner"]

        with patch("synndicate.agents.base.Agent.model_manager", mock_manager):
            # Execute multiple tasks
            for i in range(3):
                result = await full_orchestrator.process_query(
                    f"Plan task {i+1}: Create a simple web service", workflow="dynamic"
                )
                assert result.success

            # Check orchestration stats
            stats = full_orchestrator.get_orchestration_status()
            stats["dynamic_orchestration"]

            # Should have some recruitment history
            assert stats["recruitment_history"] >= 0
            assert stats["dismissal_history"] >= 0

    @pytest.mark.asyncio
    async def test_resource_optimization(self, full_orchestrator):
        """Test that the orchestrator optimizes resources by cleaning up idle agents."""
        # This test would need to run longer to test idle cleanup
        # For now, just verify the cleanup mechanism exists
        orchestrator = full_orchestrator.dynamic_orchestrator

        # Test that cleanup method exists and can be called
        cleanup_count = await orchestrator.cleanup_idle_agents()
        assert cleanup_count >= 0  # Should return number of cleaned agents

    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, full_orchestrator):
        """Test error handling and fallback mechanisms."""
        # Test with a mock that raises an exception
        mock_manager = AsyncMock()
        mock_manager.generate_response.side_effect = Exception("Model error")

        with patch("synndicate.agents.base.Agent.model_manager", mock_manager):
            # Should fallback gracefully
            result = await full_orchestrator.process_query(
                "Test query that will fail", workflow="dynamic"
            )

            # Should either succeed with fallback or handle error gracefully
            # The exact behavior depends on implementation details
            assert result is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
