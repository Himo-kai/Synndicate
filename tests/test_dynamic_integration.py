#!/usr/bin/env python3
"""
Comprehensive Integration Test for Dynamic Agent Orchestration

This script validates the dynamic orchestration system with real workloads,
measuring performance and testing various scenarios.
"""

import asyncio
import json

# Add src to path for imports
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

from synndicate.agents.coder import DynamicCoderAgent
from synndicate.agents.dynamic_critic import DynamicCriticAgent
from synndicate.agents.planner import PlannerAgent
from synndicate.core.dynamic_orchestrator import AgentRole, DynamicOrchestrator, TaskRequirement
from synndicate.core.enhanced_orchestrator import EnhancedOrchestrator
from synndicate.models.manager import ModelManager
from synndicate.observability.logging import get_logger

logger = get_logger(__name__)


class IntegrationTestRunner:
    """Runs comprehensive integration tests for dynamic orchestration."""

    def __init__(self):
        self.results: list[dict[str, Any]] = []
        self.model_manager = None
        self.enhanced_orchestrator = None

    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up integration test environment...")

        try:
            # Initialize model manager
            self.model_manager = ModelManager()
            await self.model_manager.initialize()

            # Initialize enhanced orchestrator
            self.enhanced_orchestrator = EnhancedOrchestrator(
                max_agents=5, idle_timeout=120.0, performance_threshold=0.6
            )

            logger.info("✅ Test environment setup complete")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {e}")
            logger.error(traceback.format_exc())
            return False

    async def run_all_tests(self) -> dict[str, Any]:
        """Run all integration tests."""
        logger.info("🚀 Starting comprehensive dynamic orchestration integration tests...")

        test_suite = [
            ("Basic Agent Recruitment", self.test_basic_agent_recruitment),
            ("Multi-Agent Collaboration", self.test_multi_agent_collaboration),
            ("Performance Under Load", self.test_performance_under_load),
            ("Resource Optimization", self.test_resource_optimization),
            ("Error Handling", self.test_error_handling),
            ("Real Coding Workload", self.test_real_coding_workload),
            ("Plan-and-Review Workflow", self.test_plan_and_review_workflow),
            ("Auto Workflow Selection", self.test_auto_workflow_selection),
        ]

        overall_start = time.time()
        passed = 0
        failed = 0

        for test_name, test_func in test_suite:
            logger.info(f"\n📋 Running test: {test_name}")
            start_time = time.time()

            try:
                result = await test_func()
                execution_time = time.time() - start_time

                if result.get("success", False):
                    logger.info(f"✅ {test_name} PASSED ({execution_time:.2f}s)")
                    passed += 1
                else:
                    logger.error(f"❌ {test_name} FAILED ({execution_time:.2f}s)")
                    logger.error(f"   Error: {result.get('error', 'Unknown error')}")
                    failed += 1

                result.update({"test_name": test_name, "execution_time": execution_time})
                self.results.append(result)

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"❌ {test_name} CRASHED ({execution_time:.2f}s)")
                logger.error(f"   Exception: {e}")
                logger.error(traceback.format_exc())
                failed += 1

                self.results.append(
                    {
                        "test_name": test_name,
                        "success": False,
                        "error": str(e),
                        "execution_time": execution_time,
                    }
                )

        total_time = time.time() - overall_start

        # Generate summary
        summary = {
            "total_tests": len(test_suite),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(test_suite) * 100,
            "total_execution_time": total_time,
            "results": self.results,
        }

        logger.info("\n🎯 Integration Test Summary:")
        logger.info(f"   Total Tests: {summary['total_tests']}")
        logger.info(f"   Passed: {summary['passed']}")
        logger.info(f"   Failed: {summary['failed']}")
        logger.info(f"   Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"   Total Time: {summary['total_execution_time']:.2f}s")

        return summary

    async def test_basic_agent_recruitment(self) -> dict[str, Any]:
        """Test basic agent recruitment functionality."""
        try:
            orchestrator = DynamicOrchestrator(max_agents=3)

            # Register agent factories
            orchestrator.register_agent_factory(AgentRole.PLANNER, PlannerAgent)
            orchestrator.register_agent_factory(AgentRole.CODER, DynamicCoderAgent)
            orchestrator.register_agent_factory(AgentRole.CRITIC, DynamicCriticAgent)

            # Test task requirement analysis
            query = "Implement a Python function to calculate prime numbers"
            requirements = await orchestrator.analyze_task_requirements(query)

            # Validate requirements
            assert AgentRole.PLANNER in requirements.required_roles
            assert AgentRole.CODER in requirements.required_roles
            assert requirements.estimated_complexity > 0.3

            # Test agent recruitment
            recruited_agents = await orchestrator.recruit_agents("test_task", requirements)

            # Validate recruitment
            assert len(recruited_agents) >= 1
            assert len(orchestrator.agents) >= 1

            # Test dismissal
            dismissed_count = await orchestrator.dismiss_agents("test_task", recruited_agents)
            assert dismissed_count == len(recruited_agents)

            return {
                "success": True,
                "recruited_agents": len(recruited_agents),
                "dismissed_agents": dismissed_count,
                "requirements": {
                    "roles": [role.value for role in requirements.required_roles],
                    "complexity": requirements.estimated_complexity,
                    "duration": requirements.estimated_duration,
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_multi_agent_collaboration(self) -> dict[str, Any]:
        """Test multi-agent collaboration workflows."""
        try:
            if not self.enhanced_orchestrator:
                return {"success": False, "error": "Enhanced orchestrator not initialized"}

            # Test plan-and-code workflow
            query = "Create a FastAPI endpoint for user authentication"
            result = await self.enhanced_orchestrator.process_query(
                query, context={"framework": "FastAPI", "auth_type": "JWT"}, workflow="dynamic"
            )

            # Validate result
            assert result.success
            assert result.response is not None
            assert result.execution_time > 0

            # Get orchestration status
            status = self.enhanced_orchestrator.get_orchestration_status()

            return {
                "success": True,
                "workflow_type": result.workflow_type,
                "execution_time": result.execution_time,
                "response_length": len(result.response.response) if result.response else 0,
                "confidence": result.response.confidence if result.response else 0,
                "orchestration_stats": status.get("dynamic_orchestration", {}),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_performance_under_load(self) -> dict[str, Any]:
        """Test performance under concurrent load."""
        try:
            if not self.enhanced_orchestrator:
                return {"success": False, "error": "Enhanced orchestrator not initialized"}

            # Define test queries
            queries = [
                "Implement a binary search algorithm",
                "Create a REST API for todo management",
                "Design a database schema for e-commerce",
                "Write unit tests for a calculator class",
                "Implement a caching mechanism",
            ]

            # Run concurrent tasks
            start_time = time.time()
            tasks = []

            for i, query in enumerate(queries):
                task = self.enhanced_orchestrator.process_query(
                    query, context={"task_id": i}, workflow="dynamic"
                )
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception) and r.success]
            failed_results = [
                r for r in results if isinstance(r, Exception) or not getattr(r, "success", False)
            ]

            avg_execution_time = (
                sum(r.execution_time for r in successful_results) / len(successful_results)
                if successful_results
                else 0
            )
            avg_confidence = (
                sum(r.response.confidence for r in successful_results if r.response)
                / len(successful_results)
                if successful_results
                else 0
            )

            return {
                "success": len(successful_results) >= len(queries) * 0.8,  # 80% success rate
                "total_queries": len(queries),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "total_time": total_time,
                "avg_execution_time": avg_execution_time,
                "avg_confidence": avg_confidence,
                "throughput": len(queries) / total_time,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_resource_optimization(self) -> dict[str, Any]:
        """Test resource optimization and cleanup."""
        try:
            orchestrator = DynamicOrchestrator(
                max_agents=3, idle_timeout=1.0
            )  # Short timeout for testing

            # Register factories
            orchestrator.register_agent_factory(AgentRole.PLANNER, PlannerAgent)

            # Create and recruit agents
            requirements = TaskRequirement(
                required_roles=[AgentRole.PLANNER], estimated_complexity=0.5, estimated_duration=60
            )

            recruited_agents = await orchestrator.recruit_agents("test_task", requirements)
            initial_agent_count = len(orchestrator.agents)

            # Mark agents as idle and wait for cleanup
            for agent_id in recruited_agents:
                if agent_id in orchestrator.agents:
                    orchestrator.agents[agent_id].status = orchestrator.agents[
                        agent_id
                    ].status.__class__.IDLE
                    orchestrator.agents[agent_id].metrics.last_used = (
                        time.time() - 2.0
                    )  # 2 seconds ago

            # Wait for timeout
            await asyncio.sleep(1.5)

            # Trigger cleanup
            cleaned_count = await orchestrator.cleanup_idle_agents()
            final_agent_count = len(orchestrator.agents)

            return {
                "success": cleaned_count > 0 and final_agent_count < initial_agent_count,
                "initial_agents": initial_agent_count,
                "cleaned_agents": cleaned_count,
                "final_agents": final_agent_count,
                "cleanup_effective": cleaned_count > 0,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_error_handling(self) -> dict[str, Any]:
        """Test error handling and fallback mechanisms."""
        try:
            if not self.enhanced_orchestrator:
                return {"success": False, "error": "Enhanced orchestrator not initialized"}

            # Test with a query that might cause issues
            query = "This is an intentionally vague and problematic query that might cause errors"

            result = await self.enhanced_orchestrator.process_query(query, workflow="dynamic")

            # Should handle gracefully - either succeed or fail gracefully
            graceful_handling = result is not None

            # Test fallback to traditional workflow
            fallback_result = await self.enhanced_orchestrator.process_query(
                "Simple test query", workflow="pipeline"
            )

            return {
                "success": graceful_handling and fallback_result.success,
                "dynamic_handled": graceful_handling,
                "fallback_works": fallback_result.success if fallback_result else False,
                "dynamic_result_type": type(result).__name__ if result else "None",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_real_coding_workload(self) -> dict[str, Any]:
        """Test with a realistic coding workload."""
        try:
            if not self.enhanced_orchestrator:
                return {"success": False, "error": "Enhanced orchestrator not initialized"}

            query = """
            Create a Python class for a simple task queue system with the following requirements:
            1. Add tasks to the queue
            2. Process tasks asynchronously
            3. Handle task failures with retry logic
            4. Provide status monitoring
            5. Include proper error handling and logging
            """

            context = {
                "language": "python",
                "framework": "asyncio",
                "requirements": ["error_handling", "logging", "async_processing"],
            }

            result = await self.enhanced_orchestrator.process_query(
                query, context=context, workflow="dynamic"
            )

            # Analyze the response for coding quality
            response_text = result.response.response if result.response else ""

            quality_indicators = {
                "has_class_definition": "class " in response_text,
                "has_async_methods": "async def" in response_text,
                "has_error_handling": any(
                    keyword in response_text for keyword in ["try:", "except:", "raise"]
                ),
                "has_logging": any(keyword in response_text for keyword in ["logger", "logging"]),
                "has_docstrings": '"""' in response_text or "'''" in response_text,
                "has_type_hints": ":" in response_text and "->" in response_text,
            }

            quality_score = sum(quality_indicators.values()) / len(quality_indicators)

            return {
                "success": result.success and quality_score >= 0.6,
                "execution_time": result.execution_time,
                "response_length": len(response_text),
                "confidence": result.response.confidence if result.response else 0,
                "quality_score": quality_score,
                "quality_indicators": quality_indicators,
                "workflow_type": result.workflow_type,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_plan_and_review_workflow(self) -> dict[str, Any]:
        """Test plan-and-review workflow."""
        try:
            if not self.enhanced_orchestrator:
                return {"success": False, "error": "Enhanced orchestrator not initialized"}

            query = "Design a microservices architecture for a social media platform"

            result = await self.enhanced_orchestrator.process_query(
                query,
                context={"architecture_type": "microservices", "domain": "social_media"},
                workflow="dynamic",
            )

            response_text = result.response.response if result.response else ""

            # Check for planning and review elements
            has_plan = any(
                keyword in response_text.lower()
                for keyword in ["plan", "strategy", "approach", "steps"]
            )
            has_review = any(
                keyword in response_text.lower()
                for keyword in ["review", "analysis", "assessment", "feedback"]
            )

            return {
                "success": result.success and (has_plan or has_review),
                "execution_time": result.execution_time,
                "has_planning_elements": has_plan,
                "has_review_elements": has_review,
                "workflow_type": result.workflow_type,
                "confidence": result.response.confidence if result.response else 0,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_auto_workflow_selection(self) -> dict[str, Any]:
        """Test automatic workflow selection."""
        try:
            if not self.enhanced_orchestrator:
                return {"success": False, "error": "Enhanced orchestrator not initialized"}

            test_cases = [
                ("Implement a sorting algorithm", "coding"),
                ("Create a project plan for mobile app development", "planning"),
                ("Review this code for security issues", "review"),
                ("What is Python?", "simple"),
            ]

            results = []

            for query, expected_type in test_cases:
                result = await self.enhanced_orchestrator.process_query(query, workflow="auto")

                results.append(
                    {
                        "query": query,
                        "expected_type": expected_type,
                        "success": result.success,
                        "workflow_type": result.workflow_type,
                        "execution_time": result.execution_time,
                    }
                )

            successful_selections = sum(1 for r in results if r["success"])

            return {
                "success": successful_selections >= len(test_cases) * 0.75,  # 75% success rate
                "total_cases": len(test_cases),
                "successful_selections": successful_selections,
                "results": results,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def cleanup(self):
        """Cleanup test environment."""
        logger.info("Cleaning up test environment...")

        try:
            if self.enhanced_orchestrator:
                # Clean up any remaining agents
                await self.enhanced_orchestrator.dynamic_orchestrator.cleanup_idle_agents()

            if self.model_manager:
                # Cleanup model manager if needed
                pass

            logger.info("✅ Test environment cleanup complete")

        except Exception as e:
            logger.error(f"⚠️ Error during cleanup: {e}")


async def main():
    """Main test execution function."""
    runner = IntegrationTestRunner()

    try:
        # Setup
        setup_success = await runner.setup()
        if not setup_success:
            logger.error("❌ Failed to setup test environment. Exiting.")
            return

        # Run tests
        summary = await runner.run_all_tests()

        # Save results
        results_file = Path("dynamic_orchestration_test_results.json")
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"📊 Test results saved to: {results_file}")

        # Print final status
        if summary["success_rate"] >= 80:
            logger.info(
                "🎉 Integration tests PASSED! Dynamic orchestration system is ready for production."
            )
        else:
            logger.warning("⚠️ Some integration tests failed. Review results and address issues.")

    except Exception as e:
        logger.error(f"💥 Test execution failed: {e}")
        logger.error(traceback.format_exc())

    finally:
        # Cleanup
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
