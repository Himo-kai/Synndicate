"""
Modern orchestrator with pipeline-based architecture and state management.

Improvements over original:
- Pipeline-based execution with configurable workflows
- State machine for complex execution flows
- Circuit breaker pattern for reliability
- Streaming response support
- Better error handling and recovery
- Dependency injection integration
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..agents.base import AgentResponse
from ..agents.factory import AgentFactory
from ..config.container import Container
from ..observability.logging import get_logger, set_trace_id, get_trace_id
from ..observability.metrics import get_metrics_collector
from ..observability.probe import probe
from .audit import create_trace_snapshot, save_trace_snapshot
from .determinism import ensure_deterministic_startup
from .pipeline import AgentStage, ConditionalStage, Pipeline, PipelineResult
from .state_machine import State, StateContext, StateMachine, StateType

logger = get_logger(__name__)


@dataclass
class OrchestratorResult:
    """Enhanced orchestrator result with detailed execution information."""

    success: bool
    final_response: AgentResponse | None
    pipeline_result: PipelineResult | None
    execution_time: float
    agents_used: list[str]
    execution_path: list[str]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def response_text(self) -> str:
        """Get the final response text."""
        if self.final_response:
            return self.final_response.response
        return ""


class PlanningState(State):
    """State for task planning."""

    def __init__(self, agent_factory: AgentFactory):
        super().__init__("planning", StateType.INTERMEDIATE, timeout=60.0)
        self.agent_factory = agent_factory

    async def execute(self, context: StateContext) -> str:
        """Execute planning phase."""
        query = context.get("query", "")
        planner = self.agent_factory.get_or_create_agent("planner")

        async with planner:
            response = await planner.process(query, context.data)
            context.set("plan_response", response)
            context.set("plan_confidence", response.confidence)

            # Determine next state based on confidence and complexity
            if response.confidence >= 0.8 and "analysis" in query.lower():
                return "completion"  # Skip to completion for simple analysis
            else:
                return "coding"


class CodingState(State):
    """State for code implementation."""

    def __init__(self, agent_factory: AgentFactory):
        super().__init__("coding", StateType.INTERMEDIATE, timeout=120.0)
        self.agent_factory = agent_factory

    async def execute(self, context: StateContext) -> str:
        """Execute coding phase."""
        query = context.get("query", "")
        plan_response = context.get("plan_response")

        # Build context with plan
        coding_context = context.data.copy()
        if plan_response:
            coding_context["plan"] = plan_response.response

        coder = self.agent_factory.get_or_create_agent("coder")

        async with coder:
            response = await coder.process(query, coding_context)
            context.set("code_response", response)
            context.set("code_confidence", response.confidence)

            # Determine next state based on confidence
            if response.confidence >= 0.85 and "```" in response.response:
                return "completion"  # Skip review for high-confidence code
            else:
                return "review"


class ReviewState(State):
    """State for code review."""

    def __init__(self, agent_factory: AgentFactory):
        super().__init__("review", StateType.INTERMEDIATE, timeout=90.0)
        self.agent_factory = agent_factory

    async def execute(self, context: StateContext) -> str:
        """Execute review phase."""
        code_response = context.get("code_response")
        plan_response = context.get("plan_response")

        if not code_response:
            context.set("error", "No code response to review")
            return "error"

        # Build context with plan and code
        review_context = context.data.copy()
        if plan_response:
            review_context["plan"] = plan_response.response
        review_context["code"] = code_response.response

        critic = self.agent_factory.get_or_create_agent("critic")

        async with critic:
            response = await critic.process(
                f"Review the following implementation:\n\n{code_response.response}", review_context
            )
            context.set("review_response", response)
            context.set("review_confidence", response.confidence)

            # Check if review passed
            if "approve" in response.response.lower() and "reject" not in response.response.lower():
                return "completion"
            else:
                return "revision"


class RevisionState(State):
    """State for code revision based on review feedback."""

    def __init__(self, agent_factory: AgentFactory):
        super().__init__("revision", StateType.INTERMEDIATE, timeout=120.0)
        self.agent_factory = agent_factory

    async def execute(self, context: StateContext) -> str:
        """Execute revision phase."""
        code_response = context.get("code_response")
        review_response = context.get("review_response")

        if not code_response or not review_response:
            context.set("error", "Missing code or review for revision")
            return "error"

        # Build revision context
        revision_context = context.data.copy()
        revision_context["original_code"] = code_response.response
        revision_context["review_feedback"] = review_response.response

        coder = self.agent_factory.get_or_create_agent("coder")

        async with coder:
            revision_query = f"""
            Please revise the following code based on the review feedback:
            
            Original Code:
            {code_response.response}
            
            Review Feedback:
            {review_response.response}
            
            Provide the improved version addressing the feedback.
            """

            response = await coder.process(revision_query, revision_context)
            context.set("revised_code_response", response)

            return "completion"


class CompletionState(State):
    """Final state for result compilation."""

    def __init__(self):
        super().__init__("completion", StateType.FINAL)

    async def execute(self, context: StateContext) -> str:
        """Compile final result."""
        # Determine the best response to return
        revised_response = context.get("revised_code_response")
        code_response = context.get("code_response")
        plan_response = context.get("plan_response")

        final_response = revised_response or code_response or plan_response
        context.set("final_response", final_response)

        return "completion"  # Stay in final state


class ErrorState(State):
    """Error state for handling failures."""

    def __init__(self):
        super().__init__("error", StateType.ERROR)

    async def execute(self, context: StateContext) -> str:
        """Handle error state."""
        error_msg = context.get("error", "Unknown error occurred")
        logger.error(f"Orchestrator entered error state: {error_msg}")
        return "error"  # Stay in error state


class Orchestrator:
    """
    Modern orchestrator with pipeline-based architecture.

    Improvements:
    - State machine for complex execution flows
    - Pipeline-based execution with dependency management
    - Circuit breaker pattern for reliability
    - Streaming response support
    - Better error handling and recovery
    """

    def __init__(self, container: Container):
        self.container = container
        self.agent_factory = AgentFactory(container.settings, container.get("http_client"))
        self._setup_state_machine()
        self._setup_pipelines()

    def _setup_state_machine(self):
        """Setup the orchestrator state machine."""
        self.state_machine = StateMachine("orchestrator", "planning")

        # Add states
        self.state_machine.add_state(PlanningState(self.agent_factory))
        self.state_machine.add_state(CodingState(self.agent_factory))
        self.state_machine.add_state(ReviewState(self.agent_factory))
        self.state_machine.add_state(RevisionState(self.agent_factory))
        self.state_machine.add_state(CompletionState())
        self.state_machine.add_state(ErrorState())

    def _setup_pipelines(self):
        """Setup predefined pipelines for common workflows."""
        self.pipelines = {}

        # Simple analysis pipeline
        analysis_pipeline = Pipeline(
            "analysis", [AgentStage("planner", self.agent_factory.get_or_create_agent("planner"))]
        )
        self.pipelines["analysis"] = analysis_pipeline

        # Full development pipeline
        dev_pipeline = Pipeline(
            "development",
            [
                AgentStage("planner", self.agent_factory.get_or_create_agent("planner")),
                AgentStage(
                    "coder",
                    self.agent_factory.get_or_create_agent("coder"),
                    dependencies=["planner"],
                    context_builder=lambda results: {
                        "plan": (
                            results["planner"].response.response if results.get("planner") else ""
                        )
                    },
                ),
                ConditionalStage(
                    "review_stage",
                    AgentStage(
                        "critic",
                        self.agent_factory.get_or_create_agent("critic"),
                        context_builder=lambda results: {
                            "plan": (
                                results["planner"].response.response
                                if results.get("planner")
                                else ""
                            ),
                            "code": (
                                results["coder"].response.response if results.get("coder") else ""
                            ),
                        },
                    ),
                    condition=lambda results: (
                        results.get("coder")
                        and results["coder"].response
                        and results["coder"].response.confidence < 0.85
                    ),
                    dependencies=["coder"],
                ),
            ],
        )
        self.pipelines["development"] = dev_pipeline

    async def process_query(
        self, query: str, context: dict[str, Any] | None = None, workflow: str = "auto"
    ) -> OrchestratorResult:
        """Process a query using the orchestrator."""
        # Generate trace ID if not already set
        trace_id = get_trace_id() or f"{int(time.time() * 1000):x}{hash(query) & 0xFFFF:04x}"
        set_trace_id(trace_id)
        
        start_time = time.time()
        ctx = context or {}
        ctx["trace_id"] = trace_id

        with probe("orchestrator.process_query", trace_id):
            logger.info(f"Processing query with workflow '{workflow}': {query[:100]}...", 
                       workflow=workflow, query_length=len(query))

        try:
            # Determine workflow
            if workflow == "auto":
                workflow = self._determine_workflow(query, ctx)

            # Execute based on workflow type
            if workflow == "state_machine":
                result = await self._execute_state_machine(query, ctx)
            else:
                result = await self._execute_pipeline(query, ctx, workflow)

            execution_time = time.time() - start_time

            # Record metrics
            metrics = get_metrics_collector()
            metrics.record_orchestrator_request(execution_time, result.success, result.agents_used)

            logger.info(
                f"Query processed in {execution_time:.2f}s "
                f"(success: {result.success}, agents: {len(result.agents_used)})"
            )

            # Create and save trace snapshot for audit
            try:
                snapshot = create_trace_snapshot(
                    trace_id=trace_id,
                    query=query,
                    context_keys=list(ctx.keys()),
                    agents_used=result.agents_used,
                    execution_path=result.execution_path,
                    confidence=result.confidence,
                    success=result.success,
                    additional_data={
                        "workflow": workflow,
                        "execution_time_s": execution_time,
                        "metadata": result.metadata
                    }
                )
                save_trace_snapshot(snapshot)
            except Exception as e:
                logger.warning(f"Failed to save trace snapshot: {e}")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Orchestrator failed: {e}")

            # Create and save trace snapshot for error case
            try:
                snapshot = create_trace_snapshot(
                    trace_id=trace_id,
                    query=query,
                    context_keys=list(ctx.keys()),
                    agents_used=[],
                    execution_path=["error"],
                    confidence=0.0,
                    success=False,
                    additional_data={
                        "workflow": workflow,
                        "execution_time_s": execution_time,
                        "error": str(e)
                    }
                )
                save_trace_snapshot(snapshot)
            except Exception as snapshot_error:
                logger.warning(f"Failed to save error trace snapshot: {snapshot_error}")

            return OrchestratorResult(
                success=False,
                final_response=None,
                pipeline_result=None,
                execution_time=execution_time,
                agents_used=[],
                execution_path=["error"],
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def _determine_workflow(self, query: str, context: dict[str, Any]) -> str:
        """Determine the best workflow for the query."""
        query_lower = query.lower()

        # Simple analysis tasks
        if any(word in query_lower for word in ["analyze", "explain", "describe", "what is"]):
            return "analysis"

        # Complex development tasks
        if any(word in query_lower for word in ["implement", "create", "build", "develop", "code"]):
            return "development"

        # Default to state machine for complex flows
        return "state_machine"

    async def _execute_state_machine(
        self, query: str, context: dict[str, Any]
    ) -> OrchestratorResult:
        """Execute using state machine workflow."""
        # Initialize state machine context
        initial_context = {"query": query, **context}
        await self.state_machine.start(initial_context)

        # Run to completion
        final_context = await self.state_machine.run_to_completion()

        # Extract results
        final_response = final_context.get("final_response")
        agents_used = []
        execution_path = self.state_machine.get_state_history()

        # Determine agents used from responses
        for key in ["plan_response", "code_response", "review_response", "revised_code_response"]:
            if final_context.get(key):
                agent_type = key.split("_")[0]
                if agent_type not in agents_used:
                    agents_used.append(agent_type)

        success = final_response is not None and not final_context.get("error")
        confidence = final_response.confidence if final_response else 0.0

        return OrchestratorResult(
            success=success,
            final_response=final_response,
            pipeline_result=None,
            execution_time=0.0,  # Will be set by caller
            agents_used=agents_used,
            execution_path=execution_path,
            confidence=confidence,
            metadata={
                "workflow": "state_machine",
                "state_history": execution_path,
                "final_state": self.state_machine.current_state,
            },
        )

    async def _execute_pipeline(
        self, query: str, context: dict[str, Any], pipeline_name: str
    ) -> OrchestratorResult:
        """Execute using pipeline workflow."""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        pipeline = self.pipelines[pipeline_name]
        pipeline_result = await pipeline.execute(query, context)

        # Extract orchestrator result
        agents_used = [stage.stage_name for stage in pipeline_result.completed_stages]
        execution_path = [stage.stage_name for stage in pipeline_result.stages]

        return OrchestratorResult(
            success=pipeline_result.success,
            final_response=pipeline_result.final_response,
            pipeline_result=pipeline_result,
            execution_time=0.0,  # Will be set by caller
            agents_used=agents_used,
            execution_path=execution_path,
            confidence=(
                pipeline_result.final_response.confidence if pipeline_result.final_response else 0.0
            ),
            metadata={
                "workflow": "pipeline",
                "pipeline_name": pipeline_name,
                "total_stages": len(pipeline_result.stages),
                "completed_stages": len(pipeline_result.completed_stages),
                "failed_stages": len(pipeline_result.failed_stages),
            },
        )

    async def stream_process_query(
        self, query: str, context: dict[str, Any] | None = None, workflow: str = "auto"
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream process a query with real-time updates."""
        ctx = context or {}

        # Determine workflow
        if workflow == "auto":
            workflow = self._determine_workflow(query, ctx)

        logger.info(f"Streaming query with workflow '{workflow}': {query[:100]}...")

        if workflow in self.pipelines:
            # Stream pipeline execution
            pipeline = self.pipelines[workflow]
            async for stage_result in pipeline.stream_execute(query, ctx):
                yield {
                    "type": "stage_result",
                    "stage_name": stage_result.stage_name,
                    "status": stage_result.status.value,
                    "response": stage_result.response.response if stage_result.response else None,
                    "confidence": (
                        stage_result.response.confidence if stage_result.response else 0.0
                    ),
                    "duration": stage_result.duration,
                    "error": stage_result.error,
                }
        else:
            # For state machine, we'll need to implement streaming differently
            # For now, just execute normally and yield final result
            result = await self.process_query(query, ctx, workflow)
            yield {
                "type": "final_result",
                "success": result.success,
                "response": result.response_text,
                "confidence": result.confidence,
                "agents_used": result.agents_used,
                "execution_path": result.execution_path,
            }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on orchestrator and agents."""
        try:
            # Check agent factory
            agent_health = await self.agent_factory.health_check_all()

            # Check container dependencies
            container_health = {}
            for service_name in ["http_client", "rag_retriever", "executor"]:
                try:
                    service = self.container.get(service_name)
                    if hasattr(service, "health_check"):
                        container_health[service_name] = await service.health_check()
                    else:
                        container_health[service_name] = service is not None
                except Exception:
                    container_health[service_name] = False

            overall_health = all(agent_health.values()) and all(container_health.values())

            return {
                "overall": overall_health,
                "agents": agent_health,
                "services": container_health,
                "pipelines": list(self.pipelines.keys()),
                "state_machine": {
                    "name": self.state_machine.name,
                    "current_state": self.state_machine.current_state,
                    "is_running": self.state_machine.is_running,
                },
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"overall": False, "error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup orchestrator resources."""
        try:
            await self.agent_factory.cleanup()
            if self.state_machine.is_running:
                await self.state_machine.stop()
            logger.info("Orchestrator cleanup completed")
        except Exception as e:
            logger.error(f"Error during orchestrator cleanup: {e}")

    def get_available_workflows(self) -> list[str]:
        """Get list of available workflows."""
        return ["state_machine"] + list(self.pipelines.keys())
