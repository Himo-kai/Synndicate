"""
Pipeline-based orchestration system for flexible agent workflows.

Improvements over original:
- Configurable pipeline stages with dependencies
- Parallel execution support
- Error handling and rollback capabilities
- Conditional execution based on results
- Pipeline composition and reuse
"""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..agents.base import Agent, AgentResponse
from ..observability.logging import get_logger
from ..observability.metrics import get_metrics_collector
from ..observability.tracing import trace_span

logger = get_logger(__name__)


class StageStatus(Enum):
    """Pipeline stage execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""

    stage_name: str
    status: StageStatus
    response: AgentResponse | None = None
    error: str | None = None
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""

    pipeline_name: str
    success: bool
    total_duration: float
    stages: list[StageResult]
    final_response: AgentResponse | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def completed_stages(self) -> list[StageResult]:
        """Get successfully completed stages."""
        return [s for s in self.stages if s.status == StageStatus.COMPLETED]

    @property
    def failed_stages(self) -> list[StageResult]:
        """Get failed stages."""
        return [s for s in self.stages if s.status == StageStatus.FAILED]


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(
        self,
        name: str,
        dependencies: list[str] | None = None,
        condition: Callable[[dict[str, StageResult]], bool] | None = None,
        rollback_on_failure: bool = False,
    ):
        self.name = name
        self.dependencies = dependencies or []
        self.condition = condition
        self.rollback_on_failure = rollback_on_failure

    @abstractmethod
    async def execute(
        self, query: str, context: dict[str, Any], previous_results: dict[str, StageResult]
    ) -> StageResult:
        """Execute the pipeline stage."""
        ...

    async def rollback(self, context: dict[str, Any], stage_result: StageResult) -> None:
        """Rollback the stage if needed (optional)."""
        pass

    def should_execute(self, previous_results: dict[str, StageResult]) -> bool:
        """Check if this stage should execute based on conditions."""
        if self.condition:
            return self.condition(previous_results)
        return True


class AgentStage(PipelineStage):
    """Pipeline stage that executes an agent."""

    def __init__(
        self,
        name: str,
        agent: Agent,
        dependencies: list[str] | None = None,
        condition: Callable[[dict[str, StageResult]], bool] | None = None,
        rollback_on_failure: bool = False,
        context_builder: Callable[[dict[str, StageResult]], dict[str, Any]] | None = None,
    ):
        super().__init__(name, dependencies, condition, rollback_on_failure)
        self.agent = agent
        self.context_builder = context_builder

    @trace_span("pipeline.stage.execute")
    async def execute(
        self, query: str, context: dict[str, Any], previous_results: dict[str, StageResult]
    ) -> StageResult:
        """Execute the agent stage."""
        start_time = time.time()

        try:
            # Build stage-specific context
            stage_context = context.copy()
            if self.context_builder:
                additional_context = self.context_builder(previous_results)
                stage_context.update(additional_context)

            # Execute agent
            response = await self.agent.process(query, stage_context)

            duration = time.time() - start_time

            return StageResult(
                stage_name=self.name,
                status=StageStatus.COMPLETED,
                response=response,
                duration=duration,
                metadata={
                    "agent_type": self.agent.__class__.__name__,
                    "confidence": response.confidence,
                    "execution_time": response.execution_time or duration,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Stage '{self.name}' failed: {e}")

            return StageResult(
                stage_name=self.name, status=StageStatus.FAILED, error=str(e), duration=duration
            )


class ConditionalStage(PipelineStage):
    """Pipeline stage that executes based on conditions."""

    def __init__(
        self,
        name: str,
        true_stage: PipelineStage,
        false_stage: PipelineStage | None = None,
        condition: Callable[[dict[str, StageResult]], bool] = None,
        dependencies: list[str] | None = None,
    ):
        super().__init__(name, dependencies, condition)
        self.true_stage = true_stage
        self.false_stage = false_stage

    async def execute(
        self, query: str, context: dict[str, Any], previous_results: dict[str, StageResult]
    ) -> StageResult:
        """Execute conditional logic."""
        if self.should_execute(previous_results):
            return await self.true_stage.execute(query, context, previous_results)
        elif self.false_stage:
            return await self.false_stage.execute(query, context, previous_results)
        else:
            return StageResult(
                stage_name=self.name,
                status=StageStatus.SKIPPED,
                metadata={"reason": "Condition not met"},
            )


class Pipeline:
    """
    Configurable pipeline for orchestrating agent workflows.

    Improvements:
    - Dependency-based stage ordering
    - Parallel execution where possible
    - Conditional stage execution
    - Error handling and rollback
    - Progress streaming
    """

    def __init__(self, name: str, stages: list[PipelineStage]):
        self.name = name
        self.stages = {stage.name: stage for stage in stages}
        self._execution_order = self._calculate_execution_order()

    def _calculate_execution_order(self) -> list[list[str]]:
        """Calculate stage execution order based on dependencies."""
        # Topological sort to handle dependencies
        in_degree = dict.fromkeys(self.stages, 0)
        graph = {name: [] for name in self.stages}

        # Build dependency graph
        for stage_name, stage in self.stages.items():
            for dep in stage.dependencies:
                if dep in self.stages:
                    graph[dep].append(stage_name)
                    in_degree[stage_name] += 1

        # Group stages that can run in parallel
        execution_levels = []
        remaining = set(self.stages.keys())

        while remaining:
            # Find stages with no dependencies
            ready = [name for name in remaining if in_degree[name] == 0]

            if not ready:
                # Circular dependency detected
                raise ValueError(f"Circular dependency detected in pipeline '{self.name}'")

            execution_levels.append(ready)

            # Remove ready stages and update in_degree
            for stage_name in ready:
                remaining.remove(stage_name)
                for dependent in graph[stage_name]:
                    in_degree[dependent] -= 1

        return execution_levels

    @trace_span("pipeline.execute")
    async def execute(self, query: str, context: dict[str, Any] | None = None) -> PipelineResult:
        """Execute the complete pipeline."""
        start_time = time.time()
        ctx = context or {}
        results = {}
        stage_results = []

        logger.info(f"Starting pipeline '{self.name}' with {len(self.stages)} stages")

        try:
            # Execute stages level by level
            for level, stage_names in enumerate(self._execution_order):
                logger.debug(f"Executing pipeline level {level}: {stage_names}")

                # Execute stages in parallel within each level
                tasks = []
                for stage_name in stage_names:
                    stage = self.stages[stage_name]

                    # Check if stage should execute
                    if stage.should_execute(results):
                        task = self._execute_stage(stage, query, ctx, results)
                        tasks.append((stage_name, task))
                    else:
                        # Skip stage
                        skip_result = StageResult(
                            stage_name=stage_name,
                            status=StageStatus.SKIPPED,
                            metadata={"reason": "Condition not met"},
                        )
                        results[stage_name] = skip_result
                        stage_results.append(skip_result)

                # Wait for all tasks in this level to complete
                for stage_name, task in tasks:
                    try:
                        result = await task
                        results[stage_name] = result
                        stage_results.append(result)

                        # Record metrics
                        metrics = get_metrics_collector()
                        metrics.record_agent_call(
                            stage_name,
                            result.duration,
                            result.status == StageStatus.COMPLETED,
                            result.response.confidence if result.response else 0.0,
                        )

                    except Exception as e:
                        logger.error(f"Stage '{stage_name}' failed: {e}")
                        error_result = StageResult(
                            stage_name=stage_name, status=StageStatus.FAILED, error=str(e)
                        )
                        results[stage_name] = error_result
                        stage_results.append(error_result)

                # Check if any critical stages failed
                level_failures = [r for r in stage_results if r.status == StageStatus.FAILED]
                if level_failures:
                    # Check if we should continue or abort
                    critical_failures = [
                        r for r in level_failures if self.stages[r.stage_name].rollback_on_failure
                    ]
                    if critical_failures:
                        logger.error(f"Critical stage failures in level {level}, aborting pipeline")
                        break

            # Determine overall success
            failed_results = [r for r in stage_results if r.status == StageStatus.FAILED]
            success = len(failed_results) == 0

            # Get final response (from last successful stage)
            final_response = None
            for result in reversed(stage_results):
                if result.response and result.status == StageStatus.COMPLETED:
                    final_response = result.response
                    break

            total_duration = time.time() - start_time

            pipeline_result = PipelineResult(
                pipeline_name=self.name,
                success=success,
                total_duration=total_duration,
                stages=stage_results,
                final_response=final_response,
                metadata={
                    "total_stages": len(self.stages),
                    "completed_stages": len(
                        [r for r in stage_results if r.status == StageStatus.COMPLETED]
                    ),
                    "failed_stages": len(failed_results),
                    "skipped_stages": len(
                        [r for r in stage_results if r.status == StageStatus.SKIPPED]
                    ),
                },
            )

            logger.info(
                f"Pipeline '{self.name}' completed in {total_duration:.2f}s "
                f"(success: {success}, stages: {len(stage_results)})"
            )

            return pipeline_result

        except Exception as e:
            total_duration = time.time() - start_time
            logger.error(f"Pipeline '{self.name}' failed: {e}")

            return PipelineResult(
                pipeline_name=self.name,
                success=False,
                total_duration=total_duration,
                stages=stage_results,
                metadata={"error": str(e)},
            )

    async def _execute_stage(
        self,
        stage: PipelineStage,
        query: str,
        context: dict[str, Any],
        previous_results: dict[str, StageResult],
    ) -> StageResult:
        """Execute a single pipeline stage."""
        logger.debug(f"Executing stage '{stage.name}'")

        try:
            result = await stage.execute(query, context, previous_results)

            if result.status == StageStatus.COMPLETED:
                logger.debug(f"Stage '{stage.name}' completed successfully")
            else:
                logger.warning(f"Stage '{stage.name}' failed: {result.error}")

            return result

        except Exception as e:
            logger.error(f"Stage '{stage.name}' execution failed: {e}")
            return StageResult(stage_name=stage.name, status=StageStatus.FAILED, error=str(e))

    async def stream_execute(
        self, query: str, context: dict[str, Any] | None = None
    ) -> AsyncIterator[StageResult]:
        """Execute pipeline with streaming results."""
        ctx = context or {}
        results = {}

        logger.info(f"Starting streaming pipeline '{self.name}'")

        try:
            for level, stage_names in enumerate(self._execution_order):
                # Execute stages in parallel within each level
                tasks = []
                for stage_name in stage_names:
                    stage = self.stages[stage_name]

                    if stage.should_execute(results):
                        task = self._execute_stage(stage, query, ctx, results)
                        tasks.append((stage_name, task))
                    else:
                        skip_result = StageResult(
                            stage_name=stage_name,
                            status=StageStatus.SKIPPED,
                            metadata={"reason": "Condition not met"},
                        )
                        results[stage_name] = skip_result
                        yield skip_result

                # Yield results as they complete
                for stage_name, task in tasks:
                    try:
                        result = await task
                        results[stage_name] = result
                        yield result
                    except Exception as e:
                        error_result = StageResult(
                            stage_name=stage_name, status=StageStatus.FAILED, error=str(e)
                        )
                        results[stage_name] = error_result
                        yield error_result

        except Exception as e:
            logger.error(f"Streaming pipeline '{self.name}' failed: {e}")
            yield StageResult(stage_name="pipeline_error", status=StageStatus.FAILED, error=str(e))

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the pipeline."""
        self.stages[stage.name] = stage
        self._execution_order = self._calculate_execution_order()

    def remove_stage(self, stage_name: str) -> None:
        """Remove a stage from the pipeline."""
        if stage_name in self.stages:
            del self.stages[stage_name]
            self._execution_order = self._calculate_execution_order()

    def get_stage_names(self) -> list[str]:
        """Get all stage names in execution order."""
        return [name for level in self._execution_order for name in level]
