"""
Core orchestration system with pipeline-based architecture.

Improvements over original:
- Pipeline-based orchestration with configurable stages
- State machine for execution flow
- Circuit breaker pattern for reliability
- Streaming response support
- Better error handling and recovery
"""

from .orchestrator import Orchestrator
from .pipeline import Pipeline, PipelineResult, PipelineStage
from .state_machine import State, StateMachine, Transition

__all__ = [
    "Orchestrator",
    "Pipeline",
    "PipelineStage",
    "PipelineResult",
    "StateMachine",
    "State",
    "Transition",
]
