"""
Modern agent system with protocol-based design and dependency injection.

Improvements over original:
- Protocol-based interfaces for better type safety
- Async context managers for resource management
- Confidence recalibration with multiple factors
- Streaming response support
"""

from .base import Agent, AgentProtocol, AgentResponse
from .coder import CoderAgent
from .critic import CriticAgent
from .factory import AgentFactory
from .planner import PlannerAgent

__all__ = [
    "Agent",
    "AgentResponse",
    "AgentProtocol",
    "PlannerAgent",
    "CoderAgent",
    "CriticAgent",
    "AgentFactory",
]
