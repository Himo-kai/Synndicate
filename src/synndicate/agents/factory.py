"""
Agent factory for dependency injection and configuration-driven instantiation.

Improvements over original:
- Factory pattern for clean agent instantiation
- Configuration-driven agent creation
- Proper dependency injection
- Agent lifecycle management
"""


import httpx

from ..config.settings import ModelEndpoint, Settings
from ..observability.logging import get_logger
from .base import Agent
from .coder import CoderAgent
from .critic import CriticAgent
from .planner import PlannerAgent

logger = get_logger(__name__)


class AgentFactory:
    """Factory for creating and managing agents with dependency injection."""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.http_client = http_client
        self._agent_classes: dict[str, type[Agent]] = {
            "planner": PlannerAgent,
            "coder": CoderAgent,
            "critic": CriticAgent,
        }
        self._agent_cache: dict[str, Agent] = {}

    def register_agent_type(self, name: str, agent_class: type[Agent]) -> None:
        """Register a new agent type."""
        self._agent_classes[name] = agent_class
        logger.info(f"Registered agent type: {name}")

    def create_agent(self, agent_type: str, endpoint: ModelEndpoint | None = None) -> Agent:
        """Create an agent instance of the specified type."""
        if agent_type not in self._agent_classes:
            raise ValueError(
                f"Unknown agent type: {agent_type}. Available: {list(self._agent_classes.keys())}"
            )

        # Use provided endpoint or get from settings
        if endpoint is None:
            endpoint = getattr(self.settings.models, agent_type, None)
            if endpoint is None:
                raise ValueError(f"No endpoint configuration found for agent type: {agent_type}")

        agent_class = self._agent_classes[agent_type]
        agent = agent_class(
            endpoint=endpoint, config=self.settings.agents, http_client=self.http_client
        )

        logger.info(f"Created {agent_type} agent with model {endpoint.name}")
        return agent

    def get_or_create_agent(
        self, agent_type: str, endpoint: ModelEndpoint | None = None
    ) -> Agent:
        """Get cached agent or create new one."""
        cache_key = f"{agent_type}_{endpoint.name if endpoint else 'default'}"

        if cache_key not in self._agent_cache:
            self._agent_cache[cache_key] = self.create_agent(agent_type, endpoint)

        return self._agent_cache[cache_key]

    async def health_check_all(self) -> dict[str, bool]:
        """Health check all cached agents."""
        results = {}
        for cache_key, agent in self._agent_cache.items():
            try:
                results[cache_key] = await agent.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {cache_key}: {e}")
                results[cache_key] = False

        return results

    async def cleanup(self) -> None:
        """Cleanup all cached agents."""
        for agent in self._agent_cache.values():
            try:
                await agent.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error cleaning up agent: {e}")

        self._agent_cache.clear()

        if self.http_client:
            await self.http_client.aclose()

    def get_available_agent_types(self) -> list[str]:
        """Get list of available agent types."""
        return list(self._agent_classes.keys())
