"""
Dependency injection container for managing application dependencies.

Improvements over original:
- Proper dependency injection with type safety
- Lazy initialization of expensive resources
- Lifecycle management for async resources
- Configuration-driven service instantiation
"""

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, TypeVar

from .settings import Settings, get_settings

T = TypeVar("T")


class Container:
    """Dependency injection container with async lifecycle management."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._services: dict[str, Any] = {}
        self._factories: dict[str, Any] = {}
        self._singletons: dict[str, Any] = {}
        self._async_resources: dict[str, Any] = {}

    def register_factory(self, name: str, factory: Any) -> None:
        """Register a factory function for a service."""
        self._factories[name] = factory

    def register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton instance."""
        self._singletons[name] = instance

    def get(self, name: str, default: Any = None) -> Any:
        """Get a service by name."""
        # Check singletons first
        if name in self._singletons:
            return self._singletons[name]

        # Check if already instantiated
        if name in self._services:
            return self._services[name]

        # Try to create from factory
        if name in self._factories:
            instance = self._factories[name](self)
            self._services[name] = instance
            return instance

        return default

    async def get_async(self, name: str, default: Any = None) -> Any:
        """Get an async service by name."""
        # Check if already in async resources
        if name in self._async_resources:
            return self._async_resources[name]

        # Try regular get first
        service = self.get(name, default)

        # If it's an async context manager, enter it
        if hasattr(service, "__aenter__"):
            async_service = await service.__aenter__()
            self._async_resources[name] = async_service
            return async_service

        return service

    async def cleanup(self) -> None:
        """Cleanup all async resources."""
        for name, resource in self._async_resources.items():
            if hasattr(resource, "__aexit__"):
                try:
                    await resource.__aexit__(None, None, None)
                except Exception as e:
                    # Log error but continue cleanup
                    print(f"Error cleaning up {name}: {e}")

        self._async_resources.clear()

    @asynccontextmanager
    async def lifespan(self):
        """Async context manager for container lifecycle."""
        try:
            yield self
        finally:
            await self.cleanup()


def setup_container(settings: Settings | None = None) -> Container:
    """Setup container with default service factories."""
    container = Container(settings)

    # Register core service factories
    def _http_client_factory(c: Container):
        import httpx

        return httpx.AsyncClient(
            timeout=httpx.Timeout(c.settings.models.planner.timeout),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
        )

    def _rag_retriever_factory(c: Container):
        from ..rag.retriever import RAGRetriever

        rag = c.settings.rag
        return RAGRetriever(
            embedding_model=rag.embedding_model,
            vector_store_path=str(rag.persist_directory) if rag.persist_directory else None,
            max_results=rag.max_results,
            min_relevance_score=rag.similarity_threshold,
            embedding_cache_path=str(rag.embedding_cache_path) if rag.embedding_cache_path else None,
            cache_max_entries=rag.cache_max_entries,
        )

    def _executor_factory(c: Container):
        from ..execution.executor import CodeExecutor

        return CodeExecutor(c.settings.execution)

    def _orchestrator_factory(c: Container):
        from ..core.orchestrator import Orchestrator

        return Orchestrator(container=c)

    # Register factories
    container.register_factory("http_client", _http_client_factory)
    container.register_factory("rag_retriever", _rag_retriever_factory)
    container.register_factory("executor", _executor_factory)
    container.register_factory("orchestrator", _orchestrator_factory)

    return container


@lru_cache
def get_container() -> Container:
    """Get cached container instance."""
    return setup_container()
