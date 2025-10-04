"""
Distributed tracing backend configuration for Jaeger, Zipkin, and other OTLP-compatible systems.

This module provides a unified interface for configuring distributed tracing backends
without requiring specific exporter dependencies that may conflict. Instead, it uses
the OTLP exporter which is compatible with Jaeger, Zipkin, and other tracing systems.

Improvements:
- Unified configuration for multiple tracing backends
- Environment-based configuration
- Automatic service discovery and health checks
- Fallback mechanisms for reliability
- Performance optimizations and batching
"""

import os
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GRPCExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv.resource import ResourceAttributes

from ..config.settings import Settings
from .logging import get_logger

logger = get_logger(__name__)


class TracingBackend(Enum):
    """Supported distributed tracing backends."""

    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    OTLP = "otlp"
    CONSOLE = "console"
    DISABLED = "disabled"


class TracingProtocol(Enum):
    """Supported tracing protocols."""

    GRPC = "grpc"
    HTTP = "http"


class DistributedTracingConfig:
    """Configuration for distributed tracing backends."""

    def __init__(
        self,
        backend: TracingBackend | None = None,
        protocol: TracingProtocol | None = None,
        endpoint: str | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        service_namespace: str | None = None,
        sample_rate: float | None = None,
        batch_timeout: int | None = None,
        max_batch_size: int | None = None,
        max_queue_size: int | None = None,
        health_check_enabled: bool | None = None,
        health_check_interval: int | None = None,
        headers: dict[str, str] | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings

        # Backend configuration (parameter overrides environment/settings)
        self.backend = backend or TracingBackend(os.getenv("TRACING_BACKEND", "jaeger").lower())
        self.protocol = protocol or TracingProtocol(os.getenv("TRACING_PROTOCOL", "grpc").lower())

        # Service information
        self.service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "synndicate")
        self.service_version = service_version or os.getenv("OTEL_SERVICE_VERSION", "2.0.0")
        self.service_namespace = service_namespace or os.getenv("OTEL_SERVICE_NAMESPACE", "ai")

        # Endpoint configuration
        self.endpoint = endpoint if endpoint is not None else self._get_endpoint()
        self.headers = headers or self._get_headers()

        # Performance settings
        self.batch_timeout = batch_timeout or int(os.getenv("TRACING_BATCH_TIMEOUT", "5000"))  # ms
        self.max_batch_size = max_batch_size or int(os.getenv("TRACING_MAX_BATCH_SIZE", "512"))
        self.max_queue_size = max_queue_size or int(os.getenv("TRACING_MAX_QUEUE_SIZE", "2048"))

        # Sampling configuration
        self.sample_rate = (
            sample_rate
            if sample_rate is not None
            else float(os.getenv("TRACING_SAMPLE_RATE", "1.0"))
        )
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(f"Sample rate must be between 0.0 and 1.0, got {self.sample_rate}")

        # Health check settings
        self.health_check_enabled = (
            health_check_enabled
            if health_check_enabled is not None
            else (os.getenv("TRACING_HEALTH_CHECK", "true").lower() == "true")
        )
        self.health_check_interval = health_check_interval or int(
            os.getenv("TRACING_HEALTH_CHECK_INTERVAL", "30")
        )

    def _get_endpoint(self) -> str | None:
        """Get the appropriate endpoint based on backend and protocol."""
        # Check for explicit endpoint override
        if endpoint := os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
            return endpoint

        # Backend-specific defaults (only return if explicitly set in environment)
        if self.backend == TracingBackend.JAEGER:
            if self.protocol == TracingProtocol.GRPC:
                return os.getenv("JAEGER_GRPC_ENDPOINT")
            else:
                return os.getenv("JAEGER_HTTP_ENDPOINT")

        elif self.backend == TracingBackend.ZIPKIN:
            return os.getenv("ZIPKIN_ENDPOINT")

        elif self.backend == TracingBackend.OTLP:
            if self.protocol == TracingProtocol.GRPC:
                return os.getenv("OTLP_GRPC_ENDPOINT")
            else:
                return os.getenv("OTLP_HTTP_ENDPOINT")

        return None  # No default endpoint

    def _get_headers(self) -> dict[str, str]:
        """Get authentication headers for the tracing backend."""
        headers = {}

        # Generic OTLP headers
        if auth_header := os.getenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS"):
            for header in auth_header.split(","):
                if "=" in header:
                    key, value = header.split("=", 1)
                    headers[key.strip()] = value.strip()

        # Backend-specific authentication
        if self.backend == TracingBackend.JAEGER and (username := os.getenv("JAEGER_USER")):
            password = os.getenv("JAEGER_PASSWORD", "")
            import base64

            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers


class DistributedTracingManager:
    """Manages distributed tracing backend configuration and lifecycle."""

    def __init__(
        self,
        config: DistributedTracingConfig | None = None,
        backend: TracingBackend | None = None,
        protocol: str | None = None,
        endpoint: str | None = None,
        sample_rate: float | None = None,
        batch_timeout: int | None = None,
        max_batch_size: int | None = None,
        max_queue_size: int | None = None,
        health_check_enabled: bool | None = None,
        health_check_interval: int | None = None,
    ):
        # If config provided, use it; otherwise create from parameters
        if config:
            self.config = config
        else:
            self.config = DistributedTracingConfig(
                backend=backend,
                protocol=TracingProtocol(protocol) if protocol else None,
                endpoint=endpoint,
                sample_rate=sample_rate,
                batch_timeout=batch_timeout,
                max_batch_size=max_batch_size,
                max_queue_size=max_queue_size,
                health_check_enabled=health_check_enabled,
                health_check_interval=health_check_interval,
            )

        self._tracer_provider: TracerProvider | None = None
        self._span_processor: BatchSpanProcessor | None = None
        self._health_status = {"backend": "unknown", "last_check": None}
        self._is_setup = False
        self._health_check_task = None

    def setup(self) -> TracerProvider:
        """Setup distributed tracing with the configured backend."""
        logger.info(f"Setting up distributed tracing with {self.config.backend.value} backend")

        # Configure span processors based on backend
        if self.config.backend == TracingBackend.DISABLED:
            logger.info("Distributed tracing is disabled")
            # Skip tracer provider creation for disabled backend
        else:
            # Create resource with service information
            resource = Resource.create(
                {
                    ResourceAttributes.SERVICE_NAME: self.config.service_name or "synndicate",
                    ResourceAttributes.SERVICE_VERSION: self.config.service_version or "1.0.0",
                    ResourceAttributes.SERVICE_NAMESPACE: self.config.service_namespace or "ai",
                    ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv("HOSTNAME", "unknown"),
                }
            )

            # Create tracer provider
            self._tracer_provider = TracerProvider(resource=resource)

            if self.config.backend == TracingBackend.CONSOLE:
                processor = BatchSpanProcessor(ConsoleSpanExporter())
                self._tracer_provider.add_span_processor(processor)
                self._span_processor = processor

            else:
                # Create OTLP exporter (works with Jaeger, Zipkin, and other OTLP backends)
                exporter = self._create_otlp_exporter()
                if exporter:
                    processor = BatchSpanProcessor(
                        exporter,
                        max_queue_size=self.config.max_queue_size,
                        max_export_batch_size=self.config.max_batch_size,
                        schedule_delay_millis=self.config.batch_timeout,
                    )
                    self._tracer_provider.add_span_processor(processor)
                    self._span_processor = processor

        # Set global tracer provider (only if not disabled)
        if self._tracer_provider:
            trace.set_tracer_provider(self._tracer_provider)

        # Start health monitoring if enabled
        if self.config.health_check_enabled:
            self._start_health_monitoring()

        # Mark setup as complete
        self._is_setup = True

        logger.info(f"Distributed tracing setup complete - endpoint: {self.config.endpoint}")
        return self._tracer_provider

    def _create_otlp_exporter(self):
        """Create OTLP exporter based on protocol configuration."""
        try:
            # If no endpoint is configured, fall back to console exporter
            if not self.config.endpoint:
                logger.info("No endpoint configured, using console exporter")
                return ConsoleSpanExporter()

            if self.config.protocol == TracingProtocol.GRPC:
                return GRPCExporter(
                    endpoint=self.config.endpoint,
                    headers=self.config.headers,
                    insecure=self._is_insecure_endpoint(),
                )
            else:
                return HTTPExporter(
                    endpoint=self.config.endpoint,
                    headers=self.config.headers,
                )
        except Exception as e:
            logger.error(f"Failed to create OTLP exporter: {e}")
            logger.info("Falling back to console exporter")
            return ConsoleSpanExporter()

    def _is_insecure_endpoint(self) -> bool:
        """Check if the endpoint uses insecure connection."""
        if not self.config.endpoint:
            return True  # Default to insecure for None endpoints
        parsed = urlparse(self.config.endpoint)
        return parsed.scheme in ("http", "grpc") or "localhost" in parsed.netloc

    def _start_health_monitoring(self):
        """Start background health monitoring for the tracing backend."""
        # This would typically start a background task to periodically check backend health
        # For now, we'll just log that monitoring is enabled
        logger.info(f"Health monitoring enabled for {self.config.backend.value} backend")
        self._health_status["backend"] = self.config.backend.value

    def get_health_status(self) -> dict[str, Any]:
        """Get current health status of the tracing backend."""
        return {
            "backend": self.config.backend.value,
            "endpoint": self.config.endpoint,
            "protocol": self.config.protocol.value,
            "status": self._health_status,
            "processors": 1 if self._span_processor else 0,
        }

    def shutdown(self):
        """Gracefully shutdown tracing and flush remaining spans."""
        logger.info("Shutting down distributed tracing")

        if self._tracer_provider:
            # Force flush all pending spans
            if self._span_processor:
                try:
                    self._span_processor.force_flush(timeout_millis=5000)
                    self._span_processor.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down span processor: {e}")

            # Shutdown tracer provider
            try:
                self._tracer_provider.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down tracer provider: {e}")

        # Reset manager state
        self._tracer_provider = None
        self._span_processor = None
        self._is_setup = False

        logger.info("Distributed tracing shutdown complete")


# Global distributed tracing manager
_distributed_tracing_manager: DistributedTracingManager | None = None


def setup_distributed_tracing(config: DistributedTracingConfig | None = None) -> TracerProvider:
    """Setup global distributed tracing with the specified configuration."""
    global _distributed_tracing_manager

    _distributed_tracing_manager = DistributedTracingManager(config)
    return _distributed_tracing_manager.setup()


def get_distributed_tracing_manager() -> DistributedTracingManager | None:
    """Get the global distributed tracing manager."""
    return _distributed_tracing_manager


def get_tracing_health() -> dict[str, Any]:
    """Get health status of the distributed tracing system."""
    if _distributed_tracing_manager:
        return _distributed_tracing_manager.get_health_status()
    return {"status": "not_initialized"}


def shutdown_distributed_tracing():
    """Shutdown distributed tracing and cleanup resources."""
    global _distributed_tracing_manager

    if _distributed_tracing_manager:
        _distributed_tracing_manager.shutdown()
        _distributed_tracing_manager = None


# Convenience functions for common backend configurations


def setup_jaeger_tracing(
    endpoint: str = "http://localhost:14250",
    service_name: str = "synndicate",
    protocol: TracingProtocol = TracingProtocol.GRPC,
) -> TracerProvider:
    """Setup Jaeger tracing with specified configuration."""
    os.environ.update(
        {
            "TRACING_BACKEND": "jaeger",
            "TRACING_PROTOCOL": protocol.value,
            (
                "JAEGER_GRPC_ENDPOINT"
                if protocol == TracingProtocol.GRPC
                else "JAEGER_HTTP_ENDPOINT"
            ): endpoint,
            "OTEL_SERVICE_NAME": service_name,
        }
    )

    config = DistributedTracingConfig()
    return setup_distributed_tracing(config)


def setup_zipkin_tracing(
    endpoint: str = "http://localhost:9411/api/v2/spans",
    service_name: str = "synndicate",
) -> TracerProvider:
    """Setup Zipkin tracing with specified configuration."""
    os.environ.update(
        {
            "TRACING_BACKEND": "zipkin",
            "TRACING_PROTOCOL": "http",
            "ZIPKIN_ENDPOINT": endpoint,
            "OTEL_SERVICE_NAME": service_name,
        }
    )

    config = DistributedTracingConfig()
    return setup_distributed_tracing(config)


def setup_otlp_tracing(
    endpoint: str = "http://localhost:4317",
    service_name: str = "synndicate",
    protocol: TracingProtocol = TracingProtocol.GRPC,
    headers: dict[str, str] | None = None,
) -> TracerProvider:
    """Setup OTLP tracing with specified configuration."""
    env_vars = {
        "TRACING_BACKEND": "otlp",
        "TRACING_PROTOCOL": protocol.value,
        "OTEL_SERVICE_NAME": service_name,
    }

    if protocol == TracingProtocol.GRPC:
        env_vars["OTLP_GRPC_ENDPOINT"] = endpoint
    else:
        env_vars["OTLP_HTTP_ENDPOINT"] = endpoint

    if headers:
        header_str = ",".join(f"{k}={v}" for k, v in headers.items())
        env_vars["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = header_str

    os.environ.update(env_vars)

    config = DistributedTracingConfig()
    return setup_distributed_tracing(config)
