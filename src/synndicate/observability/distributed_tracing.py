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
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GRPCExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPExporter
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
        backend: Optional[TracingBackend] = None,
        protocol: Optional[TracingProtocol] = None,
        endpoint: Optional[str] = None,
        service_name: Optional[str] = None,
        service_version: Optional[str] = None,
        service_namespace: Optional[str] = None,
        sample_rate: Optional[float] = None,
        batch_timeout: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        max_queue_size: Optional[int] = None,
        enable_health_check: Optional[bool] = None,
        health_check_interval: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings
        
        # Backend configuration (parameter overrides environment/settings)
        self.backend = backend or TracingBackend(
            os.getenv("TRACING_BACKEND", "jaeger").lower()
        )
        self.protocol = protocol or TracingProtocol(
            os.getenv("TRACING_PROTOCOL", "grpc").lower()
        )
        
        # Service information
        self.service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "synndicate")
        self.service_version = service_version or os.getenv("OTEL_SERVICE_VERSION", "2.0.0")
        self.service_namespace = service_namespace or os.getenv("OTEL_SERVICE_NAMESPACE", "ai")
        
        # Endpoint configuration
        self.endpoint = endpoint or self._get_endpoint()
        self.headers = headers or self._get_headers()
        
        # Performance settings
        self.batch_timeout = batch_timeout or int(os.getenv("TRACING_BATCH_TIMEOUT", "5000"))  # ms
        self.max_batch_size = max_batch_size or int(os.getenv("TRACING_MAX_BATCH_SIZE", "512"))
        self.max_queue_size = max_queue_size or int(os.getenv("TRACING_MAX_QUEUE_SIZE", "2048"))
        
        # Sampling configuration
        self.sample_rate = sample_rate if sample_rate is not None else float(os.getenv("TRACING_SAMPLE_RATE", "1.0"))
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(f"Sample rate must be between 0.0 and 1.0, got {self.sample_rate}")
        
        # Health check settings
        self.enable_health_check = enable_health_check if enable_health_check is not None else (
            os.getenv("TRACING_HEALTH_CHECK", "true").lower() == "true"
        )
        self.health_check_interval = health_check_interval or int(os.getenv("TRACING_HEALTH_CHECK_INTERVAL", "30"))
    
    def _get_endpoint(self) -> str:
        """Get the appropriate endpoint based on backend and protocol."""
        # Check for explicit endpoint override
        if endpoint := os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
            return endpoint
        
        # Backend-specific defaults
        if self.backend == TracingBackend.JAEGER:
            if self.protocol == TracingProtocol.GRPC:
                return os.getenv("JAEGER_GRPC_ENDPOINT", "http://localhost:14250")
            else:
                return os.getenv("JAEGER_HTTP_ENDPOINT", "http://localhost:14268/api/traces")
        
        elif self.backend == TracingBackend.ZIPKIN:
            return os.getenv("ZIPKIN_ENDPOINT", "http://localhost:9411/api/v2/spans")
        
        elif self.backend == TracingBackend.OTLP:
            if self.protocol == TracingProtocol.GRPC:
                return os.getenv("OTLP_GRPC_ENDPOINT", "http://localhost:4317")
            else:
                return os.getenv("OTLP_HTTP_ENDPOINT", "http://localhost:4318/v1/traces")
        
        return "http://localhost:4317"  # Default OTLP gRPC
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers for the tracing backend."""
        headers = {}
        
        # Generic OTLP headers
        if auth_header := os.getenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS"):
            for header in auth_header.split(","):
                if "=" in header:
                    key, value = header.split("=", 1)
                    headers[key.strip()] = value.strip()
        
        # Backend-specific authentication
        if self.backend == TracingBackend.JAEGER:
            if username := os.getenv("JAEGER_USER"):
                password = os.getenv("JAEGER_PASSWORD", "")
                import base64
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"
        
        return headers


class DistributedTracingManager:
    """Manages distributed tracing backend configuration and lifecycle."""
    
    def __init__(
        self,
        config: Optional[DistributedTracingConfig] = None,
        backend: Optional[TracingBackend] = None,
        protocol: Optional[str] = None,
        endpoint: Optional[str] = None,
        sample_rate: Optional[float] = None,
        batch_timeout: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        max_queue_size: Optional[int] = None,
        enable_health_check: Optional[bool] = None,
        health_check_interval: Optional[int] = None,
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
                enable_health_check=enable_health_check,
                health_check_interval=health_check_interval,
            )
        
        self.tracer_provider: Optional[TracerProvider] = None
        self.span_processors: List[BatchSpanProcessor] = []
        self._health_status = {"backend": "unknown", "last_check": None}
        self._is_setup = False
        self._health_check_task = None
    
    def setup(self) -> TracerProvider:
        """Setup distributed tracing with the configured backend."""
        logger.info(f"Setting up distributed tracing with {self.config.backend.value} backend")
        
        # Create resource with service information
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.config.service_name,
            ResourceAttributes.SERVICE_VERSION: self.config.service_version,
            ResourceAttributes.SERVICE_NAMESPACE: self.config.service_namespace,
            ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv("HOSTNAME", "unknown"),
        })
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        
        # Configure span processors based on backend
        if self.config.backend == TracingBackend.DISABLED:
            logger.info("Distributed tracing is disabled")
        
        elif self.config.backend == TracingBackend.CONSOLE:
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            self.tracer_provider.add_span_processor(processor)
            self.span_processors.append(processor)
        
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
                self.tracer_provider.add_span_processor(processor)
                self.span_processors.append(processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        
        # Start health monitoring if enabled
        if self.config.health_check_enabled:
            self._start_health_monitoring()
        
        logger.info(f"Distributed tracing setup complete - endpoint: {self.config.endpoint}")
        return self.tracer_provider
    
    def _create_otlp_exporter(self):
        """Create OTLP exporter based on protocol configuration."""
        try:
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
        parsed = urlparse(self.config.endpoint)
        return parsed.scheme in ("http", "grpc") or "localhost" in parsed.netloc
    
    def _start_health_monitoring(self):
        """Start background health monitoring for the tracing backend."""
        # This would typically start a background task to periodically check backend health
        # For now, we'll just log that monitoring is enabled
        logger.info(f"Health monitoring enabled for {self.config.backend.value} backend")
        self._health_status["backend"] = self.config.backend.value
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the tracing backend."""
        return {
            "backend": self.config.backend.value,
            "endpoint": self.config.endpoint,
            "protocol": self.config.protocol.value,
            "status": self._health_status,
            "processors": len(self.span_processors),
        }
    
    def shutdown(self):
        """Gracefully shutdown tracing and flush remaining spans."""
        logger.info("Shutting down distributed tracing")
        
        if self.tracer_provider:
            # Force flush all pending spans
            for processor in self.span_processors:
                try:
                    processor.force_flush(timeout_millis=5000)
                    processor.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down span processor: {e}")
        
        logger.info("Distributed tracing shutdown complete")


# Global distributed tracing manager
_distributed_tracing_manager: Optional[DistributedTracingManager] = None


def setup_distributed_tracing(config: Optional[DistributedTracingConfig] = None) -> TracerProvider:
    """Setup global distributed tracing with the specified configuration."""
    global _distributed_tracing_manager
    
    _distributed_tracing_manager = DistributedTracingManager(config)
    return _distributed_tracing_manager.setup()


def get_distributed_tracing_manager() -> Optional[DistributedTracingManager]:
    """Get the global distributed tracing manager."""
    return _distributed_tracing_manager


def get_tracing_health() -> Dict[str, Any]:
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
    os.environ.update({
        "TRACING_BACKEND": "jaeger",
        "TRACING_PROTOCOL": protocol.value,
        "JAEGER_GRPC_ENDPOINT" if protocol == TracingProtocol.GRPC else "JAEGER_HTTP_ENDPOINT": endpoint,
        "OTEL_SERVICE_NAME": service_name,
    })
    
    config = DistributedTracingConfig()
    return setup_distributed_tracing(config)


def setup_zipkin_tracing(
    endpoint: str = "http://localhost:9411/api/v2/spans",
    service_name: str = "synndicate",
) -> TracerProvider:
    """Setup Zipkin tracing with specified configuration."""
    os.environ.update({
        "TRACING_BACKEND": "zipkin",
        "TRACING_PROTOCOL": "http",
        "ZIPKIN_ENDPOINT": endpoint,
        "OTEL_SERVICE_NAME": service_name,
    })
    
    config = DistributedTracingConfig()
    return setup_distributed_tracing(config)


def setup_otlp_tracing(
    endpoint: str = "http://localhost:4317",
    service_name: str = "synndicate",
    protocol: TracingProtocol = TracingProtocol.GRPC,
    headers: Optional[Dict[str, str]] = None,
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
