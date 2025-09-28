"""
Enterprise-Grade Observability Infrastructure for Synndicate AI.

Provides production-ready observability with comprehensive trace-based monitoring,
structured logging, performance probes, and audit-ready compliance features.
Designed for enterprise deployment with deterministic behavior and full auditability.

Core Components:
- Structured Logging: Single-line JSON format with trace IDs and timing
- Performance Probes: Always-on timing with millisecond precision
- Trace Propagation: End-to-end request tracking with contextvars
- Audit Trails: Complete trace snapshots for compliance
- Metrics Collection: Prometheus and OpenTelemetry integration
- Configuration: Deterministic config hashing and validation

Features:
- Complete request lifecycle tracking from API to model inference
- Automatic trace ID generation and propagation across all components
- Performance monitoring with operation-level timing and success tracking
- Structured logs with consistent schema: t=<ISO8601> trace=<id> ms=<dur>
- Audit-ready trace snapshots with comprehensive metadata
- Circuit breaker and error handling with detailed context

Usage:
    Basic logging with trace context:
    >>> from synndicate.observability.logging import get_logger
    >>> from synndicate.observability.probe import probe
    >>>
    >>> logger = get_logger(__name__)
    >>>
    >>> @probe("my_operation")
    >>> async def my_function(data: str) -> str:
    ...     logger.info("Processing data", data_length=len(data))
    ...     result = await process_data(data)
    ...     return result

    Trace ID propagation:
    >>> from synndicate.observability.logging import set_trace_id, get_trace_id
    >>>
    >>> trace_id = "abc123def456"
    >>> set_trace_id(trace_id)
    >>> current_trace = get_trace_id()  # Returns "abc123def456"

    Performance metrics:
    >>> from synndicate.observability.probe import get_trace_metrics
    >>>
    >>> metrics = get_trace_metrics(trace_id)
    >>> # {"my_operation": {"duration_ms": 125.3, "success": true}}

Audit & Compliance:
    Every operation generates comprehensive audit data:
    - Trace snapshots: artifacts/orchestrator_trace_<trace_id>.json
    - Performance data: artifacts/perf_<trace_id>.jsonl
    - Structured logs with full context and timing
    - Configuration hashes for reproducible behavior
    - Component health and status monitoring

Configuration:
    Environment variables for observability:
    - SYN_OBSERVABILITY__LOG_LEVEL=INFO (logging level)
    - SYN_OBSERVABILITY__ENABLE_TRACING=true (trace collection)
    - SYN_OBSERVABILITY__ENABLE_METRICS=true (metrics collection)
    - SYN_OBSERVABILITY__OTLP_ENDPOINT=http://... (OpenTelemetry)
    - SYN_SEED=1337 (deterministic behavior)
"""

from .logging import get_logger, setup_logging
from .metrics import counter, gauge, histogram, timer
from .monitoring import HealthChecker, ResourceMonitor
from .tracing import get_tracer, trace_span

__all__ = [
    "get_logger",
    "setup_logging",
    "timer",
    "counter",
    "histogram",
    "gauge",
    "trace_span",
    "get_tracer",
    "HealthChecker",
    "ResourceMonitor",
]
