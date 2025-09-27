"""
OpenTelemetry tracing integration for distributed observability.

Improvements over original:
- Full OpenTelemetry integration
- Automatic span correlation
- Custom span attributes for business context
- Error tracking and span status
- Trace sampling configuration
"""

import functools
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from .logging import get_logger, trace_id_ctx

logger = get_logger(__name__)


class TracingManager:
    """Manages OpenTelemetry tracing configuration and utilities."""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self.propagator = TraceContextTextMapPropagator()

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    ) -> Span:
        """Start a new span with optional attributes."""
        span = self.tracer.start_span(name, kind=kind)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))

        # Set trace ID in logging context
        from .logging import set_trace_id
        trace_id = format(span.get_span_context().trace_id, "032x")
        set_trace_id(trace_id)

        return span

    def extract_context(self, headers: dict[str, str]) -> trace.Context:
        """Extract trace context from HTTP headers."""
        return self.propagator.extract(headers)

    def inject_context(
        self, headers: dict[str, str], context: trace.Context | None = None
    ) -> None:
        """Inject trace context into HTTP headers."""
        ctx = context or trace.get_current()
        self.propagator.inject(headers, ctx)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    ):
        """Context manager for creating spans."""
        span = self.start_span(name, attributes, kind)
        try:
            with trace.use_span(span):
                yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            span.end()


# Global tracing manager
_tracing_manager: TracingManager | None = None


def setup_tracing(tracer: Tracer) -> TracingManager:
    """Setup global tracing manager."""
    global _tracing_manager
    _tracing_manager = TracingManager(tracer)
    return _tracing_manager


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    if _tracing_manager is None:
        from opentelemetry.trace import NoOpTracer

        return NoOpTracer()
    return _tracing_manager.tracer


def get_tracing_manager() -> TracingManager:
    """Get global tracing manager."""
    if _tracing_manager is None:
        from opentelemetry.trace import NoOpTracer

        return TracingManager(NoOpTracer())
    return _tracing_manager


def trace_span(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
):
    """Decorator for automatic span creation."""

    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_tracing_manager()
            span_attributes = {
                "function.name": func.__name__,
                "function.module": func.__module__,
                **(attributes or {}),
            }

            with manager.span(span_name, span_attributes, kind) as span:
                try:
                    # Add function arguments as attributes (be careful with sensitive data)
                    if args and not any(
                        isinstance(arg, (str, int, float, bool)) for arg in args[:3]
                    ):
                        span.set_attribute("function.args_count", len(args))
                    if kwargs:
                        span.set_attribute("function.kwargs_count", len(kwargs))

                    result = await func(*args, **kwargs)

                    # Add result metadata if available
                    if hasattr(result, "__dict__"):
                        if hasattr(result, "confidence"):
                            span.set_attribute("result.confidence", result.confidence)
                        if hasattr(result, "success"):
                            span.set_attribute("result.success", result.success)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            manager = get_tracing_manager()
            span_attributes = {
                "function.name": func.__name__,
                "function.module": func.__module__,
                **(attributes or {}),
            }

            with manager.span(span_name, span_attributes, kind) as span:
                try:
                    # Add function arguments as attributes
                    if args and not any(
                        isinstance(arg, (str, int, float, bool)) for arg in args[:3]
                    ):
                        span.set_attribute("function.args_count", len(args))
                    if kwargs:
                        span.set_attribute("function.kwargs_count", len(kwargs))

                    result = func(*args, **kwargs)

                    # Add result metadata if available
                    if hasattr(result, "__dict__"):
                        if hasattr(result, "confidence"):
                            span.set_attribute("result.confidence", result.confidence)
                        if hasattr(result, "success"):
                            span.set_attribute("result.success", result.success)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # Return appropriate wrapper based on function type
        import asyncio

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def add_span_attributes(**attributes):
    """Add attributes to the current span."""
    current_span = trace.get_current_span()
    if current_span.is_recording():
        for key, value in attributes.items():
            current_span.set_attribute(key, str(value))


def add_span_event(name: str, attributes: dict[str, Any] | None = None):
    """Add an event to the current span."""
    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.add_event(name, attributes or {})


def set_span_error(error: Exception):
    """Mark the current span as having an error."""
    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.record_exception(error)
        current_span.set_status(Status(StatusCode.ERROR, str(error)))


@contextmanager
def trace_context(name: str, attributes: dict[str, Any] | None = None):
    """Context manager for creating a trace span."""
    manager = get_tracing_manager()
    with manager.span(name, attributes) as span:
        yield span
