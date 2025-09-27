"""
Performance probing and observability system.
Supports custom timers, Prometheus metrics, and OpenTelemetry traces.
"""

import time
import contextlib
from typing import Optional, Dict, Any
from .logging import get_logger

log = get_logger("syn.probe")

# OpenTelemetry support (optional)
OTEL_ENABLED = False
try:
    from opentelemetry import trace
    tracer = trace.get_tracer("synndicate")
    OTEL_ENABLED = True
except Exception:
    tracer = None

# Prometheus metrics support (optional)
PROM_ENABLED = False
try:
    from prometheus_client import Counter, Histogram
    REQS = Counter("syn_requests_total", "Total requests", ["op", "ok"])
    LAT = Histogram("syn_latency_seconds", "Request latency", ["op"])
    PROM_ENABLED = True
except Exception:
    pass

# Global metrics storage for audit trails
_METRICS_STORE: Dict[str, Dict[str, Any]] = {}


@contextlib.contextmanager
def probe(op: str, trace_id: Optional[str] = None, **labels):
    """
    Performance probe context manager.
    
    Features:
    - Always-on custom timers with structured logging
    - Optional Prometheus metrics (if available)
    - Optional OpenTelemetry traces (if enabled)
    - Metrics storage for audit trails
    
    Args:
        op: Operation name (e.g., "orchestrator.process_query")
        trace_id: Optional trace ID for correlation
        **labels: Additional labels for metrics
    """
    # Start OpenTelemetry span if available
    span_ctx = tracer.start_as_current_span(op) if OTEL_ENABLED else contextlib.nullcontext()
    
    start_time = time.perf_counter()
    ok = "true"
    error_type = None
    
    with span_ctx:
        try:
            yield
        except Exception as e:
            ok = "false"
            error_type = type(e).__name__
            raise
        finally:
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Structured logging (always on)
            log.info(
                f'op={op} ms={duration_ms:.1f} trace={trace_id or "-"} ok={ok}' +
                (f' error={error_type}' if error_type else '') +
                ''.join(f' {k}={v}' for k, v in labels.items())
            )
            
            # Prometheus metrics (if available)
            if PROM_ENABLED:
                REQS.labels(op=op, ok=ok).inc()
                LAT.labels(op=op).observe(duration_ms / 1000.0)
            
            # Store metrics for audit trails
            if trace_id:
                if trace_id not in _METRICS_STORE:
                    _METRICS_STORE[trace_id] = {}
                _METRICS_STORE[trace_id][op] = {
                    "duration_ms": duration_ms,
                    "success": ok == "true",
                    "error_type": error_type,
                    "labels": labels,
                    "timestamp": time.time()
                }


def get_trace_metrics(trace_id: str) -> Dict[str, Any]:
    """Get all metrics for a specific trace ID."""
    return _METRICS_STORE.get(trace_id, {})


def clear_trace_metrics(trace_id: str) -> None:
    """Clear metrics for a specific trace ID."""
    _METRICS_STORE.pop(trace_id, None)


def get_all_metrics() -> Dict[str, Dict[str, Any]]:
    """Get all stored metrics (for debugging)."""
    return _METRICS_STORE.copy()


# Convenience decorators
def timed(op: str, trace_id_attr: str = "trace_id"):
    """
    Decorator for timing methods.
    
    Args:
        op: Operation name
        trace_id_attr: Attribute name to get trace_id from (default: "trace_id")
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to get trace_id from self or kwargs
            trace_id = None
            if args and hasattr(args[0], trace_id_attr):
                trace_id = getattr(args[0], trace_id_attr)
            elif trace_id_attr in kwargs:
                trace_id = kwargs[trace_id_attr]
            
            with probe(op, trace_id):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def async_timed(op: str, trace_id_attr: str = "trace_id"):
    """
    Async decorator for timing methods.
    
    Args:
        op: Operation name
        trace_id_attr: Attribute name to get trace_id from (default: "trace_id")
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Try to get trace_id from self or kwargs
            trace_id = None
            if args and hasattr(args[0], trace_id_attr):
                trace_id = getattr(args[0], trace_id_attr)
            elif trace_id_attr in kwargs:
                trace_id = kwargs[trace_id_attr]
            
            with probe(op, trace_id):
                return await func(*args, **kwargs)
        return wrapper
    return decorator
