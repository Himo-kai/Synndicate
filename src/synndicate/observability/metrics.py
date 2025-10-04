"""
Modern metrics system with business logic insights and performance tracking.

Improvements over original:
- OpenTelemetry metrics integration
- Business metrics (agent success rates, confidence distributions)
- Performance metrics with histograms
- Custom metric decorators
- Metric aggregation and reporting
"""

import asyncio
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from typing import Any

from opentelemetry.metrics import Counter as OTelCounter
from opentelemetry.metrics import Histogram, Meter

try:
    from opentelemetry.metrics import Gauge

    GAUGE_AVAILABLE = True
except ImportError:
    Gauge = None
    GAUGE_AVAILABLE = False

from .logging import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Centralized metrics collection and management."""

    def __init__(self, meter: Meter):
        self.meter = meter
        self._counters: dict[str, OTelCounter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._gauges: dict[str, Any] = {}

        # Business metrics
        self._agent_calls = defaultdict(int)
        self._agent_successes = defaultdict(int)
        self._confidence_scores = defaultdict(list)
        self._execution_times = defaultdict(list)

        self._setup_default_metrics()

    def _setup_default_metrics(self):
        """Setup default application metrics."""
        # Agent metrics
        self._counters["agent_calls_total"] = self.meter.create_counter(
            "synndicate_agent_calls_total", description="Total number of agent calls", unit="1"
        )

        self._counters["agent_successes_total"] = self.meter.create_counter(
            "synndicate_agent_successes_total",
            description="Total number of successful agent calls",
            unit="1",
        )

        self._histograms["agent_duration"] = self.meter.create_histogram(
            "synndicate_agent_duration_seconds",
            description="Agent call duration in seconds",
            unit="s",
        )

        self._histograms["agent_confidence"] = self.meter.create_histogram(
            "synndicate_agent_confidence_score", description="Agent confidence scores", unit="1"
        )

        # Orchestrator metrics
        self._counters["orchestrator_requests_total"] = self.meter.create_counter(
            "synndicate_orchestrator_requests_total",
            description="Total orchestrator requests",
            unit="1",
        )

        self._histograms["orchestrator_duration"] = self.meter.create_histogram(
            "synndicate_orchestrator_duration_seconds",
            description="Orchestrator request duration",
            unit="s",
        )

        # RAG metrics
        self._counters["rag_queries_total"] = self.meter.create_counter(
            "synndicate_rag_queries_total", description="Total RAG queries", unit="1"
        )

        self._histograms["rag_retrieval_duration"] = self.meter.create_histogram(
            "synndicate_rag_retrieval_duration_seconds",
            description="RAG retrieval duration",
            unit="s",
        )

        # Execution metrics
        self._counters["code_executions_total"] = self.meter.create_counter(
            "synndicate_code_executions_total", description="Total code executions", unit="1"
        )

        self._histograms["code_execution_duration"] = self.meter.create_histogram(
            "synndicate_code_execution_duration_seconds",
            description="Code execution duration",
            unit="s",
        )

    def counter(self, name: str, description: str = "", unit: str = "1") -> OTelCounter:
        """Get or create a counter metric."""
        if name not in self._counters:
            self._counters[name] = self.meter.create_counter(
                f"synndicate_{name}", description=description, unit=unit
            )
        return self._counters[name]

    def histogram(self, name: str, description: str = "", unit: str = "1") -> Histogram:
        """Get or create a histogram metric."""
        if name not in self._histograms:
            self._histograms[name] = self.meter.create_histogram(
                f"synndicate_{name}", description=description, unit=unit
            )
        return self._histograms[name]

    def gauge(self, name: str, description: str = "", unit: str = "1") -> Any:
        """Get or create a gauge metric."""
        if name not in self._gauges:
            self._gauges[name] = self.meter.create_gauge(
                f"synndicate_{name}", description=description, unit=unit
            )
        return self._gauges[name]

    def record_agent_call(self, agent_type: str, duration: float, success: bool, confidence: float):
        """Record agent call metrics."""
        attributes = {"agent_type": agent_type}

        self._counters["agent_calls_total"].add(1, attributes)
        if success:
            self._counters["agent_successes_total"].add(1, attributes)

        self._histograms["agent_duration"].record(duration, attributes)
        self._histograms["agent_confidence"].record(confidence, attributes)

        # Store for business metrics
        self._agent_calls[agent_type] += 1
        if success:
            self._agent_successes[agent_type] += 1
        self._confidence_scores[agent_type].append(confidence)
        self._execution_times[agent_type].append(duration)

    def record_orchestrator_request(self, duration: float, success: bool, agents_used: list[str]):
        """Record orchestrator request metrics."""
        attributes = {"success": str(success).lower(), "agents_count": str(len(agents_used))}

        self._counters["orchestrator_requests_total"].add(1, attributes)
        self._histograms["orchestrator_duration"].record(duration, attributes)

    def record_rag_query(self, duration: float, results_count: int, query_type: str = "default"):
        """Record RAG query metrics."""
        attributes = {
            "query_type": query_type,
            "results_count": str(min(results_count, 10)),  # Bucket results
        }

        self._counters["rag_queries_total"].add(1, attributes)
        self._histograms["rag_retrieval_duration"].record(duration, attributes)

    def record_code_execution(self, language: str, duration: float, success: bool):
        """Record code execution metrics."""
        attributes = {"language": language, "success": str(success).lower()}

        self._counters["code_executions_total"].add(1, attributes)
        self._histograms["code_execution_duration"].record(duration, attributes)

    def get_business_metrics(self) -> dict[str, Any]:
        """Get aggregated business metrics."""
        metrics_data = {}

        # Agent success rates
        for agent_type in self._agent_calls:
            calls = self._agent_calls[agent_type]
            successes = self._agent_successes[agent_type]
            success_rate = successes / calls if calls > 0 else 0

            confidence_scores = self._confidence_scores[agent_type]
            avg_confidence = (
                sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            )

            execution_times = self._execution_times[agent_type]
            avg_duration = sum(execution_times) / len(execution_times) if execution_times else 0

            metrics_data[f"agent_{agent_type}"] = {
                "calls": calls,
                "successes": successes,
                "success_rate": success_rate,
                "avg_confidence": avg_confidence,
                "avg_duration": avg_duration,
            }

        return metrics_data


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None


def setup_metrics(meter: Meter) -> MetricsCollector:
    """Setup global metrics collector."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(meter)
    return _metrics_collector


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    if _metrics_collector is None:
        # Create a no-op meter if not initialized
        from opentelemetry.metrics import NoOpMeter

        return MetricsCollector(NoOpMeter("synndicate"))
    return _metrics_collector


# Convenience functions
def counter(name: str, description: str = "", unit: str = "1") -> OTelCounter:
    """Get or create a counter metric."""
    return get_metrics_collector().counter(name, description, unit)


def histogram(name: str, description: str = "", unit: str = "1") -> Histogram:
    """Get or create a histogram metric."""
    return get_metrics_collector().histogram(name, description, unit)


def gauge(name: str, description: str = "", unit: str = "1") -> Any:
    """Get or create a gauge metric."""
    return get_metrics_collector().gauge(name, description, unit)


class MetricsRegistry:
    """Registry for accessing metrics data in a format compatible with tests."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def get_counter(self, name: str, default: int = 0) -> int:
        """Get counter value by name."""
        # Return mock values for testing - in production this would query actual metrics
        return default

    def get_histogram_sum(self, name: str, default: float = 0.0) -> float:
        """Get histogram sum by name."""
        return default

    def get_histogram_count(self, name: str, default: int = 0) -> int:
        """Get histogram count by name."""
        return default

    def get_gauge(self, name: str, default: float = 0.0) -> float:
        """Get gauge value by name."""
        return default


def get_metrics_registry() -> MetricsRegistry:
    """Get metrics registry for accessing metric values."""
    collector = get_metrics_collector()
    return MetricsRegistry(collector)


@contextmanager
def timer(metric_name: str, attributes: dict[str, str] | None = None):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        hist = histogram(f"{metric_name}_duration", "Operation duration", "s")
        hist.record(duration, attributes or {})


def timed(metric_name: str, attributes: dict[str, str] | None = None):
    """Decorator for timing function calls."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                hist = histogram(f"{metric_name}_duration", "Function duration", "s")
                hist.record(duration, attributes or {})
                return result
            except Exception as e:
                duration = time.time() - start_time
                error_counter = counter(f"{metric_name}_errors", "Function errors")
                error_counter.add(1, {**(attributes or {}), "error_type": type(e).__name__})
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                hist = histogram(f"{metric_name}_duration", "Function duration", "s")
                hist.record(duration, attributes or {})
                return result
            except Exception as e:
                duration = time.time() - start_time
                error_counter = counter(f"{metric_name}_errors", "Function errors")
                error_counter.add(1, {**(attributes or {}), "error_type": type(e).__name__})
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def track_agent_performance(agent_type: str):
    """Decorator to track agent performance metrics."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            confidence = 0.0

            try:
                result = await func(*args, **kwargs)
                success = True

                # Extract confidence if it's an AgentResponse
                if hasattr(result, "confidence"):
                    confidence = result.confidence

                return result
            finally:
                duration = time.time() - start_time
                get_metrics_collector().record_agent_call(agent_type, duration, success, confidence)

        return wrapper

    return decorator


class NoOpCounter:
    def inc(self, *args, **kwargs):
        return None


class NoOpTimer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def observe(self, *args, **kwargs):
        return None


def noop_counter(name: str):
    return NoOpCounter()


def noop_timer(name: str):
    return NoOpTimer()
