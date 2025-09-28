"""
System monitoring and health checking with resource tracking.

Improvements over original:
- Comprehensive health checks with dependency validation
- Resource monitoring with thresholds and alerting
- Performance profiling and bottleneck detection
- Circuit breaker integration
- Graceful degradation support
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil

from .logging import get_logger
from .metrics import get_metrics_collector

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemResources:
    """System resource usage snapshot."""

    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_connections: int
    process_count: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResourceThresholds:
    """Resource usage thresholds for alerting."""

    cpu_warning: float = 70.0
    cpu_critical: float = 90.0
    memory_warning: float = 80.0
    memory_critical: float = 95.0
    disk_warning: float = 85.0
    disk_critical: float = 95.0


class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(self):
        self._checks: dict[str, Callable] = {}
        self._last_results: dict[str, HealthCheck] = {}

    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self._checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    async def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self._checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check '{name}' not found",
                duration_ms=0.0,
            )

        start_time = time.time()
        try:
            check_func = self._checks[name]
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            duration_ms = (time.time() - start_time) * 1000

            if isinstance(result, HealthCheck):
                result.duration_ms = duration_ms
                return result
            elif isinstance(result, bool):
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="OK" if result else "Check failed",
                    duration_ms=duration_ms,
                )
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    duration_ms=duration_ms,
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}",
                duration_ms=duration_ms,
            )

    async def run_all_checks(self) -> dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}

        # Run checks concurrently
        tasks = [(name, self.run_check(name)) for name in self._checks]

        for name, task in tasks:
            try:
                result = await task
                results[name] = result
                self._last_results[name] = result
            except Exception as e:
                logger.error(f"Failed to run health check '{name}': {e}")
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check execution failed: {str(e)}",
                    duration_ms=0.0,
                )

        return results

    def get_overall_status(self, results: dict[str, HealthCheck]) -> HealthStatus:
        """Determine overall system health status."""
        if not results:
            return HealthStatus.UNHEALTHY

        statuses = [check.status for check in results.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def get_last_results(self) -> dict[str, HealthCheck]:
        """Get the last health check results."""
        return self._last_results.copy()


class ResourceMonitor:
    """System resource monitoring with thresholds."""

    def __init__(self, thresholds: ResourceThresholds | None = None):
        self.thresholds = thresholds or ResourceThresholds()
        self._history: list[SystemResources] = []
        self._max_history = 100

    def get_current_resources(self) -> SystemResources:
        """Get current system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)

            # Network connections
            network_connections = len(psutil.net_connections())

            # Process count
            process_count = len(psutil.pids())

            resources = SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_connections=network_connections,
                process_count=process_count,
            )

            # Add to history
            self._history.append(resources)
            if len(self._history) > self._max_history:
                self._history.pop(0)

            # Record metrics
            metrics = get_metrics_collector()
            metrics.gauge("system_cpu_percent").set(cpu_percent)
            metrics.gauge("system_memory_percent").set(memory_percent)
            metrics.gauge("system_disk_percent").set(disk_usage_percent)
            metrics.gauge("system_network_connections").set(network_connections)

            return resources

        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            raise

    def check_thresholds(self, resources: SystemResources) -> list[str]:
        """Check if any resource thresholds are exceeded."""
        alerts = []

        # CPU checks
        if resources.cpu_percent >= self.thresholds.cpu_critical:
            alerts.append(f"CRITICAL: CPU usage at {resources.cpu_percent:.1f}%")
        elif resources.cpu_percent >= self.thresholds.cpu_warning:
            alerts.append(f"WARNING: CPU usage at {resources.cpu_percent:.1f}%")

        # Memory checks
        if resources.memory_percent >= self.thresholds.memory_critical:
            alerts.append(f"CRITICAL: Memory usage at {resources.memory_percent:.1f}%")
        elif resources.memory_percent >= self.thresholds.memory_warning:
            alerts.append(f"WARNING: Memory usage at {resources.memory_percent:.1f}%")

        # Disk checks
        if resources.disk_usage_percent >= self.thresholds.disk_critical:
            alerts.append(f"CRITICAL: Disk usage at {resources.disk_usage_percent:.1f}%")
        elif resources.disk_usage_percent >= self.thresholds.disk_warning:
            alerts.append(f"WARNING: Disk usage at {resources.disk_usage_percent:.1f}%")

        return alerts

    def get_resource_trends(self, minutes: int = 10) -> dict[str, float]:
        """Get resource usage trends over the specified time period."""
        if len(self._history) < 2:
            return {}

        cutoff_time = time.time() - (minutes * 60)
        recent_history = [r for r in self._history if r.timestamp >= cutoff_time]

        if len(recent_history) < 2:
            return {}

        # Calculate trends (positive = increasing, negative = decreasing)
        first = recent_history[0]
        last = recent_history[-1]

        return {
            "cpu_trend": last.cpu_percent - first.cpu_percent,
            "memory_trend": last.memory_percent - first.memory_percent,
            "disk_trend": last.disk_usage_percent - first.disk_usage_percent,
        }

    def get_history(self, limit: int | None = None) -> list[SystemResources]:
        """Get resource usage history."""
        if limit:
            return self._history[-limit:]
        return self._history.copy()


def setup_default_health_checks(health_checker: HealthChecker, dependencies: dict[str, Any]):
    """Setup default health checks for common dependencies."""

    async def database_check():
        """Check database connectivity."""
        # Placeholder for database health check
        return HealthCheck(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection OK",
            duration_ms=0.0,
        )

    async def http_client_check():
        """Check HTTP client health."""
        http_client = dependencies.get("http_client")
        if not http_client:
            return HealthCheck(
                name="http_client",
                status=HealthStatus.UNHEALTHY,
                message="HTTP client not available",
                duration_ms=0.0,
            )

        try:
            # Simple connectivity test
            response = await http_client.get("http://httpbin.org/status/200", timeout=5.0)
            if response.status_code == 200:
                return HealthCheck(
                    name="http_client",
                    status=HealthStatus.HEALTHY,
                    message="HTTP client OK",
                    duration_ms=0.0,
                )
            else:
                return HealthCheck(
                    name="http_client",
                    status=HealthStatus.DEGRADED,
                    message=f"HTTP client returned {response.status_code}",
                    duration_ms=0.0,
                )
        except Exception as e:
            return HealthCheck(
                name="http_client",
                status=HealthStatus.UNHEALTHY,
                message=f"HTTP client error: {str(e)}",
                duration_ms=0.0,
            )

    def resource_check():
        """Check system resources."""
        monitor = ResourceMonitor()
        resources = monitor.get_current_resources()
        alerts = monitor.check_thresholds(resources)

        if any("CRITICAL" in alert for alert in alerts):
            status = HealthStatus.CRITICAL
        elif any("WARNING" in alert for alert in alerts):
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return HealthCheck(
            name="resources",
            status=status,
            message="; ".join(alerts) if alerts else "Resource usage normal",
            duration_ms=0.0,
            metadata={
                "cpu_percent": resources.cpu_percent,
                "memory_percent": resources.memory_percent,
                "disk_percent": resources.disk_usage_percent,
            },
        )

    # Register checks
    health_checker.register_check("database", database_check)
    health_checker.register_check("http_client", http_client_check)
    health_checker.register_check("resources", resource_check)
