"""
Advanced analytics for usage patterns and system optimization.

This module provides comprehensive analytics capabilities for tracking usage patterns,
performance metrics, and optimization opportunities across the Synndicate AI system.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from ..observability.logging import get_logger
from ..observability.metrics import counter, histogram

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics for analytics tracking."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class UsagePattern(Enum):
    """Common usage patterns for analysis."""

    PEAK_HOURS = "peak_hours"
    BATCH_PROCESSING = "batch_processing"
    INTERACTIVE_SESSION = "interactive_session"
    API_HEAVY = "api_heavy"
    MULTIMODAL_USAGE = "multimodal_usage"


@dataclass
class UsageMetric:
    """Individual usage metric with metadata."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report."""

    period_start: datetime
    period_end: datetime
    total_requests: int = 0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    usage_patterns: dict[UsagePattern, float] = field(default_factory=dict)
    top_endpoints: list[tuple[str, int]] = field(default_factory=list)
    performance_insights: list[str] = field(default_factory=list)
    optimization_recommendations: list[str] = field(default_factory=list)
    resource_utilization: dict[str, float] = field(default_factory=dict)


class UsageAnalytics:
    """Advanced usage analytics and optimization system."""

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self._metrics: deque[UsageMetric] = deque()
        self._request_counts = defaultdict(int)
        self._response_times = defaultdict(list)
        self._error_counts = defaultdict(int)
        self._endpoint_usage = defaultdict(int)
        self._user_sessions = defaultdict(list)

    async def track_request(
        self,
        endpoint: str,
        user_id: str | None = None,
        response_time: float | None = None,
        status_code: int = 200,
        **metadata
    ) -> None:
        """Track API request for analytics."""
        timestamp = time.time()

        # Track basic metrics
        self._request_counts[endpoint] += 1
        self._endpoint_usage[endpoint] += 1

        if response_time:
            self._response_times[endpoint].append(response_time)
            histogram("api.response_time").observe(response_time)

        if status_code >= 400:
            self._error_counts[endpoint] += 1
            counter("api.errors").inc()

        # Track user session
        if user_id:
            self._user_sessions[user_id].append({
                'endpoint': endpoint,
                'timestamp': timestamp,
                'response_time': response_time,
                'status_code': status_code,
                **metadata
            })

        # Store metric
        metric = UsageMetric(
            name="api_request",
            value=1,
            metric_type=MetricType.COUNTER,
            timestamp=timestamp,
            labels={
                'endpoint': endpoint,
                'status_code': str(status_code),
                'user_id': user_id or 'anonymous'
            },
            metadata=metadata
        )

        self._metrics.append(metric)
        await self._cleanup_old_metrics()

    async def generate_report(
        self,
        hours_back: int = 1
    ) -> AnalyticsReport:
        """Generate comprehensive analytics report."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        # Filter metrics for time period
        cutoff_timestamp = start_time.timestamp()
        recent_metrics = [m for m in self._metrics if m.timestamp >= cutoff_timestamp]

        # Calculate basic stats
        total_requests = len([m for m in recent_metrics if m.name == "api_request"])

        # Calculate average response time
        response_times = []
        for _endpoint, times in self._response_times.items():
            response_times.extend(times)

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

        # Calculate error rate
        total_errors = sum(self._error_counts.values())
        error_rate = (total_errors / total_requests) if total_requests > 0 else 0.0

        # Analyze usage patterns
        usage_patterns = await self._analyze_usage_patterns(recent_metrics)

        # Get top endpoints
        top_endpoints = sorted(
            self._endpoint_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Generate insights and recommendations
        insights = await self._generate_performance_insights(recent_metrics)
        recommendations = await self._generate_optimization_recommendations(recent_metrics)

        # Calculate resource utilization
        resource_util = await self._calculate_resource_utilization()

        return AnalyticsReport(
            period_start=start_time,
            period_end=end_time,
            total_requests=total_requests,
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            usage_patterns=usage_patterns,
            top_endpoints=top_endpoints,
            performance_insights=insights,
            optimization_recommendations=recommendations,
            resource_utilization=resource_util
        )

    async def _analyze_usage_patterns(self, metrics: list[UsageMetric]) -> dict[UsagePattern, float]:
        """Analyze usage patterns from metrics."""
        patterns = {}

        if not metrics:
            return patterns

        # Analyze temporal patterns
        hourly_counts = defaultdict(int)
        for metric in metrics:
            hour = datetime.fromtimestamp(metric.timestamp).hour
            hourly_counts[hour] += 1

        # Detect peak hours (hours with >150% of average traffic)
        avg_hourly = sum(hourly_counts.values()) / max(len(hourly_counts), 1)
        peak_hours = sum(1 for count in hourly_counts.values() if count > avg_hourly * 1.5)
        patterns[UsagePattern.PEAK_HOURS] = peak_hours / 24.0

        # Detect batch processing (large bursts of activity)
        batch_score = 0.0
        timestamps = [m.timestamp for m in metrics]
        if len(timestamps) > 10:
            # Look for periods of high activity
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            short_intervals = sum(1 for diff in time_diffs if diff < 1.0)  # < 1 second apart
            batch_score = short_intervals / len(time_diffs)

        patterns[UsagePattern.BATCH_PROCESSING] = batch_score

        # Detect multimodal usage
        multimodal_requests = sum(
            1 for m in metrics
            if 'multimodal' in m.labels.get('endpoint', '') or
               'image' in m.labels.get('endpoint', '') or
               'code' in m.labels.get('endpoint', '')
        )
        patterns[UsagePattern.MULTIMODAL_USAGE] = multimodal_requests / max(len(metrics), 1)

        return patterns

    async def _generate_performance_insights(self, metrics: list[UsageMetric]) -> list[str]:
        """Generate performance insights from metrics."""
        insights = []

        # Response time analysis
        response_times = []
        for _, times in self._response_times.items():
            response_times.extend(times)

        if response_times:
            avg_time = sum(response_times) / len(response_times)
            p95_time = sorted(response_times)[int(len(response_times) * 0.95)]

            if avg_time > 2.0:
                insights.append(f"Average response time is high: {avg_time:.2f}s")

            if p95_time > 5.0:
                insights.append(f"95th percentile response time is concerning: {p95_time:.2f}s")

        # Error rate analysis
        total_requests = len([m for m in metrics if m.name == "api_request"])
        total_errors = sum(self._error_counts.values())

        if total_requests > 0:
            error_rate = total_errors / total_requests
            if error_rate > 0.05:  # 5% error rate
                insights.append(f"Error rate is high: {error_rate:.1%}")

        # Usage pattern insights
        endpoint_counts = defaultdict(int)
        for metric in metrics:
            endpoint = metric.labels.get('endpoint', 'unknown')
            endpoint_counts[endpoint] += 1

        if endpoint_counts:
            most_used = max(endpoint_counts.items(), key=lambda x: x[1])
            total_usage = sum(endpoint_counts.values())
            usage_percentage = most_used[1] / total_usage

            if usage_percentage > 0.7:
                insights.append(f"Heavy concentration on single endpoint: {most_used[0]} ({usage_percentage:.1%})")

        return insights

    async def _generate_optimization_recommendations(self, metrics: list[UsageMetric]) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Caching recommendations
        repeated_requests = defaultdict(int)
        for metric in metrics:
            endpoint = metric.labels.get('endpoint', '')
            user_id = metric.labels.get('user_id', '')
            key = f"{endpoint}:{user_id}"
            repeated_requests[key] += 1

        high_repeat = sum(1 for count in repeated_requests.values() if count > 5)
        if high_repeat > len(repeated_requests) * 0.3:
            recommendations.append("Consider implementing response caching for frequently repeated requests")

        # Load balancing recommendations
        response_times = []
        for times in self._response_times.values():
            response_times.extend(times)

        if response_times:
            variance = sum((t - sum(response_times)/len(response_times))**2 for t in response_times) / len(response_times)
            if variance > 1.0:  # High variance in response times
                recommendations.append("Consider load balancing to reduce response time variance")

        # Batch processing recommendations
        single_requests = sum(1 for session in self._user_sessions.values() if len(session) == 1)
        total_sessions = len(self._user_sessions)

        if total_sessions > 0 and single_requests / total_sessions > 0.8:
            recommendations.append("Many single-request sessions detected; consider batch processing APIs")

        # Multimodal optimization
        multimodal_usage = sum(
            1 for m in metrics
            if any(keyword in m.labels.get('endpoint', '') for keyword in ['multimodal', 'image', 'code'])
        )

        if multimodal_usage > len(metrics) * 0.3:
            recommendations.append("High multimodal usage detected; consider specialized processing pipelines")

        return recommendations

    async def _calculate_resource_utilization(self) -> dict[str, float]:
        """Calculate current resource utilization metrics."""
        # Placeholder for actual resource monitoring
        # In a real implementation, this would integrate with system monitoring

        return {
            "cpu_usage": 0.65,  # 65%
            "memory_usage": 0.72,  # 72%
            "disk_usage": 0.45,  # 45%
            "network_io": 0.38,  # 38%
            "gpu_usage": 0.82,  # 82% (if available)
        }

    async def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - (self.retention_hours * 3600)

        # Remove old metrics
        while self._metrics and self._metrics[0].timestamp < cutoff_time:
            self._metrics.popleft()

        # Clean up other data structures
        for endpoint in list(self._response_times.keys()):
            self._response_times[endpoint] = [
                t for t in self._response_times[endpoint]
                if time.time() - cutoff_time < self.retention_hours * 3600
            ]
            if not self._response_times[endpoint]:
                del self._response_times[endpoint]

    def get_real_time_stats(self) -> dict[str, Any]:
        """Get real-time statistics."""
        current_time = time.time()
        recent_cutoff = current_time - 300  # Last 5 minutes

        recent_metrics = [m for m in self._metrics if m.timestamp >= recent_cutoff]

        return {
            "requests_last_5min": len([m for m in recent_metrics if m.name == "api_request"]),
            "avg_response_time_5min": sum(
                t for times in self._response_times.values() for t in times[-10:]
            ) / max(sum(len(times[-10:]) for times in self._response_times.values()), 1),
            "active_endpoints": len({m.labels.get('endpoint') for m in recent_metrics}),
            "unique_users_5min": len({
                m.labels.get('user_id') for m in recent_metrics
                if m.labels.get('user_id') != 'anonymous'
            }),
            "total_metrics_stored": len(self._metrics),
        }


# Factory function
def create_usage_analytics(retention_hours: int = 24) -> UsageAnalytics:
    """Create a usage analytics instance."""
    return UsageAnalytics(retention_hours=retention_hours)
