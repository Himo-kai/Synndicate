"""
Enhanced API Server - Phase 3 Runtime Orchestration.

This module implements production-grade API patterns:
- Deadline enforcement with X-Request-Id correlation
- Backpressure with 429 responses and Retry-After
- Idempotent request handling
- Structured logging with correlation IDs
- Circuit breaker integration
- Graceful cancellation support

Runtime Invariants Enforced:
1. Every request has correlation_id and trace_id
2. Deadlines are propagated and enforced
3. Queue depth is bounded with 429 responses
4. Cancellation is authoritative and immediate
"""

import asyncio
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from synndicate.core.runtime_patterns import RequestContext, get_circuit_breaker, run_once
from synndicate.observability.logging import get_logger
from synndicate.observability.metrics import counter, gauge, histogram
from synndicate.observability.tracing import get_trace_id, trace_span

logger = get_logger(__name__)

# Configuration
MAX_INFLIGHT_JOBS = 10
MAX_QUEUE_SIZE = 20
DEFAULT_TIMEOUT_MS = 30000


class JobRequest(BaseModel):
    """Job submission request."""
    query: str = Field(..., description="Query to process")
    context: dict[str, Any] | None = Field(default=None, description="Additional context")
    workflow: str = Field(default="auto", description="Workflow type")


class JobResponse(BaseModel):
    """Job response."""
    id: str = Field(..., description="Job correlation ID")
    status: str = Field(..., description="Job status")
    result: dict[str, Any] | None = Field(default=None, description="Job result")
    created_at: float = Field(..., description="Creation timestamp")
    updated_at: float = Field(..., description="Last update timestamp")
    error: str | None = Field(default=None, description="Error message if failed")


class JobStatus(BaseModel):
    """Job status response."""
    id: str
    status: str
    progress: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float
    updated_at: float
    execution_time: float | None = None


# Global job registry and queue management
_active_jobs: dict[str, asyncio.Task] = {}
_job_results: dict[str, JobResponse] = {}
_job_semaphore = asyncio.Semaphore(MAX_INFLIGHT_JOBS)


def create_enhanced_app() -> FastAPI:
    """Create FastAPI app with Phase 3 runtime enhancements."""
    app = FastAPI(
        title="Synndicate AI - Enhanced Runtime",
        description="Production-grade AI orchestration with runtime guarantees",
        version="3.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def runtime_middleware(request: Request, call_next):
        """Runtime middleware for correlation, deadlines, and observability."""
        start_time = time.time()

        # Extract or generate correlation ID
        correlation_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        trace_id = request.headers.get("X-Trace-Id") or f"{int(start_time * 1000):x}"

        # Get trace context (trace_id already extracted from headers)
        # Note: OpenTelemetry handles trace context automatically

        # Parse timeout
        timeout_ms = request.query_params.get("timeout_ms", DEFAULT_TIMEOUT_MS)
        try:
            timeout_ms = int(timeout_ms)
        except ValueError:
            timeout_ms = DEFAULT_TIMEOUT_MS

        # Create request context
        request_ctx = RequestContext.create(
            timeout_ms=timeout_ms,
            correlation_id=correlation_id,
            trace_id=trace_id,
        )

        # Add to request state
        request.state.request_ctx = request_ctx

        # Process request
        try:
            response = await call_next(request)

            # Add correlation headers to response
            response.headers["X-Request-Id"] = correlation_id
            response.headers["X-Trace-Id"] = trace_id

            # Record metrics
            duration = time.time() - start_time
            histogram("api.request_duration_seconds").observe(duration)
            counter("api.requests_total").inc()

            return response

        except Exception as e:
            duration = time.time() - start_time
            counter("api.requests_failed_total").inc()

            logger.error(
                "Request failed",
                extra={
                    "correlation_id": correlation_id,
                    "trace_id": trace_id,
                    "duration_ms": duration * 1000,
                    "error": str(e),
                }
            )
            raise

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "active_jobs": len(_active_jobs),
            "queue_depth": MAX_INFLIGHT_JOBS - _job_semaphore._value,
        }

    @app.post("/jobs", response_model=JobResponse)
    async def submit_job(
        request: Request,
        job_request: JobRequest,
    ) -> JobResponse:
        """Submit job with idempotency and backpressure handling."""
        request_ctx: RequestContext = request.state.request_ctx
        correlation_id = request_ctx.correlation_id

        # Check if job already exists (idempotency)
        if correlation_id in _job_results:
            logger.debug(f"Returning existing job result for {correlation_id}")
            return _job_results[correlation_id]

        # Check queue capacity
        if len(_active_jobs) >= MAX_INFLIGHT_JOBS + MAX_QUEUE_SIZE:
            counter("api.requests_rejected_total").inc()
            raise HTTPException(
                status_code=429,
                detail="Queue full",
                headers={"Retry-After": "5"},
            )

        # Create job response
        job_response = JobResponse(
            id=correlation_id,
            status="accepted",
            created_at=time.time(),
            updated_at=time.time(),
        )

        _job_results[correlation_id] = job_response

        # Start job processing (single-flight)
        task = asyncio.create_task(
            run_once(
                f"job:{correlation_id}",
                lambda: _process_job(job_request, request_ctx)
            )
        )
        _active_jobs[correlation_id] = task

        # Update gauge
        gauge("api.active_jobs").set(len(_active_jobs))

        logger.info(
            "Job submitted",
            extra={
                "correlation_id": correlation_id,
                "query_preview": job_request.query[:100],
                "workflow": job_request.workflow,
            }
        )

        return job_response

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    async def get_job_status(job_id: str) -> JobStatus:
        """Get job status."""
        job_response = _job_results.get(job_id)
        if not job_response:
            raise HTTPException(status_code=404, detail="Job not found")

        # Check if job is still running
        task = _active_jobs.get(job_id)
        if task and not task.done():
            # Job still running
            return JobStatus(
                id=job_id,
                status="running",
                created_at=job_response.created_at,
                updated_at=time.time(),
            )

        # Job completed
        return JobStatus(
            id=job_id,
            status=job_response.status,
            result=job_response.result,
            error=job_response.error,
            created_at=job_response.created_at,
            updated_at=job_response.updated_at,
        )

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, str]:
        """Cancel running job."""
        task = _active_jobs.get(job_id)
        if not task:
            raise HTTPException(status_code=404, detail="Job not found")

        if task.done():
            return {"status": "already_completed"}

        # Cancel the task
        task.cancel()

        # Update job status
        if job_id in _job_results:
            job_response = _job_results[job_id]
            job_response.status = "cancelled"
            job_response.updated_at = time.time()

        logger.info(f"Job cancelled: {job_id}")
        counter("api.jobs_cancelled_total").inc()

        return {"status": "cancelled"}

    @app.get("/metrics")
    async def get_metrics():
        """Get runtime metrics."""
        return {
            "active_jobs": len(_active_jobs),
            "completed_jobs": len([j for j in _job_results.values() if j.status in ("completed", "failed", "cancelled")]),
            "queue_depth": MAX_INFLIGHT_JOBS - _job_semaphore._value,
            "circuit_breakers": {
                name: {
                    "failures": breaker.failures,
                    "open_until": breaker.open_until,
                    "is_open": time.time() < breaker.open_until,
                }
                for name, breaker in get_circuit_breaker.__self__._circuit_breakers.items()
            } if hasattr(get_circuit_breaker, '__self__') else {},
        }

    return app


async def _process_job(job_request: JobRequest, request_ctx: RequestContext) -> None:
    """Process job with deadline enforcement and error handling."""
    correlation_id = request_ctx.correlation_id

    try:
        # Acquire semaphore for concurrency control
        async with _job_semaphore:
            # Check deadline before starting
            if request_ctx.is_expired():
                raise TimeoutError("Deadline exceeded before processing")

            # Update status
            job_response = _job_results[correlation_id]
            job_response.status = "running"
            job_response.updated_at = time.time()

            # Import here to avoid circular imports
            from synndicate.container import Container
            from synndicate.core.orchestrator import Orchestrator

            # Create orchestrator with circuit breaker
            container = Container()
            orchestrator = Orchestrator(container)

            # Process with deadline
            with trace_span("job.process"):
                start_time = time.time()

                # Use circuit breaker for orchestrator calls
                breaker = get_circuit_breaker("orchestrator")

                async def orchestrate():
                    return await orchestrator.process_query(
                        job_request.query,
                        context=job_request.context or {},
                        workflow=job_request.workflow,
                    )

                # Execute with timeout and circuit breaker
                from synndicate.core.runtime_patterns import with_circuit_breaker, with_timeout

                result = await with_timeout(
                    with_circuit_breaker(breaker, orchestrate),
                    request_ctx.deadline
                )

                execution_time = time.time() - start_time

                # Update job with result
                job_response.status = "completed" if result.success else "failed"
                job_response.result = {
                    "success": result.success,
                    "response": result.response_text,
                    "confidence": result.confidence,
                    "execution_time": execution_time,
                    "agents_used": result.agents_used,
                    "execution_path": result.execution_path,
                }
                job_response.updated_at = time.time()

                counter("api.jobs_completed_total").inc()
                histogram("api.job_duration_seconds").observe(execution_time)

                logger.info(
                    "Job completed",
                    extra={
                        "correlation_id": correlation_id,
                        "success": result.success,
                        "execution_time": execution_time,
                        "agents_used": len(result.agents_used),
                    }
                )

    except TimeoutError:
        # Deadline exceeded
        job_response = _job_results[correlation_id]
        job_response.status = "timeout"
        job_response.error = "Deadline exceeded"
        job_response.updated_at = time.time()

        counter("api.jobs_timeout_total").inc()
        logger.warning(f"Job timeout: {correlation_id}")

    except asyncio.CancelledError:
        # Job was cancelled
        job_response = _job_results[correlation_id]
        job_response.status = "cancelled"
        job_response.updated_at = time.time()

        logger.info(f"Job cancelled during processing: {correlation_id}")
        raise  # Re-raise to properly handle cancellation

    except Exception as e:
        # Job failed
        job_response = _job_results[correlation_id]
        job_response.status = "failed"
        job_response.error = str(e)
        job_response.updated_at = time.time()

        counter("api.jobs_failed_total").inc()
        logger.error(
            "Job failed",
            extra={
                "correlation_id": correlation_id,
                "error": str(e),
            }
        )

    finally:
        # Clean up
        _active_jobs.pop(correlation_id, None)
        gauge("api.active_jobs").set(len(_active_jobs))


# Create the app instance
app = create_enhanced_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "synndicate.api.enhanced_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True,
    )
