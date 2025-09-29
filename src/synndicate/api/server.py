"""
Production-Ready FastAPI Server for Synndicate AI with Enterprise Observability.

Provides RESTful API endpoints for AI orchestration with comprehensive health monitoring,
trace-based observability, and deterministic behavior. Features automatic startup with
config hashing, CORS support, and structured error handling.

Endpoints:
- GET /health: Component health status with uptime and config hash
- POST /query: Process queries through multi-agent orchestration
- GET /metrics: System metrics and performance data
- GET /docs: Interactive API documentation (Swagger UI)
- GET /redoc: Alternative API documentation (ReDoc)

Features:
- 🔍 Full trace ID propagation for request tracking
- 📊 Performance probes with millisecond-precision timing
- 🎯 Deterministic startup with config SHA256 hashing
- 🛡️ CORS support and structured error handling
- 📝 Comprehensive request/response logging
- ⚙️ Graceful startup/shutdown with resource cleanup

Usage:
    Start development server:
    $ uvicorn synndicate.api.server:app --reload --host 0.0.0.0 --port 8000

    Or use make target:
    $ make dev

    Health check:
    $ curl http://localhost:8000/health
    {
      "status": "healthy",
      "config_hash": "28411d9a...",
      "components": {"orchestrator": "healthy", "models": "healthy"}
    }

    Process query:
    $ curl -X POST http://localhost:8000/query \
      -H 'Content-Type: application/json' \
      -d '{"query":"Create a Python calculator"}'
    {
      "success": true,
      "trace_id": "abc123def456",
      "agents_used": ["planner", "coder", "critic"],
      "execution_time": 2.45,
      "confidence": 0.85
    }

Observability:
    Every request generates comprehensive audit data:
    - Trace snapshots saved to artifacts/orchestrator_trace_<trace_id>.json
    - Performance metrics with operation-level timing
    - Structured logs with trace IDs and request context
    - Component health monitoring and status reporting

Configuration:
    Environment variables:
    - SYN_API__HOST=0.0.0.0 (server host)
    - SYN_API__PORT=8000 (server port)
    - SYN_API__ENABLE_CORS=true (CORS support)
    - SYN_API__ENABLE_DOCS=true (API documentation)
    - SYN_SEED=1337 (deterministic behavior)
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config.container import get_container
from ..config.settings import get_settings
from ..core.determinism import ensure_deterministic_startup, get_config_hash
from ..observability.logging import get_logger, set_trace_id
from ..observability.tracing import get_trace_id
from .auth import get_auth_manager, RateLimitTier, require_auth, UserRole, add_rate_limit_headers
from ..observability.probe import probe

logger = get_logger(__name__)

# Global instances
orchestrator = None
container = None


# Authentication dependency
async def get_current_user(request: Request):
    """Get current authenticated user from request."""
    auth_manager = get_auth_manager()
    if not auth_manager:
        raise HTTPException(status_code=503, detail="Authentication not initialized")
    
    try:
        api_key, tier = await auth_manager.authenticate_request(request)
        return {"api_key": api_key, "tier": tier}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., min_length=1, description="The query to process")
    context: dict[str, Any] | None = Field(None, description="Optional context")
    workflow: str = Field("auto", description="Workflow type (auto, development, production)")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    success: bool
    trace_id: str
    response: str | None = None
    agents_used: list[str] = []
    execution_path: list[str] = []
    confidence: float = 0.0
    execution_time: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str
    version: str
    config_hash: str
    uptime_seconds: float
    components: dict[str, str]


# Application startup/shutdown lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global container, orchestrator

    logger.info("Starting Synndicate API server...")

    # Initialize configuration and determinism
    settings = get_settings()
    seed, config_hash = ensure_deterministic_startup(settings)

    logger.info("Deterministic startup complete", seed=seed, config_hash=config_hash[:16])

    # Initialize container and orchestrator
    container = Container()
    orchestrator = Orchestrator(container)

    # Store startup time
    app.state.startup_time = time.time()
    app.state.config_hash = config_hash

    logger.info("Synndicate API server ready")

    yield

    # Cleanup
    logger.info("Shutting down Synndicate API server...")
    if orchestrator:
        await orchestrator.cleanup()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Synndicate AI",
        description="AI Orchestration System with Multi-Agent Workflows",
        version="2.0.0",
        docs_url="/docs" if settings.api.enable_docs else None,
        redoc_url="/redoc" if settings.api.enable_docs else None,
        lifespan=lifespan,
    )

    # Add CORS middleware
    if settings.api.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return app


# Create app instance
app = create_app()


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint."""
    # Add trace ID to response headers
    trace_id = get_trace_id()
    
    startup_time = getattr(app.state, "startup_time", time.time())
    uptime = max(0.0, time.time() - startup_time)

    # Check component health
    components = {
        "orchestrator": "healthy" if orchestrator else "not_initialized",
        "container": "healthy" if container else "not_initialized",
        "config": "healthy",
    }

    # Check model health if available
    try:
        if container and hasattr(container, "model_manager"):
            model_manager = container.model_manager
            if hasattr(model_manager, "health_check"):
                health_status = await model_manager.health_check()
                components["models"] = (
                    "healthy" if health_status.get("healthy", False) else "unhealthy"
                )
            else:
                components["models"] = "unknown"
        else:
            components["models"] = "not_available"
    except Exception as e:
        components["models"] = f"error: {str(e)}"

    response = HealthResponse(
        status=(
            "healthy"
            if all(
                status in ["healthy", "not_available", "unknown"] for status in components.values()
            )
            else "unhealthy"
        ),
        version="2.0.0",
        config_hash=getattr(app.state, "config_hash", get_config_hash()),
        uptime_seconds=uptime,
        components=components,
    )
    
    # Add observability headers
    from fastapi import Response
    resp = Response(content=response.model_dump_json(), media_type="application/json")
    resp.headers["x-request-id"] = trace_id
    resp.headers["trace-id"] = trace_id
    return resp


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, http_request: Request):
    """Process a query through the orchestrator."""
    # Handle authentication first - this must succeed before proceeding
    auth_manager = get_auth_manager()
    if auth_manager and getattr(auth_manager.config, 'require_api_key', True):
        try:
            api_key, tier = await auth_manager.authenticate_request(http_request)
            # Check if authentication actually succeeded
            if not api_key or tier == RateLimitTier.ANONYMOUS:
                raise HTTPException(status_code=401, detail="API key required")
            
            # Check rate limiting after successful authentication
            is_limited, limit_info = auth_manager.rate_limiter.is_rate_limited(http_request)
            if is_limited:
                error_detail = limit_info.get("error", "Rate limit exceeded")
                retry_after = limit_info.get("retry_after", 60)
                raise HTTPException(
                    status_code=429, 
                    detail=error_detail,
                    headers={"Retry-After": str(retry_after)}
                )
        except HTTPException as e:
            # Re-raise authentication errors immediately
            raise e
        except Exception as e:
            # Handle any other authentication errors
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    # Get orchestrator (try container first, then global)
    current_orchestrator = orchestrator
    if not current_orchestrator:
        try:
            container_instance = get_container()
            if container_instance and hasattr(container_instance, 'get_orchestrator'):
                current_orchestrator = container_instance.get_orchestrator()
        except Exception:
            pass
    
    if not current_orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    # Generate trace ID for this request
    trace_id = f"{int(time.time() * 1000):x}{hash(request.query) & 0xFFFF:04x}"
    set_trace_id(trace_id)

    start_time = time.time()

    with probe("api.process_query", trace_id):
        logger.info("Processing API query", trace_id=trace_id, query_length=len(request.query))

        try:
            # Process query through orchestrator
            result = await current_orchestrator.process_query(
                query=request.query, context=request.context, workflow=request.workflow
            )

            execution_time = time.time() - start_time

            # Create response
            response = QueryResponse(
                success=result.success,
                trace_id=trace_id,
                response=result.final_response,
                agents_used=result.agents_used,
                execution_path=result.execution_path,
                confidence=result.confidence,
                execution_time=execution_time,
                metadata=result.metadata,
            )

            logger.info(
                "API query processed successfully",
                trace_id=trace_id,
                execution_time=execution_time,
                success=result.success,
            )

            return response

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("API query failed", trace_id=trace_id, error=str(e))

            # Create error response
            response = QueryResponse(
                success=False,
                trace_id=trace_id,
                response=None,
                agents_used=[],
                execution_path=["api_error"],
                confidence=0.0,
                execution_time=execution_time,
                metadata={"error": str(e)},
            )

            return response


@app.get("/metrics")
async def get_metrics(request: Request):
    """Get system metrics (placeholder for Prometheus integration)."""
    # Handle authentication for admin-only endpoint
    auth_manager = get_auth_manager()
    if auth_manager and auth_manager.config.require_api_key:
        try:
            api_key, tier = await auth_manager.authenticate_request(request)
            # Check if user has admin role for metrics access
            if tier != "admin":
                raise HTTPException(status_code=401, detail="Admin access required")
        except HTTPException as e:
            if e.status_code == 401:
                raise HTTPException(status_code=401, detail="Authentication required")
            raise
    
    return {
        "message": "Metrics endpoint - integrate with Prometheus for production",
        "status": "placeholder",
        "timestamp": time.time(),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled API exception: {exc}", path=request.url.path)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if app.debug else "An unexpected error occurred",
            "trace_id": getattr(request.state, "trace_id", "unknown"),
        },
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "synndicate.api.server:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers if not settings.api.reload else 1,
    )
