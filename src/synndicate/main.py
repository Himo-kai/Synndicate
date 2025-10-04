"""
Main entry point for Synndicate AI with deterministic startup.
"""

import argparse
import sys

try:
    import uvicorn
except ImportError:
    uvicorn = None

from .config.settings import get_settings
from .core.determinism import ensure_deterministic_startup
from .observability.distributed_tracing import (DistributedTracingManager,
                                                TracingBackend)
from .observability.logging import get_logger
from .observability.tracing import TracingManager

logger = get_logger(__name__)


def main(argv=None):
    """Main application entry point with server startup."""
    parser = argparse.ArgumentParser(description="Synndicate AI Server")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--help-extended", action="store_true", help="Show extended help")
    parser.add_argument("--version", action="store_true", help="Show version")

    # Use empty list if no argv provided to avoid pytest argument conflicts
    if argv is None:
        argv = []
    args = parser.parse_args(argv)

    if args.version:
        print("Synndicate AI v2.0.0")
        return

    if args.help_extended:
        parser.print_help()
        return

    # Get settings and apply overrides
    settings = get_settings()

    # Ensure deterministic startup
    seed, config_hash = ensure_deterministic_startup(settings)

    # Initialize distributed tracing
    tracing_manager = None
    if settings.observability.enable_tracing:
        try:
            # Create distributed tracing manager
            distributed_manager = DistributedTracingManager(
                backend=TracingBackend(settings.observability.tracing_backend),
                protocol=settings.observability.tracing_protocol,
                endpoint=settings.observability.tracing_endpoint,
                sample_rate=settings.observability.tracing_sample_rate,
                batch_timeout=settings.observability.tracing_batch_timeout,
                max_batch_size=settings.observability.tracing_max_batch_size,
                max_queue_size=settings.observability.tracing_max_queue_size,
                enable_health_check=settings.observability.tracing_health_check,
                health_check_interval=settings.observability.tracing_health_check_interval,
            )

            # Create tracing manager with distributed backend
            tracing_manager = TracingManager(
                service_name=settings.observability.service_name,
                service_version=settings.observability.service_version,
                distributed_manager=distributed_manager,
            )

            # Initialize tracing
            tracing_manager.initialize(otlp_endpoint=settings.observability.otlp_endpoint)

            logger.info(
                "Distributed tracing initialized",
                backend=settings.observability.tracing_backend,
                protocol=settings.observability.tracing_protocol,
                sample_rate=settings.observability.tracing_sample_rate,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize distributed tracing: {e}")

    logger.info(
        "Synndicate AI initialized",
        seed=seed,
        config_hash=config_hash[:16] + "...",
        environment=settings.environment,
        tracing_enabled=settings.observability.enable_tracing,
    )

    if uvicorn is None:
        raise ImportError("uvicorn is required to run the server")

    # Start the server
    try:
        uvicorn.run(
            "synndicate.api.server:app",
            host=args.host or settings.api.host,
            port=args.port or settings.api.port,
            reload=args.reload or settings.api.reload,
            workers=args.workers or settings.api.workers,
        )
    finally:
        # Cleanup tracing on shutdown
        if tracing_manager:
            try:
                tracing_manager.shutdown()
                logger.info("Distributed tracing shutdown complete")
            except Exception as e:
                logger.warning(f"Error during tracing shutdown: {e}")


def cli_main():
    """CLI entry point."""
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n👋 Synndicate AI shutdown")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        print(f"❌ Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
