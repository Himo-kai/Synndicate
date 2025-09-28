"""
Main entry point for Synndicate AI with deterministic startup.
"""

import asyncio
import sys

from .config.settings import get_settings
from .core.determinism import ensure_deterministic_startup
from .observability.logging import get_logger

logger = get_logger(__name__)


async def main():
    """Main application entry point with deterministic startup."""
    print("🚀 Starting Synndicate AI...")

    # Ensure deterministic startup
    settings = get_settings()
    seed, config_hash = ensure_deterministic_startup(settings)

    logger.info(
        "Synndicate AI initialized",
        seed=seed,
        config_hash=config_hash[:16] + "...",
        environment=settings.environment,
    )

    print("✅ Deterministic startup complete:")
    print(f"   Seed: {seed}")
    print(f"   Config Hash: {config_hash}")
    print(f"   Environment: {settings.environment}")

    # For now, just demonstrate the system is ready
    # In the future, this could start different modes (CLI, API server, etc.)
    print("\n🎯 Synndicate AI is ready!")
    print("   - Run 'make dev' to start API server")
    print("   - Run 'make audit' to generate audit bundle")
    print("   - Check artifacts/ for trace snapshots and performance data")

    return True


def cli_main():
    """CLI entry point."""
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n👋 Synndicate AI shutdown")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        print(f"❌ Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
