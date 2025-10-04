"""
Global pytest configuration and fixtures for test isolation.

This module provides pytest fixtures and configuration to prevent test interaction
issues and global state contamination between tests.
"""

import asyncio
import os
import random
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


def reset_all_global_state():
    """Completely reset all global state and reseed everything."""
    # Reset random seeds to ensure deterministic testing
    random.seed(1337)
    np.random.seed(1337)
    os.environ["PYTHONHASHSEED"] = "1337"

    # Use the new reset helpers for targeted cleanup
    try:
        from synndicate.api.server import _reset_globals_for_tests

        _reset_globals_for_tests()
    except ImportError:
        pass

    try:
        from synndicate.core.determinism import _reset_config_hash_for_tests

        _reset_config_hash_for_tests()
    except ImportError:
        pass

    # Reset all global variables in critical modules (fallback)
    modules_to_reset = [
        "synndicate.api.server",
        "synndicate.api.auth",
        "synndicate.core.determinism",
        "synndicate.observability.tracing",
        "synndicate.observability.distributed_tracing",
        "synndicate.observability.metrics",
    ]

    for module_name in modules_to_reset:
        if module_name in sys.modules:
            module = sys.modules[module_name]

            # Reset specific global variables
            global_vars_to_reset = {
                "orchestrator": None,
                "container": None,
                "CONFIG_SHA256": "",  # Use empty string instead of None
                "_auth_manager": None,
                "_tracing_manager": None,
                "_distributed_tracing_manager": None,
                "_metrics_collector": None,
            }

            for var_name, reset_value in global_vars_to_reset.items():
                if hasattr(module, var_name):
                    setattr(module, var_name, reset_value)

    # Force module reload for API/core modules to prevent cached app state
    for module_name in list(sys.modules.keys()):
        if (module_name.startswith("synndicate.api.") or module_name.startswith("synndicate.core.")) and module_name in sys.modules:
            # Don't remove the module, just reset its state
            pass


@pytest.fixture(scope="session", autouse=True)
def session_setup():
    """Session-level setup that resets all global state once at the beginning."""
    print("\n🔄 Resetting all global state and reseeding for test session...")
    reset_all_global_state()
    yield
    print("\n✅ Test session cleanup complete")


@pytest.fixture(autouse=True)
def test_isolation():
    """Per-test isolation to ensure clean state for each test."""
    # Quick reset before each test
    reset_all_global_state()
    yield
    # No cleanup needed - next test will reset


@pytest.fixture
def clean_config_hash():
    """Fixture that ensures CONFIG_SHA256 is reset to a clean state."""
    with patch("synndicate.core.determinism.CONFIG_SHA256", None):
        yield


@pytest.fixture
def mock_orchestrator():
    """Fixture that provides a clean mock orchestrator."""
    mock = AsyncMock()
    mock.__bool__ = lambda self: True
    mock.health_check = AsyncMock(return_value=None)

    with patch("synndicate.api.server.orchestrator", mock):
        yield mock


@pytest.fixture
def mock_container():
    """Fixture that provides a clean mock container."""
    mock = MagicMock()
    mock.startup_time = 900.0

    # Mock model manager
    mock_model_manager = AsyncMock()
    mock_model_manager.health_check.return_value = {"healthy": True}
    mock.model_manager = mock_model_manager

    with patch("synndicate.api.server.container", mock):
        yield mock


@pytest.fixture
def isolated_api_server():
    """Fixture that provides an isolated API server environment."""
    with (
        patch("synndicate.api.server.orchestrator", None),
        patch("synndicate.api.server.container", None),
        patch("synndicate.core.determinism.CONFIG_SHA256", None),
    ):
        yield


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test function."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Suppress specific warnings that are expected during testing
@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress expected warnings during testing."""
    import warnings

    # Suppress pkg_resources deprecation warning
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

    # Suppress other test-related warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="coroutine.*was never awaited"
    )

    yield
