"""
Test isolation utilities to prevent global state contamination between tests.

This module provides utilities to clean up global state and prevent Mock objects
from persisting between test runs, which can cause formatting errors in f-strings.
"""

import sys
from typing import Any
from unittest.mock import Mock


class TestIsolation:
    """Utility class for managing test isolation and cleanup."""

    def __init__(self) -> None:
        self._original_globals: dict[str, dict[str, Any]] = {}
        self._modules_to_clean = [
            "synndicate.api.server",
            "synndicate.api.auth",
            "synndicate.observability.tracing",
            "synndicate.observability.distributed_tracing",
            "synndicate.observability.metrics",
            "synndicate.storage.artifacts",
            "synndicate.core.determinism",
        ]

    def save_global_state(self) -> None:
        """Save the current global state of critical modules."""
        for module_name in self._modules_to_clean:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                self._original_globals[module_name] = {}

                # Save global variables that are commonly mocked
                global_vars = [
                    "orchestrator",
                    "container",
                    "_auth_manager",
                    "_tracing_manager",
                    "_distributed_tracing_manager",
                    "_metrics_collector",
                    "_artifact_store",
                    "CONFIG_SHA256",
                ]

                for var_name in global_vars:
                    if hasattr(module, var_name):
                        self._original_globals[module_name][var_name] = getattr(module, var_name)

    def restore_global_state(self) -> None:
        """Restore the original global state of critical modules."""
        for module_name, saved_globals in self._original_globals.items():
            if module_name in sys.modules:
                module = sys.modules[module_name]
                for var_name, original_value in saved_globals.items():
                    setattr(module, var_name, original_value)

        self._original_globals.clear()

    def cleanup_mock_objects(self) -> None:
        """Clean up any Mock objects that might be persisting in global state."""
        for module_name in self._modules_to_clean:
            if module_name in sys.modules:
                module = sys.modules[module_name]

                # Check for Mock objects in module globals and replace with None
                for attr_name in dir(module):
                    if not attr_name.startswith("_"):
                        attr_value = getattr(module, attr_name, None)
                        if isinstance(attr_value, Mock):
                            setattr(module, attr_name, None)

    def reset_config_hash(self) -> None:
        """Reset the CONFIG_SHA256 global variable to prevent contamination."""
        if "synndicate.core.determinism" in sys.modules:
            determinism_module = sys.modules["synndicate.core.determinism"]
            # Reset to a clean state - this will be properly set by tests that need it
            determinism_module.CONFIG_SHA256 = None


# Global test isolation instance
_test_isolation = TestIsolation()


def setup_test_isolation():
    """Set up test isolation by saving current global state."""
    _test_isolation.save_global_state()


def cleanup_test_isolation():
    """Clean up test isolation by restoring original global state."""
    _test_isolation.cleanup_mock_objects()
    _test_isolation.reset_config_hash()
    _test_isolation.restore_global_state()


def safe_str_format(obj: Any) -> str:
    """Safely format an object for use in f-strings, handling Mock objects."""
    if isinstance(obj, Mock):
        return f"Mock({obj._mock_name or 'object'})"
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        try:
            return str([safe_str_format(item) for item in obj])
        except (TypeError, AttributeError):
            return str(obj)
    else:
        return str(obj)


def reset_global_state():
    """Reset all global state to clean values for test isolation."""
    # Reset API server globals
    if "synndicate.api.server" in sys.modules:
        server_module = sys.modules["synndicate.api.server"]
        server_module.orchestrator = None
        server_module.container = None

    # Reset determinism globals
    if "synndicate.core.determinism" in sys.modules:
        determinism_module = sys.modules["synndicate.core.determinism"]
        determinism_module.CONFIG_SHA256 = None

    # Reset any other global state that might cause issues
    cleanup_test_isolation()
