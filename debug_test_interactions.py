#!/usr/bin/env python3
"""
Comprehensive Test Interaction Diagnostic Tool

This script systematically investigates all potential root causes of
persistent test interaction issues that survive complete project resets.
"""

import asyncio
import gc
import inspect
import os
import signal
import sys
import threading
from typing import Any
from unittest import mock


def capture_system_state() -> dict[str, Any]:
    """Capture comprehensive system state snapshot."""
    state: dict[str, Any] = {}

    # 1. Test Framework Internal State
    state['pytest'] = {
        'plugins': getattr(sys, '_pytest_plugins', []),
        'fixtures': getattr(sys, '_pytest_fixtures', {}),
        'session_state': getattr(sys, '_pytest_session', None),
    }

    # 2. Import Hook Registration
    state['import_hooks'] = {
        'meta_path': [str(finder) for finder in sys.meta_path],
        'path_hooks': [str(hook) for hook in sys.path_hooks],
        'modules_count': len(sys.modules),
        'synndicate_modules': [m for m in sys.modules if 'synndicate' in m],
    }

    # 3. C Extension State
    state['c_extensions'] = {
        'numpy_loaded': 'numpy' in sys.modules,
        'torch_loaded': 'torch' in sys.modules,
        'swig_modules': [m for m in sys.modules if '_swig' in m.lower()],
    }

    if 'numpy' in sys.modules:
        import numpy as np
        state['c_extensions']['numpy_random_state'] = str(np.random.get_state()[1][:5])

    # 4. Process-Level State
    state['process'] = {
        'env_vars': {k: v for k, v in os.environ.items() if 'SYN' in k or 'PYTEST' in k},
        'signal_handlers': {sig: str(signal.signal(sig, signal.SIG_DFL)) for sig in [signal.SIGINT, signal.SIGTERM]},
        'cwd': os.getcwd(),
        'pid': os.getpid(),
    }

    # 5. Async Event Loop Contamination
    state['async'] = {
        'event_loop_running': False,
        'event_loop_policy': str(asyncio.get_event_loop_policy()),
        'running_tasks': 0,
    }

    try:
        loop = asyncio.get_running_loop()
        state['async']['event_loop_running'] = True
        state['async']['running_tasks'] = len([t for t in asyncio.all_tasks(loop) if not t.done()])
    except RuntimeError:
        pass

    # 6. Mock Patch Scope Issues
    state['mocks'] = {
        'active_patches': len(mock._patch._active_patches) if hasattr(mock._patch, '_active_patches') else 0,
        'mock_objects': len([obj for obj in gc.get_objects() if isinstance(obj, mock.Mock)]),
    }

    # 7. Metaclass Registration
    custom_metaclasses: list[str] = []
    state['metaclasses'] = {
        'custom_metaclasses': custom_metaclasses,
        'class_registries': {},
    }

    # Find custom metaclasses
    for obj in gc.get_objects():
        if inspect.isclass(obj) and type(obj) is not type:
            metaclass_name = type(obj).__name__
            if metaclass_name not in ['type', 'ABCMeta']:
                custom_metaclasses.append(f"{obj.__module__}.{obj.__name__}")

    # 8. Thread-Local Storage
    state['threading'] = {
        'active_threads': threading.active_count(),
        'thread_names': [t.name for t in threading.enumerate()],
        'local_objects': len([obj for obj in gc.get_objects() if isinstance(obj, threading.local)]),
    }

    # General GC state
    state['gc'] = {
        'total_objects': len(gc.get_objects()),
        'garbage': len(gc.garbage),
        'collections': gc.get_count(),
    }

    return state

def compare_states(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare two system states and identify differences."""
    differences = {}

    for category in before:
        if category not in after:
            differences[category] = {'removed': before[category]}
            continue

        cat_diff = {}

        if isinstance(before[category], dict) and isinstance(after[category], dict):
            for key in before[category].keys() | after[category].keys():
                before_val = before[category].get(key)
                after_val = after[category].get(key)

                if before_val != after_val:
                    cat_diff[key] = {'before': before_val, 'after': after_val}
        else:
            if before[category] != after[category]:
                cat_diff = {'before': before[category], 'after': after[category]}

        if cat_diff:
            differences[category] = cat_diff

    return differences

def print_state_summary(state: dict[str, Any], title: str):
    """Print a summary of system state."""
    print(f"\n=== {title} ===")

    # Key metrics
    print(f"Total objects: {state['gc']['total_objects']}")
    print(f"Synndicate modules: {len(state['import_hooks']['synndicate_modules'])}")
    print(f"Mock objects: {state['mocks']['mock_objects']}")
    print(f"Active threads: {state['threading']['active_threads']}")
    print(f"Event loop running: {state['async']['event_loop_running']}")

    # Show any concerning state
    if state['mocks']['active_patches'] > 0:
        print(f"⚠️  Active patches: {state['mocks']['active_patches']}")

    if state['async']['running_tasks'] > 0:
        print(f"⚠️  Running async tasks: {state['async']['running_tasks']}")

    if len(state['metaclasses']['custom_metaclasses']) > 0:
        print(f"⚠️  Custom metaclasses: {len(state['metaclasses']['custom_metaclasses'])}")

def print_differences(differences: dict[str, Any]):
    """Print significant state differences."""
    if not differences:
        print("✅ No significant state changes detected")
        return

    print("\n🔍 SIGNIFICANT STATE CHANGES:")
    for category, changes in differences.items():
        print(f"\n📂 {category.upper()}:")

        if isinstance(changes, dict):
            for key, change in changes.items():
                if isinstance(change, dict) and 'before' in change and 'after' in change:
                    print(f"  • {key}: {change['before']} → {change['after']}")
                else:
                    print(f"  • {key}: {change}")
        else:
            print(f"  • {changes}")

if __name__ == "__main__":
    import json
    import sys

    print("🔬 COMPREHENSIVE TEST INTERACTION DIAGNOSTIC")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        # Compare mode - load previous state and compare with current
        try:
            with open('/tmp/initial_state.json') as f:
                previous_state = json.load(f)

            print("📊 Capturing current system state for comparison...")
            current_state = capture_system_state()

            print_state_summary(current_state, "CURRENT STATE")
            print()

            print("🔍 COMPARING STATES...")
            print("=" * 30)
            differences = compare_states(previous_state, current_state)

            if differences:
                print_differences(differences)
            else:
                print("✅ No significant differences detected")

            # Save current state for next comparison
            with open('/tmp/current_state.json', 'w') as f:
                json.dump(current_state, f, indent=2, default=str)

        except FileNotFoundError:
            print("❌ No previous state found. Run without --compare first.")
            sys.exit(1)

    else:
        # Initial capture mode
        print("📊 Capturing initial system state...")
        print()

        initial_state = capture_system_state()
        print_state_summary(initial_state, "INITIAL STATE")

        print("📋 NEXT STEPS:")
        print("1. Run this script to capture initial state")
        print("2. Run a single passing test: pytest tests/test_dynamic_orchestration.py::TestDynamicOrchestrator::test_task_requirement_analysis -v")
        print("3. Run this script again to capture post-test state")
        print("4. Run the full test suite up to the failing test")
        print("5. Run this script again to capture post-suite state")
        print("6. Compare states to identify contamination source")
        print()

        # Save initial state
        with open('/tmp/initial_state.json', 'w') as f:
            json.dump(initial_state, f, indent=2, default=str)

        print("💾 Initial state saved to /tmp/initial_state.json")
        print("Run tests and then execute this script with --compare flag")
