"""
Determinism utilities for reproducible behavior.
"""

import hashlib
import json
import os
import random
from typing import Any

import numpy as np

from ..observability.logging import get_logger

logger = get_logger(__name__)

# Global config hash for audit trail - RESET FOR FRESH START
CONFIG_SHA256: str = ""


def _reset_config_hash_for_tests():
    """Reset config hash for test isolation."""
    global CONFIG_SHA256
    CONFIG_SHA256 = ""


def seed_everything(seed: int | None = None) -> int:
    """Seed all random number generators for deterministic behavior."""
    if seed is None:
        seed = int(os.getenv("SYN_SEED", "1337"))

    # Seed Python random
    random.seed(seed)

    # Seed NumPy
    np.random.seed(seed)

    # Set environment variable for child processes
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Seeded all RNGs with seed: {seed}")
    return seed


def freeze_config_and_hash(config: Any) -> str:
    """
    Freeze configuration at startup and generate deterministic hash.

    Args:
        config: Configuration object to hash

    Returns:
        SHA256 hash of the configuration
    """
    global CONFIG_SHA256

    try:
        # Convert config to dict recursively
        def to_dict(obj):
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            elif hasattr(obj, "dict"):
                return obj.dict()
            elif hasattr(obj, "model_dump"):
                return obj.model_dump()
            else:
                return str(obj)

        # Create deterministic JSON blob
        blob = json.dumps(config, default=to_dict, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )

        # Generate SHA256 hash
        CONFIG_SHA256 = hashlib.sha256(blob).hexdigest()

        logger.info(f"CONFIG_SHA256 {CONFIG_SHA256}")
        print(f"CONFIG_SHA256 {CONFIG_SHA256}")

        return CONFIG_SHA256

    except Exception as e:
        logger.error(f"Failed to hash config: {e}")
        CONFIG_SHA256 = "ERROR_HASHING_CONFIG"
        return CONFIG_SHA256


def get_config_hash() -> str:
    """Get the current config hash."""
    return CONFIG_SHA256


def ensure_deterministic_startup(config: Any) -> tuple[int, str]:
    """
    Ensure deterministic startup by seeding RNGs and hashing config.

    Args:
        config: Configuration object

    Returns:
        Tuple of (seed, config_hash)
    """
    seed = seed_everything()
    config_hash = freeze_config_and_hash(config)

    logger.info("Deterministic startup complete", seed=seed, config_hash=config_hash[:16] + "...")

    return seed, config_hash
