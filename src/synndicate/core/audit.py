"""
Audit trail and snapshot generation for observability.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from ..observability.logging import get_logger
# Import will be done locally to avoid circular imports
from .determinism import get_config_hash

logger = get_logger(__name__)


def create_trace_snapshot(
    trace_id: str,
    query: str,
    context_keys: List[str] | None = None,
    agents_used: List[str] | None = None,
    execution_path: List[str] | None = None,
    confidence: float | None = None,
    success: bool = True,
    additional_data: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Create a comprehensive trace snapshot for audit purposes.
    
    Args:
        trace_id: Unique trace identifier
        query: Original query string
        context_keys: Keys from request context
        agents_used: List of agents that processed the request
        execution_path: Execution path through the system
        confidence: Final confidence score
        success: Whether the request succeeded
        additional_data: Any additional data to include
        
    Returns:
        Complete snapshot dictionary
    """
    # Get performance metrics for this trace
    timings_dict = {}
    try:
        from ..observability.probe import get_trace_metrics
        metrics = get_trace_metrics(trace_id)
        timings_dict = {
            op: {
                "duration_ms": data["duration_ms"],
                "success": data["success"],
                "error": data.get("error")
            }
            for op, data in metrics.items()
        }
    except Exception as e:
        logger.warning(f"Failed to get trace metrics: {e}")
    
    # Create comprehensive snapshot
    snapshot = {
        "trace_id": trace_id,
        "query": query,
        "context_keys": context_keys or [],
        "agents_used": agents_used or [],
        "execution_path": execution_path or [],
        "confidence": confidence,
        "success": success,
        "config_sha256": get_config_hash(),
        "timings_ms": timings_dict,
        "metadata": {
            "total_operations": len(timings_dict),
            "total_duration_ms": sum(
                data["duration_ms"] for data in timings_dict.values()
            ),
            "failed_operations": sum(
                1 for data in timings_dict.values() if not data["success"]
            )
        }
    }
    
    # Add any additional data
    if additional_data:
        snapshot.update(additional_data)
    
    return snapshot


def save_trace_snapshot(snapshot: Dict[str, Any], artifacts_dir: Path | str = "artifacts") -> Path:
    """
    Save trace snapshot to artifacts directory.
    
    Args:
        snapshot: Snapshot dictionary from create_trace_snapshot
        artifacts_dir: Directory to save artifacts
        
    Returns:
        Path to saved snapshot file
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)
    
    trace_id = snapshot["trace_id"]
    snapshot_file = artifacts_path / f"orchestrator_trace_{trace_id}.json"
    
    try:
        snapshot_file.write_text(json.dumps(snapshot, indent=2))
        logger.info(f"Saved trace snapshot: {snapshot_file}")
        return snapshot_file
    except Exception as e:
        logger.error(f"Failed to save trace snapshot: {e}")
        raise


def save_performance_data(trace_id: str, perf_data: Dict[str, Any], artifacts_dir: Path | str = "artifacts") -> Path:
    """
    Save performance data in JSONL format.
    
    Args:
        trace_id: Trace identifier
        perf_data: Performance data dictionary
        artifacts_dir: Directory to save artifacts
        
    Returns:
        Path to saved performance file
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)
    
    perf_file = artifacts_path / f"perf_{trace_id}.jsonl"
    
    try:
        # Append to JSONL file (one JSON object per line)
        with perf_file.open("a") as f:
            f.write(json.dumps(perf_data) + "\n")
        
        logger.info(f"Saved performance data: {perf_file}")
        return perf_file
    except Exception as e:
        logger.error(f"Failed to save performance data: {e}")
        raise


def create_audit_bundle(
    trace_snapshots: List[Dict[str, Any]],
    output_dir: Path | str = "synndicate_audit"
) -> Path:
    """
    Create complete audit bundle with all required files.
    
    Args:
        trace_snapshots: List of trace snapshots to include
        output_dir: Output directory for audit bundle
        
    Returns:
        Path to audit bundle directory
    """
    bundle_path = Path(output_dir)
    bundle_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    (bundle_path / "configs").mkdir(exist_ok=True)
    (bundle_path / "artifacts").mkdir(exist_ok=True)
    (bundle_path / "logs").mkdir(exist_ok=True)
    (bundle_path / "endpoints").mkdir(exist_ok=True)
    
    logger.info(f"Created audit bundle structure: {bundle_path}")
    
    # Save trace snapshots
    for snapshot in trace_snapshots:
        trace_file = bundle_path / "artifacts" / f"orchestrator_trace_{snapshot['trace_id']}.json"
        trace_file.write_text(json.dumps(snapshot, indent=2))
    
    logger.info(f"Audit bundle ready: {bundle_path}")
    return bundle_path
