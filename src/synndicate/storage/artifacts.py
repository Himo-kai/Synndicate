"""
Artifact storage system with pluggable backends.
Supports local filesystem and S3-compatible storage.
"""

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..observability.logging import get_logger

log = get_logger("syn.storage")


@dataclass
class ArtifactRef:
    """Reference to a stored artifact."""
    uri: str  # e.g. "file:///path/to/file" or "s3://bucket/key"
    backend: str  # "local" or "s3"
    size_bytes: Optional[int] = None
    content_type: Optional[str] = None


class ArtifactStore(ABC):
    """Abstract interface for artifact storage."""
    
    @abstractmethod
    def save_text(self, relpath: str, text: str) -> ArtifactRef:
        """Save text content to storage."""
        pass
    
    @abstractmethod
    def save_json(self, relpath: str, obj: Any) -> ArtifactRef:
        """Save JSON object to storage."""
        pass
    
    @abstractmethod
    def save_blob(self, relpath: str, src_path: Path) -> ArtifactRef:
        """Save binary file to storage."""
        pass
    
    @abstractmethod
    def read_text(self, relpath: str) -> str:
        """Read text content from storage."""
        pass
    
    @abstractmethod
    def read_json(self, relpath: str) -> Any:
        """Read JSON object from storage."""
        pass
    
    @abstractmethod
    def exists(self, relpath: str) -> bool:
        """Check if artifact exists."""
        pass
    
    @abstractmethod
    def delete(self, relpath: str) -> bool:
        """Delete artifact from storage."""
        pass
    
    @abstractmethod
    def list_artifacts(self, prefix: str = "") -> list[str]:
        """List all artifacts with optional prefix filter."""
        pass


class LocalStore(ArtifactStore):
    """Local filesystem artifact storage."""
    
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        log.info("Local artifact store initialized", root=str(self.root))
    
    def _path(self, relpath: str) -> Path:
        """Convert relative path to absolute path."""
        return self.root / relpath
    
    def save_text(self, relpath: str, text: str) -> ArtifactRef:
        """Save text content to local file."""
        path = self._path(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        path.write_text(text, encoding='utf-8')
        
        log.debug("Saved text artifact", path=relpath, size=len(text))
        
        return ArtifactRef(
            uri=f"file://{path.resolve()}",
            backend="local",
            size_bytes=len(text.encode('utf-8')),
            content_type="text/plain"
        )
    
    def save_json(self, relpath: str, obj: Any) -> ArtifactRef:
        """Save JSON object to local file."""
        text = json.dumps(obj, indent=2, default=str)
        ref = self.save_text(relpath, text)
        ref.content_type = "application/json"
        return ref
    
    def save_blob(self, relpath: str, src_path: Path) -> ArtifactRef:
        """Save binary file to local storage."""
        path = self._path(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src_path, path)
        
        size = path.stat().st_size
        log.debug("Saved blob artifact", path=relpath, size=size)
        
        return ArtifactRef(
            uri=f"file://{path.resolve()}",
            backend="local",
            size_bytes=size,
            content_type="application/octet-stream"
        )
    
    def read_text(self, relpath: str) -> str:
        """Read text content from local file."""
        path = self._path(relpath)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {relpath}")
        
        return path.read_text(encoding='utf-8')
    
    def read_json(self, relpath: str) -> Any:
        """Read JSON object from local file."""
        text = self.read_text(relpath)
        return json.loads(text)
    
    def exists(self, relpath: str) -> bool:
        """Check if local file exists."""
        return self._path(relpath).exists()
    
    def delete(self, relpath: str) -> bool:
        """Delete local file."""
        path = self._path(relpath)
        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
            log.debug("Deleted artifact", path=relpath)
            return True
        return False
    
    def list_artifacts(self, prefix: str = "") -> list[str]:
        """List all local artifacts with optional prefix filter."""
        artifacts = []
        search_path = self.root / prefix if prefix else self.root
        
        if search_path.exists():
            for path in search_path.rglob("*"):
                if path.is_file():
                    relpath = path.relative_to(self.root)
                    artifacts.append(str(relpath))
        
        return sorted(artifacts)


class S3Store(ArtifactStore):
    """S3-compatible artifact storage (placeholder for future implementation)."""
    
    def __init__(self, bucket: str, prefix: str = "", **kwargs):
        self.bucket = bucket
        self.prefix = prefix
        log.info("S3 artifact store initialized", bucket=bucket, prefix=prefix)
        
        # TODO: Initialize S3 client
        raise NotImplementedError("S3 storage not yet implemented")
    
    def save_text(self, relpath: str, text: str) -> ArtifactRef:
        raise NotImplementedError("S3 storage not yet implemented")
    
    def save_json(self, relpath: str, obj: Any) -> ArtifactRef:
        raise NotImplementedError("S3 storage not yet implemented")
    
    def save_blob(self, relpath: str, src_path: Path) -> ArtifactRef:
        raise NotImplementedError("S3 storage not yet implemented")
    
    def read_text(self, relpath: str) -> str:
        raise NotImplementedError("S3 storage not yet implemented")
    
    def read_json(self, relpath: str) -> Any:
        raise NotImplementedError("S3 storage not yet implemented")
    
    def exists(self, relpath: str) -> bool:
        raise NotImplementedError("S3 storage not yet implemented")
    
    def delete(self, relpath: str) -> bool:
        raise NotImplementedError("S3 storage not yet implemented")
    
    def list_artifacts(self, prefix: str = "") -> list[str]:
        raise NotImplementedError("S3 storage not yet implemented")


# Global artifact store instance
_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Get global artifact store instance."""
    global _artifact_store
    
    if _artifact_store is None:
        # Use default local storage for now
        artifacts_root = Path("./artifacts")
        _artifact_store = LocalStore(artifacts_root)
    
    return _artifact_store


def save_trace_snapshot(trace_id: str, snapshot: Dict[str, Any]) -> ArtifactRef:
    """Save trace snapshot to artifacts."""
    store = get_artifact_store()
    return store.save_json(f"traces/trace_{trace_id}.json", snapshot)


def save_performance_data(trace_id: str, perf_data: list[Dict[str, Any]]) -> ArtifactRef:
    """Save performance data to artifacts."""
    store = get_artifact_store()
    
    # Convert to JSONL format
    jsonl_lines = [json.dumps(entry, default=str) for entry in perf_data]
    jsonl_content = "\n".join(jsonl_lines)
    
    return store.save_text(f"perf/perf_{trace_id}.jsonl", jsonl_content)


def save_audit_data(filename: str, data: Any) -> ArtifactRef:
    """Save audit data (coverage, linting, etc.)."""
    store = get_artifact_store()
    
    if isinstance(data, (dict, list)):
        return store.save_json(f"audit/{filename}", data)
    else:
        return store.save_text(f"audit/{filename}", str(data))


# Convenience functions for common artifact operations
def save_coverage_report(coverage_data: str) -> ArtifactRef:
    """Save test coverage report."""
    return save_audit_data("coverage.xml", coverage_data)


def save_lint_report(lint_data: str) -> ArtifactRef:
    """Save linting report."""
    return save_audit_data("ruff.txt", lint_data)


def save_dependency_snapshot(deps: list[Dict[str, Any]]) -> ArtifactRef:
    """Save dependency snapshot."""
    return save_audit_data("pip_freeze.json", deps)
