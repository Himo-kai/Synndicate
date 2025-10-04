"""
Artifact storage system with pluggable backends.
Supports local filesystem and S3-compatible storage.
"""

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..observability.logging import get_logger

log = get_logger("syn.storage")


@dataclass
class ArtifactRef:
    """Reference to a stored artifact."""

    uri: str  # e.g. "file:///path/to/file" or "s3://bucket/key"
    backend: str  # "local" or "s3"
    size_bytes: int | None = None
    content_type: str | None = None


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

        path.write_text(text, encoding="utf-8")

        log.debug("Saved text artifact", path=relpath, size=len(text))

        return ArtifactRef(
            uri=f"file://{path.resolve()}",
            backend="local",
            size_bytes=len(text.encode("utf-8")),
            content_type="text/plain",
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
            content_type="application/octet-stream",
        )

    def read_text(self, relpath: str) -> str:
        """Read text content from local file."""
        path = self._path(relpath)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {relpath}")

        return path.read_text(encoding="utf-8")

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
    """S3-compatible artifact storage with full implementation."""

    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1", **kwargs):
        """Initialize S3 store with boto3 client.

        Args:
            bucket: S3 bucket name
            prefix: Optional prefix for all keys
            region: AWS region (default: us-east-1)
            **kwargs: Additional boto3 client parameters (endpoint_url, etc.)
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region = region

        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError

            # Store exceptions for later use
            self.ClientError = ClientError
            self.NoCredentialsError = NoCredentialsError

            # Initialize S3 client with provided parameters
            client_kwargs = {"region_name": region}
            client_kwargs.update(kwargs)

            self.s3_client = boto3.client("s3", **client_kwargs)

            # Test connection by checking if bucket exists
            try:
                self.s3_client.head_bucket(Bucket=bucket)
                log.info(
                    "S3 artifact store initialized", bucket=bucket, prefix=prefix, region=region
                )
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "404":
                    log.warning(
                        "S3 bucket does not exist, will attempt to create on first write",
                        bucket=bucket,
                    )
                else:
                    log.error("Failed to access S3 bucket", bucket=bucket, error=str(e))
                    raise

        except ImportError:
            log.error("boto3 not installed. Install with: pip install boto3")
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
        except NoCredentialsError:
            log.error("AWS credentials not found. Configure with AWS CLI or environment variables")
            raise

    def _key(self, relpath: str) -> str:
        """Convert relative path to S3 key with prefix."""
        if self.prefix:
            return f"{self.prefix}/{relpath.lstrip('/')}"
        return relpath.lstrip("/")

    def _ensure_bucket_exists(self):
        """Ensure S3 bucket exists, create if necessary."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except self.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                try:
                    if self.region == "us-east-1":
                        # us-east-1 doesn't need LocationConstraint
                        self.s3_client.create_bucket(Bucket=self.bucket)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket,
                            CreateBucketConfiguration={"LocationConstraint": self.region},
                        )
                    log.info("Created S3 bucket", bucket=self.bucket)
                except self.ClientError as create_error:
                    log.error(
                        "Failed to create S3 bucket", bucket=self.bucket, error=str(create_error)
                    )
                    raise
            else:
                raise

    def save_text(self, relpath: str, text: str) -> ArtifactRef:
        """Save text content to S3."""
        self._ensure_bucket_exists()
        key = self._key(relpath)

        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=text.encode("utf-8"),
                ContentType="text/plain",
                Metadata={"synndicate-type": "text"},
            )

            size_bytes = len(text.encode("utf-8"))
            log.debug("Saved text artifact to S3", key=key, size=size_bytes)

            return ArtifactRef(
                uri=f"s3://{self.bucket}/{key}",
                backend="s3",
                size_bytes=size_bytes,
                content_type="text/plain",
            )
        except self.ClientError as e:
            log.error("Failed to save text to S3", key=key, error=str(e))
            raise

    def save_json(self, relpath: str, obj: Any) -> ArtifactRef:
        """Save JSON object to S3."""
        self._ensure_bucket_exists()
        key = self._key(relpath)

        try:
            json_text = json.dumps(obj, indent=2, ensure_ascii=False)

            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json_text.encode("utf-8"),
                ContentType="application/json",
                Metadata={"synndicate-type": "json"},
            )

            size_bytes = len(json_text.encode("utf-8"))
            log.debug("Saved JSON artifact to S3", key=key, size=size_bytes)

            return ArtifactRef(
                uri=f"s3://{self.bucket}/{key}",
                backend="s3",
                size_bytes=size_bytes,
                content_type="application/json",
            )
        except self.ClientError as e:
            log.error("Failed to save JSON to S3", key=key, error=str(e))
            raise

    def save_blob(self, relpath: str, src_path: Path) -> ArtifactRef:
        """Save binary file to S3."""
        self._ensure_bucket_exists()
        key = self._key(relpath)

        try:
            # Determine content type from file extension
            import mimetypes

            content_type, _ = mimetypes.guess_type(str(src_path))
            if not content_type:
                content_type = "application/octet-stream"

            # Upload file
            self.s3_client.upload_file(
                str(src_path),
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type, "Metadata": {"synndicate-type": "blob"}},
            )

            size_bytes = src_path.stat().st_size
            log.debug("Saved blob artifact to S3", key=key, size=size_bytes)

            return ArtifactRef(
                uri=f"s3://{self.bucket}/{key}",
                backend="s3",
                size_bytes=size_bytes,
                content_type=content_type,
            )
        except self.ClientError as e:
            log.error("Failed to save blob to S3", key=key, error=str(e))
            raise

    def read_text(self, relpath: str) -> str:
        """Read text content from S3."""
        key = self._key(relpath)

        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            content = response["Body"].read().decode("utf-8")
            log.debug("Read text artifact from S3", key=key, size=len(content))
            return content
        except self.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                log.warning("Text artifact not found in S3", key=key)
                raise FileNotFoundError(f"Artifact not found: {relpath}")
            log.error("Failed to read text from S3", key=key, error=str(e))
            raise

    def read_json(self, relpath: str) -> Any:
        """Read JSON object from S3."""
        text_content = self.read_text(relpath)
        try:
            return json.loads(text_content)
        except json.JSONDecodeError as e:
            log.error("Failed to parse JSON from S3", key=self._key(relpath), error=str(e))
            raise

    def exists(self, relpath: str) -> bool:
        """Check if artifact exists in S3."""
        key = self._key(relpath)

        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            log.error("Failed to check S3 object existence", key=key, error=str(e))
            raise

    def delete(self, relpath: str) -> bool:
        """Delete artifact from S3."""
        key = self._key(relpath)

        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=key)
            log.debug("Deleted artifact from S3", key=key)
            return True
        except self.ClientError as e:
            log.error("Failed to delete from S3", key=key, error=str(e))
            return False

    def list_artifacts(self, prefix: str = "") -> list[str]:
        """List all artifacts in S3 with optional prefix filter."""
        # Combine store prefix with search prefix
        full_prefix = self._key(prefix) if prefix else (self.prefix + "/" if self.prefix else "")

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=full_prefix)

            artifacts = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        # Remove store prefix to get relative path
                        if self.prefix and key.startswith(self.prefix + "/"):
                            relpath = key[len(self.prefix) + 1 :]
                        else:
                            relpath = key
                        artifacts.append(relpath)

            log.debug("Listed S3 artifacts", count=len(artifacts), prefix=full_prefix)
            return artifacts

        except self.ClientError as e:
            log.error("Failed to list S3 artifacts", prefix=full_prefix, error=str(e))
            raise


# Global artifact store instance
_artifact_store: ArtifactStore | None = None


def get_artifact_store() -> ArtifactStore:
    """Get global artifact store instance."""
    global _artifact_store

    if _artifact_store is None:
        # Use default local storage for now
        artifacts_root = Path("./artifacts")
        _artifact_store = LocalStore(artifacts_root)

    return _artifact_store


def save_trace_snapshot(trace_id: str, snapshot: dict[str, Any]) -> ArtifactRef:
    """Save trace snapshot to artifacts."""
    store = get_artifact_store()
    return store.save_json(f"traces/trace_{trace_id}.json", snapshot)


def save_performance_data(trace_id: str, perf_data: list[dict[str, Any]]) -> ArtifactRef:
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


def save_dependency_snapshot(deps: list[dict[str, Any]]) -> ArtifactRef:
    """Save dependency snapshot."""
    return save_audit_data("pip_freeze.json", deps)
