"""
Comprehensive test suite for storage artifacts system.
Tests LocalStore, S3Store, and convenience functions.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from synndicate.storage.artifacts import (
    ArtifactRef,
    ArtifactStore,
    LocalStore,
    S3Store,
    get_artifact_store,
    save_audit_data,
    save_coverage_report,
    save_dependency_snapshot,
    save_lint_report,
    save_performance_data,
    save_trace_snapshot,
)


class TestArtifactRef:
    """Test ArtifactRef dataclass."""

    def test_artifact_ref_creation(self):
        """Test basic ArtifactRef creation."""
        ref = ArtifactRef(
            uri="file:///tmp/test.txt",
            backend="local",
            size_bytes=1024,
            content_type="text/plain"
        )
        assert ref.uri == "file:///tmp/test.txt"
        assert ref.backend == "local"
        assert ref.size_bytes == 1024
        assert ref.content_type == "text/plain"

    def test_artifact_ref_minimal(self):
        """Test ArtifactRef with minimal parameters."""
        ref = ArtifactRef(uri="s3://bucket/key", backend="s3")
        assert ref.uri == "s3://bucket/key"
        assert ref.backend == "s3"
        assert ref.size_bytes is None
        assert ref.content_type is None


class TestLocalStore:
    """Test LocalStore implementation."""

    def test_local_store_initialization(self):
        """Test LocalStore initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStore(Path(tmpdir))
            assert store.root == Path(tmpdir)
            assert store.root.exists()

    def test_local_store_creates_directory(self):
        """Test LocalStore creates root directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_storage"
            store = LocalStore(new_dir)
            assert new_dir.exists()
            assert store.root == new_dir

    def test_save_and_read_text(self):
        """Test saving and reading text content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStore(Path(tmpdir))
            
            # Save text
            content = "Hello, World!"
            ref = store.save_text("test.txt", content)
            
            assert ref.backend == "local"
            assert ref.uri.startswith("file://")
            assert ref.size_bytes == len(content.encode())
            assert ref.content_type == "text/plain"
            
            # Read text back
            read_content = store.read_text("test.txt")
            assert read_content == content

    def test_save_and_read_json(self):
        """Test saving and reading JSON content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStore(Path(tmpdir))
            
            # Save JSON
            data = {"key": "value", "number": 42, "list": [1, 2, 3]}
            ref = store.save_json("test.json", data)
            
            assert ref.backend == "local"
            assert ref.content_type == "application/json"
            
            # Read JSON back
            read_data = store.read_json("test.json")
            assert read_data == data

    def test_save_blob(self):
        """Test saving binary files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStore(Path(tmpdir))
            
            # Create source file
            src_file = Path(tmpdir) / "source.bin"
            binary_data = b"Binary content\x00\x01\x02"
            src_file.write_bytes(binary_data)
            
            # Save blob
            ref = store.save_blob("target.bin", src_file)
            
            assert ref.backend == "local"
            assert ref.size_bytes == len(binary_data)
            assert ref.content_type == "application/octet-stream"
            
            # Verify file was copied
            target_file = store._path("target.bin")
            assert target_file.read_bytes() == binary_data

    def test_exists(self):
        """Test file existence checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStore(Path(tmpdir))
            
            # File doesn't exist initially
            assert not store.exists("nonexistent.txt")
            
            # Save file and check existence
            store.save_text("exists.txt", "content")
            assert store.exists("exists.txt")

    def test_delete(self):
        """Test file deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStore(Path(tmpdir))
            
            # Save file
            store.save_text("delete_me.txt", "content")
            assert store.exists("delete_me.txt")
            
            # Delete file
            result = store.delete("delete_me.txt")
            assert result is True
            assert not store.exists("delete_me.txt")
            
            # Try to delete non-existent file
            result = store.delete("nonexistent.txt")
            assert result is False

    def test_list_artifacts(self):
        """Test listing artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStore(Path(tmpdir))
            
            # Save multiple files
            store.save_text("file1.txt", "content1")
            store.save_text("file2.txt", "content2")
            store.save_text("subdir/file3.txt", "content3")
            
            # List all artifacts
            artifacts = store.list_artifacts()
            assert "file1.txt" in artifacts
            assert "file2.txt" in artifacts
            assert "subdir/file3.txt" in artifacts
            
            # List with prefix filter
            subdir_artifacts = store.list_artifacts("subdir/")
            assert "subdir/file3.txt" in subdir_artifacts
            assert "file1.txt" not in subdir_artifacts

    def test_nested_directories(self):
        """Test handling of nested directory structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStore(Path(tmpdir))
            
            # Save file in nested directory
            nested_path = "level1/level2/level3/deep.txt"
            store.save_text(nested_path, "deep content")
            
            # Verify file exists and can be read
            assert store.exists(nested_path)
            content = store.read_text(nested_path)
            assert content == "deep content"

    def test_error_handling(self):
        """Test error handling for various scenarios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStore(Path(tmpdir))
            
            # Try to read non-existent file
            with pytest.raises(FileNotFoundError):
                store.read_text("nonexistent.txt")
            
            # Try to read invalid JSON
            store.save_text("invalid.json", "not json content")
            with pytest.raises(json.JSONDecodeError):
                store.read_json("invalid.json")


class TestS3Store:
    """Test S3Store implementation."""

    @patch('boto3.client')
    def test_s3_store_initialization_success(self, mock_boto3_client):
        """Test successful S3Store initialization."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Mock successful bucket check
        mock_client.head_bucket.return_value = None
        
        store = S3Store("test-bucket", "test-prefix", "us-west-2")
        
        assert store.bucket == "test-bucket"
        assert store.prefix == "test-prefix"
        assert store.region == "us-west-2"
        mock_boto3_client.assert_called_once_with("s3", region_name="us-west-2")
        mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")

    @patch('boto3.client')
    def test_s3_store_initialization_bucket_not_found(self, mock_boto3_client):
        """Test S3Store initialization when bucket doesn't exist."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Mock bucket not found error
        from botocore.exceptions import ClientError
        error_response = {"Error": {"Code": "404"}}
        mock_client.head_bucket.side_effect = ClientError(error_response, "HeadBucket")
        
        # Should not raise exception, just log warning
        store = S3Store("nonexistent-bucket")
        assert store.bucket == "nonexistent-bucket"

    @patch('boto3.client')
    def test_s3_store_initialization_access_denied(self, mock_boto3_client):
        """Test S3Store initialization with access denied error."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Mock access denied error
        from botocore.exceptions import ClientError
        error_response = {"Error": {"Code": "403"}}
        mock_client.head_bucket.side_effect = ClientError(error_response, "HeadBucket")
        
        # Should raise exception
        with pytest.raises(ClientError):
            S3Store("forbidden-bucket")

    def test_s3_store_initialization_no_boto3(self):
        """Test S3Store initialization when boto3 is not available."""
        with patch.dict('sys.modules', {'boto3': None}):
            with pytest.raises(ImportError, match="boto3 is required"):
                S3Store("test-bucket")

    @patch('boto3.client')
    def test_s3_store_initialization_no_credentials(self, mock_boto3_client):
        """Test S3Store initialization with no AWS credentials."""
        from botocore.exceptions import NoCredentialsError
        mock_boto3_client.side_effect = NoCredentialsError()
        
        with pytest.raises(NoCredentialsError):
            S3Store("test-bucket")

    @patch('boto3.client')
    def test_s3_key_generation(self, mock_boto3_client):
        """Test S3 key generation with prefix."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.head_bucket.return_value = None
        
        # Test with prefix
        store = S3Store("bucket", "my/prefix")
        key = store._key("file.txt")
        assert key == "my/prefix/file.txt"
        
        # Test without prefix
        store_no_prefix = S3Store("bucket", "")
        key_no_prefix = store_no_prefix._key("file.txt")
        assert key_no_prefix == "file.txt"

    @patch('boto3.client')
    def test_s3_save_and_read_text(self, mock_boto3_client):
        """Test S3 text save and read operations."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.head_bucket.return_value = None
        
        store = S3Store("test-bucket", "test-prefix")
        
        # Mock put_object response
        mock_client.put_object.return_value = {"ETag": '"abc123"'}
        
        # Save text
        content = "Hello S3!"
        ref = store.save_text("test.txt", content)
        
        assert ref.backend == "s3"
        assert ref.uri == "s3://test-bucket/test-prefix/test.txt"
        assert ref.size_bytes == len(content.encode())
        assert ref.content_type == "text/plain"
        
        # Verify put_object was called correctly
        mock_client.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test-prefix/test.txt",
            Body=content.encode(),
            ContentType="text/plain"
        )
        
        # Mock get_object for reading
        mock_response = {
            'Body': MagicMock(),
            'ContentLength': len(content.encode()),
            'ContentType': 'text/plain'
        }
        mock_response['Body'].read.return_value = content.encode()
        mock_client.get_object.return_value = mock_response
        
        # Read text back
        read_content = store.read_text("test.txt")
        assert read_content == content
        
        mock_client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test-prefix/test.txt"
        )

    @patch('boto3.client')
    def test_s3_save_and_read_json(self, mock_boto3_client):
        """Test S3 JSON save and read operations."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.head_bucket.return_value = None
        
        store = S3Store("test-bucket")
        
        # Mock put_object response
        mock_client.put_object.return_value = {"ETag": '"def456"'}
        
        # Save JSON
        data = {"test": "data", "number": 123}
        ref = store.save_json("data.json", data)
        
        assert ref.content_type == "application/json"
        
        # Mock get_object for reading
        json_content = json.dumps(data)
        mock_response = {
            'Body': MagicMock(),
            'ContentLength': len(json_content.encode()),
            'ContentType': 'application/json'
        }
        mock_response['Body'].read.return_value = json_content.encode()
        mock_client.get_object.return_value = mock_response
        
        # Read JSON back
        read_data = store.read_json("data.json")
        assert read_data == data

    @patch('boto3.client')
    def test_s3_save_blob(self, mock_boto3_client):
        """Test S3 blob save operation."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.head_bucket.return_value = None
        
        store = S3Store("test-bucket")
        
        # Mock put_object response
        mock_client.put_object.return_value = {"ETag": '"ghi789"'}
        
        # Create temporary source file
        with tempfile.NamedTemporaryFile() as tmp_file:
            binary_data = b"Binary data\x00\x01\x02"
            tmp_file.write(binary_data)
            tmp_file.flush()
            
            # Save blob
            ref = store.save_blob("binary.bin", Path(tmp_file.name))
            
            assert ref.content_type == "application/octet-stream"
            assert ref.size_bytes == len(binary_data)
            
            # Verify put_object was called with file content
            mock_client.put_object.assert_called_once()
            call_args = mock_client.put_object.call_args
            assert call_args[1]["Bucket"] == "test-bucket"
            assert call_args[1]["Key"] == "binary.bin"
            assert call_args[1]["ContentType"] == "application/octet-stream"

    @patch('boto3.client')
    def test_s3_exists(self, mock_boto3_client):
        """Test S3 existence checking."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.head_bucket.return_value = None
        
        store = S3Store("test-bucket")
        
        # Mock head_object for existing file
        mock_client.head_object.return_value = {"ContentLength": 100}
        assert store.exists("existing.txt") is True
        
        # Mock head_object for non-existing file
        from botocore.exceptions import ClientError
        error_response = {"Error": {"Code": "404"}}
        mock_client.head_object.side_effect = ClientError(error_response, "HeadObject")
        assert store.exists("nonexistent.txt") is False

    @patch('boto3.client')
    def test_s3_delete(self, mock_boto3_client):
        """Test S3 file deletion."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.head_bucket.return_value = None
        
        store = S3Store("test-bucket")
        
        # Mock successful deletion
        mock_client.delete_object.return_value = {"DeleteMarker": True}
        result = store.delete("delete_me.txt")
        assert result is True
        
        mock_client.delete_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="delete_me.txt"
        )

    @patch('boto3.client')
    def test_s3_list_artifacts(self, mock_boto3_client):
        """Test S3 artifact listing."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.head_bucket.return_value = None
        
        store = S3Store("test-bucket", "prefix")
        
        # Mock list_objects_v2 response
        mock_response = {
            "Contents": [
                {"Key": "prefix/file1.txt"},
                {"Key": "prefix/file2.txt"},
                {"Key": "prefix/subdir/file3.txt"}
            ]
        }
        mock_client.list_objects_v2.return_value = mock_response
        
        # List all artifacts
        artifacts = store.list_artifacts()
        expected = ["file1.txt", "file2.txt", "subdir/file3.txt"]
        assert artifacts == expected
        
        # List with additional prefix
        artifacts_filtered = store.list_artifacts("subdir/")
        mock_client.list_objects_v2.assert_called_with(
            Bucket="test-bucket",
            Prefix="prefix/subdir/"
        )

    @patch('boto3.client')
    def test_s3_ensure_bucket_exists(self, mock_boto3_client):
        """Test S3 bucket creation when it doesn't exist."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Mock bucket doesn't exist initially
        from botocore.exceptions import ClientError
        error_response = {"Error": {"Code": "404"}}
        mock_client.head_bucket.side_effect = ClientError(error_response, "HeadBucket")
        
        store = S3Store("new-bucket")
        
        # Mock successful bucket creation
        mock_client.create_bucket.return_value = None
        store._ensure_bucket_exists()
        
        mock_client.create_bucket.assert_called_once_with(Bucket="new-bucket")


class TestGlobalArtifactStore:
    """Test global artifact store management."""

    def test_get_artifact_store_default(self):
        """Test getting default artifact store."""
        with patch('synndicate.storage.artifacts._artifact_store', None):
            with patch('synndicate.storage.artifacts.LocalStore') as mock_local:
                mock_instance = MagicMock()
                mock_local.return_value = mock_instance
                
                store = get_artifact_store()
                assert store == mock_instance
                mock_local.assert_called_once()

    def test_get_artifact_store_cached(self):
        """Test getting cached artifact store."""
        mock_store = MagicMock()
        with patch('synndicate.storage.artifacts._artifact_store', mock_store):
            store = get_artifact_store()
            assert store == mock_store


class TestConvenienceFunctions:
    """Test convenience functions for common operations."""

    @patch('synndicate.storage.artifacts.get_artifact_store')
    def test_save_trace_snapshot(self, mock_get_store):
        """Test save_trace_snapshot convenience function."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_ref = ArtifactRef("uri", "local")
        mock_store.save_json.return_value = mock_ref
        
        trace_id = "trace-123"
        snapshot = {"spans": [], "duration": 1.5}
        
        result = save_trace_snapshot(trace_id, snapshot)
        
        assert result == mock_ref
        mock_store.save_json.assert_called_once_with(
            f"traces/trace_{trace_id}.json",
            snapshot
        )

    @patch('synndicate.storage.artifacts.get_artifact_store')
    def test_save_performance_data(self, mock_get_store):
        """Test save_performance_data convenience function."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_ref = ArtifactRef("uri", "local")
        mock_store.save_text.return_value = mock_ref
        
        trace_id = "trace-456"
        perf_data = [{"metric": "cpu", "value": 0.8}]
        
        result = save_performance_data(trace_id, perf_data)
        
        assert result == mock_ref
        mock_store.save_text.assert_called_once_with(
            f"perf/perf_{trace_id}.jsonl",
            '{"metric": "cpu", "value": 0.8}'
        )

    @patch('synndicate.storage.artifacts.get_artifact_store')
    def test_save_audit_data(self, mock_get_store):
        """Test save_audit_data convenience function."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_ref = ArtifactRef("uri", "local")
        mock_store.save_json.return_value = mock_ref
        
        filename = "coverage.xml"
        data = {"coverage": 85.5}
        
        result = save_audit_data(filename, data)
        
        assert result == mock_ref
        mock_store.save_json.assert_called_once_with(
            f"audit/{filename}",
            data
        )

    @patch('synndicate.storage.artifacts.save_audit_data')
    def test_save_coverage_report(self, mock_save_audit):
        """Test save_coverage_report convenience function."""
        mock_ref = ArtifactRef("uri", "local")
        mock_save_audit.return_value = mock_ref
        
        coverage_data = "<coverage>...</coverage>"
        result = save_coverage_report(coverage_data)
        
        assert result == mock_ref
        mock_save_audit.assert_called_once_with("coverage.xml", coverage_data)

    @patch('synndicate.storage.artifacts.save_audit_data')
    def test_save_lint_report(self, mock_save_audit):
        """Test save_lint_report convenience function."""
        mock_ref = ArtifactRef("uri", "local")
        mock_save_audit.return_value = mock_ref
        
        lint_data = "lint results..."
        result = save_lint_report(lint_data)
        
        assert result == mock_ref
        mock_save_audit.assert_called_once_with("ruff.txt", lint_data)

    @patch('synndicate.storage.artifacts.save_audit_data')
    def test_save_dependency_snapshot(self, mock_save_audit):
        """Test save_dependency_snapshot convenience function."""
        mock_ref = ArtifactRef("uri", "local")
        mock_save_audit.return_value = mock_ref
        
        deps = [{"name": "pytest", "version": "7.0.0"}]
        result = save_dependency_snapshot(deps)
        
        assert result == mock_ref
        mock_save_audit.assert_called_once_with("pip_freeze.json", deps)


if __name__ == "__main__":
    pytest.main([__file__])
