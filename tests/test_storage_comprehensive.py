"""
Comprehensive test suite for storage/artifacts module.

Tests both LocalStore and S3Store implementations, including:
- Basic CRUD operations (save, read, delete, exists, list)
- Text, JSON, and binary blob operations
- Error handling and edge cases
- Global store management
- Convenience functions for traces, performance, audit data
- S3 backend with mocking for AWS services
- Local filesystem backend with temporary directories
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from synndicate.storage.artifacts import (
    ArtifactRef,
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


class TestArtifactRef(unittest.TestCase):
    """Test ArtifactRef dataclass."""

    def test_artifact_ref_creation(self):
        """Test creating ArtifactRef instances."""
        # Basic creation
        ref = ArtifactRef(uri="file:///tmp/test.txt", backend="local")
        self.assertEqual(ref.uri, "file:///tmp/test.txt")
        self.assertEqual(ref.backend, "local")
        self.assertIsNone(ref.size_bytes)
        self.assertIsNone(ref.content_type)

    def test_artifact_ref_with_metadata(self):
        """Test ArtifactRef with size and content type."""
        ref = ArtifactRef(
            uri="s3://bucket/key", backend="s3", size_bytes=1024, content_type="application/json"
        )
        self.assertEqual(ref.uri, "s3://bucket/key")
        self.assertEqual(ref.backend, "s3")
        self.assertEqual(ref.size_bytes, 1024)
        self.assertEqual(ref.content_type, "application/json")

    def test_artifact_ref_equality(self):
        """Test ArtifactRef equality comparison."""
        ref1 = ArtifactRef(uri="file:///test", backend="local")
        ref2 = ArtifactRef(uri="file:///test", backend="local")
        ref3 = ArtifactRef(uri="file:///other", backend="local")

        self.assertEqual(ref1, ref2)
        self.assertNotEqual(ref1, ref3)


class TestLocalStore(unittest.TestCase):
    """Test LocalStore filesystem backend."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = LocalStore(Path(self.temp_dir))

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_local_store_initialization(self):
        """Test LocalStore initialization."""
        self.assertEqual(self.store.root, Path(self.temp_dir))
        self.assertTrue(self.store.root.exists())
        self.assertTrue(self.store.root.is_dir())

    def test_local_store_initialization_creates_directory(self):
        """Test LocalStore creates directory if it doesn't exist."""
        new_temp_dir = Path(self.temp_dir) / "new_subdir"
        self.assertFalse(new_temp_dir.exists())

        LocalStore(new_temp_dir)
        self.assertTrue(new_temp_dir.exists())
        self.assertTrue(new_temp_dir.is_dir())

    def test_save_and_read_text(self):
        """Test saving and reading text content."""
        content = "Hello, World!"
        ref = self.store.save_text("test.txt", content)

        # Verify reference
        self.assertIn("file://", ref.uri)
        self.assertEqual(ref.backend, "local")
        self.assertEqual(ref.size_bytes, len(content.encode()))
        self.assertEqual(ref.content_type, "text/plain")

        # Verify content
        read_content = self.store.read_text("test.txt")
        self.assertEqual(read_content, content)

    def test_save_and_read_json(self):
        """Test saving and reading JSON objects."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        ref = self.store.save_json("test.json", data)

        # Verify reference
        self.assertIn("file://", ref.uri)
        self.assertEqual(ref.backend, "local")
        self.assertEqual(ref.content_type, "application/json")

        # Verify content
        read_data = self.store.read_json("test.json")
        self.assertEqual(read_data, data)

    def test_save_and_read_blob(self):
        """Test saving and reading binary files."""
        # Create a temporary source file
        src_file = Path(self.temp_dir) / "source.bin"
        src_content = b"Binary content\x00\x01\x02"
        src_file.write_bytes(src_content)

        ref = self.store.save_blob("test.bin", src_file)

        # Verify reference
        self.assertIn("file://", ref.uri)
        self.assertEqual(ref.backend, "local")
        self.assertEqual(ref.size_bytes, len(src_content))
        self.assertEqual(ref.content_type, "application/octet-stream")

        # Verify content by reading the saved file directly
        saved_path = self.store._path("test.bin")
        saved_content = saved_path.read_bytes()
        self.assertEqual(saved_content, src_content)

    def test_exists_and_delete(self):
        """Test file existence checking and deletion."""
        # Initially doesn't exist
        self.assertFalse(self.store.exists("nonexistent.txt"))

        # Save a file
        self.store.save_text("test.txt", "content")
        self.assertTrue(self.store.exists("test.txt"))

        # Delete the file
        self.store.delete("test.txt")
        self.assertFalse(self.store.exists("test.txt"))

    def test_list_artifacts(self):
        """Test listing artifacts."""
        # Initially empty
        artifacts = self.store.list_artifacts()
        self.assertEqual(len(artifacts), 0)

        # Save some files
        self.store.save_text("file1.txt", "content1")
        self.store.save_text("file2.txt", "content2")
        self.store.save_text("subdir/file3.txt", "content3")

        # List all artifacts (returns list of strings, not ArtifactRef objects)
        artifacts = self.store.list_artifacts()
        self.assertEqual(len(artifacts), 3)

        # Check artifact paths (artifacts are strings, not objects with .uri)
        self.assertIn("file1.txt", artifacts)
        self.assertIn("file2.txt", artifacts)
        self.assertIn("subdir/file3.txt", artifacts)

    def test_list_artifacts_with_prefix(self):
        """Test listing artifacts with prefix filter."""
        # Save files with different prefixes
        self.store.save_text("logs/app.log", "log content")
        self.store.save_text("logs/error.log", "error content")
        self.store.save_text("data/file.json", "json content")

        # List with prefix (returns relative paths from prefix)
        log_artifacts = self.store.list_artifacts("logs/")
        self.assertEqual(len(log_artifacts), 2)

        data_artifacts = self.store.list_artifacts("data/")
        self.assertEqual(len(data_artifacts), 1)

    def test_nested_directory_creation(self):
        """Test creating nested directories automatically."""
        nested_path = "deep/nested/directory/file.txt"
        content = "nested content"

        ref = self.store.save_text(nested_path, content)
        self.assertIn("file://", ref.uri)

        # Verify content can be read
        read_content = self.store.read_text(nested_path)
        self.assertEqual(read_content, content)

    def test_read_nonexistent_file_error(self):
        """Test reading nonexistent file raises appropriate error."""
        with self.assertRaises(FileNotFoundError):
            self.store.read_text("nonexistent.txt")

        with self.assertRaises(FileNotFoundError):
            self.store.read_json("nonexistent.json")

    def test_delete_nonexistent_file_behavior(self):
        """Test deleting nonexistent file returns False instead of raising error."""
        # LocalStore.delete() returns False for nonexistent files, doesn't raise exception
        result = self.store.delete("nonexistent.txt")
        self.assertFalse(result)

    def test_save_blob_nonexistent_source_error(self):
        """Test saving blob from nonexistent source file."""
        nonexistent_source = Path("/nonexistent/source.bin")
        with self.assertRaises(FileNotFoundError):
            self.store.save_blob("test.bin", nonexistent_source)


class TestS3Store(unittest.TestCase):
    """Test S3Store cloud backend with mocking."""

    def setUp(self):
        """Set up S3Store with mocked boto3."""
        try:
            # Try to import boto3 to see if it's available
            import importlib.util
            if importlib.util.find_spec("boto3") is None:
                raise ImportError("boto3 not available")
            if importlib.util.find_spec("botocore") is None:
                raise ImportError("botocore not available")

            # Mock boto3 and S3 client
            self.mock_boto3 = MagicMock()
            self.mock_s3_client = MagicMock()
            self.mock_boto3.client.return_value = self.mock_s3_client

            # Mock successful bucket head operation
            self.mock_s3_client.head_bucket.return_value = {}

            # Patch boto3 import
            self.boto3_patcher = patch("synndicate.storage.artifacts.boto3", self.mock_boto3)
            self.boto3_patcher.start()

            self.store = S3Store(bucket="test-bucket", prefix="test-prefix")
            self.boto3_available = True

        except ImportError:
            self.boto3_available = False
            self.skipTest("boto3 not available for S3Store testing")

    def tearDown(self):
        """Clean up patches."""
        if hasattr(self, "boto3_patcher"):
            self.boto3_patcher.stop()

    def test_s3_store_initialization(self):
        """Test S3Store initialization."""
        self.assertEqual(self.store.bucket, "test-bucket")
        self.assertEqual(self.store.prefix, "test-prefix")
        self.assertEqual(self.store.region, "us-east-1")

        # Verify boto3 client was created
        self.mock_boto3.client.assert_called_once()
        self.mock_s3_client.head_bucket.assert_called_once_with(Bucket="test-bucket")

    def test_s3_store_initialization_with_custom_region(self):
        """Test S3Store initialization with custom region."""
        with patch("synndicate.storage.artifacts.boto3", self.mock_boto3):
            from botocore.exceptions import ClientError, NoCredentialsError

            with (
                patch("synndicate.storage.artifacts.ClientError", ClientError),
                patch("synndicate.storage.artifacts.NoCredentialsError", NoCredentialsError),
            ):

                store = S3Store(bucket="test-bucket", region="eu-west-1")
                self.assertEqual(store.region, "eu-west-1")

    def test_s3_store_initialization_missing_boto3(self):
        """Test S3Store initialization fails without boto3."""
        with patch(
            "synndicate.storage.artifacts.boto3", side_effect=ImportError("No module named 'boto3'")
        ):
            with self.assertRaises(ImportError) as cm:
                S3Store(bucket="test-bucket")

            self.assertIn("boto3 is required", str(cm.exception))

    def test_s3_store_initialization_no_credentials(self):
        """Test S3Store initialization fails without AWS credentials."""
        from botocore.exceptions import NoCredentialsError

        mock_boto3_no_creds = MagicMock()
        mock_boto3_no_creds.client.side_effect = NoCredentialsError()

        with (
            patch("synndicate.storage.artifacts.boto3", mock_boto3_no_creds),
            patch("synndicate.storage.artifacts.NoCredentialsError", NoCredentialsError),
            self.assertRaises(NoCredentialsError),
        ):
            S3Store(bucket="test-bucket")

    def test_s3_store_key_generation(self):
        """Test S3 key generation with prefix."""
        key = self.store._key("test/file.txt")
        self.assertEqual(key, "test-prefix/test/file.txt")

        # Test without prefix
        store_no_prefix = S3Store.__new__(S3Store)
        store_no_prefix.prefix = ""
        key_no_prefix = store_no_prefix._key("test/file.txt")
        self.assertEqual(key_no_prefix, "test/file.txt")

    def test_save_and_read_text_s3(self):
        """Test saving and reading text content to S3."""
        content = "S3 text content"

        # Mock S3 put_object response
        self.mock_s3_client.put_object.return_value = {
            "ETag": '"abc123"',
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

        # Mock S3 get_object response
        mock_response = {
            "Body": MagicMock(),
            "ContentLength": len(content.encode()),
            "ContentType": "text/plain",
        }
        mock_response["Body"].read.return_value = content.encode()
        self.mock_s3_client.get_object.return_value = mock_response

        # Save text
        ref = self.store.save_text("test.txt", content)

        # Verify reference
        self.assertIn("s3://", ref.uri)
        self.assertEqual(ref.backend, "s3")
        self.assertEqual(ref.content_type, "text/plain")

        # Verify S3 put_object was called
        self.mock_s3_client.put_object.assert_called_once()
        call_args = self.mock_s3_client.put_object.call_args
        self.assertEqual(call_args[1]["Bucket"], "test-bucket")
        self.assertEqual(call_args[1]["Key"], "test-prefix/test.txt")
        self.assertEqual(call_args[1]["Body"], content.encode())

        # Read text
        read_content = self.store.read_text("test.txt")
        self.assertEqual(read_content, content)

        # Verify S3 get_object was called
        self.mock_s3_client.get_object.assert_called_with(
            Bucket="test-bucket", Key="test-prefix/test.txt"
        )

    def test_save_and_read_json_s3(self):
        """Test saving and reading JSON objects to S3."""
        data = {"s3": "json", "test": True}
        json_content = json.dumps(data, indent=2)

        # Mock S3 responses
        self.mock_s3_client.put_object.return_value = {"ETag": '"def456"'}

        mock_response = {
            "Body": MagicMock(),
            "ContentLength": len(json_content.encode()),
            "ContentType": "application/json",
        }
        mock_response["Body"].read.return_value = json_content.encode()
        self.mock_s3_client.get_object.return_value = mock_response

        # Save and read JSON
        ref = self.store.save_json("test.json", data)
        self.assertEqual(ref.content_type, "application/json")

        read_data = self.store.read_json("test.json")
        self.assertEqual(read_data, data)

    def test_exists_and_delete_s3(self):
        """Test checking existence and deleting objects in S3."""
        # Mock head_object for existence check
        self.mock_s3_client.head_object.return_value = {
            "ContentLength": 100,
            "LastModified": "2023-01-01",
        }

        # Test exists
        exists = self.store.exists("test.txt")
        self.assertTrue(exists)
        self.mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket", Key="test-prefix/test.txt"
        )

        # Mock delete_object
        self.mock_s3_client.delete_object.return_value = {"DeleteMarker": True}

        # Test delete
        self.store.delete("test.txt")
        self.mock_s3_client.delete_object.assert_called_with(
            Bucket="test-bucket", Key="test-prefix/test.txt"
        )

    def test_exists_s3_not_found(self):
        """Test checking existence of non-existent S3 object."""
        # Use the ClientError stored in the S3Store instance
        client_error_class = self.store.ClientError

        # Mock ClientError for 404
        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        self.mock_s3_client.head_object.side_effect = client_error_class(error_response, "HeadObject")

        exists = self.store.exists("nonexistent.txt")
        self.assertFalse(exists)

    def test_list_artifacts_s3(self):
        """Test listing artifacts in S3."""
        # Mock list_objects_v2 response
        mock_response = {
            "Contents": [
                {"Key": "test-prefix/file1.txt", "Size": 100, "LastModified": "2023-01-01"},
                {"Key": "test-prefix/file2.json", "Size": 200, "LastModified": "2023-01-02"},
            ],
            "IsTruncated": False,
        }
        self.mock_s3_client.list_objects_v2.return_value = mock_response

        artifacts = self.store.list_artifacts()
        self.assertEqual(len(artifacts), 2)

        # Verify artifact references
        self.assertTrue(any("file1.txt" in ref.uri for ref in artifacts))
        self.assertTrue(any("file2.json" in ref.uri for ref in artifacts))

        # Verify S3 call
        self.mock_s3_client.list_objects_v2.assert_called_with(
            Bucket="test-bucket", Prefix="test-prefix/"
        )

    def test_list_artifacts_s3_with_prefix(self):
        """Test listing artifacts in S3 with additional prefix."""
        mock_response = {"Contents": [], "IsTruncated": False}
        self.mock_s3_client.list_objects_v2.return_value = mock_response

        self.store.list_artifacts("logs/")

        # Verify S3 call with combined prefix
        self.mock_s3_client.list_objects_v2.assert_called_with(
            Bucket="test-bucket", Prefix="test-prefix/logs/"
        )

    def test_s3_error_handling(self):
        """Test S3 error handling for various operations."""
        # Use the ClientError stored in the S3Store instance
        client_error_class = self.store.ClientError

        # Mock ClientError for various operations
        error_response = {"Error": {"Code": "500", "Message": "Internal Error"}}
        client_error = client_error_class(error_response, "Operation")

        # Test save_text error
        self.mock_s3_client.put_object.side_effect = client_error
        with self.assertRaises(client_error_class):
            self.store.save_text("test.txt", "content")

        # Test read_text error
        self.mock_s3_client.get_object.side_effect = client_error
        with self.assertRaises(client_error_class):
            self.store.read_text("test.txt")


class TestGlobalStoreManagement(unittest.TestCase):
    """Test global artifact store management."""

    def setUp(self):
        """Reset global store state."""
        import synndicate.storage.artifacts as artifacts_module

        artifacts_module._artifact_store = None

    def tearDown(self):
        """Clean up global store state."""
        import synndicate.storage.artifacts as artifacts_module

        artifacts_module._artifact_store = None

    def test_get_artifact_store_default(self):
        """Test getting default artifact store."""
        store = get_artifact_store()
        self.assertIsInstance(store, LocalStore)

        # Should return the same instance on subsequent calls
        store2 = get_artifact_store()
        self.assertIs(store, store2)

    def test_get_artifact_store_creates_artifacts_directory(self):
        """Test that get_artifact_store creates artifacts directory."""
        # The actual implementation creates a LocalStore with ./artifacts directory
        import os
        import tempfile

        # Change to a temporary directory to avoid creating artifacts in project root
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)

            try:
                # Reset global store
                import synndicate.storage.artifacts as artifacts_module

                artifacts_module._artifact_store = None

                store = get_artifact_store()
                self.assertIsInstance(store, LocalStore)

                # Verify artifacts directory was created
                artifacts_path = Path("./artifacts")
                self.assertTrue(artifacts_path.exists())
                self.assertTrue(artifacts_path.is_dir())

            finally:
                os.chdir(original_cwd)
                # Reset global store
                artifacts_module._artifact_store = None


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for common artifact operations."""

    def setUp(self):
        """Set up mock artifact store."""
        self.mock_store = MagicMock()

        # Mock get_artifact_store to return our mock
        self.patcher = patch("synndicate.storage.artifacts.get_artifact_store")
        self.mock_get_store = self.patcher.start()
        self.mock_get_store.return_value = self.mock_store

    def tearDown(self):
        """Clean up patches."""
        self.patcher.stop()

    def test_save_trace_snapshot(self):
        """Test saving trace snapshot."""
        trace_id = "trace-123"
        snapshot = {"spans": [], "duration": 100}

        save_trace_snapshot(trace_id, snapshot)

        self.mock_store.save_json.assert_called_once_with(f"traces/trace_{trace_id}.json", snapshot)

    def test_save_performance_data(self):
        """Test saving performance data."""
        trace_id = "trace-456"
        perf_data = [{"operation": "query", "duration": 50}, {"operation": "index", "duration": 30}]

        save_performance_data(trace_id, perf_data)

        # Performance data is saved as JSONL text, not JSON
        expected_content = (
            '{"operation": "query", "duration": 50}\n{"operation": "index", "duration": 30}'
        )
        self.mock_store.save_text.assert_called_once_with(
            f"perf/perf_{trace_id}.jsonl", expected_content
        )

    def test_save_audit_data(self):
        """Test saving audit data."""
        filename = "audit_report.json"
        data = {"coverage": 85, "linting": "passed"}

        save_audit_data(filename, data)

        self.mock_store.save_json.assert_called_once_with(f"audit/{filename}", data)

    def test_save_coverage_report(self):
        """Test saving coverage report."""
        coverage_data = "Coverage report content"

        # Mock save_audit_data since save_coverage_report calls it
        with patch("synndicate.storage.artifacts.save_audit_data") as mock_save_audit:
            save_coverage_report(coverage_data)

            mock_save_audit.assert_called_once_with("coverage.xml", coverage_data)

    def test_save_lint_report(self):
        """Test saving lint report."""
        lint_data = "Linting report content"

        # Mock save_audit_data since save_lint_report calls it
        with patch("synndicate.storage.artifacts.save_audit_data") as mock_save_audit:
            save_lint_report(lint_data)

            mock_save_audit.assert_called_once_with("ruff.txt", lint_data)

    def test_save_dependency_snapshot(self):
        """Test saving dependency snapshot."""
        deps = [{"name": "pytest", "version": "7.0.0"}, {"name": "numpy", "version": "1.21.0"}]

        # Mock save_audit_data since save_dependency_snapshot calls it
        with patch("synndicate.storage.artifacts.save_audit_data") as mock_save_audit:
            save_dependency_snapshot(deps)

            mock_save_audit.assert_called_once_with("pip_freeze.json", deps)


class TestStorageEdgeCases(unittest.TestCase):
    """Test edge cases and error scenarios for storage system."""

    def setUp(self):
        """Set up temporary directory for edge case tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = LocalStore(Path(self.temp_dir))

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_content_handling(self):
        """Test handling of empty content."""
        # Empty text
        ref = self.store.save_text("empty.txt", "")
        self.assertEqual(ref.size_bytes, 0)
        content = self.store.read_text("empty.txt")
        self.assertEqual(content, "")

        # Empty JSON
        ref = self.store.save_json("empty.json", {})
        content = self.store.read_json("empty.json")
        self.assertEqual(content, {})

    def test_unicode_content_handling(self):
        """Test handling of Unicode content."""
        unicode_content = "Hello 世界! 🌍 Ñoño"
        ref = self.store.save_text("unicode.txt", unicode_content)

        content = self.store.read_text("unicode.txt")
        self.assertEqual(content, unicode_content)

        # Verify size is in bytes, not characters
        self.assertEqual(ref.size_bytes, len(unicode_content.encode("utf-8")))

    def test_large_json_handling(self):
        """Test handling of large JSON objects."""
        large_data = {"items": [{"id": i, "data": f"item_{i}" * 100} for i in range(1000)]}

        ref = self.store.save_json("large.json", large_data)
        self.assertGreater(ref.size_bytes, 1000)  # Should be substantial

        read_data = self.store.read_json("large.json")
        self.assertEqual(read_data, large_data)

    def test_special_characters_in_paths(self):
        """Test handling of special characters in file paths."""
        # Note: Some characters may not be valid on all filesystems
        safe_special_path = "test-file_with.special-chars/file.txt"
        content = "Special path content"

        ref = self.store.save_text(safe_special_path, content)
        self.assertIn("file://", ref.uri)

        read_content = self.store.read_text(safe_special_path)
        self.assertEqual(read_content, content)

    def test_concurrent_access_simulation(self):
        """Test simulation of concurrent access patterns."""
        # Save multiple files rapidly
        for i in range(10):
            self.store.save_text(f"concurrent_{i}.txt", f"content_{i}")

        # Verify all files exist
        for i in range(10):
            self.assertTrue(self.store.exists(f"concurrent_{i}.txt"))
            content = self.store.read_text(f"concurrent_{i}.txt")
            self.assertEqual(content, f"content_{i}")

    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        # These should be handled safely by Path normalization
        safe_paths = ["normal/path.txt", "./relative/path.txt", "path/with/../normalization.txt"]

        for path in safe_paths:
            ref = self.store.save_text(path, "safe content")
            self.assertIn("file://", ref.uri)
            # Verify the file was created within the store root
            actual_path = self.store._path(path)
            self.assertTrue(str(actual_path).startswith(str(self.store.root)))


if __name__ == "__main__":
    unittest.main()
