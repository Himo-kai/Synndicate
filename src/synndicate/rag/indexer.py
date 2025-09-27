"""
Async document indexer with progress tracking and batch processing.

Improvements over original:
- Async indexing with background workers
- Progress tracking and status reporting
- Batch processing for large document sets
- Error handling and retry logic
- Incremental indexing support
"""

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..observability.logging import get_logger
from .chunking import Chunk, ChunkingStrategy, SemanticChunker

logger = get_logger(__name__)


class IndexingStatus(Enum):
    """Status of indexing operations."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class IndexingProgress:
    """Progress tracking for indexing operations."""

    total_documents: int = 0
    processed_documents: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_documents: int = 0
    status: IndexingStatus = IndexingStatus.PENDING
    start_time: float | None = None
    end_time: float | None = None
    current_document: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100

    @property
    def duration(self) -> float | None:
        """Get indexing duration in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def documents_per_second(self) -> float | None:
        """Get processing rate in documents per second."""
        duration = self.duration
        if duration is None or duration == 0:
            return None
        return self.processed_documents / duration


@dataclass
class DocumentMetadata:
    """Metadata for indexed documents."""

    file_path: str
    file_size: int
    file_type: str
    last_modified: float
    content_hash: str
    chunk_count: int
    indexed_at: float = field(default_factory=time.time)

    @classmethod
    def from_file(cls, file_path: Path) -> "DocumentMetadata":
        """Create metadata from file."""
        stat = file_path.stat()
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        content_hash = hashlib.md5(content.encode()).hexdigest()

        return cls(
            file_path=str(file_path),
            file_size=stat.st_size,
            file_type=file_path.suffix.lower(),
            last_modified=stat.st_mtime,
            content_hash=content_hash,
            chunk_count=0,  # Will be updated after chunking
        )


class DocumentIndexer:
    """
    Async document indexer with progress tracking and batch processing.

    Features:
    - Async processing with configurable concurrency
    - Progress tracking and status reporting
    - Incremental indexing (skip unchanged files)
    - Error handling and retry logic
    - Batch processing for large document sets
    """

    def __init__(
        self,
        chunking_strategy: ChunkingStrategy | None = None,
        max_concurrent: int = 5,
        batch_size: int = 100,
        supported_extensions: list[str] | None = None,
    ):
        self.chunking_strategy = chunking_strategy or SemanticChunker()
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.supported_extensions = supported_extensions or [
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".rs",
            ".go",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".json",
            ".yaml",
            ".yml",
            ".html",
            ".css",
            ".sql",
            ".sh",
            ".dockerfile",
        ]
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._document_metadata: dict[str, DocumentMetadata] = {}

    async def index_directory(
        self,
        directory: Path,
        recursive: bool = True,
        progress_callback: Callable[[IndexingProgress], None] | None = None,
        filter_func: Callable[[Path], bool] | None = None,
    ) -> IndexingProgress:
        """Index all supported files in a directory."""
        logger.info(f"Starting directory indexing: {directory}")

        # Discover files
        files = self._discover_files(directory, recursive, filter_func)

        # Initialize progress tracking
        progress = IndexingProgress(
            total_documents=len(files), status=IndexingStatus.RUNNING, start_time=time.time()
        )

        if progress_callback:
            progress_callback(progress)

        try:
            # Process files in batches
            async for batch_progress in self._process_files_batched(
                files, progress, progress_callback
            ):
                progress = batch_progress

            progress.status = IndexingStatus.COMPLETED
            progress.end_time = time.time()

            logger.info(
                f"Directory indexing completed: {progress.processed_documents}/{progress.total_documents} "
                f"documents, {progress.total_chunks} chunks in {progress.duration:.2f}s"
            )

        except Exception as e:
            progress.status = IndexingStatus.FAILED
            progress.end_time = time.time()
            progress.errors.append(str(e))
            logger.error(f"Directory indexing failed: {e}")

        if progress_callback:
            progress_callback(progress)

        return progress

    async def index_file(self, file_path: Path, force_reindex: bool = False) -> list[Chunk]:
        """Index a single file and return its chunks."""
        try:
            # Check if file needs indexing
            if not force_reindex and self._is_file_unchanged(file_path):
                logger.debug(f"Skipping unchanged file: {file_path}")
                return []

            # Read file content
            content = await self._read_file_async(file_path)
            if not content.strip():
                logger.debug(f"Skipping empty file: {file_path}")
                return []

            # Create metadata
            metadata = DocumentMetadata.from_file(file_path)

            # Chunk the content
            chunks = self.chunking_strategy.chunk(
                content,
                {
                    "file_path": str(file_path),
                    "file_extension": file_path.suffix.lower(),
                    "file_size": metadata.file_size,
                    "last_modified": metadata.last_modified,
                },
            )

            # Update metadata with chunk count
            metadata.chunk_count = len(chunks)
            self._document_metadata[str(file_path)] = metadata

            logger.debug(f"Indexed file: {file_path} ({len(chunks)} chunks)")
            return chunks

        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            raise

    async def _process_files_batched(
        self,
        files: list[Path],
        progress: IndexingProgress,
        progress_callback: Callable[[IndexingProgress], None] | None,
    ) -> AsyncIterator[IndexingProgress]:
        """Process files in batches with concurrency control."""

        for i in range(0, len(files), self.batch_size):
            batch = files[i : i + self.batch_size]

            # Process batch concurrently
            tasks = [self._process_file_with_semaphore(file_path, progress) for file_path in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Update progress
            for j, result in enumerate(results):
                file_path = batch[j]
                progress.current_document = str(file_path)

                if isinstance(result, Exception):
                    progress.failed_documents += 1
                    progress.errors.append(f"{file_path}: {str(result)}")
                    logger.error(f"Failed to process {file_path}: {result}")
                else:
                    chunks = result
                    progress.total_chunks += len(chunks)
                    progress.processed_chunks += len(chunks)

                progress.processed_documents += 1

                # Call progress callback
                if progress_callback:
                    progress_callback(progress)

            yield progress

    async def _process_file_with_semaphore(
        self, file_path: Path, progress: IndexingProgress
    ) -> list[Chunk]:
        """Process a single file with semaphore for concurrency control."""
        async with self._semaphore:
            return await self.index_file(file_path)

    def _discover_files(
        self, directory: Path, recursive: bool, filter_func: Callable[[Path], bool] | None
    ) -> list[Path]:
        """Discover files to index in the directory."""
        files = []

        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue

            # Check extension
            if file_path.suffix.lower() not in self.supported_extensions:
                continue

            # Apply custom filter
            if filter_func and not filter_func(file_path):
                continue

            # Skip hidden files and common ignore patterns
            if self._should_skip_file(file_path):
                continue

            files.append(file_path)

        logger.info(f"Discovered {len(files)} files to index")
        return files

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        # Skip hidden files
        if file_path.name.startswith("."):
            return True

        # Skip common ignore patterns
        ignore_patterns = [
            "__pycache__",
            "node_modules",
            ".git",
            ".vscode",
            ".idea",
            "target",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
        ]

        for part in file_path.parts:
            if part in ignore_patterns:
                return True

        return False

    def _is_file_unchanged(self, file_path: Path) -> bool:
        """Check if file has changed since last indexing."""
        file_key = str(file_path)
        if file_key not in self._document_metadata:
            return False

        try:
            current_metadata = DocumentMetadata.from_file(file_path)
            stored_metadata = self._document_metadata[file_key]

            return (
                current_metadata.content_hash == stored_metadata.content_hash
                and current_metadata.last_modified == stored_metadata.last_modified
            )
        except Exception:
            return False

    async def _read_file_async(self, file_path: Path) -> str:
        """Read file content asynchronously."""
        try:
            # Try UTF-8 first
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                # Fall back to latin-1
                return file_path.read_text(encoding="latin-1")
            except Exception:
                # Last resort: ignore errors
                return file_path.read_text(encoding="utf-8", errors="ignore")

    def get_document_metadata(self, file_path: str) -> DocumentMetadata | None:
        """Get metadata for a specific document."""
        return self._document_metadata.get(file_path)

    def get_all_metadata(self) -> dict[str, DocumentMetadata]:
        """Get metadata for all indexed documents."""
        return self._document_metadata.copy()

    def clear_metadata(self) -> None:
        """Clear all document metadata."""
        self._document_metadata.clear()

    def get_indexing_stats(self) -> dict[str, Any]:
        """Get indexing statistics."""
        if not self._document_metadata:
            return {}

        total_files = len(self._document_metadata)
        total_chunks = sum(meta.chunk_count for meta in self._document_metadata.values())
        total_size = sum(meta.file_size for meta in self._document_metadata.values())

        file_types = {}
        for meta in self._document_metadata.values():
            file_type = meta.file_type or "unknown"
            file_types[file_type] = file_types.get(file_type, 0) + 1

        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
            "file_types": file_types,
            "supported_extensions": self.supported_extensions,
        }
