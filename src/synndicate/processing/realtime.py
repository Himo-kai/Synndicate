"""
Real-time document processing pipeline for live analysis and updates.

This module provides infrastructure for processing documents in real-time,
enabling live analysis, incremental updates, and streaming processing capabilities.

Key features:
- Live document monitoring and change detection
- Incremental processing for efficiency
- Streaming analysis with backpressure handling
- Event-driven architecture with pub/sub patterns
- Multi-format document support
- Real-time collaboration features
"""

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..agents.multimodal import MultiModalContent, MultiModalOutput
from ..observability.logging import get_logger
from ..observability.metrics import counter, histogram, timer
from ..observability.tracing import trace_span

logger = get_logger(__name__)


class DocumentEventType(Enum):
    """Types of document events for real-time processing."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    ACCESSED = "accessed"
    MOVED = "moved"
    CONTENT_CHANGED = "content_changed"
    METADATA_CHANGED = "metadata_changed"


class ProcessingStatus(Enum):
    """Status of document processing operations."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class DocumentEvent:
    """Event representing a change to a document."""

    event_type: DocumentEventType
    document_id: str
    timestamp: float = field(default_factory=time.time)
    file_path: str | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Get age of event in seconds."""
        return time.time() - self.timestamp


@dataclass
class ProcessingTask:
    """Task for processing a document or content."""

    task_id: str
    document_id: str
    content: MultiModalContent | None = None
    priority: int = 0  # Higher values = higher priority
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def processing_time(self) -> float | None:
        """Get processing time in seconds if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def wait_time(self) -> float:
        """Get time spent waiting to be processed."""
        start_time = self.started_at or time.time()
        return start_time - self.created_at


class DocumentProcessor(ABC):
    """Abstract base class for document processors."""

    @abstractmethod
    async def process_document(
        self,
        content: MultiModalContent,
        metadata: dict[str, Any] | None = None
    ) -> MultiModalOutput:
        """Process a document and return results."""
        pass

    @abstractmethod
    async def can_process(self, content: MultiModalContent) -> bool:
        """Check if this processor can handle the given content."""
        pass

    @property
    @abstractmethod
    def processor_name(self) -> str:
        """Get the name of this processor."""
        pass


class EventListener(ABC):
    """Abstract base class for document event listeners."""

    @abstractmethod
    async def on_event(self, event: DocumentEvent) -> None:
        """Handle a document event."""
        pass

    @abstractmethod
    def should_handle(self, event: DocumentEvent) -> bool:
        """Check if this listener should handle the event."""
        pass


class RealTimeDocumentProcessor:
    """
    Real-time document processing engine with event-driven architecture.

    Provides live document monitoring, incremental processing, and streaming
    analysis capabilities with support for multiple document formats.
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        task_queue_size: int = 1000,
        enable_incremental: bool = True,
        batch_size: int = 5,
        batch_timeout: float = 1.0,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue_size = task_queue_size
        self.enable_incremental = enable_incremental
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        # Processing infrastructure
        self._task_queue: asyncio.Queue[ProcessingTask] = asyncio.Queue(maxsize=task_queue_size)
        self._active_tasks: dict[str, ProcessingTask] = {}
        self._completed_tasks: deque[ProcessingTask] = deque(maxlen=1000)
        self._processing_semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Event system
        self._event_listeners: list[EventListener] = []
        self._event_queue: asyncio.Queue[DocumentEvent] = asyncio.Queue()

        # Processors and content tracking
        self._processors: list[DocumentProcessor] = []
        self._document_hashes: dict[str, str] = {}
        self._document_metadata: dict[str, dict[str, Any]] = {}

        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Background tasks
        self._worker_tasks: list[asyncio.Task] = []
        self._event_handler_task: asyncio.Task | None = None
        self._batch_processor_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the real-time processing engine."""
        if self._running:
            logger.warning("Real-time processor already running")
            return

        logger.info("Starting real-time document processor")
        self._running = True
        self._shutdown_event.clear()

        # Start worker tasks
        for i in range(self.max_concurrent_tasks):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._worker_tasks.append(task)

        # Start event handler
        self._event_handler_task = asyncio.create_task(self._event_handler_loop())

        # Start batch processor if enabled
        if self.batch_size > 1:
            self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())

        logger.info(f"Started {len(self._worker_tasks)} worker tasks")

    async def stop(self) -> None:
        """Stop the real-time processing engine gracefully."""
        if not self._running:
            return

        logger.info("Stopping real-time document processor")
        self._running = False
        self._shutdown_event.set()

        # Cancel and wait for all tasks
        all_tasks = self._worker_tasks.copy()
        if self._event_handler_task:
            all_tasks.append(self._event_handler_task)
        if self._batch_processor_task:
            all_tasks.append(self._batch_processor_task)

        for task in all_tasks:
            task.cancel()

        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        self._worker_tasks.clear()
        self._event_handler_task = None
        self._batch_processor_task = None

        logger.info("Real-time document processor stopped")

    def add_processor(self, processor: DocumentProcessor) -> None:
        """Add a document processor to the pipeline."""
        self._processors.append(processor)
        logger.info(f"Added processor: {processor.processor_name}")

    def add_event_listener(self, listener: EventListener) -> None:
        """Add an event listener for document events."""
        self._event_listeners.append(listener)
        logger.info(f"Added event listener: {type(listener).__name__}")

    async def submit_document(
        self,
        document_id: str,
        content: MultiModalContent,
        priority: int = 0,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Submit a document for processing."""
        with trace_span("realtime_processor.submit_document"):
            # Check for incremental processing
            if self.enable_incremental and await self._should_skip_processing(document_id, content):
                logger.debug(f"Skipping unchanged document: {document_id}")
                counter("realtime_processor.documents_skipped").inc()
                return f"skipped-{document_id}"

            # Create processing task
            task_id = f"{document_id}-{int(time.time() * 1000)}"
            task = ProcessingTask(
                task_id=task_id,
                document_id=document_id,
                content=content,
                priority=priority,
                metadata=metadata or {}
            )

            # Submit to queue
            try:
                await self._task_queue.put(task)
                counter("realtime_processor.documents_submitted").inc()
                logger.debug(f"Submitted document for processing: {document_id}")
                return task_id
            except asyncio.QueueFull as e:
                counter("realtime_processor.queue_full_errors").inc()
                raise RuntimeError("Processing queue is full") from e

    async def emit_event(self, event: DocumentEvent) -> None:
        """Emit a document event for processing."""
        try:
            await self._event_queue.put(event)
            counter("realtime_processor.events_emitted").inc()
        except asyncio.QueueFull:
            counter("realtime_processor.event_queue_full_errors").inc()
            logger.warning(f"Event queue full, dropping event: {event.event_type}")

    async def get_task_status(self, task_id: str) -> ProcessingTask | None:
        """Get the status of a processing task."""
        # Check active tasks
        if task_id in self._active_tasks:
            return self._active_tasks[task_id]

        # Check completed tasks
        for task in self._completed_tasks:
            if task.task_id == task_id:
                return task

        return None

    async def stream_results(
        self,
        document_id: str | None = None
    ) -> AsyncIterator[tuple[ProcessingTask, MultiModalOutput]]:
        """Stream processing results in real-time."""
        processed_tasks = set()

        while self._running or self._active_tasks:
            # Check completed tasks
            for task in list(self._completed_tasks):
                if task.task_id in processed_tasks:
                    continue

                if (document_id is None or task.document_id == document_id) and task.status == ProcessingStatus.COMPLETED and task.metadata.get("result"):
                    processed_tasks.add(task.task_id)
                    yield task, task.metadata["result"]

            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

    async def _worker_loop(self, worker_name: str) -> None:
        """Main worker loop for processing documents."""
        logger.debug(f"Started worker: {worker_name}")

        while self._running:
            try:
                # Get next task with timeout
                try:
                    task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                # Process the task
                await self._process_task(task, worker_name)

            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                counter("realtime_processor.worker_errors").inc()

        logger.debug(f"Worker stopped: {worker_name}")

    async def _process_task(self, task: ProcessingTask, worker_name: str) -> None:
        """Process a single document task."""
        async with self._processing_semaphore:
            task.status = ProcessingStatus.PROCESSING
            task.started_at = time.time()
            self._active_tasks[task.task_id] = task

            try:
                with trace_span("realtime_processor.process_task", {"task_id": task.task_id}), timer("realtime_processor.task_processing_time"):
                    # Find appropriate processor
                    processor = await self._find_processor(task.content)
                    if not processor:
                        raise ValueError(f"No processor found for content type: {type(task.content)}")

                    # Process the document
                    result = await processor.process_document(task.content, task.metadata)

                    # Update task
                    task.status = ProcessingStatus.COMPLETED
                    task.completed_at = time.time()
                    task.metadata["result"] = result
                    task.metadata["processor"] = processor.processor_name
                    task.metadata["worker"] = worker_name

                    # Update document tracking
                    if task.content:
                        content_hash = self._calculate_content_hash(task.content)
                        self._document_hashes[task.document_id] = content_hash

                    counter("realtime_processor.documents_processed").inc()
                    histogram("realtime_processor.processing_time").observe(task.processing_time or 0)

                    logger.debug(f"Processed document: {task.document_id} in {task.processing_time:.2f}s")

            except Exception as e:
                task.status = ProcessingStatus.FAILED
                task.error = str(e)
                task.retry_count += 1

                counter("realtime_processor.processing_errors").inc()
                logger.error(f"Failed to process document {task.document_id}: {e}")

                # Retry if under limit
                if task.retry_count <= task.max_retries:
                    task.status = ProcessingStatus.RETRYING
                    await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                    await self._task_queue.put(task)
                    counter("realtime_processor.task_retries").inc()

            finally:
                # Move to completed tasks
                if task.task_id in self._active_tasks:
                    del self._active_tasks[task.task_id]
                self._completed_tasks.append(task)

    async def _event_handler_loop(self) -> None:
        """Handle document events from the event queue."""
        logger.debug("Started event handler loop")

        while self._running:
            try:
                # Get next event with timeout
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                # Dispatch to listeners
                for listener in self._event_listeners:
                    if listener.should_handle(event):
                        try:
                            await listener.on_event(event)
                        except Exception as e:
                            logger.error(f"Event listener error: {e}")
                            counter("realtime_processor.event_handler_errors").inc()

                counter("realtime_processor.events_processed").inc()

            except Exception as e:
                logger.error(f"Event handler error: {e}")
                counter("realtime_processor.event_handler_errors").inc()

        logger.debug("Event handler loop stopped")

    async def _batch_processor_loop(self) -> None:
        """Process documents in batches for efficiency."""
        logger.debug("Started batch processor loop")
        batch = []
        last_batch_time = time.time()

        while self._running:
            try:
                # Try to get a task
                try:
                    task = await asyncio.wait_for(self._task_queue.get(), timeout=0.1)
                    batch.append(task)
                except TimeoutError:
                    pass

                # Process batch if conditions met
                current_time = time.time()
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and current_time - last_batch_time >= self.batch_timeout)
                )

                if should_process and batch:
                    await self._process_batch(batch)
                    batch.clear()
                    last_batch_time = current_time

            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                counter("realtime_processor.batch_processor_errors").inc()

        # Process remaining batch
        if batch:
            await self._process_batch(batch)

        logger.debug("Batch processor loop stopped")

    async def _process_batch(self, batch: list[ProcessingTask]) -> None:
        """Process a batch of tasks together."""
        with trace_span("realtime_processor.process_batch", {"batch_size": len(batch)}):
            # Group by processor type for efficiency
            processor_groups = defaultdict(list)

            for task in batch:
                processor = await self._find_processor(task.content)
                if processor:
                    processor_groups[processor].append(task)

            # Process each group
            for _, tasks in processor_groups.items():
                await asyncio.gather(
                    *[self._process_task(task, "batch-worker") for task in tasks],
                    return_exceptions=True
                )

            counter("realtime_processor.batches_processed").inc()
            histogram("realtime_processor.batch_size").observe(len(batch))

    async def _find_processor(self, content: MultiModalContent | None) -> DocumentProcessor | None:
        """Find an appropriate processor for the given content."""
        if not content:
            return None

        for processor in self._processors:
            if await processor.can_process(content):
                return processor

        return None

    async def _should_skip_processing(self, document_id: str, content: MultiModalContent) -> bool:
        """Check if document processing should be skipped (incremental processing)."""
        if not self.enable_incremental:
            return False

        # Calculate content hash
        content_hash = self._calculate_content_hash(content)

        # Check if content has changed
        previous_hash = self._document_hashes.get(document_id)
        return previous_hash == content_hash

    def _calculate_content_hash(self, content: MultiModalContent) -> str:
        """Calculate hash of content for change detection."""
        if hasattr(content, 'content'):
            # Text or code content
            return hashlib.sha256(content.content.encode('utf-8')).hexdigest()
        elif hasattr(content, 'data'):
            # Image or binary content
            return hashlib.sha256(content.data).hexdigest()
        else:
            # Fallback
            return hashlib.sha256(str(content).encode('utf-8')).hexdigest()

    @property
    def stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "running": self._running,
            "active_tasks": len(self._active_tasks),
            "queue_size": self._task_queue.qsize(),
            "completed_tasks": len(self._completed_tasks),
            "processors": len(self._processors),
            "event_listeners": len(self._event_listeners),
            "event_queue_size": self._event_queue.qsize(),
        }


class FileSystemEventListener(EventListener):
    """Event listener for file system changes."""

    def __init__(self, processor: RealTimeDocumentProcessor, watch_extensions: set[str] | None = None):
        self.processor = processor
        self.watch_extensions = watch_extensions or {'.txt', '.md', '.py', '.js', '.html', '.css'}

    def should_handle(self, event: DocumentEvent) -> bool:
        """Check if this listener should handle file system events."""
        if not event.file_path:
            return False

        file_path = Path(event.file_path)
        return file_path.suffix.lower() in self.watch_extensions

    async def on_event(self, event: DocumentEvent) -> None:
        """Handle file system events by submitting for processing."""
        if event.event_type in (DocumentEventType.CREATED, DocumentEventType.MODIFIED):
            logger.info(f"File system event: {event.event_type} - {event.file_path}")

            # In a real implementation, you would read the file content here
            # For now, we'll create a placeholder
            from ..agents.multimodal import TextContent

            content = TextContent(
                content=f"File content from {event.file_path}",
                metadata={"file_path": event.file_path, "event_type": event.event_type.value}
            )

            await self.processor.submit_document(
                document_id=event.document_id,
                content=content,
                metadata={"source": "filesystem", "event": event.event_type.value}
            )


# Factory functions
def create_realtime_processor(**kwargs) -> RealTimeDocumentProcessor:
    """Create a real-time document processor with default settings."""
    return RealTimeDocumentProcessor(**kwargs)


async def create_filesystem_watcher(
    processor: RealTimeDocumentProcessor,
    watch_paths: list[str] | None = None,
    **kwargs
) -> FileSystemEventListener:
    """Create a file system watcher for real-time processing."""
    listener = FileSystemEventListener(processor, **kwargs)
    processor.add_event_listener(listener)
    return listener
