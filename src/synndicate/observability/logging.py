"""
Structured logging for Synndicate AI system with trace ID support.
"""

import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

# Context variable for trace ID propagation
trace_id_ctx: ContextVar[str | None] = ContextVar("trace_id", default=None)

# Global logger cache
_loggers: dict[str, "StructuredLogger"] = {}


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logs with trace ID support."""

    def format(self, record: logging.LogRecord) -> str:
        # Get current trace ID from context
        trace_id = trace_id_ctx.get() or getattr(record, "trace_id", None) or "-"

        # Extract module and operation from logger name
        parts = record.name.split(".")
        mod = parts[-1] if parts else record.name
        op = getattr(record, "op", getattr(record, "funcName", "-"))

        # Get duration if available
        duration = getattr(record, "ms", getattr(record, "duration_ms", None))
        ms_part = f" ms={duration:.1f}" if duration is not None else ""

        # Build structured log line
        timestamp = datetime.now(UTC).isoformat()
        level = record.levelname
        msg = record.getMessage()

        # Additional fields from record
        extra_fields = ""
        for key, value in getattr(record, "__dict__", {}).items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "trace_id",
                "op",
                "ms",
                "duration_ms",
            }:
                extra_fields += f" {key}={value}"

        return f't={timestamp} level={level} trace={trace_id} mod={mod} op={op}{ms_part} msg="{msg}"{extra_fields}'


class StructuredLogger:
    """Structured logger with trace ID and operation support."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

    def _log(self, level: int, msg: str, **kwargs):
        """Internal logging method with structured fields."""
        # Filter out reserved LogRecord attributes to avoid KeyError
        reserved_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "getMessage",
            "exc_info",
            "exc_text",
            "stack_info",
            "message",
        }
        extra = {k: v for k, v in kwargs.items() if k not in reserved_attrs}
        extra["trace_id"] = trace_id_ctx.get()
        self.logger.log(level, msg, extra=extra)

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)

    def timed(self, msg: str, duration_ms: float, **kwargs):
        """Log with timing information."""
        kwargs["ms"] = duration_ms
        self.info(msg, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for the given module."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration with structured formatter."""
    # Create handler with structured formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))

    # Disable other loggers to avoid noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)


def set_trace_id(trace_id: str) -> None:
    """Set trace ID in current context."""
    trace_id_ctx.set(trace_id)


def get_trace_id() -> str | None:
    """Get current trace ID from context."""
    return trace_id_ctx.get()


def clear_trace_id() -> None:
    """Clear trace ID from current context."""
    trace_id_ctx.set(None)


# Convenience function for getting current trace context
def get_trace_context() -> dict[str, str | None]:
    """Get current trace context."""
    return {"trace_id": get_trace_id()}
