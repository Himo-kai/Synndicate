"""
Synndicate Python SDK - Official client library for the Synndicate AI system.

This SDK provides a comprehensive interface for interacting with Synndicate's
advanced AI capabilities, including multi-modal agents, real-time processing,
and analytics.

Key Features:
- Multi-modal agent interactions (text, code, image)
- Real-time document processing
- Usage analytics and optimization insights
- Async/await support for high performance
- Type hints for better developer experience
- Comprehensive error handling and retries
"""

from .client import SynndicateClient
from .exceptions import (
    SynndicateError,
    AuthenticationError,
    RateLimitError,
    ProcessingError,
    ValidationError,
)
from .models import (
    AgentResponse,
    MultiModalInput,
    MultiModalOutput,
    ProcessingTask,
    AnalyticsReport,
    TextContent,
    CodeContent,
    ImageContent,
)
from .agents import (
    TextAgent,
    CodeAgent,
    MultiModalAgent,
    VisionAgent,
)

__version__ = "1.0.0"
__author__ = "Synndicate Team"
__email__ = "support@synndicate.ai"

__all__ = [
    # Core client
    "SynndicateClient",
    
    # Exceptions
    "SynndicateError",
    "AuthenticationError", 
    "RateLimitError",
    "ProcessingError",
    "ValidationError",
    
    # Data models
    "AgentResponse",
    "MultiModalInput",
    "MultiModalOutput", 
    "ProcessingTask",
    "AnalyticsReport",
    "TextContent",
    "CodeContent",
    "ImageContent",
    
    # Agent interfaces
    "TextAgent",
    "CodeAgent", 
    "MultiModalAgent",
    "VisionAgent",
]
