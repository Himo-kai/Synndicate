"""
Multi-modal agent support for text, code, and image processing.

This module extends the base agent architecture to support multiple input/output modalities:
- Text: Natural language processing and generation
- Code: Programming language analysis, generation, and execution
- Image: Visual content analysis and generation

Key features:
- Unified multi-modal input/output data models
- Content type detection and validation
- Modality-specific processing pipelines
- Cross-modal reasoning capabilities
"""

import base64
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..observability.logging import get_logger

logger = get_logger(__name__)


class ModalityType(Enum):
    """Supported modality types for multi-modal agents."""

    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    AUDIO = "audio"  # Future extension
    VIDEO = "video"  # Future extension


class CodeLanguage(Enum):
    """Supported programming languages for code modality."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    RUST = "rust"
    GO = "go"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"
    YAML = "yaml"
    JSON = "json"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class ImageFormat(Enum):
    """Supported image formats for image modality."""

    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"
    SVG = "svg"
    BMP = "bmp"
    TIFF = "tiff"


@dataclass
class TextContent:
    """Text content with metadata."""

    content: str
    language: str = "en"  # ISO language code
    encoding: str = "utf-8"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return character count."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return len(self.content.split())


@dataclass
class CodeContent:
    """Code content with language detection and metadata."""

    content: str
    language: CodeLanguage = CodeLanguage.UNKNOWN
    file_path: str | None = None
    line_numbers: tuple[int, int] | None = None  # (start, end)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-detect language if not specified."""
        if self.language == CodeLanguage.UNKNOWN and self.file_path:
            self.language = self._detect_language_from_path(self.file_path)

    def _detect_language_from_path(self, file_path: str) -> CodeLanguage:
        """Detect programming language from file extension."""
        suffix = Path(file_path).suffix.lower()

        language_map = {
            '.py': CodeLanguage.PYTHON,
            '.js': CodeLanguage.JAVASCRIPT,
            '.ts': CodeLanguage.TYPESCRIPT,
            '.java': CodeLanguage.JAVA,
            '.cpp': CodeLanguage.CPP,
            '.cc': CodeLanguage.CPP,
            '.cxx': CodeLanguage.CPP,
            '.c': CodeLanguage.C,
            '.rs': CodeLanguage.RUST,
            '.go': CodeLanguage.GO,
            '.html': CodeLanguage.HTML,
            '.htm': CodeLanguage.HTML,
            '.css': CodeLanguage.CSS,
            '.sql': CodeLanguage.SQL,
            '.sh': CodeLanguage.BASH,
            '.bash': CodeLanguage.BASH,
            '.yaml': CodeLanguage.YAML,
            '.yml': CodeLanguage.YAML,
            '.json': CodeLanguage.JSON,
            '.md': CodeLanguage.MARKDOWN,
        }

        return language_map.get(suffix, CodeLanguage.UNKNOWN)

    @property
    def line_count(self) -> int:
        """Get line count."""
        return len(self.content.splitlines())


@dataclass
class ImageContent:
    """Image content with format detection and metadata."""

    data: bytes
    format: ImageFormat | None = None
    width: int | None = None
    height: int | None = None
    file_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-detect format if not specified."""
        if self.format is None:
            self.format = self._detect_format()

    def _detect_format(self) -> ImageFormat:
        """Detect image format from data or file path."""
        # Try file path first
        if self.file_path:
            mime_type, _ = mimetypes.guess_type(self.file_path)
            if mime_type:
                format_map = {
                    'image/jpeg': ImageFormat.JPEG,
                    'image/png': ImageFormat.PNG,
                    'image/gif': ImageFormat.GIF,
                    'image/webp': ImageFormat.WEBP,
                    'image/svg+xml': ImageFormat.SVG,
                    'image/bmp': ImageFormat.BMP,
                    'image/tiff': ImageFormat.TIFF,
                }
                if mime_type in format_map:
                    return format_map[mime_type]

        # Try magic bytes detection
        if len(self.data) >= 8:
            # JPEG
            if self.data[:2] == b'\xff\xd8':
                return ImageFormat.JPEG
            # PNG
            elif self.data[:8] == b'\x89PNG\r\n\x1a\n':
                return ImageFormat.PNG
            # GIF
            elif self.data[:6] in (b'GIF87a', b'GIF89a'):
                return ImageFormat.GIF
            # WebP
            elif self.data[:4] == b'RIFF' and self.data[8:12] == b'WEBP':
                return ImageFormat.WEBP
            # BMP
            elif self.data[:2] == b'BM':
                return ImageFormat.BMP

        # Default fallback
        return ImageFormat.PNG

    def to_base64(self) -> str:
        """Convert image data to base64 string."""
        return base64.b64encode(self.data).decode('utf-8')

    @classmethod
    def from_base64(cls, base64_str: str, **kwargs) -> 'ImageContent':
        """Create ImageContent from base64 string."""
        data = base64.b64decode(base64_str)
        return cls(data=data, **kwargs)

    @property
    def size_bytes(self) -> int:
        """Get image size in bytes."""
        return len(self.data)


# Union type for all supported content types
MultiModalContent = TextContent | CodeContent | ImageContent


@dataclass
class MultiModalInput:
    """Multi-modal input containing multiple content types."""

    contents: list[MultiModalContent] = field(default_factory=list)
    primary_modality: ModalityType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_text(self, content: str, **kwargs) -> 'MultiModalInput':
        """Add text content."""
        self.contents.append(TextContent(content=content, **kwargs))
        if self.primary_modality is None:
            self.primary_modality = ModalityType.TEXT
        return self

    def add_code(self, content: str, **kwargs) -> 'MultiModalInput':
        """Add code content."""
        self.contents.append(CodeContent(content=content, **kwargs))
        if self.primary_modality is None:
            self.primary_modality = ModalityType.CODE
        return self

    def add_image(self, data: bytes, **kwargs) -> 'MultiModalInput':
        """Add image content."""
        self.contents.append(ImageContent(data=data, **kwargs))
        if self.primary_modality is None:
            self.primary_modality = ModalityType.IMAGE
        return self

    def get_contents_by_type(self, content_type: type) -> list[MultiModalContent]:
        """Get all contents of a specific type."""
        return [content for content in self.contents if isinstance(content, content_type)]

    @property
    def text_contents(self) -> list[TextContent]:
        """Get all text contents."""
        return self.get_contents_by_type(TextContent)

    @property
    def code_contents(self) -> list[CodeContent]:
        """Get all code contents."""
        return self.get_contents_by_type(CodeContent)

    @property
    def image_contents(self) -> list[ImageContent]:
        """Get all image contents."""
        return self.get_contents_by_type(ImageContent)

    @property
    def modalities(self) -> set[ModalityType]:
        """Get set of all modalities present in input."""
        modalities = set()
        for content in self.contents:
            if isinstance(content, TextContent):
                modalities.add(ModalityType.TEXT)
            elif isinstance(content, CodeContent):
                modalities.add(ModalityType.CODE)
            elif isinstance(content, ImageContent):
                modalities.add(ModalityType.IMAGE)
        return modalities


@dataclass
class MultiModalOutput:
    """Multi-modal output containing multiple content types."""

    contents: list[MultiModalContent] = field(default_factory=list)
    primary_content: MultiModalContent | None = None
    confidence: float = 0.0
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_text(self, content: str, **kwargs) -> 'MultiModalOutput':
        """Add text content to output."""
        text_content = TextContent(content=content, **kwargs)
        self.contents.append(text_content)
        if self.primary_content is None:
            self.primary_content = text_content
        return self

    def add_code(self, content: str, **kwargs) -> 'MultiModalOutput':
        """Add code content to output."""
        code_content = CodeContent(content=content, **kwargs)
        self.contents.append(code_content)
        if self.primary_content is None:
            self.primary_content = code_content
        return self

    def add_image(self, data: bytes, **kwargs) -> 'MultiModalOutput':
        """Add image content to output."""
        image_content = ImageContent(data=data, **kwargs)
        self.contents.append(image_content)
        if self.primary_content is None:
            self.primary_content = image_content
        return self

    @property
    def primary_text(self) -> str:
        """Get primary content as text representation."""
        if self.primary_content is None:
            return ""

        if isinstance(self.primary_content, (TextContent, CodeContent)):
            return self.primary_content.content
        elif isinstance(self.primary_content, ImageContent):
            format_value = self.primary_content.format.value if self.primary_content.format else "unknown"
            return f"[Image: {format_value}, {self.primary_content.size_bytes} bytes]"

        return str(self.primary_content)


class MultiModalProcessor(ABC):
    """Abstract base class for multi-modal content processors."""

    @abstractmethod
    async def process_text(self, content: TextContent) -> TextContent:
        """Process text content."""
        pass

    @abstractmethod
    async def process_code(self, content: CodeContent) -> CodeContent:
        """Process code content."""
        pass

    @abstractmethod
    async def process_image(self, content: ImageContent) -> ImageContent:
        """Process image content."""
        pass

    async def process_multimodal(self, input_data: MultiModalInput) -> MultiModalOutput:
        """Process multi-modal input and return multi-modal output."""
        output = MultiModalOutput()

        # Process each content type
        for content in input_data.contents:
            if isinstance(content, TextContent):
                processed = await self.process_text(content)
                output.add_text(processed.content, **processed.metadata)
            elif isinstance(content, CodeContent):
                processed = await self.process_code(content)
                output.add_code(processed.content, language=processed.language, **processed.metadata)
            elif isinstance(content, ImageContent):
                processed = await self.process_image(content)
                output.add_image(processed.data, format=processed.format, **processed.metadata)

        return output


def create_text_input(content: str, **kwargs) -> MultiModalInput:
    """Convenience function to create text-only input."""
    return MultiModalInput().add_text(content, **kwargs)


def create_code_input(content: str, **kwargs) -> MultiModalInput:
    """Convenience function to create code-only input."""
    return MultiModalInput().add_code(content, **kwargs)


def create_image_input(data: bytes, **kwargs: Any) -> MultiModalInput:
    """Convenience function to create image-only input."""
    return MultiModalInput().add_image(data, **kwargs)
