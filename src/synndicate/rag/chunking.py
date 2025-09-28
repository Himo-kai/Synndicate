"""
Smart chunking strategies for document processing.

Improvements over original:
- Semantic chunking based on content structure
- Language-aware chunking for code files
- Adaptive chunk sizing based on content type
- Overlap optimization for better context preservation
- Metadata extraction during chunking
"""

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..observability.logging import get_logger

logger = get_logger(__name__)


class ChunkType(Enum):
    """Types of content chunks."""

    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    DOCUMENTATION = "documentation"
    COMMENT = "comment"


@dataclass
class Chunk:
    """A chunk of content with metadata."""

    content: str
    chunk_type: ChunkType
    start_index: int
    end_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Get chunk size in characters."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return len(self.content.split())


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    @abstractmethod
    def chunk(self, content: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk content into smaller pieces."""
        ...

    def _create_chunk(
        self,
        content: str,
        chunk_type: ChunkType,
        start_index: int,
        end_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        """Create a chunk with metadata."""
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata.update(
            {
                "chunk_size": len(content),
                "word_count": len(content.split()),
                "start_index": start_index,
                "end_index": end_index,
            }
        )

        return Chunk(
            content=content.strip(),
            chunk_type=chunk_type,
            start_index=start_index,
            end_index=end_index,
            metadata=chunk_metadata,
        )


class FixedSizeChunker(ChunkingStrategy):
    """Simple fixed-size chunking with overlap."""

    def chunk(self, content: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk content into fixed-size pieces."""
        if not content.strip():
            return []

        chunks = []
        start = 0

        while start < len(content):
            end = min(start + self.max_chunk_size, len(content))

            # Try to break at word boundaries
            if end < len(content):
                # Look for the last space within the chunk
                last_space = content.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk_content = content[start:end]
            if chunk_content.strip():
                chunk = self._create_chunk(chunk_content, ChunkType.TEXT, start, end, metadata)
                chunks.append(chunk)

            # Move start position with overlap
            start = max(start + 1, end - self.overlap)

        return chunks


class SemanticChunker(ChunkingStrategy):
    """Semantic chunking based on content structure and meaning."""

    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        super().__init__(max_chunk_size, overlap)
        self.sentence_endings = re.compile(r"[.!?]+\s+")
        self.paragraph_breaks = re.compile(r"\n\s*\n")

    def chunk(self, content: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk content based on semantic boundaries."""
        if not content.strip():
            return []

        # Determine content type
        content_type = self._detect_content_type(content, metadata)

        if content_type == ChunkType.CODE:
            return self._chunk_code(content, metadata)
        elif content_type == ChunkType.MARKDOWN:
            return self._chunk_markdown(content, metadata)
        else:
            return self._chunk_text(content, metadata)

    def _detect_content_type(self, content: str, metadata: dict[str, Any] | None) -> ChunkType:
        """Detect the type of content for appropriate chunking."""
        if metadata:
            file_ext = metadata.get("file_extension", "").lower()
            if file_ext in [".py", ".js", ".ts", ".rs", ".go", ".java", ".cpp", ".c"]:
                return ChunkType.CODE
            elif file_ext in [".md", ".markdown"]:
                return ChunkType.MARKDOWN

        # Heuristic detection
        code_indicators = ["def ", "function ", "class ", "import ", "from ", "#!/"]
        if any(indicator in content for indicator in code_indicators):
            return ChunkType.CODE

        if content.startswith("#") or "```" in content:
            return ChunkType.MARKDOWN

        return ChunkType.TEXT

    def _chunk_text(self, content: str, metadata: dict[str, Any] | None) -> list[Chunk]:
        """Chunk plain text content."""
        chunks = []

        # Split by paragraphs first
        paragraphs = self.paragraph_breaks.split(content)
        current_chunk = ""
        start_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed max size, finalize current chunk
            if current_chunk and len(current_chunk) + len(paragraph) > self.max_chunk_size:
                chunk = self._create_chunk(
                    current_chunk,
                    ChunkType.TEXT,
                    start_index,
                    start_index + len(current_chunk),
                    metadata,
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + paragraph
                start_index = start_index + len(current_chunk) - len(overlap_text)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk,
                ChunkType.TEXT,
                start_index,
                start_index + len(current_chunk),
                metadata,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_code(self, content: str, metadata: dict[str, Any] | None) -> list[Chunk]:
        """Chunk code content preserving logical boundaries."""

        # Try to parse as Python first
        try:
            tree = ast.parse(content)
            return self._chunk_python_ast(content, tree, metadata)
        except SyntaxError:
            pass

        # Fall back to line-based chunking for other languages
        return self._chunk_code_lines(content, metadata)

    def _chunk_python_ast(
        self, content: str, tree: ast.AST, metadata: dict[str, Any] | None
    ) -> list[Chunk]:
        """Chunk Python code using AST analysis."""
        chunks = []
        lines = content.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, "end_lineno") else start_line + 10

                # Extract the function/class with some context
                chunk_lines = lines[max(0, start_line - 1) : min(len(lines), end_line + 1)]
                chunk_content = "\n".join(chunk_lines)

                if len(chunk_content) <= self.max_chunk_size:
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update(
                        {
                            "node_type": type(node).__name__,
                            "node_name": node.name,
                            "start_line": start_line,
                            "end_line": end_line,
                        }
                    )

                    chunk = self._create_chunk(
                        chunk_content,
                        ChunkType.CODE,
                        0,  # We'd need to calculate actual character positions
                        len(chunk_content),
                        chunk_metadata,
                    )
                    chunks.append(chunk)

        # If no AST nodes found or chunks are empty, fall back to line-based
        if not chunks:
            return self._chunk_code_lines(content, metadata)

        return chunks

    def _chunk_code_lines(self, content: str, metadata: dict[str, Any] | None) -> list[Chunk]:
        """Chunk code by logical line groupings."""
        lines = content.split("\n")
        chunks = []
        current_chunk_lines = []
        current_size = 0
        start_line = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline

            # Check if adding this line would exceed max size
            if current_size + line_size > self.max_chunk_size and current_chunk_lines:
                # Create chunk from current lines
                chunk_content = "\n".join(current_chunk_lines)
                chunk = self._create_chunk(chunk_content, ChunkType.CODE, start_line, i, metadata)
                chunks.append(chunk)

                # Start new chunk with some overlap
                overlap_lines = max(1, self.overlap // 50)  # Rough estimate
                current_chunk_lines = current_chunk_lines[-overlap_lines:] + [line]
                current_size = sum(len(line_text) + 1 for line_text in current_chunk_lines)
                start_line = i - overlap_lines
            else:
                current_chunk_lines.append(line)
                current_size += line_size

        # Add final chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunk = self._create_chunk(
                chunk_content, ChunkType.CODE, start_line, len(lines), metadata
            )
            chunks.append(chunk)

        return chunks

    def _chunk_markdown(self, content: str, metadata: dict[str, Any] | None) -> list[Chunk]:
        """Chunk Markdown content preserving structure."""
        chunks = []

        # Split by headers
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        sections = header_pattern.split(content)

        current_chunk = ""
        start_index = 0

        for i in range(0, len(sections), 3):  # sections come in groups of 3 from split
            if i + 2 < len(sections):
                header_level = sections[i + 1]
                header_text = sections[i + 2]
                section_content = sections[i] if i > 0 else ""

                full_section = f"{header_level} {header_text}\n{section_content}".strip()

                if len(current_chunk) + len(full_section) > self.max_chunk_size and current_chunk:
                    # Finalize current chunk
                    chunk = self._create_chunk(
                        current_chunk,
                        ChunkType.MARKDOWN,
                        start_index,
                        start_index + len(current_chunk),
                        metadata,
                    )
                    chunks.append(chunk)

                    current_chunk = full_section
                    start_index = start_index + len(current_chunk)
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + full_section
                    else:
                        current_chunk = full_section
            else:
                # Handle remaining content
                remaining = sections[i] if i < len(sections) else ""
                if remaining.strip():
                    current_chunk += remaining

        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk,
                ChunkType.MARKDOWN,
                start_index,
                start_index + len(current_chunk),
                metadata,
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= self.overlap:
            return text

        # Try to find a good breaking point for overlap
        overlap_start = len(text) - self.overlap

        # Look for sentence boundary
        sentence_match = self.sentence_endings.search(text, overlap_start)
        if sentence_match:
            return text[sentence_match.end() :]

        # Look for word boundary
        space_index = text.find(" ", overlap_start)
        if space_index != -1:
            return text[space_index + 1 :]

        # Fall back to character-based overlap
        return text[-self.overlap :]


class CodeAwareChunker(SemanticChunker):
    """Enhanced chunker with deep code understanding."""

    def __init__(self, max_chunk_size: int = 1500, overlap: int = 150):
        super().__init__(max_chunk_size, overlap)
        self.language_patterns = {
            "python": {
                "function": re.compile(r"^(async\s+)?def\s+\w+\s*\("),
                "class": re.compile(r"^class\s+\w+"),
                "import": re.compile(r"^(from\s+\w+\s+)?import\s+"),
                "comment": re.compile(r"^\s*#"),
                "docstring": re.compile(r'^\s*"""'),
            },
            "javascript": {
                "function": re.compile(r"^(async\s+)?function\s+\w+\s*\(|^\w+\s*=\s*(async\s+)?\("),
                "class": re.compile(r"^class\s+\w+"),
                "import": re.compile(r"^import\s+|^from\s+"),
                "comment": re.compile(r"^\s*//|^\s*/\*"),
            },
            "rust": {
                "function": re.compile(r"^(pub\s+)?(async\s+)?fn\s+\w+"),
                "struct": re.compile(r"^(pub\s+)?struct\s+\w+"),
                "impl": re.compile(r"^impl\s+"),
                "use": re.compile(r"^use\s+"),
                "comment": re.compile(r"^\s*//"),
            },
        }

    def _detect_language(self, content: str, metadata: dict[str, Any] | None) -> str | None:
        """Detect programming language from content and metadata."""
        if metadata:
            file_ext = metadata.get("file_extension", "").lower()
            ext_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "javascript",
                ".rs": "rust",
                ".go": "go",
                ".java": "java",
            }
            if file_ext in ext_map:
                return ext_map[file_ext]

        # Heuristic detection
        if "def " in content and "import " in content:
            return "python"
        elif "function " in content or "=>" in content:
            return "javascript"
        elif "fn " in content and "use " in content:
            return "rust"

        return None

    def _chunk_code(self, content: str, metadata: dict[str, Any] | None) -> list[Chunk]:
        """Enhanced code chunking with language awareness."""
        language = self._detect_language(content, metadata)

        if language and language in self.language_patterns:
            return self._chunk_by_language_patterns(content, language, metadata)
        else:
            return super()._chunk_code(content, metadata)

    def _chunk_by_language_patterns(
        self, content: str, language: str, metadata: dict[str, Any] | None
    ) -> list[Chunk]:
        """Chunk code using language-specific patterns."""
        patterns = self.language_patterns[language]
        lines = content.split("\n")
        chunks = []

        current_chunk_lines = []
        current_function = None
        current_size = 0
        start_line = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1

            # Check for function/class definitions
            for pattern_name, pattern in patterns.items():
                if (
                    pattern.match(line.strip())
                    and current_chunk_lines
                    and pattern_name in ["function", "class", "struct"]
                ):
                    # If we have a current chunk and this is a new function/class
                    # Finalize current chunk
                    chunk_content = "\n".join(current_chunk_lines)
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update(
                        {
                            "language": language,
                            "contains_function": current_function,
                            "start_line": start_line,
                            "end_line": i,
                        }
                    )

                    chunk = self._create_chunk(
                        chunk_content, ChunkType.CODE, start_line, i, chunk_metadata
                    )
                    chunks.append(chunk)

                    # Start new chunk
                    current_chunk_lines = [line]
                    current_size = line_size
                    start_line = i
                    current_function = line.strip()
                    continue

            # Add line to current chunk
            current_chunk_lines.append(line)
            current_size += line_size

            # Check size limit
            if current_size > self.max_chunk_size:
                chunk_content = "\n".join(current_chunk_lines)
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update(
                    {
                        "language": language,
                        "contains_function": current_function,
                        "start_line": start_line,
                        "end_line": i + 1,
                    }
                )

                chunk = self._create_chunk(
                    chunk_content, ChunkType.CODE, start_line, i + 1, chunk_metadata
                )
                chunks.append(chunk)

                # Reset for next chunk
                current_chunk_lines = []
                current_size = 0
                start_line = i + 1
                current_function = None

        # Add final chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update(
                {
                    "language": language,
                    "contains_function": current_function,
                    "start_line": start_line,
                    "end_line": len(lines),
                }
            )

            chunk = self._create_chunk(
                chunk_content, ChunkType.CODE, start_line, len(lines), chunk_metadata
            )
            chunks.append(chunk)

        return chunks
