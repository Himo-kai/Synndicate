"""
Enhanced Coder Agent with improved code generation and validation.

Improvements over original:
- Better code quality assessment
- Language-specific validation
- Security checks
- Performance considerations
- Test generation capabilities
"""

import ast
import re
from dataclasses import dataclass
from enum import Enum

from ..observability.logging import get_logger
from .base import Agent, AgentResponse

logger = get_logger(__name__)


class CodeQuality(Enum):
    """Code quality levels."""

    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


@dataclass
class CodeBlock:
    """Extracted code block with metadata."""

    language: str
    code: str
    line_count: int
    has_comments: bool
    has_error_handling: bool
    has_type_hints: bool
    complexity_score: float


@dataclass
class CodeAnalysis:
    """Analysis of generated code."""

    blocks: list[CodeBlock]
    total_lines: int
    languages: set[str]
    quality: CodeQuality
    security_issues: list[str]
    performance_notes: list[str]
    test_coverage: bool


class CoderAgent(Agent):
    """
    Enhanced coder agent with improved code generation and validation.

    Improvements:
    - Language-specific code validation
    - Security vulnerability detection
    - Code quality assessment
    - Performance optimization suggestions
    - Test generation capabilities
    """

    def system_prompt(self) -> str:
        return """You are an expert Coding Agent responsible for implementing high-quality, secure, and maintainable code.

Your responsibilities:
1. Write clean, readable, and well-documented code
2. Follow language-specific best practices and conventions
3. Include proper error handling and input validation
4. Add type hints where applicable (Python, TypeScript)
5. Write secure code that avoids common vulnerabilities
6. Consider performance implications
7. Include relevant tests when appropriate
8. Provide clear explanations of your implementation choices

Code Quality Standards:
- Use descriptive variable and function names
- Add docstrings for functions and classes
- Handle edge cases and errors gracefully
- Follow DRY (Don't Repeat Yourself) principles
- Use appropriate data structures and algorithms
- Include input validation for user-facing functions
- Avoid hardcoded values; use constants or configuration

Security Considerations:
- Validate and sanitize all inputs
- Avoid SQL injection, XSS, and other common vulnerabilities
- Use secure random number generation when needed
- Don't log sensitive information
- Use parameterized queries for database operations
- Implement proper authentication and authorization

Performance Guidelines:
- Choose efficient algorithms and data structures
- Avoid unnecessary loops and computations
- Use lazy evaluation when appropriate
- Consider memory usage for large datasets
- Profile and optimize bottlenecks

Always explain your implementation approach and any trade-offs you made."""

    def _calculate_confidence_factors(self, response: str) -> dict[str, float]:
        """Calculate confidence factors specific to coding."""
        factors = {}

        # Code structure factor
        structure_score = 0.0
        code_blocks = re.findall(r"```(\w+)?\n(.*?)\n```", response, re.DOTALL)

        if code_blocks:
            structure_score += 0.3

            # Analyze code quality
            for lang, code in code_blocks:
                if lang and lang.lower() in ["python", "javascript", "typescript", "rust", "go"]:
                    structure_score += 0.1

                # Check for good practices
                if re.search(r"def \w+\(.*\):", code) or re.search(r"function \w+\(", code):
                    structure_score += 0.1

                # Check for error handling
                if any(keyword in code.lower() for keyword in ["try", "except", "catch", "error"]):
                    structure_score += 0.1

                # Check for comments/docstrings
                if '"""' in code or "'''" in code or "//" in code or "/*" in code:
                    structure_score += 0.1

        factors["structure"] = min(1.0, structure_score)

        # Implementation completeness
        completeness_score = 0.0

        # Check for imports/includes
        if any(keyword in response for keyword in ["import ", "from ", "#include", "use "]):
            completeness_score += 0.2

        # Check for main function or entry point
        if any(pattern in response for pattern in ["if __name__", "main()", "fn main"]):
            completeness_score += 0.2

        # Check for type hints (Python/TypeScript)
        if re.search(r":\s*\w+", response) or "->" in response:
            completeness_score += 0.2

        # Check for tests
        if any(keyword in response.lower() for keyword in ["test_", "assert", "expect", "should"]):
            completeness_score += 0.2

        # Check for documentation
        explanation_keywords = [
            "explanation",
            "approach",
            "implementation",
            "solution",
            "algorithm",
        ]
        if any(keyword in response.lower() for keyword in explanation_keywords):
            completeness_score += 0.2

        factors["completeness"] = min(1.0, completeness_score)

        # Security awareness
        security_score = 0.0

        # Positive security indicators
        security_good = ["validate", "sanitize", "escape", "parameterized", "secure"]
        security_mentions = sum(1 for term in security_good if term in response.lower())
        security_score += min(0.3, security_mentions * 0.1)

        # Negative security indicators (reduce confidence)
        security_bad = ["eval(", "exec(", "system(", "shell=True", "innerHTML"]
        security_issues = sum(1 for term in security_bad if term in response)
        security_score -= security_issues * 0.2

        factors["security"] = max(0.0, min(1.0, security_score + 0.5))  # Base 0.5

        return factors

    def extract_code_blocks(self, response: str) -> list[CodeBlock]:
        """Extract and analyze code blocks from response."""
        blocks = []
        code_pattern = r"```(\w+)?\n(.*?)\n```"
        matches = re.findall(code_pattern, response, re.DOTALL)

        for lang, code in matches:
            if not code.strip():
                continue

            language = lang.lower() if lang else "unknown"
            lines = code.split("\n")
            line_count = len([line for line in lines if line.strip()])

            # Analyze code characteristics
            has_comments = any(
                line.strip().startswith(("#", "//", "/*", '"""', "'''")) for line in lines
            )

            has_error_handling = any(
                keyword in code.lower() for keyword in ["try", "except", "catch", "error", "panic"]
            )

            has_type_hints = bool(re.search(r":\s*\w+", code)) or "->" in code

            # Simple complexity score based on control structures
            complexity_indicators = ["if", "for", "while", "match", "switch", "loop"]
            complexity_score = (
                sum(
                    len(re.findall(rf"\b{indicator}\b", code, re.IGNORECASE))
                    for indicator in complexity_indicators
                )
                / max(1, line_count)
                * 10
            )

            blocks.append(
                CodeBlock(
                    language=language,
                    code=code,
                    line_count=line_count,
                    has_comments=has_comments,
                    has_error_handling=has_error_handling,
                    has_type_hints=has_type_hints,
                    complexity_score=complexity_score,
                )
            )

        return blocks

    def analyze_code_quality(self, blocks: list[CodeBlock]) -> CodeAnalysis:
        """Analyze overall code quality."""
        if not blocks:
            return CodeAnalysis(
                blocks=[],
                total_lines=0,
                languages=set(),
                quality=CodeQuality.POOR,
                security_issues=[],
                performance_notes=[],
                test_coverage=False,
            )

        total_lines = sum(block.line_count for block in blocks)
        languages = {block.language for block in blocks}

        # Calculate quality score
        quality_score = 0.0
        for block in blocks:
            block_score = 0.0

            if block.has_comments:
                block_score += 0.25
            if block.has_error_handling:
                block_score += 0.25
            if block.has_type_hints:
                block_score += 0.25
            if block.complexity_score < 5:  # Not too complex
                block_score += 0.25

            quality_score += block_score

        quality_score /= len(blocks)  # Average across blocks

        # Determine quality level
        if quality_score >= 0.8:
            quality = CodeQuality.EXCELLENT
        elif quality_score >= 0.6:
            quality = CodeQuality.GOOD
        elif quality_score >= 0.4:
            quality = CodeQuality.FAIR
        else:
            quality = CodeQuality.POOR

        # Check for security issues
        security_issues = []
        for block in blocks:
            if "eval(" in block.code:
                security_issues.append("Use of eval() function detected")
            if "exec(" in block.code:
                security_issues.append("Use of exec() function detected")
            if "shell=True" in block.code:
                security_issues.append("Shell injection risk with shell=True")
            if "innerHTML" in block.code:
                security_issues.append("XSS risk with innerHTML")

        # Performance notes
        performance_notes = []
        for block in blocks:
            if block.complexity_score > 10:
                performance_notes.append(f"High complexity in {block.language} code")
            if re.search(r"for.*in.*for.*in", block.code):
                performance_notes.append("Nested loops detected - consider optimization")

        # Check for test coverage
        test_coverage = any(
            any(keyword in block.code.lower() for keyword in ["test_", "assert", "expect"])
            for block in blocks
        )

        return CodeAnalysis(
            blocks=blocks,
            total_lines=total_lines,
            languages=languages,
            quality=quality,
            security_issues=security_issues,
            performance_notes=performance_notes,
            test_coverage=test_coverage,
        )

    def validate_python_syntax(self, code: str) -> list[str]:
        """Validate Python code syntax."""
        issues = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e.msg} at line {e.lineno}")
        except Exception as e:
            issues.append(f"Parse error: {str(e)}")

        return issues

    async def process(self, query: str, context: dict | None = None) -> AgentResponse:
        """Process coding request with enhanced analysis."""
        response = await super().process(query, context)

        # Extract and analyze code blocks
        code_blocks = self.extract_code_blocks(response.response)
        analysis = self.analyze_code_quality(code_blocks)

        # Add analysis to metadata
        response.metadata.update(
            {
                "code_analysis": analysis,
                "total_code_lines": analysis.total_lines,
                "languages_used": list(analysis.languages),
                "code_quality": analysis.quality.value,
                "security_issues": analysis.security_issues,
                "performance_notes": analysis.performance_notes,
                "has_tests": analysis.test_coverage,
            }
        )

        # Validate Python code if present
        for block in code_blocks:
            if block.language == "python":
                syntax_issues = self.validate_python_syntax(block.code)
                if syntax_issues:
                    response.metadata["python_syntax_issues"] = syntax_issues
                    # Reduce confidence for syntax errors
                    response.confidence *= 0.7

        # Adjust confidence based on code quality
        quality_multipliers = {
            CodeQuality.EXCELLENT: 1.1,
            CodeQuality.GOOD: 1.0,
            CodeQuality.FAIR: 0.9,
            CodeQuality.POOR: 0.7,
        }
        response.confidence *= quality_multipliers[analysis.quality]
        response.confidence = min(0.95, response.confidence)

        # Reduce confidence for security issues
        if analysis.security_issues:
            response.confidence *= 0.8 ** len(analysis.security_issues)

        return response
