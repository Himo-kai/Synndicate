"""
Dynamic Coder Agent for implementation and development tasks.

This agent specializes in:
- Code generation and implementation
- Following coding best practices
- Integrating with existing codebases
- Writing clean, maintainable code
"""

import re
from typing import Any

from ..config.settings import AgentConfig, ModelEndpoint
from ..observability.logging import get_logger
from .base import Agent, AgentResponse

logger = get_logger(__name__)


class DynamicCoderAgent(Agent):
    """
    Specialized agent for coding and implementation tasks.

    Capabilities:
    - Code generation and implementation
    - Following coding standards and best practices
    - Integration with existing codebases
    - Multiple programming language support
    """

    def __init__(
        self,
        specialization: str | None = None,
        endpoint: ModelEndpoint | None = None,
        config: AgentConfig | None = None,
        http_client=None,
        model_manager=None,
    ):
        # Provide sensible defaults so tests can instantiate without DI
        endpoint = endpoint or ModelEndpoint(name="mock-coder", base_url="local")
        config = config or AgentConfig()
        super().__init__(
            endpoint=endpoint, config=config, http_client=http_client, model_manager=model_manager
        )
        self.specialization = specialization  # e.g., "python", "javascript", "rust"

    def system_prompt(self) -> str:
        base_prompt = """You are an expert Coder Agent specialized in software implementation and development.

Your responsibilities:
1. Write clean, maintainable, and efficient code
2. Follow established coding standards and best practices
3. Implement features according to specifications
4. Integrate seamlessly with existing codebases
5. Add appropriate error handling and logging
6. Write self-documenting code with clear variable names
7. Consider performance and security implications

Coding Standards:
- Use clear, descriptive variable and function names
- Add docstrings and comments for complex logic
- Follow language-specific conventions (PEP 8 for Python, etc.)
- Implement proper error handling and validation
- Write modular, reusable code
- Consider edge cases and error conditions

Always provide:
- Complete, runnable code implementations
- Brief explanations of key design decisions
- Any dependencies or setup requirements
- Testing considerations or example usage"""

        if self.specialization:
            base_prompt += f"\n\nSpecialization: You are particularly expert in {self.specialization} development."

        return base_prompt

    def _calculate_confidence_factors(self, response: str) -> dict[str, float]:
        """Calculate confidence factors specific to coding tasks."""
        factors = {}

        # Code structure factor
        structure_score = 0.0

        # Check for proper code blocks
        if "```" in response:
            structure_score += 0.3

        # Check for function/class definitions
        if re.search(r"(def |class |function |const |let |var )", response):
            structure_score += 0.2

        # Check for imports/includes
        if re.search(r"(import |from |#include|require\()", response):
            structure_score += 0.1

        # Check for error handling
        if re.search(r"(try:|except:|catch|throw|error)", response, re.IGNORECASE):
            structure_score += 0.2

        # Check for documentation
        if re.search(r'("""|\'\'\')|(//|#)\s*\w+', response):
            structure_score += 0.2

        factors["code_structure"] = min(1.0, structure_score)

        # Implementation completeness factor
        completeness_score = 0.0

        # Check for complete function implementations
        function_matches = re.findall(r"def \w+\([^)]*\):", response)
        if function_matches:
            # Check if functions have bodies (not just pass/...)
            complete_functions = len(
                [
                    f
                    for f in function_matches
                    if not re.search(r"def \w+\([^)]*\):\s*(pass|\.\.\.)", response)
                ]
            )
            completeness_score += min(0.4, complete_functions * 0.1)

        # Check for return statements
        if re.search(r"return \w+", response):
            completeness_score += 0.2

        # Check for variable assignments
        if re.search(r"\w+\s*=\s*\w+", response):
            completeness_score += 0.2

        # Check for control flow
        if re.search(r"(if |for |while |with )", response):
            completeness_score += 0.2

        factors["implementation_completeness"] = min(1.0, completeness_score)

        # Best practices factor
        practices_score = 0.0

        # Check for type hints (Python)
        if re.search(r":\s*(str|int|float|bool|List|Dict|Optional)", response):
            practices_score += 0.2

        # Check for logging
        if re.search(r"(logger\.|logging\.|console\.log)", response):
            practices_score += 0.2

        # Check for validation
        if re.search(r"(if not |assert |validate|check)", response):
            practices_score += 0.2

        # Check for constants/configuration
        if re.search(r"[A-Z_]{3,}", response):
            practices_score += 0.2

        # Check for modular design
        if len(function_matches) >= 2:
            practices_score += 0.2

        factors["best_practices"] = min(1.0, practices_score)

        return factors

    async def process(self, query: str, context: dict[str, Any] | None = None) -> AgentResponse:
        """Process a coding request with enhanced context awareness."""
        # Enhance the query with coding-specific context
        enhanced_query = self._enhance_coding_query(query, context)

        # Process with the base agent
        response = await super().process(enhanced_query, context)

        # Post-process to add coding-specific metadata
        if response.metadata is None:
            response.metadata = {}

        response.metadata.update(
            {
                "agent_type": "coder",
                "specialization": self.specialization,
                "code_analysis": self._analyze_code_response(response.response),
            }
        )

        return response

    def _enhance_coding_query(self, query: str, context: dict[str, Any] | None) -> str:
        """Enhance the query with coding-specific context and requirements."""
        enhanced_parts = [query]

        # Add context about existing codebase if available
        if context:
            if "file_path" in context:
                enhanced_parts.append(f"File path: {context['file_path']}")

            if "existing_code" in context:
                enhanced_parts.append(
                    f"Existing code context:\n```\n{context['existing_code']}\n```"
                )

            if "dependencies" in context:
                enhanced_parts.append(
                    f"Available dependencies: {', '.join(context['dependencies'])}"
                )

            if "coding_standards" in context:
                enhanced_parts.append(f"Coding standards: {context['coding_standards']}")

        # Add coding-specific requirements
        enhanced_parts.append(
            """
Requirements:
- Provide complete, runnable code
- Include proper error handling
- Add clear documentation/comments
- Follow best practices for the language
- Consider edge cases and validation
"""
        )

        return "\n\n".join(enhanced_parts)

    def _analyze_code_response(self, response: str) -> dict[str, Any]:
        """Analyze the code response for quality metrics."""
        analysis = {
            "has_code_blocks": "```" in response,
            "function_count": len(re.findall(r"def \w+\(", response)),
            "class_count": len(re.findall(r"class \w+", response)),
            "has_error_handling": bool(re.search(r"(try:|except:|catch)", response, re.IGNORECASE)),
            "has_documentation": bool(re.search(r'("""|\'\'\')|(//|#)\s*\w+', response)),
            "has_type_hints": bool(re.search(r":\s*(str|int|float|bool|List|Dict)", response)),
            "has_imports": bool(re.search(r"(import |from |#include)", response)),
            "estimated_lines": len(
                [
                    line
                    for line in response.split("\n")
                    if line.strip() and not line.strip().startswith("#")
                ]
            ),
        }

        # Calculate quality score
        quality_factors = [
            analysis["has_code_blocks"],
            analysis["has_error_handling"],
            analysis["has_documentation"],
            analysis["function_count"] > 0,
            analysis["has_imports"],
        ]

        analysis["quality_score"] = sum(quality_factors) / len(quality_factors)

        return analysis
