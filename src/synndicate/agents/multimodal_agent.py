"""
Multi-modal agent implementation extending the base agent architecture.

This module provides concrete implementations of multi-modal agents that can process
text, code, and image inputs simultaneously, enabling cross-modal reasoning and
generation capabilities.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config.settings import AgentConfig, ModelEndpoint
from ..observability.logging import get_logger
from ..observability.tracing import trace_span
from .base import Agent, AgentResponse
from .multimodal import (
    CodeContent,
    ImageContent,
    ModalityType,
    MultiModalInput,
    MultiModalOutput,
    MultiModalProcessor,
    TextContent,
)

logger = get_logger(__name__)


@dataclass
class MultiModalAgentResponse(AgentResponse):
    """Enhanced agent response with multi-modal support."""

    multimodal_output: MultiModalOutput | None = None
    primary_modality: ModalityType | None = None
    cross_modal_reasoning: str = ""
    modality_confidence: dict[ModalityType, float] = field(default_factory=dict)

    @property
    def has_multimodal_content(self) -> bool:
        """Check if response contains multi-modal content."""
        return self.multimodal_output is not None and len(self.multimodal_output.contents) > 0

    @property
    def supported_modalities(self) -> set[ModalityType]:
        """Get set of modalities present in the response."""
        if not self.multimodal_output:
            return set()

        modalities = set()
        for content in self.multimodal_output.contents:
            if isinstance(content, TextContent):
                modalities.add(ModalityType.TEXT)
            elif isinstance(content, CodeContent):
                modalities.add(ModalityType.CODE)
            elif isinstance(content, ImageContent):
                modalities.add(ModalityType.IMAGE)

        return modalities


class MultiModalAgent(Agent, MultiModalProcessor):
    """
    Multi-modal agent capable of processing text, code, and image inputs.

    This agent extends the base Agent class to support multiple input/output modalities
    while maintaining compatibility with the existing agent ecosystem.
    """

    def __init__(
        self,
        endpoint: ModelEndpoint,
        config: AgentConfig,
        http_client: httpx.AsyncClient | None = None,
        model_manager: Any | None = None,
        supported_modalities: set[ModalityType] | None = None,
    ):
        super().__init__(endpoint, config, http_client, model_manager)
        self.supported_modalities = supported_modalities or {
            ModalityType.TEXT,
            ModalityType.CODE,
            ModalityType.IMAGE,
        }

    @abstractmethod
    async def process_multimodal_query(
        self,
        input_data: MultiModalInput,
        context: dict[str, Any] | None = None
    ) -> MultiModalAgentResponse:
        """
        Process a multi-modal query and return a multi-modal response.

        This is the main entry point for multi-modal processing.
        Subclasses must implement this method to define their specific
        multi-modal processing logic.
        """
        pass

    async def process_text(self, content: TextContent) -> TextContent:
        """Process text content using the agent's text processing capabilities."""
        with trace_span("multimodal_agent.process_text"):
            # Use the existing text processing pipeline
            response = await self.process(content.content)

            return TextContent(
                content=response.response,
                metadata={
                    "confidence": response.confidence,
                    "reasoning": response.reasoning,
                    "execution_time": response.execution_time,
                    **content.metadata,
                }
            )

    async def process_code(self, content: CodeContent) -> CodeContent:
        """Process code content with language-specific handling."""
        with trace_span("multimodal_agent.process_code"):
            # Build code-specific prompt
            code_prompt = self._build_code_prompt(content)
            response = await self.process(code_prompt)

            # Extract code from response (assuming it's wrapped in code blocks)
            processed_code = self._extract_code_from_response(response.response, content.language)

            return CodeContent(
                content=processed_code,
                language=content.language,
                file_path=content.file_path,
                line_numbers=content.line_numbers,
                metadata={
                    "confidence": response.confidence,
                    "reasoning": response.reasoning,
                    "execution_time": response.execution_time,
                    "original_language": content.language.value if content.language else "unknown",
                    **content.metadata,
                }
            )

    async def process_image(self, content: ImageContent) -> ImageContent:
        """Process image content (placeholder for vision model integration)."""
        with trace_span("multimodal_agent.process_image"):
            # For now, return the image with analysis metadata
            # In a full implementation, this would integrate with vision models

            analysis = await self._analyze_image_metadata(content)

            return ImageContent(
                data=content.data,
                format=content.format,
                width=content.width,
                height=content.height,
                file_path=content.file_path,
                metadata={
                    "analysis": analysis,
                    "size_bytes": content.size_bytes,
                    "format": content.format.value if content.format else "unknown",
                    **content.metadata,
                }
            )

    def _build_code_prompt(self, content: CodeContent) -> str:
        """Build a code-specific prompt for processing."""
        language_name = content.language.value if content.language else "unknown"

        prompt_parts = [
            f"Analyze and process the following {language_name} code:",
            "",
            "```" + language_name,
            content.content,
            "```",
            "",
            "Provide analysis, suggestions, or modifications as appropriate.",
        ]

        if content.file_path:
            prompt_parts.insert(1, f"File: {content.file_path}")

        if content.line_numbers:
            start, end = content.line_numbers
            prompt_parts.insert(-3, f"Lines {start}-{end}:")

        return "\n".join(prompt_parts)

    def _extract_code_from_response(self, response: str, language: Any | None) -> str:
        """Extract code blocks from agent response."""
        import re

        language_name = language.value if language else ""

        # Try to find code blocks with language specification
        pattern1 = rf"```{re.escape(language_name)}\n(.*?)\n```"
        matches_lang: list[str] = re.findall(pattern1, response, re.DOTALL | re.IGNORECASE)

        if matches_lang:
            return matches_lang[0].strip()

        # Try to find any code blocks
        pattern2 = r"```.*?\n(.*?)\n```"
        matches_any: list[str] = re.findall(pattern2, response, re.DOTALL)

        if matches_any:
            return matches_any[0].strip()

        # If no code blocks found, return the original response
        return response.strip()

    async def _analyze_image_metadata(self, content: ImageContent) -> dict[str, Any]:
        """Analyze image metadata (placeholder for vision model integration)."""
        # Basic metadata analysis - explicitly typed to match actual values
        analysis: dict[str, int | float | str | bool] = {
            "format": content.format.value if content.format else "unknown",
            "size_bytes": content.size_bytes,
            "has_dimensions": content.width is not None and content.height is not None,
        }

        if content.width and content.height:
            analysis.update({
                "width": content.width,
                "height": content.height,
                "aspect_ratio": float(content.width / content.height),
                "total_pixels": content.width * content.height,
            })

        # Placeholder for actual vision model analysis
        analysis["vision_analysis"] = "Vision model integration pending"

        return analysis

    def _calculate_multimodal_confidence(
        self,
        input_data: MultiModalInput,
        output: MultiModalOutput
    ) -> dict[str, float]:
        """Calculate confidence scores for multi-modal processing."""
        factors = {}

        # Base confidence from modality support
        supported_count = len(input_data.modalities.intersection(self.supported_modalities))
        total_count = len(input_data.modalities)

        if total_count > 0:
            factors["modality_support"] = supported_count / total_count
        else:
            factors["modality_support"] = 1.0

        # Content complexity factor
        total_content_size = sum(
            len(content.content) if hasattr(content, 'content') else content.size_bytes
            for content in input_data.contents
        )

        # Normalize complexity (higher complexity = lower confidence)
        if total_content_size > 0:
            complexity_factor = min(1.0, 1000 / total_content_size)  # Arbitrary scaling
            factors["content_complexity"] = complexity_factor

        # Cross-modal consistency (placeholder)
        factors["cross_modal_consistency"] = 0.8  # Default value

        return factors

    async def process(self, query: str, context: dict[str, Any] | None = None) -> MultiModalAgentResponse:
        """
        Process a text query with multi-modal capabilities.

        This method maintains compatibility with the base Agent interface
        while providing enhanced multi-modal functionality.
        """
        # Convert text query to multi-modal input
        input_data = MultiModalInput().add_text(query)

        # Process using multi-modal pipeline
        return await self.process_multimodal_query(input_data, context)


class TextCodeAgent(MultiModalAgent):
    """Specialized multi-modal agent for text and code processing."""

    def __init__(
        self,
        endpoint: ModelEndpoint,
        config: AgentConfig,
        http_client: httpx.AsyncClient | None = None,
        model_manager: Any | None = None,
    ):
        super().__init__(
            endpoint=endpoint,
            config=config,
            http_client=http_client,
            model_manager=model_manager,
            supported_modalities={ModalityType.TEXT, ModalityType.CODE},
        )

    async def process_multimodal_query(
        self,
        input_data: MultiModalInput,
        context: dict[str, Any] | None = None
    ) -> MultiModalAgentResponse:
        """Process text and code inputs with cross-modal reasoning."""
        with trace_span("text_code_agent.process_multimodal_query"):
            # Process the multi-modal input
            output = await self.process_multimodal(input_data)

            # Build cross-modal reasoning
            cross_modal_reasoning = await self._build_cross_modal_reasoning(input_data, output)

            # Calculate confidence factors
            confidence_factors = self._calculate_multimodal_confidence(input_data, output)
            base_confidence = self._combine_confidence_factors(confidence_factors)

            # Create enhanced response
            return MultiModalAgentResponse(
                response=output.primary_text,
                confidence=base_confidence,
                reasoning=output.reasoning,
                multimodal_output=output,
                primary_modality=input_data.primary_modality,
                cross_modal_reasoning=cross_modal_reasoning,
                modality_confidence={
                    ModalityType.TEXT: confidence_factors.get("text_confidence", base_confidence),
                    ModalityType.CODE: confidence_factors.get("code_confidence", base_confidence),
                },
                confidence_factors=confidence_factors,
                metadata={
                    "input_modalities": list(input_data.modalities),
                    "output_modalities": list(output.contents),
                    "supported_modalities": list(self.supported_modalities),
                }
            )

    @property
    def system_prompt(self) -> str:
        """Get the system prompt for text-code agent."""
        return (
            "You are a specialized multi-modal AI agent capable of processing both text and code. "
            "You can understand natural language requirements and translate them into code, "
            "analyze existing code and explain it in natural language, and perform cross-modal "
            "reasoning between textual descriptions and code implementations."
        )

    def _calculate_confidence_factors(self, response: str) -> dict[str, float]:
        """Calculate confidence factors specific to text-code processing."""
        factors = {}

        # Code block detection
        import re
        code_blocks = re.findall(r'```.*?```', response, re.DOTALL)
        if code_blocks:
            factors['has_code_blocks'] = 0.8
        else:
            factors['has_code_blocks'] = 0.4

        # Technical terminology
        tech_terms = ['function', 'class', 'method', 'variable', 'import', 'return']
        tech_count = sum(1 for term in tech_terms if term.lower() in response.lower())
        factors['technical_content'] = min(1.0, tech_count / 10.0)

        return factors

    async def _build_cross_modal_reasoning(
        self,
        input_data: MultiModalInput,
        output: MultiModalOutput
    ) -> str:
        """Build reasoning that explains cross-modal relationships."""
        reasoning_parts = []

        text_contents = input_data.text_contents
        code_contents = input_data.code_contents

        if text_contents and code_contents:
            reasoning_parts.append(
                f"Analyzed {len(text_contents)} text input(s) and {len(code_contents)} code input(s). "
                "Cross-modal reasoning applied to understand relationships between natural language "
                "requirements and code implementation."
            )
        elif text_contents:
            reasoning_parts.append(
                f"Processed {len(text_contents)} text input(s) with code generation capabilities."
            )
        elif code_contents:
            reasoning_parts.append(
                f"Analyzed {len(code_contents)} code input(s) with natural language explanation."
            )

        return " ".join(reasoning_parts)


class VisionCodeAgent(MultiModalAgent):
    """Specialized multi-modal agent for image and code processing."""

    def __init__(
        self,
        endpoint: ModelEndpoint,
        config: AgentConfig,
        http_client: httpx.AsyncClient | None = None,
        model_manager: Any | None = None,
    ):
        super().__init__(
            endpoint=endpoint,
            config=config,
            http_client=http_client,
            model_manager=model_manager,
            supported_modalities={ModalityType.IMAGE, ModalityType.CODE, ModalityType.TEXT},
        )

    async def process_multimodal_query(
        self,
        input_data: MultiModalInput,
        context: dict[str, Any] | None = None
    ) -> MultiModalAgentResponse:
        """Process image and code inputs for visual code analysis."""
        with trace_span("vision_code_agent.process_multimodal_query"):
            # Process the multi-modal input
            output = await self.process_multimodal(input_data)

            # Build vision-code reasoning
            cross_modal_reasoning = await self._build_vision_code_reasoning(input_data, output)

            # Calculate confidence factors
            confidence_factors = self._calculate_multimodal_confidence(input_data, output)
            base_confidence = self._combine_confidence_factors(confidence_factors)

            return MultiModalAgentResponse(
                response=output.primary_text,
                confidence=base_confidence,
                reasoning=output.reasoning,
                multimodal_output=output,
                primary_modality=input_data.primary_modality,
                cross_modal_reasoning=cross_modal_reasoning,
                modality_confidence={
                    ModalityType.IMAGE: confidence_factors.get("image_confidence", base_confidence),
                    ModalityType.CODE: confidence_factors.get("code_confidence", base_confidence),
                    ModalityType.TEXT: confidence_factors.get("text_confidence", base_confidence),
                },
                confidence_factors=confidence_factors,
                metadata={
                    "input_modalities": list(input_data.modalities),
                    "vision_analysis_enabled": True,
                    "supported_modalities": list(self.supported_modalities),
                }
            )

    @property
    def system_prompt(self) -> str:
        """Get the system prompt for vision-code agent."""
        return (
            "You are a specialized multi-modal AI agent capable of processing images, code, and text. "
            "You can analyze visual content like diagrams, screenshots, and UI mockups, then generate "
            "corresponding code implementations. You can also analyze code and create visual "
            "representations or explanations of how the code would appear or function visually."
        )

    def _calculate_confidence_factors(self, response: str) -> dict[str, float]:
        """Calculate confidence factors specific to vision-code processing."""
        factors = {}

        # Visual terminology
        visual_terms = ['image', 'visual', 'display', 'render', 'ui', 'interface', 'layout']
        visual_count = sum(1 for term in visual_terms if term.lower() in response.lower())
        factors['visual_content'] = min(1.0, visual_count / 5.0)

        # Code analysis
        import re
        code_blocks = re.findall(r'```.*?```', response, re.DOTALL)
        factors['has_code_blocks'] = 0.8 if code_blocks else 0.3

        # Cross-modal indicators
        cross_modal_terms = ['based on', 'from the image', 'visual analysis', 'code representation']
        cross_modal_count = sum(1 for term in cross_modal_terms if term.lower() in response.lower())
        factors['cross_modal_reasoning'] = min(1.0, cross_modal_count / 3.0)

        return factors

    async def _build_vision_code_reasoning(
        self,
        input_data: MultiModalInput,
        output: MultiModalOutput
    ) -> str:
        """Build reasoning for vision-code cross-modal analysis."""
        reasoning_parts = []

        image_contents = input_data.image_contents
        code_contents = input_data.code_contents
        text_contents = input_data.text_contents

        if image_contents and code_contents:
            reasoning_parts.append(
                f"Analyzed {len(image_contents)} image(s) and {len(code_contents)} code input(s). "
                "Applied vision-code cross-modal reasoning to understand visual elements and "
                "their code representations."
            )
        elif image_contents:
            reasoning_parts.append(
                f"Processed {len(image_contents)} image(s) with code generation from visual analysis."
            )

        if text_contents:
            reasoning_parts.append(
                f"Incorporated {len(text_contents)} text input(s) for contextual understanding."
            )

        return " ".join(reasoning_parts)


# Factory functions for creating specialized multi-modal agents
def create_text_code_agent(
    endpoint: ModelEndpoint,
    config: AgentConfig | None = None,
    **kwargs
) -> TextCodeAgent:
    """Create a text-code specialized multi-modal agent."""
    if config is None:
        config = AgentConfig()

    return TextCodeAgent(endpoint=endpoint, config=config, **kwargs)


def create_vision_code_agent(
    endpoint: ModelEndpoint,
    config: AgentConfig | None = None,
    **kwargs
) -> VisionCodeAgent:
    """Create a vision-code specialized multi-modal agent."""
    if config is None:
        config = AgentConfig()

    return VisionCodeAgent(endpoint=endpoint, config=config, **kwargs)
