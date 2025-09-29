"""
Dynamic Critic Agent for code review and quality assessment.

This agent specializes in:
- Code review and quality analysis
- Identifying potential issues and improvements
- Providing constructive feedback
- Ensuring best practices compliance
"""

import re
from typing import Any

from ..observability.logging import get_logger
from ..config.settings import ModelEndpoint, AgentConfig
from .base import Agent, AgentResponse

logger = get_logger(__name__)


class DynamicCriticAgent(Agent):
    """
    Specialized agent for code review and quality assessment.
    
    Capabilities:
    - Code review and analysis
    - Quality assessment and scoring
    - Best practices validation
    - Security and performance analysis
    - Constructive feedback generation
    """
    
    def __init__(
        self,
        focus_area: str | None = None,
        endpoint: ModelEndpoint | None = None,
        config: AgentConfig | None = None,
        http_client=None,
        model_manager=None,
    ):
        # Provide sensible defaults so tests can instantiate without DI
        endpoint = endpoint or ModelEndpoint(name="mock-critic", base_url="local")
        config = config or AgentConfig()
        super().__init__(endpoint=endpoint, config=config, http_client=http_client, model_manager=model_manager)
        self.focus_area = focus_area  # e.g., "security", "performance", "maintainability"
    
    def system_prompt(self) -> str:
        base_prompt = """You are an expert Critic Agent specialized in code review and quality assessment.

Your responsibilities:
1. Analyze code for quality, maintainability, and best practices
2. Identify potential bugs, security issues, and performance problems
3. Provide constructive, actionable feedback
4. Suggest specific improvements with examples
5. Assess code readability and documentation quality
6. Evaluate error handling and edge case coverage
7. Check for proper testing and validation

Review Criteria:
- Code structure and organization
- Naming conventions and clarity
- Error handling and validation
- Security considerations
- Performance implications
- Documentation and comments
- Testing coverage
- Maintainability and extensibility

Always provide:
- Specific issues with line references when possible
- Concrete suggestions for improvement
- Priority levels (critical, important, minor)
- Positive feedback for good practices
- Overall quality assessment and score"""

        if self.focus_area:
            base_prompt += f"\n\nSpecial Focus: Pay particular attention to {self.focus_area} aspects of the code."
        
        return base_prompt
    
    def _calculate_confidence_factors(self, response: str) -> dict[str, float]:
        """Calculate confidence factors specific to code review tasks."""
        factors = {}
        
        # Analysis depth factor
        depth_score = 0.0
        
        # Check for specific issue identification
        if re.search(r"(issue|problem|bug|error|warning)", response, re.IGNORECASE):
            depth_score += 0.2
        
        # Check for improvement suggestions
        if re.search(r"(suggest|recommend|improve|consider|should)", response, re.IGNORECASE):
            depth_score += 0.2
        
        # Check for code examples or snippets
        if "```" in response:
            depth_score += 0.2
        
        # Check for priority/severity indicators
        if re.search(r"(critical|important|minor|high|medium|low)", response, re.IGNORECASE):
            depth_score += 0.2
        
        # Check for positive feedback
        if re.search(r"(good|well|excellent|proper|correct)", response, re.IGNORECASE):
            depth_score += 0.2
        
        factors["analysis_depth"] = min(1.0, depth_score)
        
        # Constructiveness factor
        constructive_score = 0.0
        
        # Check for specific suggestions
        suggestion_count = len(re.findall(r"(suggest|recommend|consider|try|use)", response, re.IGNORECASE))
        constructive_score += min(0.4, suggestion_count * 0.1)
        
        # Check for examples
        if re.search(r"(example|for instance|like this)", response, re.IGNORECASE):
            constructive_score += 0.2
        
        # Check for explanations
        if re.search(r"(because|since|due to|reason)", response, re.IGNORECASE):
            constructive_score += 0.2
        
        # Check for alternatives
        if re.search(r"(instead|alternatively|better|prefer)", response, re.IGNORECASE):
            constructive_score += 0.2
        
        factors["constructiveness"] = min(1.0, constructive_score)
        
        # Coverage factor
        coverage_score = 0.0
        
        # Check for different types of analysis
        analysis_types = [
            r"security",
            r"performance",
            r"maintainability",
            r"readability",
            r"error.handling",
            r"testing",
            r"documentation"
        ]
        
        for analysis_type in analysis_types:
            if re.search(analysis_type, response, re.IGNORECASE):
                coverage_score += 0.14  # 1.0 / 7 types
        
        factors["coverage"] = min(1.0, coverage_score)
        
        return factors
    
    async def process(self, query: str, context: dict[str, Any] | None = None) -> AgentResponse:
        """Process a code review request with enhanced analysis."""
        # Enhance the query with review-specific context
        enhanced_query = self._enhance_review_query(query, context)
        
        # Process with the base agent
        response = await super().process(enhanced_query, context)
        
        # Post-process to add review-specific metadata
        if response.metadata is None:
            response.metadata = {}
        
        response.metadata.update({
            "agent_type": "critic",
            "focus_area": self.focus_area,
            "review_analysis": self._analyze_review_response(response.response)
        })
        
        return response
    
    def _enhance_review_query(self, query: str, context: dict[str, Any] | None) -> str:
        """Enhance the query with review-specific context and requirements."""
        enhanced_parts = [query]
        
        # Add context about the code being reviewed
        if context:
            if "code_to_review" in context:
                enhanced_parts.append(f"Code to review:\n```\n{context['code_to_review']}\n```")
            
            if "file_path" in context:
                enhanced_parts.append(f"File: {context['file_path']}")
            
            if "author" in context:
                enhanced_parts.append(f"Author: {context['author']}")
            
            if "purpose" in context:
                enhanced_parts.append(f"Purpose: {context['purpose']}")
            
            if "review_focus" in context:
                enhanced_parts.append(f"Review Focus: {context['review_focus']}")
        
        # Add review-specific requirements
        enhanced_parts.append("""
Review Requirements:
- Identify specific issues with clear explanations
- Provide constructive suggestions for improvement
- Include code examples where helpful
- Prioritize issues by severity (critical, important, minor)
- Highlight both problems and good practices
- Consider security, performance, and maintainability
- Assess documentation and testing adequacy
""")
        
        return "\n\n".join(enhanced_parts)
    
    def _analyze_review_response(self, response: str) -> dict[str, Any]:
        """Analyze the review response for quality metrics."""
        analysis = {
            "issues_identified": len(re.findall(r"(issue|problem|bug|error)", response, re.IGNORECASE)),
            "suggestions_made": len(re.findall(r"(suggest|recommend|consider)", response, re.IGNORECASE)),
            "has_code_examples": "```" in response,
            "has_priorities": bool(re.search(r"(critical|important|minor|high|medium|low)", response, re.IGNORECASE)),
            "has_positive_feedback": bool(re.search(r"(good|well|excellent|proper)", response, re.IGNORECASE)),
            "covers_security": bool(re.search(r"security", response, re.IGNORECASE)),
            "covers_performance": bool(re.search(r"performance", response, re.IGNORECASE)),
            "covers_maintainability": bool(re.search(r"(maintainability|maintainable)", response, re.IGNORECASE)),
            "response_length": len(response.split())
        }
        
        # Calculate review quality score
        quality_factors = [
            analysis["issues_identified"] > 0,
            analysis["suggestions_made"] > 0,
            analysis["has_code_examples"],
            analysis["has_priorities"],
            analysis["has_positive_feedback"],
            analysis["response_length"] > 50  # Substantial response
        ]
        
        analysis["review_quality_score"] = sum(quality_factors) / len(quality_factors)
        
        # Determine review thoroughness
        coverage_areas = [
            analysis["covers_security"],
            analysis["covers_performance"], 
            analysis["covers_maintainability"]
        ]
        analysis["thoroughness_score"] = sum(coverage_areas) / len(coverage_areas)
        
        return analysis
