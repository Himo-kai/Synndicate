"""
Dynamic Critic Agent for code review and quality assessment.

This agent specializes in:
- Code review and quality analysis
- Identifying potential issues and improvements
- Providing constructive feedback
- Ensuring best practices compliance
"""

import re
from typing import TYPE_CHECKING, Any

from ..config.settings import AgentConfig, ModelEndpoint
from ..observability.logging import get_logger
from .base import Agent, AgentResponse

if TYPE_CHECKING:
    from .critic import ReviewIssue

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
        review_focus: str | None = None,
        endpoint: ModelEndpoint | None = None,
        config: AgentConfig | None = None,
        http_client=None,
        model_manager=None,
    ):
        # Provide sensible defaults so tests can instantiate without DI
        endpoint = endpoint or ModelEndpoint(name="mock-critic", base_url="local")
        config = config or AgentConfig()
        super().__init__(
            endpoint=endpoint, config=config, http_client=http_client, model_manager=model_manager
        )
        self.review_focus = review_focus  # e.g., "security", "performance", "maintainability"

    def system_prompt(self) -> str:
        base_prompt = """You are an expert Code Review Agent specialized in comprehensive analysis and quality assessment.

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

        if self.review_focus:
            focus_details = {
                "security": "You are particularly focused on security aspects of code review, including identifying security vulnerabilities, authentication issues, data validation problems, and potential attack vectors.",
                "performance": "You are particularly focused on performance aspects of code review, including identifying bottlenecks, inefficient algorithms, memory usage issues, and optimization opportunities.",
                "maintainability": "You are particularly focused on maintainability aspects of code review, including code organization, readability, documentation quality, and long-term sustainability.",
            }

            focus_text = focus_details.get(
                self.review_focus,
                f"You are particularly focused on {self.review_focus} aspects of code review.",
            )
            base_prompt += f"\n\nSpecialization: {focus_text}"

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
        suggestion_count = len(
            re.findall(r"(suggest|recommend|consider|try|use)", response, re.IGNORECASE)
        )
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
            r"documentation",
        ]

        for analysis_type in analysis_types:
            if re.search(analysis_type, response, re.IGNORECASE):
                coverage_score += 0.14  # 1.0 / 7 types

        factors["coverage"] = min(1.0, coverage_score)

        return factors

    def extract_review_issues(self, review_text: str) -> list["ReviewIssue"]:
        """Extract review issues from review text."""
        from .critic import ReviewCategory, ReviewIssue, ReviewSeverity

        issues = []

        # Look for structured issue patterns with severity levels
        # Pattern: - SEVERITY: Description
        structured_pattern = r"^\s*-\s*(HIGH|MEDIUM|LOW|INFO|CRITICAL)\s*:\s*(.+)$"

        lines = review_text.split("\n")
        for i, line in enumerate(lines):
            match = re.match(structured_pattern, line.strip(), re.IGNORECASE)
            if match:
                severity_text = match.group(1).upper()
                description = match.group(2).strip()

                # Map severity text to ReviewSeverity enum
                severity_map = {
                    "HIGH": ReviewSeverity.HIGH,
                    "CRITICAL": ReviewSeverity.CRITICAL,
                    "MEDIUM": ReviewSeverity.MEDIUM,
                    "LOW": ReviewSeverity.LOW,
                    "INFO": ReviewSeverity.INFO,
                }

                # Determine category based on keywords in description
                category = ReviewCategory.MAINTAINABILITY  # default
                description_lower = description.lower()
                if any(
                    word in description_lower
                    for word in ["error", "bug", "issue", "problem", "missing", "validation"]
                ):
                    category = ReviewCategory.CORRECTNESS
                elif any(
                    word in description_lower for word in ["security", "vulnerability", "unsafe"]
                ):
                    category = ReviewCategory.SECURITY
                elif any(
                    word in description_lower for word in ["performance", "slow", "inefficient"]
                ):
                    category = ReviewCategory.PERFORMANCE
                elif any(word in description_lower for word in ["style", "naming", "format"]):
                    category = ReviewCategory.STYLE
                elif any(word in description_lower for word in ["test", "testing", "coverage"]):
                    category = ReviewCategory.TESTING
                elif any(
                    word in description_lower for word in ["complete", "missing", "docstring"]
                ):
                    category = ReviewCategory.COMPLETENESS

                issues.append(
                    ReviewIssue(
                        category=category,
                        severity=severity_map.get(severity_text, ReviewSeverity.MEDIUM),
                        title=f"{severity_text}: {description[:50]}...",
                        description=description,
                        line_reference=str(i + 1),
                    )
                )

        # Fallback: if no structured issues found, use generic patterns
        if not issues:
            issue_patterns = [
                (r"\b(error|bug|issue|problem)\b", ReviewCategory.CORRECTNESS, ReviewSeverity.HIGH),
                (
                    r"\b(warning|concern|potential)\b",
                    ReviewCategory.MAINTAINABILITY,
                    ReviewSeverity.MEDIUM,
                ),
                (
                    r"\b(improvement|suggestion|consider)\b",
                    ReviewCategory.MAINTAINABILITY,
                    ReviewSeverity.LOW,
                ),
                (
                    r"\b(security|vulnerability|unsafe)\b",
                    ReviewCategory.SECURITY,
                    ReviewSeverity.HIGH,
                ),
                (
                    r"\b(performance|slow|inefficient)\b",
                    ReviewCategory.PERFORMANCE,
                    ReviewSeverity.MEDIUM,
                ),
            ]

            for i, line in enumerate(lines):
                for pattern, category, severity in issue_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(
                            ReviewIssue(
                                category=category,
                                severity=severity,
                                title=line.strip()[:50] + "...",
                                description=line.strip(),
                                line_reference=str(i + 1),
                            )
                        )

        return issues

    def determine_recommendation(self, review_text: str, issues: list) -> str:
        """Determine overall recommendation based on review and issues."""
        if not issues:
            return "APPROVE"

        # Count critical/high severity issues
        critical_count = 0
        for issue in issues:
            if hasattr(issue, "severity"):
                # ReviewIssue object
                if issue.severity.value in ["critical", "high"]:
                    critical_count += 1
            elif isinstance(issue, dict) and issue.get("severity") in ["critical", "high"]:
                # Dictionary format (fallback)
                critical_count += 1

        if critical_count > 0:
            return "REJECT"
        elif len(issues) > 5:
            return "NEEDS_WORK"
        else:
            return "APPROVE_WITH_SUGGESTIONS"

    def calculate_overall_score(self, issues: list) -> float:
        """Calculate overall quality score based on issues."""
        if not issues:
            return 1.0

        # Start with perfect score
        score = 1.0

        # Deduct points based on issue severity
        for issue in issues:
            if hasattr(issue, "severity"):
                # ReviewIssue object
                severity = issue.severity.value
            elif isinstance(issue, dict):
                # Dictionary format (fallback)
                severity = issue.get("severity", "medium")
            else:
                severity = "medium"  # default

            if severity == "critical":
                score -= 0.6  # Critical issues heavily penalize score
            elif severity == "high":
                score -= 0.3
            elif severity == "medium":
                score -= 0.1
            else:
                score -= 0.05

        return max(0.0, score)

    def extract_strengths(self, review_text: str) -> list[str]:
        """Extract positive aspects from review text."""
        import re

        strengths = []

        # Look for positive patterns
        positive_patterns = [
            r"\b(good|excellent|well|clean|clear|efficient)\b",
            r"\b(follows|adheres|implements)\b.*\b(best practices|standards)\b",
            r"\b(readable|maintainable|documented)\b",
        ]

        lines = review_text.split("\n")
        for line in lines:
            for pattern in positive_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    strengths.append(line.strip())
                    break

        return strengths

    async def process(self, query: str, context: dict[str, Any] | None = None) -> AgentResponse:
        """Process a code review request with enhanced analysis."""
        # Enhance the query with review-specific context
        enhanced_query = self._enhance_review_query(query, context)

        # Process with the base agent
        response = await super().process(enhanced_query, context)

        # Extract review issues and perform analysis
        issues = self.extract_review_issues(response.response)
        recommendation = self.determine_recommendation(response.response, issues)
        overall_score = self.calculate_overall_score(issues)
        strengths = self.extract_strengths(response.response)

        # Post-process to add review-specific metadata
        if response.metadata is None:
            response.metadata = {}

        response.metadata.update(
            {
                "agent_type": "critic",
                "focus_area": self.review_focus,
                "review_analysis": self._analyze_review_response(response.response),
                "review_result": {
                    "issues": [
                        {
                            "category": issue.category.value,
                            "severity": issue.severity.value,
                            "title": issue.title,
                            "description": issue.description,
                        }
                        for issue in issues
                    ],
                    "strengths": strengths,
                    "recommendation": recommendation,
                    "score": overall_score,
                },
                "issues_count": len(issues),
                "recommendation": recommendation,
                "overall_score": overall_score,
            }
        )

        # Adjust confidence based on review thoroughness
        # Only adjust if we have substantial review content (not just basic responses)
        if len(issues) > 2 or len(strengths) > 2:
            # More thorough review increases confidence
            response.confidence = min(1.0, response.confidence + 0.1)

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
        enhanced_parts.append(
            """
Review Requirements:
- Identify specific issues with clear explanations
- Provide constructive suggestions for improvement
- Include code examples where helpful
- Prioritize issues by severity (critical, important, minor)
- Highlight both problems and good practices
- Consider security, performance, and maintainability
- Assess documentation and testing adequacy
"""
        )

        return "\n\n".join(enhanced_parts)

    def _analyze_review_response(self, response: str) -> dict[str, Any]:
        """Analyze the review response for quality metrics."""
        analysis = {
            "issues_identified": len(
                re.findall(r"(issue|problem|bug|error)", response, re.IGNORECASE)
            ),
            "suggestions_made": len(
                re.findall(r"(suggest|recommend|consider)", response, re.IGNORECASE)
            ),
            "has_code_examples": "```" in response,
            "has_priorities": bool(
                re.search(r"(critical|important|minor|high|medium|low)", response, re.IGNORECASE)
            ),
            "has_positive_feedback": bool(
                re.search(r"(good|well|excellent|proper)", response, re.IGNORECASE)
            ),
            "covers_security": bool(re.search(r"security", response, re.IGNORECASE)),
            "covers_performance": bool(re.search(r"performance", response, re.IGNORECASE)),
            "covers_maintainability": bool(
                re.search(r"(maintainability|maintainable)", response, re.IGNORECASE)
            ),
            "response_length": len(response.split()),
        }

        # Calculate review quality score
        quality_factors = [
            analysis["issues_identified"] > 0,
            analysis["suggestions_made"] > 0,
            analysis["has_code_examples"],
            analysis["has_priorities"],
            analysis["has_positive_feedback"],
            analysis["response_length"] > 50,  # Substantial response
        ]

        analysis["review_quality_score"] = sum(quality_factors) / len(quality_factors)

        # Determine review thoroughness
        coverage_areas = [
            analysis["covers_security"],
            analysis["covers_performance"],
            analysis["covers_maintainability"],
        ]
        analysis["thoroughness_score"] = sum(coverage_areas) / len(coverage_areas)

        return analysis
