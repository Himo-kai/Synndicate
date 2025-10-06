"""
Enhanced Critic Agent with comprehensive code and plan review capabilities.

Improvements over original:
- Multi-dimensional review criteria
- Structured feedback with severity levels
- Integration with plan context
- Security-focused analysis
- Performance impact assessment
"""

import re
from dataclasses import dataclass
from enum import Enum

from ..observability.logging import get_logger
from .base import Agent, AgentResponse

logger = get_logger(__name__)


class ReviewSeverity(Enum):
    """Severity levels for review issues."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewCategory(Enum):
    """Categories of review feedback."""

    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    COMPLETENESS = "completeness"
    TESTING = "testing"


@dataclass
class ReviewIssue:
    """Individual review issue with metadata."""

    category: ReviewCategory
    severity: ReviewSeverity
    title: str
    description: str
    suggestion: str | None = None
    line_reference: str | None = None


@dataclass
class ReviewResult:
    """Comprehensive review result."""

    overall_score: float  # 0.0 to 1.0
    recommendation: str  # APPROVE, APPROVE_WITH_CHANGES, REJECT
    issues: list[ReviewIssue]
    strengths: list[str]
    summary: str
    confidence: float


class CriticAgent(Agent):
    """
    Enhanced critic agent with comprehensive review capabilities.

    Improvements:
    - Multi-dimensional analysis (correctness, security, performance, etc.)
    - Structured feedback with severity levels
    - Integration with plan and code analysis
    - Actionable suggestions for improvement
    """

    def system_prompt(self) -> str:
        return """You are an expert Code Review Agent responsible for comprehensive analysis of plans and implementations.

Your responsibilities:
1. Evaluate correctness and completeness against requirements
2. Identify security vulnerabilities and risks
3. Assess performance implications and optimizations
4. Review code maintainability and readability
5. Check adherence to best practices and conventions
6. Validate error handling and edge cases
7. Assess test coverage and quality
8. Provide actionable improvement suggestions

Review Criteria:

CORRECTNESS:
- Does the solution address the original requirements?
- Are there logical errors or bugs?
- Are edge cases handled properly?
- Is the algorithm/approach sound?

SECURITY:
- Are inputs properly validated and sanitized?
- Are there injection vulnerabilities (SQL, XSS, etc.)?
- Is sensitive data handled securely?
- Are authentication/authorization checks present?

PERFORMANCE:
- Are algorithms efficient for the expected data size?
- Are there unnecessary computations or memory usage?
- Could caching or optimization improve performance?
- Are database queries optimized?

MAINTAINABILITY:
- Is the code readable and well-documented?
- Are functions/classes appropriately sized?
- Is the code structure logical and modular?
- Are naming conventions consistent?

TESTING:
- Are there adequate tests for the functionality?
- Do tests cover edge cases and error conditions?
- Are tests maintainable and reliable?

Provide structured feedback with:
- Overall recommendation (APPROVE/APPROVE_WITH_CHANGES/REJECT)
- Specific issues categorized by type and severity
- Actionable suggestions for improvement
- Recognition of good practices and strengths"""

    def _calculate_confidence_factors(self, response: str) -> dict[str, float]:
        """Calculate confidence factors specific to code review."""
        factors = {}

        # Structure factor - well-structured reviews have higher confidence
        structure_score = 0.0

        # Check for structured sections
        sections = ["correctness", "security", "performance", "maintainability", "testing"]
        section_count = sum(1 for section in sections if section in response.lower())
        structure_score += min(0.4, section_count * 0.08)

        # Check for recommendation
        recommendations = ["approve", "reject", "changes"]
        if any(rec in response.lower() for rec in recommendations):
            structure_score += 0.2

        # Check for specific issues mentioned
        if re.search(r"issue|problem|concern|vulnerability", response, re.IGNORECASE):
            structure_score += 0.2

        # Check for suggestions
        if re.search(r"suggest|recommend|improve|consider", response, re.IGNORECASE):
            structure_score += 0.2

        factors["structure"] = min(1.0, structure_score)

        # Depth of analysis
        depth_score = 0.0

        # Technical depth indicators
        tech_terms = [
            "algorithm",
            "complexity",
            "optimization",
            "vulnerability",
            "injection",
            "validation",
            "sanitization",
            "authentication",
            "authorization",
            "encryption",
            "performance",
            "memory",
            "cpu",
            "database",
            "query",
            "index",
        ]
        tech_mentions = sum(1 for term in tech_terms if term in response.lower())
        depth_score += min(0.5, tech_mentions * 0.05)

        # Code-specific analysis
        if re.search(r"line \d+|function \w+|class \w+", response):
            depth_score += 0.2

        # Security analysis depth
        security_terms = ["xss", "sql injection", "csrf", "buffer overflow", "privilege escalation"]
        security_mentions = sum(1 for term in security_terms if term in response.lower())
        depth_score += min(0.3, security_mentions * 0.1)

        factors["depth"] = min(1.0, depth_score)

        # Actionability factor
        actionability_score = 0.0

        # Look for specific suggestions
        suggestion_patterns = [
            r"should \w+",
            r"could \w+",
            r"consider \w+",
            r"recommend \w+",
            r"use \w+",
            r"add \w+",
            r"remove \w+",
            r"change \w+",
        ]
        suggestion_count = sum(
            len(re.findall(pattern, response, re.IGNORECASE)) for pattern in suggestion_patterns
        )
        actionability_score += min(0.6, suggestion_count * 0.1)

        # Check for code examples or specific fixes
        if "```" in response:
            actionability_score += 0.2

        # Check for references to best practices
        if any(term in response.lower() for term in ["best practice", "convention", "standard"]):
            actionability_score += 0.2

        factors["actionability"] = min(1.0, actionability_score)

        return factors

    def extract_review_issues(self, response: str) -> list[ReviewIssue]:
        """Extract structured issues from review response."""
        issues = []

        # Pattern to match issue descriptions
        issue_patterns = [
            (
                ReviewCategory.SECURITY,
                r"(?:security|vulnerability|injection|xss|csrf|auth)",
                ReviewSeverity.HIGH,
            ),
            (
                ReviewCategory.CORRECTNESS,
                r"(?:bug|error|incorrect|wrong|fail)",
                ReviewSeverity.HIGH,
            ),
            (
                ReviewCategory.PERFORMANCE,
                r"(?:slow|inefficient|performance|optimization)",
                ReviewSeverity.MEDIUM,
            ),
            (
                ReviewCategory.MAINTAINABILITY,
                r"(?:readable|maintainable|complex|confusing)",
                ReviewSeverity.MEDIUM,
            ),
            (ReviewCategory.TESTING, r"(?:test|coverage|assertion)", ReviewSeverity.MEDIUM),
            (ReviewCategory.STYLE, r"(?:style|convention|formatting|naming)", ReviewSeverity.LOW),
        ]

        sentences = re.split(r"[.!?]+", response)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue

            for category, pattern, default_severity in issue_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Determine severity based on keywords
                    severity = default_severity
                    if any(word in sentence.lower() for word in ["critical", "severe", "major"]):
                        severity = ReviewSeverity.CRITICAL
                    elif any(word in sentence.lower() for word in ["minor", "small", "trivial"]):
                        severity = ReviewSeverity.LOW

                    # Extract title (first few words)
                    title = " ".join(sentence.split()[:8]) + "..."

                    issues.append(
                        ReviewIssue(
                            category=category,
                            severity=severity,
                            title=title,
                            description=sentence,
                        )
                    )
                    break  # Only categorize each sentence once

        return issues

    def determine_recommendation(self, response: str, issues: list[ReviewIssue]) -> str:
        """Determine overall recommendation based on response and issues."""
        response_lower = response.lower()

        # Explicit recommendations
        if "reject" in response_lower:
            return "REJECT"
        elif "approve" in response_lower and "changes" not in response_lower:
            return "APPROVE"
        elif any(word in response_lower for word in ["approve with changes", "changes needed"]):
            return "APPROVE_WITH_CHANGES"

        # Infer from issues
        critical_issues = [i for i in issues if i.severity == ReviewSeverity.CRITICAL]
        high_issues = [i for i in issues if i.severity == ReviewSeverity.HIGH]

        if critical_issues or len(high_issues) > 2:
            return "REJECT"
        elif high_issues or len(issues) > 5:
            return "APPROVE_WITH_CHANGES"
        else:
            return "APPROVE"

    def calculate_overall_score(self, issues: list[ReviewIssue]) -> float:
        """Calculate overall quality score based on issues."""
        base_score = 1.0

        severity_penalties = {
            ReviewSeverity.CRITICAL: 0.3,
            ReviewSeverity.HIGH: 0.2,
            ReviewSeverity.MEDIUM: 0.1,
            ReviewSeverity.LOW: 0.05,
            ReviewSeverity.INFO: 0.01,
        }

        for issue in issues:
            base_score -= severity_penalties.get(issue.severity, 0.05)

        return max(0.0, min(1.0, base_score))

    def extract_strengths(self, response: str) -> list[str]:
        """Extract mentioned strengths from the response."""
        strengths = []

        # Look for positive indicators
        positive_patterns = [
            r"good (?:use of|implementation of|approach to) ([^.!?]+)",
            r"well (?:structured|organized|documented|implemented) ([^.!?]*)",
            r"excellent ([^.!?]+)",
            r"properly (?:handles|implements|validates) ([^.!?]+)",
        ]

        for pattern in positive_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if match.strip():
                    strengths.append(match.strip())

        # Look for explicit strength mentions
        strength_section = re.search(
            r"(?:strengths?|positives?|good points?):\s*(.*?)(?:\n\n|\n[A-Z]|$)",
            response,
            re.DOTALL | re.IGNORECASE,
        )

        if strength_section:
            strength_text = strength_section.group(1)
            bullet_strengths = re.findall(r"[-*•]\s*([^\n]+)", strength_text)
            strengths.extend([s.strip() for s in bullet_strengths])

        return strengths[:5]  # Limit to top 5 strengths

    async def process(self, query: str, context: dict | None = None) -> AgentResponse:
        """Process review request with enhanced analysis."""
        response: AgentResponse = await super().process(query, context)

        # Extract structured review information
        issues = self.extract_review_issues(response.response)
        recommendation = self.determine_recommendation(response.response, issues)
        overall_score = self.calculate_overall_score(issues)
        strengths = self.extract_strengths(response.response)

        # Create review result
        review_result = ReviewResult(
            overall_score=overall_score,
            recommendation=recommendation,
            issues=issues,
            strengths=strengths,
            summary=(
                response.response[:200] + "..."
                if len(response.response) > 200
                else response.response
            ),
            confidence=response.confidence,
        )

        # Add review analysis to metadata
        response.metadata.update(
            {
                "review_result": review_result,
                "recommendation": recommendation,
                "overall_score": overall_score,
                "issue_count": len(issues),
                "critical_issues": len(
                    [i for i in issues if i.severity == ReviewSeverity.CRITICAL]
                ),
                "high_issues": len([i for i in issues if i.severity == ReviewSeverity.HIGH]),
                "strengths_count": len(strengths),
            }
        )

        # Adjust confidence based on review thoroughness
        if len(issues) > 0 or len(strengths) > 0:
            # Boost confidence for thorough reviews
            response.confidence = min(0.95, response.confidence + 0.05)

        # Adjust confidence based on recommendation clarity
        if recommendation in ["APPROVE", "REJECT"]:
            response.confidence = min(0.95, response.confidence + 0.03)

        return response
