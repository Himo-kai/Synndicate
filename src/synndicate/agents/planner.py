"""
Enhanced Planner Agent with improved task decomposition and analysis.

Improvements over original:
- Better task complexity analysis
- Structured plan output with validation
- Resource estimation
- Dependency detection
"""

import re
from dataclasses import dataclass
from enum import Enum

from ..observability.logging import get_logger
from .base import Agent, AgentResponse

logger = get_logger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class TaskType(Enum):
    """Task type categories."""

    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    RESEARCH = "research"


@dataclass
class PlanStep:
    """Individual step in a plan."""

    step_number: int
    description: str
    estimated_time: str | None = None
    dependencies: list[int] = None
    complexity: TaskComplexity | None = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class Plan:
    """Structured plan representation."""

    task_type: TaskType
    complexity: TaskComplexity
    steps: list[PlanStep]
    acceptance_criteria: list[str]
    estimated_total_time: str | None = None
    required_resources: list[str] = None
    risks: list[str] = None

    def __post_init__(self):
        if self.required_resources is None:
            self.required_resources = []
        if self.risks is None:
            self.risks = []


class PlannerAgent(Agent):
    """
    Enhanced planner agent with structured planning capabilities.

    Improvements:
    - Structured plan output with validation
    - Task complexity and type detection
    - Resource and time estimation
    - Risk assessment
    """

    def system_prompt(self) -> str:
        return """You are an expert Planning Agent responsible for breaking down complex tasks into actionable steps.

Your responsibilities:
1. Analyze the task to determine its type and complexity
2. Break down the task into clear, numbered steps
3. Identify dependencies between steps
4. Estimate time and resources needed
5. Define clear acceptance criteria
6. Identify potential risks or challenges

Task Types:
- ANALYSIS: Understanding, reviewing, or examining something
- IMPLEMENTATION: Building, coding, or creating something new
- DEBUGGING: Finding and fixing problems
- OPTIMIZATION: Improving performance or efficiency
- DOCUMENTATION: Writing docs, comments, or explanations
- TESTING: Creating or running tests
- RESEARCH: Investigating or learning about something

Complexity Levels:
- TRIVIAL: < 15 minutes, single step
- SIMPLE: 15-60 minutes, 2-3 steps
- MEDIUM: 1-4 hours, 3-6 steps
- COMPLEX: 4-24 hours, 6-10 steps
- VERY_COMPLEX: > 1 day, 10+ steps

Always structure your response with clear sections and be specific about requirements."""

    def _calculate_confidence_factors(self, response: str) -> dict[str, float]:
        """Calculate confidence factors specific to planning."""
        factors = {}

        # Structure factor - well-structured plans have higher confidence
        structure_score = 0.0
        if "Task Type:" in response or "TASK TYPE:" in response:
            structure_score += 0.2
        if "Complexity:" in response or "COMPLEXITY:" in response:
            structure_score += 0.2
        if re.search(r"\d+\.", response):  # Numbered steps
            structure_score += 0.3
        if "Acceptance Criteria:" in response or "ACCEPTANCE CRITERIA:" in response:
            structure_score += 0.3

        factors["structure"] = min(1.0, structure_score)

        # Content depth factor
        content_score = 0.0
        step_count = len(re.findall(r"\d+\.", response))
        if step_count >= 3:
            content_score += 0.3
        if step_count >= 6:
            content_score += 0.2

        # Check for time estimates
        if re.search(r"\d+\s*(minute|hour|day)", response, re.IGNORECASE):
            content_score += 0.2

        # Check for dependency mentions
        if any(word in response.lower() for word in ["depend", "require", "after", "before"]):
            content_score += 0.2

        # Check for risk assessment
        if any(
            word in response.lower()
            for word in ["risk", "challenge", "difficult", "potential issue"]
        ):
            content_score += 0.1

        factors["content"] = min(1.0, content_score)

        # Specificity factor
        specificity_score = 0.0
        # Avoid vague language
        vague_words = ["somehow", "maybe", "perhaps", "generally", "usually"]
        vague_count = sum(1 for word in vague_words if word in response.lower())
        specificity_score = max(0.0, 0.8 - (vague_count * 0.1))

        # Reward specific technical terms
        tech_terms = ["function", "class", "method", "variable", "database", "api", "endpoint"]
        tech_count = sum(1 for term in tech_terms if term in response.lower())
        specificity_score += min(0.2, tech_count * 0.05)

        factors["specificity"] = min(1.0, specificity_score)

        return factors

    def extract_plan_structure(self, response: str) -> Plan | None:
        """Extract structured plan from response text."""
        try:
            # Extract task type
            task_type = TaskType.ANALYSIS  # default
            type_match = re.search(r"(?:Task Type|TASK TYPE):\s*(\w+)", response, re.IGNORECASE)
            if type_match:
                try:
                    task_type = TaskType(type_match.group(1).lower())
                except ValueError:
                    pass

            # Extract complexity
            complexity = TaskComplexity.MEDIUM  # default
            complexity_match = re.search(
                r"(?:Complexity|COMPLEXITY):\s*(\w+)", response, re.IGNORECASE
            )
            if complexity_match:
                try:
                    complexity = TaskComplexity(complexity_match.group(1).lower())
                except ValueError:
                    pass

            # Extract steps
            steps = []
            step_pattern = r"(\d+)\.\s*([^\n]+)"
            step_matches = re.findall(step_pattern, response)

            for step_num, description in step_matches:
                steps.append(PlanStep(step_number=int(step_num), description=description.strip()))

            # Extract acceptance criteria
            criteria = []
            criteria_section = re.search(
                r"(?:Acceptance Criteria|ACCEPTANCE CRITERIA):\s*(.*?)(?:\n\n|\n[A-Z]|$)",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if criteria_section:
                criteria_text = criteria_section.group(1)
                # Extract bullet points or numbered items
                criteria_items = re.findall(r"[-*•]\s*([^\n]+)", criteria_text)
                if not criteria_items:
                    criteria_items = re.findall(r"\d+\.\s*([^\n]+)", criteria_text)
                criteria = [item.strip() for item in criteria_items]

            # Extract time estimate
            time_estimate = None
            time_match = re.search(
                r"(?:Estimated Time|ESTIMATED TIME):\s*([^\n]+)", response, re.IGNORECASE
            )
            if time_match:
                time_estimate = time_match.group(1).strip()

            return Plan(
                task_type=task_type,
                complexity=complexity,
                steps=steps,
                acceptance_criteria=criteria,
                estimated_total_time=time_estimate,
            )

        except Exception as e:
            logger.warning(f"Failed to extract plan structure: {e}")
            return None

    async def process(self, query: str, context: dict | None = None) -> AgentResponse:
        """Process planning request with enhanced structure extraction."""
        response = await super().process(query, context)

        # Extract structured plan
        plan = self.extract_plan_structure(response.response)
        if plan:
            response.metadata["plan"] = plan
            response.metadata["task_type"] = plan.task_type.value
            response.metadata["complexity"] = plan.complexity.value
            response.metadata["step_count"] = len(plan.steps)

            # Boost confidence if we successfully extracted structure
            if response.confidence < 0.8:
                response.confidence = min(0.9, response.confidence + 0.1)

        return response
