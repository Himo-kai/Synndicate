"""
Comprehensive test suite for dynamic agents system.
Tests DynamicCoderAgent and DynamicCriticAgent.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synndicate.agents.base import AgentResponse
from synndicate.agents.dynamic_coder import DynamicCoderAgent
from synndicate.agents.dynamic_critic import DynamicCriticAgent
from synndicate.config.settings import AgentConfig, ModelEndpoint


class TestDynamicCoderAgent:
    """Test DynamicCoderAgent implementation."""

    def test_dynamic_coder_initialization_defaults(self):
        """Test DynamicCoderAgent initialization with defaults."""
        agent = DynamicCoderAgent()
        
        assert agent.specialization is None
        assert agent.endpoint.name == "mock-coder"
        assert agent.endpoint.base_url == "local"
        assert isinstance(agent.config, AgentConfig)

    def test_dynamic_coder_initialization_custom(self):
        """Test DynamicCoderAgent initialization with custom parameters."""
        endpoint = ModelEndpoint(name="custom-coder", base_url="http://localhost:8080")
        config = AgentConfig(max_retries=5, timeout=60.0)
        
        agent = DynamicCoderAgent(
            specialization="python",
            endpoint=endpoint,
            config=config
        )
        
        assert agent.specialization == "python"
        assert agent.endpoint.name == "custom-coder"
        assert agent.endpoint.base_url == "http://localhost:8080"
        assert agent.config.max_retries == 5
        assert agent.config.timeout == 60.0

    def test_dynamic_coder_system_prompt_base(self):
        """Test DynamicCoderAgent system prompt generation."""
        agent = DynamicCoderAgent()
        prompt = agent.system_prompt()
        
        assert "expert Coder Agent" in prompt
        assert "software implementation" in prompt
        assert "best practices" in prompt
        assert "clean, maintainable code" in prompt

    def test_dynamic_coder_system_prompt_specialized(self):
        """Test DynamicCoderAgent system prompt with specialization."""
        agent = DynamicCoderAgent(specialization="python")
        prompt = agent.system_prompt()
        
        assert "python" in prompt.lower()
        assert "specialized in python" in prompt.lower()

    @pytest.mark.asyncio
    async def test_dynamic_coder_process_basic(self):
        """Test DynamicCoderAgent basic processing."""
        agent = DynamicCoderAgent()
        
        # Mock the parent process method
        mock_response = AgentResponse(
            response="def hello_world():\n    return 'Hello, World!'",
            confidence=0.9,
            metadata={"language": "python"}
        )
        
        with patch.object(agent.__class__.__bases__[0], 'process', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = mock_response
            
            result = await agent.process("Write a hello world function in Python")
            
            assert isinstance(result, AgentResponse)
            assert "def hello_world" in result.response
            assert result.confidence == 0.9
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_dynamic_coder_extract_code_blocks(self):
        """Test code block extraction from response."""
        agent = DynamicCoderAgent()
        
        response_text = """Here's the implementation:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

And here's a test:

```python
def test_fibonacci():
    assert fibonacci(5) == 5
```
"""
        
        code_blocks = agent.extract_code_blocks(response_text)
        assert len(code_blocks) == 2
        assert "def fibonacci(n):" in code_blocks[0]
        assert "def test_fibonacci():" in code_blocks[1]

    def test_dynamic_coder_detect_language(self):
        """Test programming language detection."""
        agent = DynamicCoderAgent()
        
        # Test Python detection
        python_code = "def hello():\n    print('Hello')"
        assert agent.detect_language(python_code) == "python"
        
        # Test JavaScript detection
        js_code = "function hello() {\n    console.log('Hello');\n}"
        assert agent.detect_language(js_code) == "javascript"
        
        # Test Java detection
        java_code = "public class Hello {\n    public static void main(String[] args) {}\n}"
        assert agent.detect_language(java_code) == "java"
        
        # Test unknown language
        unknown_code = "some random text without clear language markers"
        assert agent.detect_language(unknown_code) == "unknown"

    def test_dynamic_coder_validate_syntax_python(self):
        """Test Python syntax validation."""
        agent = DynamicCoderAgent()
        
        # Valid Python code
        valid_code = "def hello():\n    return 'Hello, World!'"
        assert agent.validate_syntax(valid_code, "python") is True
        
        # Invalid Python code
        invalid_code = "def hello(\n    return 'Hello'"
        assert agent.validate_syntax(invalid_code, "python") is False

    def test_dynamic_coder_validate_syntax_non_python(self):
        """Test syntax validation for non-Python languages."""
        agent = DynamicCoderAgent()
        
        # For non-Python languages, should return True (basic validation)
        js_code = "function hello() { return 'Hello'; }"
        assert agent.validate_syntax(js_code, "javascript") is True
        
        java_code = "public class Test {}"
        assert agent.validate_syntax(java_code, "java") is True

    def test_dynamic_coder_calculate_complexity_score(self):
        """Test code complexity scoring."""
        agent = DynamicCoderAgent()
        
        # Simple code
        simple_code = "def hello():\n    return 'Hello'"
        simple_score = agent.calculate_complexity_score(simple_code)
        assert 0.0 <= simple_score <= 1.0
        
        # Complex code with loops and conditions
        complex_code = """
def complex_function(data):
    result = []
    for item in data:
        if item > 0:
            for i in range(item):
                if i % 2 == 0:
                    result.append(i * 2)
                else:
                    result.append(i)
        elif item < 0:
            result.append(abs(item))
    return result
"""
        complex_score = agent.calculate_complexity_score(complex_code)
        assert complex_score > simple_score
        assert 0.0 <= complex_score <= 1.0

    @pytest.mark.asyncio
    async def test_dynamic_coder_enhanced_processing(self):
        """Test enhanced processing with code analysis."""
        agent = DynamicCoderAgent(specialization="python")
        
        # Mock response with code
        code_response = """Here's the implementation:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```
"""
        
        mock_base_response = AgentResponse(
            response=code_response,
            confidence=0.8,
            metadata={}
        )
        
        with patch.object(agent.__class__.__bases__[0], 'process', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = mock_base_response
            
            result = await agent.process("Write a factorial function")
            
            # Should have enhanced metadata
            assert "code_blocks" in result.metadata
            assert "languages_detected" in result.metadata
            assert "syntax_valid" in result.metadata
            assert "complexity_score" in result.metadata
            
            # Confidence should be adjusted based on code quality
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0


class TestDynamicCriticAgent:
    """Test DynamicCriticAgent implementation."""

    def test_dynamic_critic_initialization_defaults(self):
        """Test DynamicCriticAgent initialization with defaults."""
        agent = DynamicCriticAgent()
        
        assert agent.review_focus is None
        assert agent.endpoint.name == "mock-critic"
        assert agent.endpoint.base_url == "local"
        assert isinstance(agent.config, AgentConfig)

    def test_dynamic_critic_initialization_custom(self):
        """Test DynamicCriticAgent initialization with custom parameters."""
        endpoint = ModelEndpoint(name="custom-critic", base_url="http://localhost:8080")
        config = AgentConfig(max_retries=3, timeout=30.0)
        
        agent = DynamicCriticAgent(
            review_focus="security",
            endpoint=endpoint,
            config=config
        )
        
        assert agent.review_focus == "security"
        assert agent.endpoint.name == "custom-critic"
        assert agent.endpoint.base_url == "http://localhost:8080"
        assert agent.config.max_retries == 3
        assert agent.config.timeout == 30.0

    def test_dynamic_critic_system_prompt_base(self):
        """Test DynamicCriticAgent system prompt generation."""
        agent = DynamicCriticAgent()
        prompt = agent.system_prompt()
        
        assert "expert Code Review Agent" in prompt
        assert "comprehensive analysis" in prompt
        assert "security" in prompt
        assert "performance" in prompt
        assert "maintainability" in prompt

    def test_dynamic_critic_system_prompt_focused(self):
        """Test DynamicCriticAgent system prompt with focus."""
        agent = DynamicCriticAgent(review_focus="security")
        prompt = agent.system_prompt()
        
        assert "security" in prompt.lower()
        assert "security vulnerabilities" in prompt.lower()

    @pytest.mark.asyncio
    async def test_dynamic_critic_process_basic(self):
        """Test DynamicCriticAgent basic processing."""
        agent = DynamicCriticAgent()
        
        # Mock the parent process method
        mock_response = AgentResponse(
            response="The code looks good overall. Consider adding error handling.",
            confidence=0.85,
            metadata={"review_type": "general"}
        )
        
        with patch.object(agent.__class__.__bases__[0], 'process', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = mock_response
            
            result = await agent.process("Review this Python function")
            
            assert isinstance(result, AgentResponse)
            assert "code looks good" in result.response
            assert result.confidence == 0.85
            mock_process.assert_called_once()

    def test_dynamic_critic_extract_review_issues(self):
        """Test extraction of review issues from response."""
        agent = DynamicCriticAgent()
        
        review_text = """
        Issues found:
        - HIGH: Missing input validation
        - MEDIUM: No error handling for edge cases
        - LOW: Variable naming could be improved
        - INFO: Consider adding docstrings
        """
        
        issues = agent.extract_review_issues(review_text)
        assert len(issues) >= 3  # Should find at least the major issues
        
        # Check that different severity levels are detected
        severities = [issue.severity.value for issue in issues]
        assert "high" in severities or "critical" in severities
        assert "medium" in severities
        assert "low" in severities or "info" in severities

    def test_dynamic_critic_determine_recommendation(self):
        """Test recommendation determination logic."""
        agent = DynamicCriticAgent()
        
        # Test APPROVE recommendation
        good_review = "The code is well-written and follows best practices."
        issues = []  # No issues
        recommendation = agent.determine_recommendation(good_review, issues)
        assert recommendation == "APPROVE"
        
        # Test REJECT recommendation
        bad_review = "Critical security vulnerabilities found."
        from synndicate.agents.critic import ReviewIssue, ReviewSeverity, ReviewCategory
        critical_issues = [
            ReviewIssue(
                category=ReviewCategory.SECURITY,
                severity=ReviewSeverity.CRITICAL,
                title="SQL Injection",
                description="Vulnerable to SQL injection attacks"
            )
        ]
        recommendation = agent.determine_recommendation(bad_review, critical_issues)
        assert recommendation == "REJECT"

    def test_dynamic_critic_calculate_overall_score(self):
        """Test overall score calculation."""
        agent = DynamicCriticAgent()
        
        # Test with no issues (should be high score)
        no_issues = []
        score = agent.calculate_overall_score(no_issues)
        assert 0.8 <= score <= 1.0
        
        # Test with critical issues (should be low score)
        from synndicate.agents.critic import ReviewIssue, ReviewSeverity, ReviewCategory
        critical_issues = [
            ReviewIssue(
                category=ReviewCategory.SECURITY,
                severity=ReviewSeverity.CRITICAL,
                title="Critical Issue",
                description="Critical security flaw"
            )
        ]
        critical_score = agent.calculate_overall_score(critical_issues)
        assert critical_score < 0.5

    def test_dynamic_critic_extract_strengths(self):
        """Test extraction of code strengths from review."""
        agent = DynamicCriticAgent()
        
        review_text = """
        Strengths:
        - Good error handling
        - Clear variable names
        - Well-documented functions
        - Efficient algorithm
        """
        
        strengths = agent.extract_strengths(review_text)
        assert len(strengths) >= 3
        assert any("error handling" in strength.lower() for strength in strengths)
        assert any("variable names" in strength.lower() for strength in strengths)

    @pytest.mark.asyncio
    async def test_dynamic_critic_enhanced_processing(self):
        """Test enhanced processing with structured review."""
        agent = DynamicCriticAgent(review_focus="performance")
        
        # Mock response with review content
        review_response = """
        Code Review Results:
        
        Issues:
        - MEDIUM: Loop could be optimized
        - LOW: Consider using list comprehension
        
        Strengths:
        - Good variable naming
        - Clear logic flow
        
        Overall: The code is functional but has room for performance improvements.
        """
        
        mock_base_response = AgentResponse(
            response=review_response,
            confidence=0.75,
            metadata={}
        )
        
        with patch.object(agent.__class__.__bases__[0], 'process', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = mock_base_response
            
            result = await agent.process("Review this code for performance")
            
            # Should have enhanced metadata
            assert "review_result" in result.metadata
            assert "issues_count" in result.metadata
            assert "recommendation" in result.metadata
            assert "overall_score" in result.metadata
            
            # Confidence should be adjusted based on review thoroughness
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0


class TestDynamicAgentsIntegration:
    """Test integration between dynamic agents."""

    @pytest.mark.asyncio
    async def test_coder_critic_workflow(self):
        """Test workflow between coder and critic agents."""
        coder = DynamicCoderAgent(specialization="python")
        critic = DynamicCriticAgent(review_focus="general")
        
        # Mock coder response
        code_response = AgentResponse(
            response="def add(a, b):\n    return a + b",
            confidence=0.9,
            metadata={"language": "python"}
        )
        
        # Mock critic response
        review_response = AgentResponse(
            response="The function is simple and correct. Consider adding type hints.",
            confidence=0.8,
            metadata={"recommendation": "APPROVE_WITH_CHANGES"}
        )
        
        with patch.object(coder.__class__.__bases__[0], 'process', new_callable=AsyncMock) as mock_coder:
            mock_coder.return_value = code_response
            
            with patch.object(critic.__class__.__bases__[0], 'process', new_callable=AsyncMock) as mock_critic:
                mock_critic.return_value = review_response
                
                # Generate code
                code_result = await coder.process("Write an add function")
                assert "def add" in code_result.response
                
                # Review code
                review_result = await critic.process(f"Review this code: {code_result.response}")
                assert "simple and correct" in review_result.response
                
                # Both agents should have been called
                mock_coder.assert_called_once()
                mock_critic.assert_called_once()

    def test_dynamic_agents_specialization_consistency(self):
        """Test that agents maintain their specialization consistently."""
        python_coder = DynamicCoderAgent(specialization="python")
        js_coder = DynamicCoderAgent(specialization="javascript")
        security_critic = DynamicCriticAgent(review_focus="security")
        performance_critic = DynamicCriticAgent(review_focus="performance")
        
        # Check specializations are maintained
        assert python_coder.specialization == "python"
        assert js_coder.specialization == "javascript"
        assert security_critic.review_focus == "security"
        assert performance_critic.review_focus == "performance"
        
        # Check system prompts reflect specializations
        assert "python" in python_coder.system_prompt().lower()
        assert "javascript" in js_coder.system_prompt().lower()
        assert "security" in security_critic.system_prompt().lower()
        assert "performance" in performance_critic.system_prompt().lower()


if __name__ == "__main__":
    pytest.main([__file__])
