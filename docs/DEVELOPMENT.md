# Synndicate AI Development Guide

## 🚀 **Getting Started**

### **Prerequisites**

- Python 3.11+ (3.13 recommended)
- Git for version control
- Make for build automation
- curl for API testing

### **Development Setup**

```bash
# 1. Clone repository
git clone <repository-url>
cd Synndicate

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install development dependencies
pip install -e .
pip install pytest pytest-cov ruff black mypy

# 4. Verify installation
python -m synndicate.main
make test
```

### **Distributed Tracing Development Setup**

```bash
# 1. Start local tracing backend for development
cd config/tracing

# Option A: Start Jaeger (recommended for development)
docker-compose -f jaeger-docker-compose.yml up -d
# Access UI: http://localhost:16686

# Option B: Start Zipkin (lightweight alternative)
docker-compose -f zipkin-docker-compose.yml up -d
# Access UI: http://localhost:9411

# Option C: Start full stack (all backends)
docker-compose up -d

cd ../..

# 2. Configure tracing for development
export SYN_OBSERVABILITY__TRACING_BACKEND="jaeger"
export SYN_OBSERVABILITY__TRACING_SAMPLE_RATE="1.0"  # 100% sampling for dev
export SYN_OBSERVABILITY__TRACING_HEALTH_CHECK="true"

# 3. Test tracing integration
python -c "
from synndicate.observability.distributed_tracing import DistributedTracingManager
from synndicate.observability.tracing import TracingManager
manager = DistributedTracingManager()
print(f'✅ Tracing backend: {manager.config.backend}')
"

# 4. Verify traces are being sent
# Run a query and check the tracing UI for spans
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "test"}'
```

## 🛠️ **Development Workflow**

### **Code Quality Standards**

```bash
# Linting with Ruff
make lint
ruff check src/ --fix

# Code formatting with Black
make format
black src/ tests/

# Type checking with MyPy (enhanced configuration)
mypy src/ --config-file pyproject.toml
# Enhanced type safety with warn_return_any and warn_unused_ignores

# Run all quality checks
make audit
```

### **Testing Strategy**

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/synndicate --cov-report=html

# Run specific test modules
pytest tests/test_models_comprehensive.py  # Models system tests
pytest tests/test_main_entry.py            # Main entry point tests
pytest tests/test_state_machine_focused.py # State machine tests
pytest tests/test_orchestrator_focused.py  # Orchestrator tests

# Run with verbose output and coverage
pytest -v --cov=src/synndicate --cov-report=term-missing
```

### **Development Server**

```bash
# Start API server with auto-reload
make dev
# or
uvicorn synndicate.api.server:app --reload --host 0.0.0.0 --port 8000

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Hello world"}'
```

## 🏗️ **Architecture Patterns**

### **Dependency Injection**

```python
# Container-based dependency injection
from synndicate.config.container import Container

container = Container()
orchestrator = Orchestrator(container)
agent = container.get_agent("planner")
```

### **Protocol-Based Design**

```python
# Define interfaces with protocols
from typing import Protocol

class AgentInterface(Protocol):
    async def process(self, query: str, context: dict) -> AgentResponse:
        ...

# Implement with concrete classes
class PlannerAgent(AgentInterface):
    async def process(self, query: str, context: dict) -> AgentResponse:
        # Implementation
        pass
```

### **Observability Integration**

```python
# Add observability to any function
from synndicate.observability.probe import probe
from synndicate.observability.logging import get_logger

logger = get_logger(__name__)

@probe("my_operation")
async def my_function(data: str) -> str:
    logger.info("Processing data", data_length=len(data))
    result = await process_data(data)
    return result
```

### **Configuration Management**

```python
# Pydantic-based configuration
from pydantic import BaseModel, Field
from synndicate.config.settings import get_settings

class MyConfig(BaseModel):
    timeout: float = Field(30.0, description="Request timeout")
    retries: int = Field(3, description="Max retries")

# Environment variable support
# SYN_MY_CONFIG__TIMEOUT=60.0
# SYN_MY_CONFIG__RETRIES=5
```

## 🧪 **Testing**

### **Distributed Tracing Testing**

```bash
# Test distributed tracing functionality
pytest tests/test_distributed_tracing.py -v

# Test with different backends
SYN_OBSERVABILITY__TRACING_BACKEND=console pytest tests/test_distributed_tracing.py
SYN_OBSERVABILITY__TRACING_BACKEND=disabled pytest tests/test_distributed_tracing.py

# Integration testing with tracing enabled
SYN_OBSERVABILITY__TRACING_BACKEND=jaeger pytest tests/test_orchestrator.py -v
```

### **Running Tests**

### **Test Structure**

```python
# tests/test_my_module.py
import pytest
from synndicate.my_module import MyClass

class TestMyClass:
    @pytest.fixture
    def my_instance(self):
        return MyClass()
    
    async def test_basic_functionality(self, my_instance):
        result = await my_instance.process("test")
        assert result.success
        assert "test" in result.content
```

### **Integration Testing**

```python
# Test with observability
from synndicate.observability.probe import get_trace_metrics

async def test_with_observability():
    trace_id = "test_trace_123"
    
    with probe("test_operation", trace_id):
        result = await my_operation()
    
    metrics = get_trace_metrics(trace_id)
    assert "test_operation" in metrics
    assert metrics["test_operation"]["success"]
```

### **Mock External Dependencies**

```python
# Mock model calls for testing
from unittest.mock import AsyncMock, patch

@patch('synndicate.models.manager.ModelManager.generate_text')
async def test_agent_processing(mock_generate):
    mock_generate.return_value = ModelResponse(
        content="Generated response",
        metadata={}
    )
    
    agent = PlannerAgent()
    result = await agent.process("test query")
    assert result.content == "Generated response"
```

## 📊 **Debugging & Monitoring**

### **Trace-Based Debugging**

```python
# Every request has a trace ID
trace_id = get_trace_id()
logger.info("Debug info", trace_id=trace_id, extra_data=data)

# View trace snapshots
cat artifacts/orchestrator_trace_<trace_id>.json

# Performance analysis
cat artifacts/perf_<trace_id>.jsonl
```

### **Log Analysis**

```bash
# Structured log filtering
grep "trace=abc123" logs/latest.log
grep "level=ERROR" logs/latest.log | jq .

# Performance monitoring
grep "ms=" logs/latest.log | sort -k3 -n
```

### **Health Monitoring**

```bash
# Component health
curl http://localhost:8000/health | jq .

# Model status
curl http://localhost:8000/health | jq .components.models
```

## 🔧 **Adding New Features**

### **Creating a New Agent**

```python
# 1. Define agent interface
class MyAgent(AgentInterface):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.name = "my_agent"
    
    async def process(self, query: str, context: dict) -> AgentResponse:
        with probe(f"agent.{self.name}.process"):
            # Agent logic here
            response = await self.model_manager.generate_text(
                prompt=f"Process: {query}",
                model_name="default"
            )
            return AgentResponse(content=response.content)

# 2. Register in agent factory
# synndicate/agents/factory.py
def create_agent(self, agent_type: str) -> AgentInterface:
    if agent_type == "my_agent":
        return MyAgent(self.model_manager)
```

### **Adding Model Providers**

```python
# 1. Implement model interface
class MyModelProvider(LanguageModel):
    async def load(self) -> None:
        # Load model logic
        pass
    
    async def generate(self, prompt: str, config: GenerationConfig) -> ModelResponse:
        with probe(f"model.{self.config.name}.generate"):
            # Generation logic
            return ModelResponse(content="Generated text")

# 2. Register in model manager
# synndicate/models/manager.py
def _create_language_model(self, config: ModelConfig) -> LanguageModel:
    if config.format == ModelFormat.MY_FORMAT:
        return MyModelProvider(config)
```

### **Extending RAG Capabilities**

```python
# 1. Create new chunking strategy
class MyChunkingStrategy(ChunkingStrategy):
    def create_chunks(self, content: str, metadata: dict) -> list[Chunk]:
        # Custom chunking logic
        return chunks

# 2. Add retrieval mode
class MyRetriever(RAGRetriever):
    async def retrieve(self, query: str, mode: str = "my_mode") -> list[Chunk]:
        if mode == "my_mode":
            # Custom retrieval logic
            return results
```

## 🚀 **Deployment**

### **Environment Configuration**

```bash
# Production environment variables
export SYN_ENVIRONMENT=production
export SYN_OBSERVABILITY__LOG_LEVEL=INFO
export SYN_API__HOST=0.0.0.0
export SYN_API__PORT=8000
export SYN_SEED=42

# Model configuration
export SYN_MODELS__PLANNER__BASE_URL=https://api.openai.com/v1
export SYN_MODELS__PLANNER__API_KEY=sk-...
```

### **Docker Deployment**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "synndicate.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Health Checks**

```bash
# Kubernetes health check
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

## 📚 **Code Style Guide**

### **Naming Conventions**

- Classes: `PascalCase` (e.g., `AgentFactory`)
- Functions/Variables: `snake_case` (e.g., `process_query`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `CONFIG_SHA256`)
- Private methods: `_leading_underscore`

### **Documentation Standards**

```python
async def process_query(
    self, 
    query: str, 
    context: dict[str, Any] | None = None,
    workflow: str = "auto"
) -> OrchestratorResult:
    """
    Process a query through the orchestration system.
    
    Args:
        query: The input query to process
        context: Optional context dictionary
        workflow: Workflow type (auto, development, production)
        
    Returns:
        OrchestratorResult with processing results and metadata
        
    Raises:
        ValueError: If query is empty
        RuntimeError: If orchestrator is not initialized
    """
```

### **Error Handling**

```python
# Use specific exceptions
class SyndicateError(Exception):
    """Base exception for Synndicate errors."""
    pass

class ModelNotFoundError(SyndicateError):
    """Raised when a requested model is not available."""
    pass

# Log errors with context
try:
    result = await risky_operation()
except Exception as e:
    logger.error("Operation failed", 
                error=str(e), 
                trace_id=get_trace_id(),
                operation="risky_operation")
    raise
```

## 🔍 **Performance Guidelines**

### **Async Best Practices**

```python
# Use async/await consistently
async def process_items(items: list[str]) -> list[str]:
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results

# Avoid blocking operations
# Bad: time.sleep(1)
# Good: await asyncio.sleep(1)
```

### **Memory Management**

```python
# Use context managers for resources
async with model_manager.get_model("gpt-4") as model:
    response = await model.generate(prompt)

# Clean up large objects
del large_data_structure
gc.collect()
```

### **Caching Strategies**

```python
# Use functools.lru_cache for expensive computations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(input_data: str) -> str:
    # Expensive operation
    return result
```

## 🤝 **Contributing**

### **Pull Request Process**

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run quality checks: `make audit`
5. Commit with descriptive messages
6. Push and create pull request

### **Commit Message Format**

feat: add new agent type for code review
fix: resolve trace ID propagation issue
docs: update API documentation
test: add integration tests for RAG system
refactor: improve model manager error handling

### **Code Review Checklist**

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Observability integrated
- [ ] Error handling implemented
- [ ] Performance considerations addressed
- [ ] Security implications reviewed

---

Happy coding! 🚀
