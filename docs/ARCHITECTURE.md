# Synndicate AI Architecture Guide

## 🏗️ **System Overview**

Synndicate AI is built on a modern, scalable architecture designed for enterprise AI orchestration with comprehensive observability and audit capabilities.

## 🎯 **Core Design Principles**

### **1. Observability First**

- **Distributed Tracing**: Multi-backend support (Jaeger, Zipkin, OTLP) with automatic span creation
- **Trace IDs**: Every operation is traced with unique trace IDs across all components
- **Structured Logging**: Single-line JSON format with trace correlation and consistent schema
- **Performance Probes**: Always-on timing and success metrics on all hot paths
- **Health Monitoring**: Backend health checks, automatic failover, and graceful shutdown
- **Complete Audit Trails**: Trace snapshots and audit bundles for compliance and debugging

### **2. Deterministic Behavior**

- Seeded random number generators
- Configuration hashing for reproducibility
- Immutable trace snapshots
- Environment-based configuration

### **3. Protocol-Based Design**

- Abstract interfaces for all components
- Dependency injection for testability
- Async-first architecture
- Pluggable storage backends

## 🏛️ **Architecture Layers**

┌─────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                    │
├─────────────────────────────────────────────────────────────┤
│                 Orchestration Layer                        │
│  ┌─────────────────┐  ┌──────────────────────────────────┐  │
│  │   Orchestrator  │  │        Agent Factory            │  │
│  │   - Pipelines   │  │  - Dynamic Agent Creation       │  │
│  │   - State Mgmt  │  │  - Lifecycle Management         │  │
│  └─────────────────┘  └──────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Agent Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Planner   │  │    Coder    │  │       Critic        │  │
│  │   Agent     │  │    Agent    │  │       Agent         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 Intelligence Layer                         │
│  ┌─────────────────┐  ┌──────────────────────────────────┐  │
│  │  Model Manager  │  │           RAG Engine            │  │
│  │  - LLM Models   │  │  - Hybrid Retrieval             │  │
│  │  - Embeddings   │  │  - Context Integration          │  │
│  └─────────────────┘  └──────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                Infrastructure Layer                        │
│  ┌─────────────────┐  ┌──────────────────────────────────┐  │
│  │  Observability  │  │       Storage & Config          │  │
│  │  - Logging      │  │  - Artifact Storage             │  │
│  │  - Tracing      │  │  - Configuration Mgmt           │  │
│  │  - Metrics      │  │  - Dependency Injection         │  │
│  └─────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

## 📊 **Distributed Tracing Architecture**

### **Multi-Backend Support**

┌─────────────────────────────────────────────────────────────┐
│                    Synndicate Application                   │
├─────────────────────────────────────────────────────────────┤
│  DistributedTracingManager  │  TracingManager (OpenTelemetry)│
├─────────────────────────────────────────────────────────────┤
│                    OTLP Exporters                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │    gRPC     │  │    HTTP     │  │      Console        │  │
│  │  Exporter   │  │  Exporter   │  │     Exporter        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  Tracing Backends                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Jaeger    │  │   Zipkin    │  │   Custom OTLP       │  │
│  │ :14250/16686│  │   :9411     │  │   Collector         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

### **Trace Flow**

```python
# 1. Application startup - initialize distributed tracing
distributed_manager = DistributedTracingManager(
    backend=TracingBackend.JAEGER,
    sample_rate=1.0,
    batch_timeout=5000
)
tracing_manager = TracingManager(distributed_manager=distributed_manager)
tracing_manager.initialize()

# 2. Request processing - automatic span creation
with tracing_manager.start_span("api.query") as span:
    span.set_attribute("query.length", len(query))
    result = await process_query(query)
    span.set_attribute("result.success", True)

# 3. Agent processing - nested spans
with tracing_manager.start_span("agent.planner.process") as span:
    span.set_attribute("agent.type", "planner")
    plan = await planner.process(query)
```

### **Configuration Flexibility**

- **Environment Variables**: `SYN_OBSERVABILITY__TRACING_*` for runtime configuration
- **Settings Files**: Pydantic-based configuration with validation
- **Programmatic Setup**: Direct API for custom integrations
- **Docker Integration**: Ready-to-use compose files for all backends

## 🔄 **Request Flow**

### **1. API Request Processing**

```python
# 1. Request arrives at FastAPI endpoint
POST /query {"query": "Create a Python function"}

# 2. Trace ID generation and context setup
trace_id = generate_trace_id()
set_trace_id(trace_id)

# 3. Orchestrator invocation
result = await orchestrator.process_query(query, context, workflow)
```

### **2. Orchestration Flow**

```python
# 1. Workflow determination
workflow = determine_workflow(query)  # "development", "production", etc.

# 2. Pipeline execution
if workflow == "development":
    pipeline = [PlannerAgent, CoderAgent, CriticAgent]
    
# 3. Agent execution with observability
for agent in pipeline:
    with probe(f"agent.{agent.name}.process", trace_id):
        response = await agent.process(query, context)
```

### **3. Agent Processing**

```python
# 1. Agent initialization with dependency injection
agent = container.get_agent(agent_type)

# 2. Context retrieval via RAG
context = await rag_engine.retrieve_context(query, agent_type)

# 3. Model inference with observability
with probe(f"model.{model_name}.generate", trace_id):
    response = await model_manager.generate_text(prompt, model_name)
```

### **4. Audit Trail Generation**

```python
# 1. Trace snapshot creation
snapshot = create_trace_snapshot(
    trace_id=trace_id,
    query=query,
    agents_used=result.agents_used,
    execution_path=result.execution_path,
    timings_ms=get_trace_metrics(trace_id)
)

# 2. Artifact storage
save_trace_snapshot(snapshot)
save_performance_data(trace_id, perf_metrics)
```

## 🧩 **Component Details**

### **Orchestrator**

- **Purpose**: Workflow management and agent coordination
- **Key Features**: Pipeline execution, state management, error handling
- **Observability**: Full trace propagation, performance monitoring
- **Configuration**: Workflow definitions, early exit thresholds

### **Agent System**

- **Base Protocol**: `AgentInterface` with lifecycle methods
- **Specializations**: Planner (strategy), Coder (implementation), Critic (review)
- **Features**: Confidence scoring, circuit breakers, async processing
- **Context**: Agent-specific RAG context integration

### **Model Manager**

- **Purpose**: Unified interface for language and embedding models
- **Providers**: Local (llama.cpp), API (OpenAI), Embedding (sentence-transformers)
- **Features**: Health monitoring, automatic fallback, performance tracking
- **Configuration**: Model endpoints, parameters, retry policies

### Models System

The models system provides a unified interface for language and embedding models with support for local and remote providers.

### Components

- **ModelManager**: Central model lifecycle management ([manager.py](../src/synndicate/models/manager.py))
- **Providers**: LocalLlamaProvider, LocalBGEProvider, OpenAIProvider ([providers.py](../src/synndicate/models/providers.py))
- **Interfaces**: Abstract base classes for type safety ([interfaces.py](../src/synndicate/models/interfaces.py))
- **Test Coverage**: 88% interfaces, 42% manager, 46% providers ([test_models_comprehensive.py](../tests/test_models_comprehensive.py))

### **RAG Engine**

- **Retrieval**: Hybrid search (vector + keyword + semantic)
- **Chunking**: Semantic, code-aware, and adaptive strategies
- **Context**: Agent-specific formatting and priority selection
- **Storage**: ChromaDB, FAISS, in-memory fallback

### **Observability Stack**

- **Logging**: Structured JSON with trace IDs and timing
- **Tracing**: End-to-end request tracking with contextvars
- **Metrics**: Performance probes with Prometheus integration
- **Audit**: Complete trace snapshots and artifact storage

## 🔧 **Configuration Architecture**

### **Layered Configuration**

```python
# 1. Base configuration (settings.py)
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SYN_")
    
# 2. Environment overrides
SYN_MODELS__PLANNER__NAME="gpt-4"
SYN_OBSERVABILITY__LOG_LEVEL="DEBUG"

# 3. YAML configuration files
# config/development.yaml
# config/production.yaml

# 4. Runtime configuration
config_hash = freeze_config_and_hash(settings)
```

### **Deterministic Startup**

```python
# 1. Seed all RNGs
seed = int(os.getenv("SYN_SEED", "1337"))
random.seed(seed)
np.random.seed(seed)

# 2. Hash configuration
config_blob = json.dumps(config, sort_keys=True)
config_hash = hashlib.sha256(config_blob).hexdigest()

# 3. Log for audit trail
logger.info(f"CONFIG_SHA256 {config_hash}")
```

## 📊 **Data Flow Patterns**

### **Trace Propagation**

```python
# Context variable-based trace propagation
trace_id: ContextVar[str] = ContextVar('trace_id')

# Automatic propagation in async contexts
async def process_with_trace():
    current_trace = get_trace_id()  # Inherited from parent context
    with probe("operation", current_trace):
        result = await some_operation()
```

### **Performance Monitoring**

```python
# Always-on performance probes
@probe("orchestrator.process_query")
async def process_query(self, query: str):
    # Automatic timing and success tracking
    result = await self._execute_pipeline(query)
    return result

# Metrics collection
metrics = get_trace_metrics(trace_id)
# {"orchestrator.process_query": {"duration_ms": 1250, "success": true}}
```

### **Audit Trail Generation**

```python
# Complete request lifecycle capture
snapshot = {
    "trace_id": trace_id,
    "query": query,
    "agents_used": ["planner", "coder", "critic"],
    "execution_path": ["planning", "coding", "review"],
    "config_sha256": config_hash,
    "timings_ms": trace_metrics,
    "metadata": {"workflow": "development", "success": true}
}
```

## 🚀 **Scalability Considerations**

### **Horizontal Scaling**

- Stateless API servers with FastAPI
- Shared artifact storage (S3, GCS)
- Distributed tracing with OpenTelemetry
- Load balancing with consistent trace routing

### **Performance Optimization**

- Model caching and connection pooling
- Async processing throughout the stack
- Efficient RAG indexing and retrieval
- Streaming responses for long operations

### **Resource Management**

- Memory-efficient model loading
- Configurable context windows
- Circuit breakers for external services
- Graceful degradation and fallbacks

## 🔒 **Security Architecture**

### **Current Implementation**

- Environment-based configuration
- Structured logging without sensitive data
- Configurable CORS and API security
- Deterministic behavior for audit compliance

### **Planned Enhancements**

- Rust-based code execution sandbox
- Input validation and sanitization
- Rate limiting and authentication
- Encrypted artifact storage

## 📈 **Monitoring & Alerting**

### **Health Checks**

- Component health monitoring
- Model availability and performance
- Storage and configuration validation
- End-to-end request testing

### **Metrics & Alerts**

- Request latency and throughput
- Model inference performance
- Error rates and failure patterns
- Resource utilization tracking

---

This architecture provides a solid foundation for enterprise AI orchestration with comprehensive observability, deterministic behavior, and audit-ready compliance.
