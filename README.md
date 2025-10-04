# Synndicate

**A Multi-Agent AI Orchestration System**

An AI orchestration platform with comprehensive observability, deterministic behavior, and audit-ready architecture. Features local language model integration (TinyLlama), advanced RAG capabilities with distributed vector store, and full trace-based monitoring.

## **Key Features**

### **Multi-Agent Intelligence**

- **Planner Agent**: Strategic task decomposition and workflow planning
- **Coder Agent**: Code generation and implementation with best practices
- **Critic Agent**: Quality assurance, review, and improvement suggestions
- **Agent Factory**: Dynamic agent creation with dependency injection

### **Language Model Integration**

- **Model Manager**: Comprehensive lifecycle management with health monitoring ([manager.py](src/synndicate/models/manager.py))
- **Local Models**: TinyLlama 1.1B (638MB) with llama.cpp integration ([providers.py](src/synndicate/models/providers.py))
- **Embedding Models**: BGE-small-en-v1.5 (384-dimensional embeddings) with SentenceTransformers
- **API Fallback**: OpenAI API support for production scaling with async client management
- **Model Interfaces**: Type-safe abstractions for language and embedding models ([interfaces.py](src/synndicate/models/interfaces.py))
- **Test Coverage**: 88% interfaces, 42% manager, 46% providers with comprehensive test suite ([test_models_comprehensive.py](tests/test_models_comprehensive.py))
- **Performance**: 9.4 words/sec average throughput with full observability

### **Advanced RAG System**

- **Hybrid Retrieval**: Vector, keyword, semantic, and hybrid search modes
- **Distributed Vector Store**: HTTP API with authentication and persistence
- **Smart Chunking**: Semantic, code-aware, and content-adaptive strategies
- **Context Integration**: Agent-specific formatting and priority-based selection
- **Multi-turn Context**: Conversation history preservation and expansion
- **Embedding Cache**: Persistent caching for improved performance

### **Enterprise Observability**

- **Distributed Tracing**: Full Jaeger, Zipkin, and OTLP backend support with Docker deployments ([tracing.py](src/synndicate/observability/tracing.py))
- **Trace IDs**: End-to-end request tracking across all components and services ([logging.py](src/synndicate/observability/logging.py))
- **Structured Logging**: Single-line JSON format with full context and trace correlation
- **Performance Probes**: Always-on timing and success metrics with Prometheus integration ([probe.py](src/synndicate/observability/probe.py))
- **Audit Trails**: Complete trace snapshots for compliance and debugging ([audit.py](src/synndicate/core/audit.py))
- **Health Monitoring**: Backend health checks, automatic failover, and graceful shutdown
- **Production Ready**: Configurable sampling, batching, and resource limits

### **Distributed Tracing Backend**

- **Multi-Backend Support**: Jaeger, Zipkin, OTLP, Console, or Disabled modes ([distributed_tracing.py](src/synndicate/observability/distributed_tracing.py))
- **Docker Integration**: Ready-to-use Docker Compose configurations ([config/tracing/](config/tracing/))
- **Flexible Configuration**: Environment variables, settings files ([settings.py](src/synndicate/config/settings.py)), or programmatic setup
- **Performance Optimization**: Configurable sampling rates, batching, and resource limits
- **Health Monitoring**: Automatic backend health checks and failover mechanisms
- **Production Deployment**: Kubernetes manifests and production hardening guides
- **Comprehensive Documentation**: Complete setup and troubleshooting guide ([distributed-tracing.md](docs/distributed-tracing.md))

### **Production Infrastructure**

- **FastAPI Server**: RESTful API with `/health` and `/query` endpoints ([server.py](src/synndicate/api/server.py))
- **Vector Store API**: Authenticated HTTP vector store with persistence ([vectorstore_server.py](scripts/vectorstore_server.py))
- **Deterministic Behavior**: Seeded RNGs and config hashing for reproducibility ([audit.py](src/synndicate/core/audit.py))
- **Configuration Management**: Pydantic-based settings with environment overrides ([settings.py](src/synndicate/config/settings.py))
- **Artifact Storage**: Pluggable storage for traces, performance data, and audit bundles ([storage/](src/synndicate/storage/))

## **Quick Start**

### **System Requirements**

**Supported Platforms:**

- **Arch Linux** (primary development platform)
- **Ubuntu/Debian** (CI/CD and production)
- **macOS** (development)
- **Windows** (via WSL2 recommended)

**Optional Dependencies:**

- **Docker** (for distributed tracing backends)
- **Docker Compose** (for multi-service deployments)

**Dependencies:**

```bash
# Arch Linux
sudo pacman -S python python-pip python-virtualenv git cmake

# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-pip python3-venv git cmake build-essential

# macOS (with Homebrew)
brew install python git cmake

# Windows (WSL2 Ubuntu)
sudo apt update && sudo apt install python3 python3-pip python3-venv git cmake build-essential
```

### **Installation**

1. **Clone and Setup Environment**

```bash
git clone https://github.com/Perihelionys/Synndicate.git
cd Synndicate
python -m venv venv

# Activate virtual environment
source venv/bin/activate          # Linux/macOS
# or
venv\Scripts\activate            # Windows
```

2. **Install Dependencies**

```bash
pip install -e .
```

3. **Configure Environment (Optional)**

```bash
# Vector store configuration
export SYN_RAG_VECTOR_API="http://localhost:8080"
export SYN_RAG_VECTOR_API_KEY="your-secret-key"

# Embedding cache
export SYN_EMBEDDING_CACHE_PATH="$HOME/.synndicate/emb_cache.json"

# Distributed tracing configuration
export SYN_OBSERVABILITY__TRACING_BACKEND="jaeger"  # jaeger, zipkin, otlp, console, disabled
export SYN_OBSERVABILITY__TRACING_SAMPLE_RATE="1.0"  # 0.0-1.0
export SYN_OBSERVABILITY__TRACING_ENDPOINT="http://localhost:14250"  # optional custom endpoint

# Deterministic behavior
export SYN_SEED="42"
```

### **Basic Usage**

**Start Distributed Tracing Backend (Optional)**

```bash
# Start Jaeger (recommended) - see config/tracing/jaeger-docker-compose.yml
cd config/tracing
docker-compose -f jaeger-docker-compose.yml up -d
# Access Jaeger UI at: http://localhost:16686

# OR start Zipkin - see config/tracing/zipkin-docker-compose.yml
docker-compose -f zipkin-docker-compose.yml up -d
# Access Zipkin UI at: http://localhost:9411

# OR start full stack - see config/tracing/docker-compose.yml
docker-compose up -d
cd ../..
```

*Start Vector Store** (Optional for distributed RAG)

```bash
# Generate API key
export SYN_VECTORSTORE_API_KEY="$(openssl rand -hex 32)"
export SYN_VECTORSTORE_PERSIST_PATH="$HOME/.synndicate/vectorstore.json"

# Start server
uvicorn --app-dir scripts vectorstore_server:app --host 0.0.0.0 --port 8080
```

**Run Synndicate**

```bash
# Initialize with deterministic startup
python -m synndicate.main

# Start API server
uvicorn synndicate.api.server:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Generate audit bundle
python -c "from synndicate.core.audit import generate_audit_bundle; generate_audit_bundle()"
```

### **API Usage**

```bash
# Health check
curl http://localhost:8000/health

# Process query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Create a Python function to parse log files"}'
```

### **Vector Store CLI**

Use the [vector_cli.py](scripts/vector_cli.py) tool for document management:

```bash
# Health check
python scripts/vector_cli.py health

# Add documents
python scripts/vector_cli.py add --text "Python tutorial" --id "doc1" --metadata '{"topic":"python"}'

# Search
python scripts/vector_cli.py query --text "programming tutorial" --limit 5

# Delete
python scripts/vector_cli.py delete --ids "doc1,doc2"
```

## **Architecture Overview**

### **Core Components**

- **Orchestrator**: Pipeline and state machine-based workflow management
- **Agent System**: Protocol-based agents with lifecycle management
- **Model Manager**: Unified interface for language and embedding models
- **RAG Engine**: Hybrid retrieval with distributed vector store support
- **Vector Store**: HTTP API with authentication, persistence, and CRUD operations
- **Observability Stack**: Logging, tracing, metrics, and audit trails

### **Data Flow**

```
API Request → Orchestrator → Agent Pipeline → Model Inference → RAG Context → Response
     ↓              ↓              ↓              ↓              ↓
Trace ID    Performance    Agent State    Model Metrics    Context Logs
     ↓              ↓              ↓              ↓              ↓
           Audit Trail → Trace Snapshot → Artifact Storage
```

### **Vector Store Architecture**

```
RAG Retriever → HTTP Client → Vector Store API → In-Memory Index → Persistence Layer
      ↓              ↓              ↓              ↓              ↓
  Embedding     Auth Headers    CRUD Endpoints   Cosine Search   JSON Snapshots
```

### **Current Metrics**

- **Language Model**: TinyLlama 1.1B at 9.4 words/sec average
- **Embedding Model**: BGE-small-en-v1.5 with persistent caching
- **Vector Store**: In-memory cosine similarity with JSON persistence
- **Test Coverage**: 53% overall, 83% for RAG components
- **CI/CD**: Automated testing on Ubuntu runners, cross-platform compatibility

### **Observability Features**

- **Trace IDs**: UUID4-based request tracking
- **Structured Logs**: JSON format with timestamp, level, component, trace_id
- **Performance Probes**: Sub-millisecond timing for all operations
- **Audit Bundles**: Complete system state snapshots with deterministic hashing

## **Development & Testing**

### **Test Coverage**: 20% overall with major improvements in critical modules ([test coverage report](build/coverage/))

- **Models System**: 88% interfaces, 42% manager, 46% providers ([test_models_comprehensive.py](tests/test_models_comprehensive.py))
- **Main Entry**: 92% success rate with comprehensive lifecycle testing ([test_main_entry.py](tests/test_main_entry.py))
- **State Machine**: 71% coverage with 100% test success rate ([test_state_machine_focused.py](tests/test_state_machine_focused.py))
- **Orchestrator**: 43% coverage with complete workflow testing ([test_orchestrator_focused.py](tests/test_orchestrator_focused.py))
- **CI/CD Pipeline**: GitHub Actions with automated testing, linting, and Docker builds ([.github/workflows/](.github/workflows/))
- **Code Quality**: Black, Ruff, MyPy with enhanced type checking and comprehensive pre-commit hooks ([pyproject.toml](pyproject.toml))
- **Type Safety**: Strict MyPy configuration with `warn_return_any` and `warn_unused_ignores` for production-grade type safety
- **Documentation**: Architecture guides, API docs, and troubleshooting resources ([docs/](docs/))

## **Docker & Deployment**

### **Vector Store Container**

Build using the [Dockerfile.vectorstore](Dockerfile.vectorstore):

```bash
# Build image
docker build -f Dockerfile.vectorstore -t synndicate-vectorstore .

# Run with persistence and auth
docker run -d \
  -p 8080:8080 \
  -e SYN_VECTORSTORE_API_KEY="your-secret" \
  -e SYN_VECTORSTORE_PERSIST_PATH="/data/vectorstore.json" \
  -v $(pwd)/data:/data \
  synndicate-vectorstore
```

### **Docker Compose**

Use the [docker-compose.vectorstore.yml](docker-compose.vectorstore.yml) configuration:

```bash
# Start vector store stack
docker-compose -f docker-compose.vectorstore.yml up -d

# Configure client
export SYN_RAG_VECTOR_API="http://localhost:8080"
export SYN_RAG_VECTOR_API_KEY="your-secret"
```

## **Development and Testing**

### **Development Setup**

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linting and formatting
ruff check src/ tests/
black src/ tests/
mypy src/
```

### **Testing**

Comprehensive test suites in [`tests/`](tests/):

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_rag_basic.py    # RAG system tests
pytest tests/test_dynamic_orchestration.py  # Dynamic orchestration tests
pytest tests/test_models.py       # Model integration tests

# Run with coverage
pytest --cov=synndicate --cov-report=html
```

### **Platform-Specific Notes**

**Arch Linux:**

- Uses system Python (3.13+) with excellent package availability
- cmake and build tools included in base-devel group
- Recommended for development due to cutting-edge packages

**Ubuntu/Debian:**

- Stable LTS versions recommended for production
- May need `python3-dev` for some native extensions
- Used in CI/CD for consistency

**macOS:**

- Use Homebrew for dependencies
- May need Xcode command line tools for cmake
- M1/M2 Macs: ensure compatible Python builds

**Windows:**

- WSL2 with Ubuntu strongly recommended
- Native Windows support via conda environments
- PowerShell scripts available for setup

## **Configuration**

### **Environment Variables**

```bash
# Core system
SYN_ENVIRONMENT=development|production
SYN_SEED=42                                    # Deterministic behavior
SYN_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR

# Distributed Tracing
SYN_OBSERVABILITY__TRACING_BACKEND=jaeger      # jaeger, zipkin, otlp, console, disabled
SYN_OBSERVABILITY__TRACING_PROTOCOL=grpc       # grpc, http
SYN_OBSERVABILITY__TRACING_ENDPOINT=http://localhost:14250  # Custom endpoint (optional)
SYN_OBSERVABILITY__TRACING_SAMPLE_RATE=1.0     # Sampling rate (0.0-1.0)
SYN_OBSERVABILITY__TRACING_BATCH_TIMEOUT=5000  # Batch timeout (ms)
SYN_OBSERVABILITY__TRACING_MAX_BATCH_SIZE=512  # Max batch size
SYN_OBSERVABILITY__TRACING_HEALTH_CHECK=true   # Enable health checks
SYN_OBSERVABILITY__SERVICE_NAME=synndicate      # Service identification
SYN_OBSERVABILITY__SERVICE_VERSION=2.0.0       # Service version

# RAG and Vector Store
SYN_RAG_VECTOR_API=http://localhost:8080       # Vector store URL
SYN_RAG_VECTOR_API_KEY=your-secret-key        # Client auth key
SYN_EMBEDDING_CACHE_PATH=~/.synndicate/cache.json

# Vector Store Server
SYN_VECTORSTORE_API_KEY=your-secret-key       # Server auth key
SYN_VECTORSTORE_PERSIST_PATH=/data/store.json # Persistence file

# Language Models
SYN_MODEL_PATH=/path/to/models                 # Local model directory
OPENAI_API_KEY=sk-...                         # OpenAI fallback
```

### **Configuration Files**

- [`pyproject.toml`](pyproject.toml): Project metadata and dependencies
- [`docker-compose.vectorstore.yml`](docker-compose.vectorstore.yml): Vector store deployment
- [`config/deployment/`](config/deployment/): Deployment configurations (nginx.conf, docker-compose.yml)
- [`config/tracing/`](config/tracing/): Distributed tracing Docker Compose configurations
- [`examples/`](examples/): Demo scripts and usage examples (demo_synndicate.py)
- [`.github/workflows/`](.github/workflows/): CI/CD pipelines for Ubuntu runners
- [`scripts/`](scripts/): CLI tools and development utilities

## **Project Structure**

```
synndicate/
├── src/synndicate/          # Core application code
│   ├── agents/              # Multi-agent system (planner, coder, critic) [agents/](src/synndicate/agents/)
│   ├── api/                 # FastAPI server and authentication [api/](src/synndicate/api/)
│   ├── config/              # Configuration and dependency injection [config/](src/synndicate/config/)
│   ├── core/                # Orchestrator, state machine, pipelines [core/](src/synndicate/core/)
│   ├── models/              # Language model integration and management [models/](src/synndicate/models/)
│   ├── observability/       # Tracing, logging, metrics, monitoring [observability/](src/synndicate/observability/)
│   ├── rag/                 # Retrieval-augmented generation system [rag/](src/synndicate/rag/)
│   └── storage/             # Artifact storage and audit trails [storage/](src/synndicate/storage/)
├── tests/                   # Comprehensive test suite [tests/](tests/)
├── scripts/                 # Utility scripts and examples [scripts/](scripts/)
├── config/                  # Deployment configurations [config/](config/)
│   ├── deployment/          # Docker, Kubernetes manifests [deployment/](config/deployment/)
│   └── tracing/             # Distributed tracing configurations [tracing/](config/tracing/)
├── docs/                    # Documentation and guides [docs/](docs/)
├── examples/                # Usage examples and demos [examples/](examples/)
├── validation/              # Integration and validation scripts [validation/](validation/)
└── build/                   # Build artifacts and coverage reports [build/](build/)
```

### **Development Workflow**

1. Fork and clone the repository
2. Create a feature branch
3. Install development dependencies
4. Make changes with tests
5. Run linting and tests locally
6. Submit pull request

### **Cross-Platform Testing**

- **Local**: Test on your platform (Arch, Ubuntu, macOS, Windows)
- **CI/CD**: Automated testing on Ubuntu runners
- **Docker**: Container testing for deployment scenarios

### **Code Standards**

- **Formatting**: Black with 100-character line length
- **Linting**: Ruff with strict settings
- **Type Checking**: MyPy with gradual typing
- **Testing**: pytest with >80% coverage target

## **License**

Copyright 2025 Himokai. All Rights Reserved.

This software is proprietary and confidential. See [LICENSE](LICENSE) file for full terms.
For licensing inquiries, contact: <himokai@proton.me>

## **Documentation**

### **Comprehensive Guides**

- **[Architecture Guide](docs/ARCHITECTURE.md)**: System design, component interactions, and data flow
- **[Development Guide](docs/DEVELOPMENT.md)**: Setup, coding standards, testing, and contribution workflow
- **[Model Setup Guide](docs/MODEL_SETUP.md)**: Language model configuration and deployment options
- **[Distributed Tracing Guide](docs/distributed-tracing.md)**: Complete setup, configuration, and troubleshooting for Jaeger, Zipkin, and OTLP backends

### **API Documentation**

- **Interactive API Docs**: `http://localhost:8000/docs` (when server is running)
- **OpenAPI Spec**: Auto-generated from FastAPI endpoints
- **Vector Store API**: RESTful endpoints for document management

### **Code Documentation**

- **In-code Docstrings**: Comprehensive function and class documentation
- **Type Hints**: Full type annotations throughout the codebase
- **Configuration Schema**: Pydantic models with validation and documentation

## **Support**

- **Contact**: <himokai@proton.me> for inquiries
- **Documentation**: Comprehensive guides and in-code documentation
- **Platform Support**: All major Linux distributions, macOS, Windows (WSL2)
- **Portfolio**: This project demonstrates professional AI system development

---

**Synndicate AI** - Professional Multi-Agent Orchestration System  
Copyright © 2025 Himokai. All Rights Reserved.
