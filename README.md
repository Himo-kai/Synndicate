# Synndicate AI

🚀 **Enterprise-Grade Multi-Agent AI Orchestration System**

A production-ready AI orchestration platform with comprehensive observability, deterministic behavior, and audit-ready architecture. Features local language model integration (TinyLlama), advanced RAG capabilities, and full trace-based monitoring.

## 🎯 **Key Features**

### **🤖 Multi-Agent Intelligence**
- **Planner Agent**: Strategic task decomposition and workflow planning
- **Coder Agent**: Code generation and implementation with best practices
- **Critic Agent**: Quality assurance, review, and improvement suggestions
- **Agent Factory**: Dynamic agent creation with dependency injection

### **🧠 Language Model Integration**
- **Local Models**: TinyLlama 1.1B (638MB) with llama.cpp integration
- **Embedding Models**: BGE-small-en-v1.5 (384-dimensional embeddings)
- **API Fallback**: OpenAI API support for production scaling
- **Performance**: 9.4 words/sec average throughput with full observability

### **📚 Advanced RAG System**
- **Hybrid Retrieval**: Vector, keyword, semantic, and hybrid search modes
- **Smart Chunking**: Semantic, code-aware, and content-adaptive strategies
- **Context Integration**: Agent-specific formatting and priority-based selection
- **Multi-turn Context**: Conversation history preservation and expansion

### **🔍 Enterprise Observability**
- **Trace IDs**: End-to-end request tracking across all components
- **Structured Logging**: Single-line JSON format with full context
- **Performance Probes**: Always-on timing and success metrics
- **Audit Trails**: Complete trace snapshots for compliance and debugging

### **⚙️ Production Infrastructure**
- **FastAPI Server**: RESTful API with `/health` and `/query` endpoints
- **Deterministic Behavior**: Seeded RNGs and config hashing for reproducibility
- **Configuration Management**: Pydantic-based settings with environment overrides
- **Artifact Storage**: Pluggable storage for traces, performance data, and audit bundles

## 🚀 **Quick Start**

### **Installation**
```bash
# Clone and setup
git clone <repository>
cd Synndicate
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e .
```

### **Basic Usage**
```bash
# Initialize with deterministic startup
python -m synndicate.main

# Start API server
make dev
# or
uvicorn synndicate.api.server:app --reload --host 0.0.0.0 --port 8000

# Run comprehensive tests
make test

# Generate audit bundle
make audit
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

## 🏗️ **Architecture Overview**

### **Core Components**
- **Orchestrator**: Pipeline and state machine-based workflow management
- **Agent System**: Protocol-based agents with lifecycle management
- **Model Manager**: Unified interface for language and embedding models
- **RAG Engine**: Hybrid retrieval with context integration
- **Observability Stack**: Logging, tracing, metrics, and audit trails

### **Data Flow**
```
API Request → Orchestrator → Agent Pipeline → Model Inference → RAG Context → Response
     ↓              ↓              ↓              ↓              ↓
Trace ID    Performance    Agent State    Model Metrics    Context Logs
     ↓              ↓              ↓              ↓              ↓
           Audit Trail → Trace Snapshot → Artifact Storage
```

## 📊 **Performance & Observability**

### **Current Metrics**
- **Language Model**: TinyLlama 1.1B at 9.4 words/sec average
- **Embedding Model**: BGE 384-dim with <100ms encoding
- **API Response**: <3s end-to-end for complex queries
- **Trace Coverage**: 100% with comprehensive timing data

### **Audit Features**
- **Deterministic Config**: SHA256 hashing for reproducible builds
- **Trace Snapshots**: Complete request lifecycle in JSON format
- **Performance Data**: JSONL format with operation-level metrics
- **Dependency Tracking**: Full pip freeze and environment capture

## 🛠️ **Development**

### **Requirements**
- Python 3.11+
- FastAPI & Uvicorn for API server
- Sentence-transformers for embeddings
- llama.cpp for local language models
- Pydantic for configuration management

### **Development Tools**
```bash
# Code quality
make lint      # Ruff linting
make format    # Black formatting
make test      # Pytest with coverage
make audit     # Comprehensive audit

# Development server
make dev       # Start with auto-reload
```

### **Project Structure**
```
src/synndicate/
├── agents/          # Multi-agent system
├── api/             # FastAPI server
├── config/          # Configuration management
├── core/            # Orchestrator and workflows
├── models/          # Language model integration
├── observability/   # Logging, tracing, metrics
├── rag/             # Retrieval-augmented generation
└── storage/         # Artifact and data storage
```

## 🔒 **Security & Compliance**

- **Audit Ready**: Complete trace snapshots and performance data
- **Deterministic**: Reproducible behavior with seeded RNGs
- **Configurable**: Environment-based settings with validation
- **Sandboxed Execution**: Rust executor integration (planned)

## 📈 **Status**

✅ **Production Ready** - Audit-ready architecture with enterprise observability

### **Completed Features**
- ✅ Multi-agent orchestration with full observability
- ✅ Local language model integration (TinyLlama)
- ✅ Advanced RAG with hybrid retrieval
- ✅ FastAPI server with health and query endpoints
- ✅ Deterministic startup and configuration management
- ✅ Comprehensive audit trail and trace snapshots
- ✅ Performance monitoring and metrics collection

### **Next Phase**
- 🚧 Rust executor for secure code execution
- 🚧 Additional language models (Phi-3, Llama)
- 🚧 Advanced reflection and self-improvement loops
- 🚧 Production deployment and scaling

## 📚 **Documentation**

- [Model Setup Guide](docs/MODEL_SETUP.md) - Language model configuration
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Architecture Guide](docs/ARCHITECTURE.md) - System design details
- [Development Guide](docs/DEVELOPMENT.md) - Contributing guidelines

---

**Built with ❤️ for enterprise AI orchestration**
