"""
Synndicate AI - Enterprise-Grade Multi-Agent AI Orchestration System

A production-ready AI orchestration platform with comprehensive observability,
deterministic behavior, and audit-ready architecture. Features local language
model integration, advanced RAG capabilities, and full trace-based monitoring.

Key Features:
- 🤖 Multi-agent workflows (Planner, Coder, Critic) with specialized capabilities
- 🧠 Local language models (TinyLlama 1.1B) with llama.cpp integration
- 📚 Advanced RAG with hybrid retrieval (vector + keyword + semantic)
- 🔍 Enterprise observability with end-to-end trace IDs and audit trails
- ⚙️ FastAPI server with /health and /query endpoints
- 🎯 Deterministic configuration with SHA256 hashing for reproducibility
- 📊 Performance monitoring with 9.4 words/sec average throughput

Quick Start:
    >>> from synndicate import Orchestrator
    >>> from synndicate.config.container import Container
    >>>
    >>> # Initialize with deterministic startup
    >>> container = Container()
    >>> orchestrator = Orchestrator(container)
    >>>
    >>> # Process query with full observability
    >>> result = await orchestrator.process_query(
    ...     "Create a Python function to parse log files"
    ... )
    >>> print(f"Success: {result.success}, Agents: {result.agents_used}")

API Server:
    Start the FastAPI server with deterministic behavior:
    $ python -m synndicate.main  # Initialize with config hash
    $ make dev  # Start development server
    # or
    $ uvicorn synndicate.api.server:app --host 0.0.0.0 --port 8000

    Health check with component status:
    $ curl http://localhost:8000/health | jq .
    {
      "status": "healthy",
      "config_hash": "28411d9a...",
      "components": {"orchestrator": "healthy", "models": "healthy"}
    }

    Process query with trace ID:
    $ curl -X POST http://localhost:8000/query \
      -H 'Content-Type: application/json' \
      -d '{"query":"Create a calculator class"}' | jq .
    {
      "success": true,
      "trace_id": "abc123def456",
      "agents_used": ["planner", "coder", "critic"],
      "execution_time": 2.45
    }

Observability & Audit:
    Every request generates comprehensive audit data:
    - 📄 Trace snapshots: artifacts/orchestrator_trace_<trace_id>.json
    - 📈 Performance data: artifacts/perf_<trace_id>.jsonl
    - 📋 Coverage reports: artifacts/coverage.xml
    - 🔍 Structured logs with trace IDs: t=<ISO8601> trace=<id> ms=<dur>

    Generate audit bundle:
    $ make audit-bundle
    $ ls synndicate_audit/
    configs/ artifacts/ logs/ endpoints/ tree.txt

Configuration:
    Deterministic configuration with environment variables:
    - SYN_SEED=1337 (reproducible RNG seeding)
    - SYN_MODELS__PLANNER__NAME=gpt-4 (model endpoints)
    - SYN_OBSERVABILITY__LOG_LEVEL=INFO (logging configuration)
    - SYN_API__PORT=8000 (server configuration)

    Config hash for audit compliance:
    CONFIG_SHA256 28411d9ae8a1861a86fe220d625fddfdc524d8317297a85ceec37280002f22b2

Architecture:
    Layered architecture with comprehensive observability:
    API Layer (FastAPI) → Orchestration → Agents → Models → RAG → Storage
           ↓                    ↓         ↓        ↓       ↓        ↓
    Trace IDs → Performance → Agent State → Timing → Context → Artifacts

Performance:
    Production-ready performance with monitoring:
    - Language Model: TinyLlama 1.1B at 9.4 words/sec average
    - Embedding Model: BGE 384-dim with <100ms encoding
    - API Response: <3s end-to-end for complex queries
    - Trace Coverage: 100% with comprehensive timing data
"""

__version__ = "2.0.0"
__author__ = "himokai"

from .agents.base import Agent, AgentResponse
from .config.settings import Settings
from .core.orchestrator import Orchestrator
from .core.pipeline import Pipeline, PipelineStage

__all__ = [
    "Orchestrator",
    "Pipeline",
    "PipelineStage",
    "Agent",
    "AgentResponse",
    "Settings",
]
