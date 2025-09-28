"""
Modern configuration system with Pydantic Settings and validation.

Improvements over original:
- Type-safe configuration with Pydantic v2
- Environment variable validation and parsing
- Nested configuration with proper defaults
- Configuration profiles (dev, staging, prod)
- Secrets management integration
"""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelEndpoint(BaseModel):
    """Configuration for a model endpoint."""

    name: str = Field(..., description="Model name (e.g., 'mistral:7b-instruct')")
    base_url: str = Field("http://localhost:11434", description="Base URL for the model API")
    api_key: str | None = Field(None, description="API key if required")
    timeout: float = Field(120.0, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")

    @validator("base_url")
    def validate_base_url(self, v):
        if v != "local" and not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v.rstrip("/")


class AgentConfig(BaseModel):
    """Configuration for agent behavior."""

    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    max_context_length: int = Field(4096, gt=0)
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    enable_streaming: bool = Field(True)


class ModelsConfig(BaseModel):
    """Configuration for all model endpoints."""

    planner: ModelEndpoint = Field(
        default_factory=lambda: ModelEndpoint(name="mistral:7b-instruct")
    )
    coder: ModelEndpoint = Field(
        default_factory=lambda: ModelEndpoint(name="qwen2.5-coder:7b-instruct")
    )
    critic: ModelEndpoint = Field(
        default_factory=lambda: ModelEndpoint(name="llama3.1:8b-instruct")
    )
    embeddings: ModelEndpoint = Field(
        default_factory=lambda: ModelEndpoint(name="all-MiniLM-L6-v2", base_url="local")
    )


class OrchestratorConfig(BaseModel):
    """Configuration for orchestrator behavior."""

    early_exit_threshold: float = Field(0.8, ge=0.0, le=1.0)
    critic_skip_threshold: float = Field(0.85, ge=0.0, le=1.0)
    max_iterations: int = Field(3, gt=0)
    enable_parallel_execution: bool = Field(True)
    enable_circuit_breaker: bool = Field(True)
    circuit_breaker_failure_threshold: int = Field(5, gt=0)
    circuit_breaker_timeout: float = Field(60.0, gt=0)


class RAGConfig(BaseModel):
    """Configuration for RAG system."""

    collection_name: str = Field("synndicate")
    embedding_model: str = Field("all-MiniLM-L6-v2")
    persist_directory: Path = Field(Path("./data/chroma"))
    chunk_size: int = Field(512, gt=0)
    chunk_overlap: int = Field(50, ge=0)
    max_results: int = Field(10, gt=0)
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    enable_hybrid_search: bool = Field(True)
    enable_reranking: bool = Field(True)


class ExecutionConfig(BaseModel):
    """Configuration for code execution environment."""

    enable_execution: bool = Field(True)
    max_memory_mb: int = Field(512, gt=0)
    max_cpu_percent: float = Field(50.0, gt=0, le=100)
    allowed_languages: list[str] = Field(
        default_factory=lambda: ["python", "rust", "javascript", "typescript", "go"]
    )
    docker_image_prefix: str = Field("synndicate-exec")
    enable_networking: bool = Field(False)
    enable_filesystem_access: bool = Field(False)


class ObservabilityConfig(BaseModel):
    """Configuration for observability and monitoring."""

    enable_tracing: bool = Field(True)
    enable_metrics: bool = Field(True)
    enable_logging: bool = Field(True)
    log_level: str = Field("INFO")
    log_format: str = Field("json")  # json or console

    # OpenTelemetry configuration
    otlp_endpoint: str | None = Field(None)
    service_name: str = Field("synndicate")
    service_version: str = Field("2.0.0")

    # Metrics configuration
    metrics_port: int = Field(9090, gt=0, le=65535)
    enable_custom_metrics: bool = Field(True)


class APIConfig(BaseModel):
    """Configuration for API server."""

    host: str = Field("0.0.0.0")
    port: int = Field(8000, gt=0, le=65535)
    reload: bool = Field(False)
    workers: int = Field(1, gt=0)
    enable_cors: bool = Field(True)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    enable_docs: bool = Field(True)
    docs_url: str = Field("/docs")
    redoc_url: str = Field("/redoc")


class Settings(BaseSettings):
    """Main application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="SYN_", env_nested_delimiter="__", case_sensitive=False, extra="ignore"
    )

    # Core configuration sections
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # Global settings
    environment: str = Field(
        "development", description="Environment: development, staging, production"
    )
    debug: bool = Field(False)
    data_directory: Path = Field(Path("./data"))

    def model_post_init(self, __context) -> None:
        """Post-initialization setup."""
        # Ensure data directories exist
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.rag.persist_directory.mkdir(parents=True, exist_ok=True)

    @validator("environment")
    def validate_environment(self, v):
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
