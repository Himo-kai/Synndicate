# Synndicate AI - Production Docker Image
# Multi-stage build for optimized production deployment

# Build stage - Install dependencies and build
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files needed for installation
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

# Production stage - Minimal runtime image
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    SYN_ENVIRONMENT=production \
    SYN_API__HOST=0.0.0.0 \
    SYN_API__PORT=8000 \
    SYN_SEED=1337

# Install minimal runtime dependencies and security updates
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security with restricted permissions
RUN groupadd -r synndicate --gid=1001 && \
    useradd -r -g synndicate --uid=1001 --home-dir=/app --shell=/bin/false synndicate

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy source code
COPY src/ /app/src/
COPY README.md pyproject.toml Makefile /app/
COPY docs/ /app/docs/

# Create directories for runtime data
RUN mkdir -p /app/artifacts /app/logs /app/configs && \
    chown -R synndicate:synndicate /app

# Switch to non-root user
USER synndicate

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Default command - run API server
CMD ["python", "-m", "uvicorn", "synndicate.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Labels for metadata
LABEL maintainer="Synndicate AI Team" \
      version="2.0.0" \
      description="Enterprise-grade multi-agent AI orchestration system" \
      org.opencontainers.image.title="Synndicate AI" \
      org.opencontainers.image.description="Production-ready AI orchestration with observability" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.vendor="Synndicate AI"
