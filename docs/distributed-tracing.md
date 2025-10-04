# Distributed Tracing Guide

This guide covers setting up and using distributed tracing with Synndicate.

## 📊 System Integration Status

✅ **Distributed Tracing**: Fully integrated and production-ready

- **Backend Support**: Jaeger, Zipkin, OTLP, Console ([distributed_tracing.py](../src/synndicate/observability/distributed_tracing.py))
- **Configuration**: Environment variables and settings integration ([settings.py](../src/synndicate/config/settings.py))
- **Docker Deployments**: Ready-to-use compose files ([config/tracing/](../config/tracing/))
- **Main Integration**: Startup lifecycle management ([main.py](../src/synndicate/main.py))

## Overview

Synndicate's distributed tracing backend provides:

- **Multiple Backend Support**: Jaeger, Zipkin, OTLP, Console, or Disabled
- **Flexible Configuration**: Environment variables, code configuration, or settings files
- **Health Monitoring**: Backend health checks and automatic failover
- **Production Ready**: Batching, sampling, resource limits, and graceful shutdown
- **Docker Integration**: Ready-to-use Docker Compose configurations

## Quick Start

### 1. Start Tracing Backend

Choose your preferred tracing backend:

#### Jaeger (Recommended)

```bash
cd config/tracing
docker-compose -f jaeger-docker-compose.yml up -d
```

Access Jaeger UI at: <http://localhost:16686>

#### Zipkin

```bash
cd config/tracing
docker-compose -f zipkin-docker-compose.yml up -d
```

Access Zipkin UI at: <http://localhost:9411>

#### Full Stack (Jaeger + Zipkin + OTLP Collector)

```bash
cd config/tracing
docker-compose up -d
```

- Jaeger UI: <http://localhost:16686>
- Zipkin UI: <http://localhost:9411>
- OTLP Collector: <http://localhost:4317> (gRPC), <http://localhost:4318> (HTTP)

### 2. Configure Synndicate

Set environment variables:

```bash
# Basic configuration
export SYN_OBSERVABILITY__TRACING_BACKEND=jaeger
export SYN_OBSERVABILITY__TRACING_PROTOCOL=grpc
export SYN_OBSERVABILITY__TRACING_SAMPLE_RATE=1.0

# Advanced configuration
export SYN_OBSERVABILITY__TRACING_ENDPOINT=http://localhost:14250
export SYN_OBSERVABILITY__TRACING_BATCH_TIMEOUT=5000
export SYN_OBSERVABILITY__TRACING_MAX_BATCH_SIZE=512
export SYN_OBSERVABILITY__TRACING_HEALTH_CHECK=true
```

### 3. Start Synndicate

```bash
python -m synndicate.main
```

Tracing will be automatically initialized and spans will be sent to your chosen backend.

## Configuration Reference

### Tracing Backends

| Backend | Description | Default Endpoint | UI Port |
|---------|-------------|------------------|---------|
| `jaeger` | Jaeger all-in-one | `http://localhost:14250` (gRPC) | 16686 |
| `zipkin` | Zipkin server | `http://localhost:4318` (HTTP via OTLP) | 9411 |
| `otlp` | Custom OTLP endpoint | `http://localhost:4317` (gRPC) | N/A |
| `console` | Console output (debug) | N/A | N/A |
| `disabled` | No tracing | N/A | N/A |

### Environment Variables

All configuration can be set via environment variables with the `SYN_OBSERVABILITY__` prefix:

```bash
# Backend selection
SYN_OBSERVABILITY__TRACING_BACKEND=jaeger|zipkin|otlp|console|disabled

# Protocol (for OTLP backends)
SYN_OBSERVABILITY__TRACING_PROTOCOL=grpc|http

# Custom endpoint (overrides backend defaults)
SYN_OBSERVABILITY__TRACING_ENDPOINT=http://custom-endpoint:4317

# Sampling configuration
SYN_OBSERVABILITY__TRACING_SAMPLE_RATE=1.0  # 0.0-1.0

# Batching configuration
SYN_OBSERVABILITY__TRACING_BATCH_TIMEOUT=5000  # milliseconds
SYN_OBSERVABILITY__TRACING_MAX_BATCH_SIZE=512
SYN_OBSERVABILITY__TRACING_MAX_QUEUE_SIZE=2048

# Health monitoring
SYN_OBSERVABILITY__TRACING_HEALTH_CHECK=true
SYN_OBSERVABILITY__TRACING_HEALTH_CHECK_INTERVAL=30  # seconds

# Service identification
SYN_OBSERVABILITY__SERVICE_NAME=synndicate
SYN_OBSERVABILITY__SERVICE_VERSION=2.0.0
```

### Settings File Configuration

```python
# config/settings.py or environment-specific config
observability:
  enable_tracing: true
  tracing_backend: "jaeger"
  tracing_protocol: "grpc"
  tracing_endpoint: null  # Use backend default
  tracing_sample_rate: 1.0
  tracing_batch_timeout: 5000
  tracing_max_batch_size: 512
  tracing_max_queue_size: 2048
  tracing_health_check: true
  tracing_health_check_interval: 30
  service_name: "synndicate"
  service_version: "2.0.0"
```

## Advanced Usage

### Programmatic Configuration

```python
from synndicate.observability.distributed_tracing import (
    DistributedTracingManager,
    TracingBackend,
    setup_jaeger_tracing,
    setup_zipkin_tracing,
    setup_otlp_tracing,
)

# Quick setup functions
jaeger_manager = setup_jaeger_tracing()
zipkin_manager = setup_zipkin_tracing(endpoint="http://zipkin:9411")
otlp_manager = setup_otlp_tracing(endpoint="http://otel-collector:4317")

# Custom configuration
manager = DistributedTracingManager(
    backend=TracingBackend.JAEGER,
    protocol="grpc",
    endpoint="http://jaeger:14250",
    sample_rate=0.1,  # 10% sampling
    batch_timeout=10000,  # 10 seconds
    max_batch_size=1024,
    enable_health_check=True,
)

# Initialize
manager.setup()

# Cleanup
manager.shutdown()
```

### Integration with TracingManager

```python
from synndicate.observability.tracing import TracingManager
from synndicate.observability.distributed_tracing import DistributedTracingManager

# Create distributed backend
distributed_manager = DistributedTracingManager(
    backend=TracingBackend.JAEGER
)

# Create tracing manager with distributed backend
tracing_manager = TracingManager(
    service_name="my-service",
    service_version="1.0.0",
    distributed_manager=distributed_manager
)

# Initialize (sets up both managers)
tracing_manager.initialize()

# Use tracing
with tracing_manager.start_span("my-operation") as span:
    span.set_attribute("key", "value")
    # Your code here

# Cleanup
tracing_manager.shutdown()
```

## Production Deployment

### Docker Compose Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  synndicate:
    image: synndicate:latest
    environment:
      - SYN_OBSERVABILITY__TRACING_BACKEND=jaeger
      - SYN_OBSERVABILITY__TRACING_ENDPOINT=http://jaeger:14250
      - SYN_OBSERVABILITY__TRACING_SAMPLE_RATE=0.1
    depends_on:
      - jaeger
    networks:
      - tracing

  jaeger:
    image: jaegertracing/all-in-one:1.60
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "16686:16686"
    networks:
      - tracing
    volumes:
      - jaeger-data:/badger
    restart: unless-stopped

networks:
  tracing:
    driver: bridge

volumes:
  jaeger-data:
```

### Kubernetes Deployment

```yaml
# k8s-tracing.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: synndicate-tracing-config
data:
  SYN_OBSERVABILITY__TRACING_BACKEND: "jaeger"
  SYN_OBSERVABILITY__TRACING_ENDPOINT: "http://jaeger-collector:14250"
  SYN_OBSERVABILITY__TRACING_SAMPLE_RATE: "0.1"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: synndicate
spec:
  replicas: 3
  selector:
    matchLabels:
      app: synndicate
  template:
    metadata:
      labels:
        app: synndicate
    spec:
      containers:
      - name: synndicate
        image: synndicate:latest
        envFrom:
        - configMapRef:
            name: synndicate-tracing-config
```

### Performance Tuning

For high-throughput production environments:

```bash
# Reduce sampling rate
export SYN_OBSERVABILITY__TRACING_SAMPLE_RATE=0.01  # 1%

# Increase batch sizes
export SYN_OBSERVABILITY__TRACING_MAX_BATCH_SIZE=2048
export SYN_OBSERVABILITY__TRACING_MAX_QUEUE_SIZE=8192

# Reduce batch timeout for faster export
export SYN_OBSERVABILITY__TRACING_BATCH_TIMEOUT=1000  # 1 second

# Disable health checks if not needed
export SYN_OBSERVABILITY__TRACING_HEALTH_CHECK=false
```

## Troubleshooting

### Common Issues

#### 1. Connection Refused

ERROR: Failed to initialize distributed tracing: Connection refused

**Solution**: Ensure tracing backend is running and accessible:

```bash
# Check if Jaeger is running
curl http://localhost:16686/

# Check if Zipkin is running  
curl http://localhost:9411/health

# Check Docker containers
docker ps | grep -E "(jaeger|zipkin|otel)"
```

#### 2. No Traces Appearing

**Possible causes**:

- Sampling rate too low
- Backend not receiving traces
- Network connectivity issues

**Debug steps**:

```bash
# Enable console tracing for debugging
export SYN_OBSERVABILITY__TRACING_BACKEND=console

# Increase sampling rate
export SYN_OBSERVABILITY__TRACING_SAMPLE_RATE=1.0

# Check backend logs
docker logs synndicate-jaeger
docker logs synndicate-zipkin
```

#### 3. High Memory Usage

**Solution**: Tune batch settings:

```bash
# Reduce queue sizes
export SYN_OBSERVABILITY__TRACING_MAX_QUEUE_SIZE=512
export SYN_OBSERVABILITY__TRACING_MAX_BATCH_SIZE=128

# Reduce batch timeout
export SYN_OBSERVABILITY__TRACING_BATCH_TIMEOUT=2000
```

### Health Checks

The distributed tracing backend includes health monitoring:

```python
from synndicate.observability.distributed_tracing import DistributedTracingManager

manager = DistributedTracingManager(enable_health_check=True)
manager.setup()

# Health checks run automatically in background
# Check logs for health status updates
```

### Monitoring

Monitor tracing backend performance:

1. **Jaeger Metrics**: Available at <http://localhost:16686/metrics>
2. **Zipkin Metrics**: Available at <http://localhost:9411/metrics>
3. **OTLP Collector Metrics**: Available at <http://localhost:8888/metrics>

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from synndicate.observability.distributed_tracing import setup_jaeger_tracing
from synndicate.observability.tracing import get_tracing_manager

app = FastAPI()

# Initialize tracing on startup
@app.on_event("startup")
async def startup_event():
    setup_jaeger_tracing()

@app.get("/api/query")
async def query_endpoint():
    tracing_manager = get_tracing_manager()
    
    with tracing_manager.start_span("query_processing") as span:
        span.set_attribute("endpoint", "/api/query")
        # Process query
        return {"result": "success"}
```

### Agent Integration

```python
from synndicate.agents.base import Agent
from synndicate.observability.tracing import get_tracing_manager

class MyAgent(Agent):
    async def process(self, query: str) -> str:
        tracing_manager = get_tracing_manager()
        
        with tracing_manager.start_span("agent_process") as span:
            span.set_attribute("agent_type", self.__class__.__name__)
            span.set_attribute("query_length", len(query))
            
            # Process query
            result = await self._do_processing(query)
            
            span.set_attribute("result_length", len(result))
            return result
```

This comprehensive distributed tracing backend provides enterprise-grade observability for Synndicate with flexible configuration, multiple backend support, and production-ready features.
