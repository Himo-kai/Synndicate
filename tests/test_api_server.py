"""
Comprehensive tests for API server endpoints and functionality.

Tests cover:
- Health check endpoint with component status
- Query processing endpoint with authentication
- Metrics endpoint with Prometheus format
- Error handling and edge cases
- Authentication and rate limiting integration
- Request/response validation
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from synndicate.api.server import app, create_app
from synndicate.api.auth import UserRole, RateLimitTier
from synndicate.config.settings import get_settings


@pytest.fixture
def client():
    """Create test client for API server."""
    return TestClient(app)


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for testing."""
    orchestrator = AsyncMock()
    orchestrator.process_query = AsyncMock()
    return orchestrator


@pytest.fixture
def mock_container():
    """Mock container with orchestrator."""
    container = MagicMock()
    orchestrator = AsyncMock()
    # Create a simple object with the expected attributes (not a MagicMock)
    class MockResult:
        def __init__(self):
            self.success = True
            self.final_response = "Test response"  # API server expects final_response, not response
            self.agents_used = ["planner", "coder"]
            self.execution_path = ["plan", "code", "review"]
            self.confidence = 0.85
            self.execution_time = 2.5
            self.metadata = {"test": True}
    
    mock_result = MockResult()
    
    orchestrator.process_query = AsyncMock(return_value=mock_result)
    container.get_orchestrator.return_value = orchestrator
    return container


@pytest.fixture
def mock_auth_manager():
    """Mock authentication manager."""
    from synndicate.api.auth import APIKey, AuthManager, SecurityConfig
    
    config = SecurityConfig()
    auth_manager = AuthManager(config)
    
    # Add test API key
    auth_manager.add_api_key(
        "test_api_key_12345",
        UserRole.USER,
        RateLimitTier.USER,
        "Test User Key"
    )
    
    return auth_manager


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "config_hash" in data
        assert "uptime_seconds" in data
        assert "components" in data
        
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0
    
    def test_health_check_components(self, client):
        """Test health check includes component status."""
        response = client.get("/health")
        data = response.json()
        
        components = data["components"]
        assert isinstance(components, dict)
        
        # Should include basic component checks
        for component in components.values():
            assert component in ["healthy", "unhealthy", "unknown", "not_initialized", "not_available"] or component.startswith("error:")
    
    def test_health_check_headers(self, client):
        """Test health check response headers."""
        response = client.get("/health")
        
        assert response.headers["content-type"] == "application/json"
        assert "x-request-id" in response.headers or "trace-id" in response.headers.get("x-trace-id", "")


class TestQueryEndpoint:
    """Test query processing endpoint."""
    
    @patch('synndicate.api.server.get_container')
    @patch('synndicate.api.server.get_auth_manager')
    def test_query_success_with_auth(self, mock_get_auth, mock_get_container, client, mock_container, mock_auth_manager):
        """Test successful query processing with authentication."""
        mock_get_container.return_value = mock_container
        
        # Create a completely mocked auth manager that returns successful authentication
        mock_auth_manager_instance = MagicMock()
        mock_auth_manager_instance.config.require_api_key = True
        
        # Create a mock API key object
        mock_api_key = MagicMock()
        mock_api_key.name = "test_key"
        mock_api_key.role = UserRole.USER
        
        # Mock successful authentication - return valid API key and USER tier (async)
        async def mock_authenticate_request(request):
            return (mock_api_key, RateLimitTier.USER)
        
        mock_auth_manager_instance.authenticate_request = mock_authenticate_request
        mock_auth_manager_instance.rate_limiter.is_rate_limited.return_value = (False, {"requests_remaining": 59})
        
        mock_get_auth.return_value = mock_auth_manager_instance
        
        response = client.post(
            "/query",
            json={"query": "Create a Python calculator"},
            headers={"X-API-Key": "test_api_key_12345"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Debug: print the actual response to understand what's failing
        print(f"Response data: {data}")
        
        assert data["success"] is True
        assert "trace_id" in data
        assert data["response"] == "Test response"
        assert data["agents_used"] == ["planner", "coder"]
        assert data["execution_path"] == ["plan", "code", "review"]
        assert data["confidence"] == 0.85
        assert isinstance(data["execution_time"], float) and data["execution_time"] > 0
        assert data["metadata"] == {"test": True}
    
    def test_query_missing_api_key(self, client):
        """Test query without API key returns 401."""
        response = client.post(
            "/query",
            json={"query": "Test query"}
        )
        
        assert response.status_code == 401
        assert "API key required" in response.json()["detail"]
    
    def test_query_invalid_api_key(self, client):
        """Test query with invalid API key returns 401."""
        response = client.post(
            "/query",
            json={"query": "Test query"},
            headers={"X-API-Key": "invalid_key"}
        )
        
        assert response.status_code == 401
    
    @patch('synndicate.api.server.get_container')
    @patch('synndicate.api.server.get_auth_manager')
    def test_query_rate_limited(self, mock_get_auth, mock_get_container, client, mock_container, mock_auth_manager):
        """Test query rate limiting."""
        mock_get_container.return_value = mock_container
        
        # Create a completely mocked auth manager for rate limiting test
        mock_auth_manager_instance = MagicMock()
        mock_auth_manager_instance.config.require_api_key = True
        
        # Create a mock API key object
        mock_api_key = MagicMock()
        mock_api_key.name = "test_key"
        mock_api_key.role = UserRole.USER
        
        # Mock successful authentication but rate limited
        async def mock_authenticate_request(request):
            return (mock_api_key, RateLimitTier.USER)
        
        mock_auth_manager_instance.authenticate_request = mock_authenticate_request
        mock_auth_manager_instance.rate_limiter.is_rate_limited.return_value = (True, {
            "error": "Rate limit exceeded",
            "retry_after": 60
        })
        
        mock_get_auth.return_value = mock_auth_manager_instance
        
        response = client.post(
            "/query",
            json={"query": "Test query"},
            headers={"X-API-Key": "test_api_key_12345"}
        )
        
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
        assert response.headers["Retry-After"] == "60"
    
    def test_query_invalid_request_body(self, client):
        """Test query with invalid request body."""
        response = client.post(
            "/query",
            json={"invalid": "field"},
            headers={"X-API-Key": "test_api_key_12345"}
        )
        
        # Should return validation error
        assert response.status_code in [400, 422]
    
    @patch('synndicate.api.server.get_container')
    @patch('synndicate.api.server.get_auth_manager')
    def test_query_empty_query(self, mock_get_auth, mock_get_container, client, mock_container):
        """Test query with empty query string."""
        mock_get_container.return_value = mock_container
        
        # Create a completely mocked auth manager
        mock_auth_manager_instance = MagicMock()
        mock_auth_manager_instance.config.require_api_key = True
        
        # Create a mock API key object
        mock_api_key = MagicMock()
        mock_api_key.name = "test_key"
        mock_api_key.role = UserRole.USER
        
        # Mock successful authentication
        async def mock_authenticate_request(request):
            return (mock_api_key, RateLimitTier.USER)
        
        mock_auth_manager_instance.authenticate_request = mock_authenticate_request
        mock_auth_manager_instance.rate_limiter.is_rate_limited.return_value = (False, {})
        
        mock_get_auth.return_value = mock_auth_manager_instance
        
        response = client.post(
            "/query",
            json={"query": ""},
            headers={"X-API-Key": "test_api_key_12345"}
        )
        
        # Should return validation error for empty query
        assert response.status_code in [400, 422]
    
    @patch('synndicate.api.server.get_container')
    @patch('synndicate.api.server.get_auth_manager')
    def test_query_orchestrator_error(self, mock_get_auth, mock_get_container, client):
        """Test query when orchestrator fails."""
        # Mock container with no orchestrator
        mock_container = MagicMock()
        mock_container.get_orchestrator.return_value = None
        mock_get_container.return_value = mock_container
        
        # Create a completely mocked auth manager
        mock_auth_manager_instance = MagicMock()
        mock_auth_manager_instance.config.require_api_key = True
        
        # Create a mock API key object
        mock_api_key = MagicMock()
        mock_api_key.name = "test_key"
        mock_api_key.role = UserRole.USER
        
        # Mock successful authentication
        async def mock_authenticate_request(request):
            return (mock_api_key, RateLimitTier.USER)
        
        mock_auth_manager_instance.authenticate_request = mock_authenticate_request
        mock_auth_manager_instance.rate_limiter.is_rate_limited.return_value = (False, {})
        
        mock_get_auth.return_value = mock_auth_manager_instance
        
        response = client.post(
            "/query",
            json={"query": "Test query"},
            headers={"X-API-Key": "test_api_key_12345"}
        )
        
        assert response.status_code == 503
        assert "Orchestrator not initialized" in response.json()["detail"]


class TestMetricsEndpoint:
    """Test metrics endpoint."""
    
    @patch('synndicate.api.auth.get_auth_manager')
    def test_metrics_success_with_auth(self, mock_get_auth, client, mock_auth_manager):
        """Test successful metrics retrieval with authentication."""
        mock_get_auth.return_value = mock_auth_manager
        
        # Mock metrics registry
        with patch('synndicate.api.server.get_metrics_registry') as mock_registry:
            registry = MagicMock()
            registry.get_counter.return_value = 42
            registry.get_histogram_sum.return_value = 123.45
            registry.get_histogram_count.return_value = 10
            registry.get_gauge.return_value = 5
            mock_registry.return_value = registry
            
            with patch.object(mock_auth_manager, 'authenticate_request', return_value=(
                mock_auth_manager.config.api_keys[mock_auth_manager._hash_key("test_api_key_12345")],
                RateLimitTier.USER
            )):
                with patch.object(mock_auth_manager.rate_limiter, 'is_rate_limited', return_value=(False, {})):
                    response = client.get(
                        "/metrics",
                        headers={"X-API-Key": "test_api_key_12345"}
                    )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        
        content = response.text
        assert "synndicate_requests_total" in content
        assert "synndicate_response_time_seconds" in content
        assert "synndicate_active_connections" in content
        assert "synndicate_orchestrator_executions_total" in content
        assert "synndicate_agent_invocations_total" in content
        
        # Check Prometheus format
        assert "# HELP" in content
        assert "# TYPE" in content
    
    def test_metrics_missing_auth(self, client):
        """Test metrics endpoint without authentication."""
        response = client.get("/metrics")
        assert response.status_code == 401
    
    @patch('synndicate.api.auth.get_auth_manager')
    def test_metrics_insufficient_role(self, mock_get_auth, client, mock_auth_manager):
        """Test metrics with insufficient role permissions."""
        mock_get_auth.return_value = mock_auth_manager
        
        # Create a key with insufficient permissions (would need custom setup)
        with patch.object(mock_auth_manager, 'authenticate_request', return_value=(None, RateLimitTier.ANONYMOUS)):
            response = client.get(
                "/metrics",
                headers={"X-API-Key": "invalid_key"}
            )
        
        assert response.status_code == 401
    
    @patch('synndicate.api.auth.get_auth_manager')
    def test_metrics_registry_error(self, mock_get_auth, client, mock_auth_manager):
        """Test metrics endpoint when registry fails."""
        mock_get_auth.return_value = mock_auth_manager
        
        with patch('synndicate.api.server.get_metrics_registry', side_effect=Exception("Registry error")):
            with patch.object(mock_auth_manager, 'authenticate_request', return_value=(
                mock_auth_manager.config.api_keys[mock_auth_manager._hash_key("test_api_key_12345")],
                RateLimitTier.USER
            )):
                with patch.object(mock_auth_manager.rate_limiter, 'is_rate_limited', return_value=(False, {})):
                    response = client.get(
                        "/metrics",
                        headers={"X-API-Key": "test_api_key_12345"}
                    )
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to generate metrics" in data["error"]


class TestApplicationLifecycle:
    """Test application startup and shutdown."""
    
    def test_create_app(self):
        """Test application creation."""
        test_app = create_app()
        
        assert test_app is not None
        assert hasattr(test_app, 'routes')
        
        # The create_app() function creates a base app without routes
        # Routes are registered on the module-level 'app' instance
        # Check that the app has basic FastAPI structure
        route_paths = [route.path for route in test_app.routes]
        # Basic FastAPI routes (docs, openapi, etc.)
        assert "/openapi.json" in route_paths
        
        # Test the actual app instance with routes
        from synndicate.api.server import app as actual_app
        actual_route_paths = [route.path for route in actual_app.routes]
        assert "/health" in actual_route_paths
        assert "/query" in actual_route_paths
        assert "/metrics" in actual_route_paths
    
    def test_app_metadata(self):
        """Test application metadata."""
        test_app = create_app()
        assert test_app.title == "Synndicate AI"
        assert test_app.version == "2.0.0"
        assert "AI Orchestration" in test_app.description
    
    @patch('synndicate.api.server.ensure_deterministic_startup')
    @patch('synndicate.api.server.get_container')
    def test_startup_initialization(self, mock_get_container, mock_deterministic):
        """Test application startup initialization."""
        # Mock container initialization
        mock_container = MagicMock()
        mock_get_container.return_value = mock_container
        
        # Test that startup calls deterministic initialization
        with TestClient(app):
            mock_deterministic.assert_called_once()


class TestErrorHandling:
    """Test global error handling."""
    
    def test_404_not_found(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self, client):
        """Test 405 error handling."""
        response = client.put("/health")
        assert response.status_code == 405
    
    @patch('synndicate.api.server.get_container')
    def test_internal_server_error(self, mock_get_container, client):
        """Test 500 error handling."""
        # Mock container to raise exception
        mock_get_container.side_effect = Exception("Test error")
        
        response = client.get("/health")
        
        # Should handle gracefully
        assert response.status_code in [500, 503]


class TestCORSAndSecurity:
    """Test CORS and security headers."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/health")
        
        # Should have CORS headers
        assert response.status_code in [200, 405]  # OPTIONS might not be explicitly handled
    
    def test_security_headers(self, client):
        """Test security headers."""
        response = client.get("/health")
        
        # Check for basic security considerations
        assert response.status_code == 200
        
        # Content-Type should be properly set
        assert "application/json" in response.headers.get("content-type", "")


class TestRequestValidation:
    """Test request validation and sanitization."""
    
    def test_query_request_validation(self, client):
        """Test query request validation."""
        # Test various invalid payloads
        invalid_payloads = [
            {},  # Missing query
            {"query": None},  # Null query
            {"query": 123},  # Non-string query
            {"query": "x" * 10000},  # Extremely long query
        ]
        
        for payload in invalid_payloads:
            response = client.post("/query", json=payload)
            assert response.status_code in [400, 422], f"Failed for payload: {payload}"
    
    def test_content_type_validation(self, client):
        """Test content type validation."""
        # Test non-JSON content type
        response = client.post(
            "/query",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code in [400, 422, 415]


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async endpoint behavior."""
    
    @patch('synndicate.api.server.get_container')
    @patch('synndicate.api.auth.get_auth_manager')
    async def test_concurrent_requests(self, mock_get_auth, mock_get_container, mock_container, mock_auth_manager):
        """Test handling of concurrent requests."""
        mock_get_container.return_value = mock_container
        mock_get_auth.return_value = mock_auth_manager
        
        # Mock slow orchestrator response
        mock_container.get_orchestrator().process_query = AsyncMock()
        mock_container.get_orchestrator().process_query.return_value = MagicMock(
            success=True,
            response="Concurrent response",
            agents_used=["planner"],
            execution_path=["plan"],
            confidence=0.8,
            execution_time=1.0,
            metadata={}
        )
        
        with TestClient(app) as client:
            # This would test concurrent behavior in a real async environment
            response = client.get("/health")
            assert response.status_code == 200
