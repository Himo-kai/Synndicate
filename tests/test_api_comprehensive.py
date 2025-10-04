"""
Comprehensive test suite for API system covering authentication, rate limiting, and server endpoints.

This test suite provides complete coverage for:
- API authentication and RBAC (Role-Based Access Control)
- Rate limiting with sliding window algorithm
- Security audit logging and IP blocking
- FastAPI server endpoints and lifecycle
- Health checks, metrics, and query processing
- Error handling and global exception management

Test Categories:
1. Authentication System Tests
2. Rate Limiting Tests
3. Security and Audit Tests
4. API Server Endpoint Tests
5. Lifecycle and Configuration Tests
6. Integration Tests

Coverage Goals:
- API Auth: 0% → 80%+ coverage
- API Server: 0% → 80%+ coverage
- 100% test success rate
"""

import json
import time
from collections import deque
from datetime import datetime
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from synndicate.api.auth import (
    APIKey,
    AuthManager,
    RateLimitConfig,
    RateLimiter,
    RateLimitTier,
    SecurityConfig,
    UserRole,
    add_rate_limit_headers,
)
from synndicate.api.server import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    app,
    create_app,
    get_current_user,
    global_exception_handler,
)


class TestAPIAuthentication:
    """Test API authentication and RBAC functionality."""

    def test_user_role_enum(self):
        """Test UserRole enum values."""
        assert UserRole.ADMIN == "admin"
        assert UserRole.USER == "user"
        assert UserRole.READONLY == "readonly"

        # Test enum comparison
        assert UserRole.ADMIN != UserRole.USER
        assert UserRole.USER != UserRole.READONLY

    def test_rate_limit_tier_enum(self):
        """Test RateLimitTier enum values."""
        assert RateLimitTier.ADMIN == "admin"
        assert RateLimitTier.USER == "user"
        assert RateLimitTier.READONLY == "readonly"
        assert RateLimitTier.ANONYMOUS == "anonymous"

    def test_api_key_model(self):
        """Test APIKey model creation and validation."""
        api_key = APIKey(
            key_hash="test_hash",
            role=UserRole.USER,
            tier=RateLimitTier.USER,
            name="test_key",
            created_at=datetime.now()
        )

        assert api_key.key_hash == "test_hash"
        assert api_key.role == UserRole.USER
        assert api_key.tier == RateLimitTier.USER
        assert api_key.name == "test_key"
        assert api_key.is_active is True
        assert api_key.last_used is None
        assert api_key.rate_limit_override is None

    def test_rate_limit_config_model(self):
        """Test RateLimitConfig model creation and validation."""
        config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=2000,
            burst_size=20,
            window_size_seconds=120
        )

        assert config.requests_per_minute == 100
        assert config.requests_per_hour == 2000
        assert config.burst_size == 20
        assert config.window_size_seconds == 120

    def test_rate_limit_config_validation(self):
        """Test RateLimitConfig validation rules."""
        # Test positive value validation
        with pytest.raises(ValueError):
            RateLimitConfig(requests_per_minute=0)

        with pytest.raises(ValueError):
            RateLimitConfig(requests_per_hour=-1)

        with pytest.raises(ValueError):
            RateLimitConfig(burst_size=0)

    def test_security_config_defaults(self):
        """Test SecurityConfig default values."""
        config = SecurityConfig()

        assert config.require_api_key is True
        assert config.key_header_name == "X-API-Key"
        assert config.enable_ip_blocking is True
        assert config.max_failed_attempts == 5
        assert config.block_duration_minutes == 15
        assert config.enable_audit_logging is True

        # Test default rate limits
        assert RateLimitTier.ADMIN in config.rate_limits
        assert RateLimitTier.USER in config.rate_limits
        assert RateLimitTier.READONLY in config.rate_limits
        assert RateLimitTier.ANONYMOUS in config.rate_limits

        # Test admin tier has highest limits
        admin_limits = config.rate_limits[RateLimitTier.ADMIN]
        user_limits = config.rate_limits[RateLimitTier.USER]
        assert admin_limits.requests_per_minute > user_limits.requests_per_minute
        assert admin_limits.requests_per_hour > user_limits.requests_per_hour

    def test_auth_manager_initialization(self):
        """Test AuthManager initialization."""
        config = SecurityConfig()
        auth_manager = AuthManager(config)

        assert auth_manager.config == config
        assert auth_manager.rate_limiter is not None
        assert auth_manager.security is not None

    def test_auth_manager_hash_key(self):
        """Test API key hashing."""
        config = SecurityConfig()
        auth_manager = AuthManager(config)

        key1 = "test_key_123"
        key2 = "test_key_456"

        hash1 = auth_manager._hash_key(key1)
        hash2 = auth_manager._hash_key(key2)

        # Hashes should be different for different keys
        assert hash1 != hash2

        # Same key should produce same hash
        assert auth_manager._hash_key(key1) == hash1

        # Hash should be hex string
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex length

    def test_auth_manager_add_api_key(self):
        """Test adding API keys to auth manager."""
        config = SecurityConfig()
        auth_manager = AuthManager(config)

        api_key = "test_api_key_123"
        auth_manager.add_api_key(
            api_key=api_key,
            role=UserRole.USER,
            tier=RateLimitTier.USER,
            name="test_key"
        )

        key_hash = auth_manager._hash_key(api_key)
        assert key_hash in auth_manager.config.api_keys

        stored_key = auth_manager.config.api_keys[key_hash]
        assert stored_key.role == UserRole.USER
        assert stored_key.tier == RateLimitTier.USER
        assert stored_key.name == "test_key"
        assert stored_key.is_active is True

    def test_auth_manager_validate_api_key(self):
        """Test API key validation."""
        config = SecurityConfig()
        auth_manager = AuthManager(config)

        # Add a test key
        api_key = "valid_test_key"
        auth_manager.add_api_key(
            api_key=api_key,
            role=UserRole.ADMIN,
            tier=RateLimitTier.ADMIN,
            name="admin_key"
        )

        # Test valid key
        result = auth_manager.validate_api_key(api_key)
        assert result is not None
        assert result.role == UserRole.ADMIN
        assert result.tier == RateLimitTier.ADMIN
        assert result.name == "admin_key"

        # Test invalid key
        result = auth_manager.validate_api_key("invalid_key")
        assert result is None

        # Test inactive key
        key_hash = auth_manager._hash_key(api_key)
        auth_manager.config.api_keys[key_hash].is_active = False
        result = auth_manager.validate_api_key(api_key)
        assert result is None


class TestRateLimiting(TestCase):
    """Test rate limiting functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SecurityConfig()
        self.rate_limiter = RateLimiter(self.config)

    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        config = SecurityConfig()
        rate_limiter = RateLimiter(config)

        assert rate_limiter.config == config
        assert isinstance(rate_limiter.request_history, dict)
        assert isinstance(rate_limiter.blocked_ips, dict)
        assert isinstance(rate_limiter.failed_attempts, dict)

    def test_rate_limiter_get_client_ip(self):
        """Test IP extraction from request."""
        # Mock request with X-Forwarded-For header
        mock_request = MagicMock()
        mock_request.headers = {"X-Forwarded-For": "203.0.113.1, 192.168.1.100"}

        result = self.rate_limiter._get_client_ip(mock_request)
        self.assertEqual(result, "203.0.113.1")

        # Test fallback to X-Real-IP
        mock_request.headers = {"X-Real-IP": "192.168.1.100"}
        result = self.rate_limiter._get_client_ip(mock_request)
        self.assertEqual(result, "192.168.1.100")

        # Test fallback to client.host
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.100"
        result = self.rate_limiter._get_client_ip(mock_request)
        self.assertEqual(result, "192.168.1.100")

    def test_rate_limiter_get_client_key(self):
        """Test client key generation."""
        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {}

        # Test with API key - should return hashed key format
        result = self.rate_limiter._get_client_key(mock_request, "test_api_key")
        self.assertTrue(result.startswith("key:"))
        self.assertEqual(len(result), 20)  # "key:" + 16 char hash

        # Test without API key - should return IP format
        result = self.rate_limiter._get_client_key(mock_request)
        self.assertEqual(result, "ip:192.168.1.100")

    def test_rate_limiter_cleanup_old_requests(self):
        """Test cleanup of old requests from sliding window."""
        config = SecurityConfig()
        rate_limiter = RateLimiter(config)

        history = deque()

        current_time = time.time()
        # Add some old and new requests
        history.append(current_time - 120)  # 2 minutes ago (should be removed)
        history.append(current_time - 30)   # 30 seconds ago (should remain)
        history.append(current_time - 10)   # 10 seconds ago (should remain)

        rate_limiter._cleanup_old_requests(history, window_seconds=60)

        # Should have removed the old request
        assert len(history) == 2
        assert current_time - 120 not in history

    def test_rate_limiter_is_rate_limited(self):
        """Test rate limiting logic."""
        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {}

        # Test not rate limited - returns tuple of (bool, dict)
        is_limited, info = self.rate_limiter.is_rate_limited(
            mock_request, RateLimitTier.USER
        )
        self.assertFalse(is_limited)
        self.assertIsInstance(info, dict)
        self.assertIn("requests_remaining", info)

    @patch('synndicate.api.auth.Request')
    def test_rate_limiter_failed_attempts(self, mock_request):
        """Test failed attempt tracking and IP blocking."""
        config = SecurityConfig()
        rate_limiter = RateLimiter(config)

        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {}

        # Record failed attempts
        for _ in range(config.max_failed_attempts):
            rate_limiter.record_failed_attempt(mock_request)

        client_ip = rate_limiter._get_client_ip(mock_request)

        # Should be blocked after max attempts
        assert client_ip in rate_limiter.blocked_ips

        # Test successful auth resets attempts
        rate_limiter.record_successful_auth(mock_request)
        assert rate_limiter.failed_attempts[client_ip] == 0


class TestAPIServerEndpoints(TestCase):
    """Test API server endpoint functionality."""

    def test_query_request_model(self):
        """Test QueryRequest model validation."""
        # Valid request
        request = QueryRequest(
            query="Test query",
            context={"key": "value"},
            workflow="development"
        )

        assert request.query == "Test query"
        assert request.context == {"key": "value"}
        assert request.workflow == "development"

        # Test validation
        with pytest.raises(ValueError):
            QueryRequest(query="")  # Empty query should fail

        with pytest.raises(ValueError):
            QueryRequest(query="x" * 5001)  # Too long query should fail

    def test_query_response_model(self):
        """Test QueryResponse model creation."""
        response = QueryResponse(
            success=True,
            trace_id="test_trace_123",
            response="Test response",
            agents_used=["planner", "coder"],
            execution_path=["analyze", "code", "review"],
            confidence=0.85,
            execution_time=2.5,
            metadata={"key": "value"}
        )

        assert response.success is True
        assert response.trace_id == "test_trace_123"
        assert response.response == "Test response"
        assert response.agents_used == ["planner", "coder"]
        assert response.execution_path == ["analyze", "code", "review"]
        assert response.confidence == 0.85
        assert response.execution_time == 2.5
        assert response.metadata == {"key": "value"}

    def test_health_response_model(self):
        """Test HealthResponse model creation."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            config_hash="abc123",
            uptime_seconds=3600.0,
            components={"orchestrator": "healthy", "models": "healthy"}
        )

        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.config_hash == "abc123"
        assert response.uptime_seconds == 3600.0
        assert response.components == {"orchestrator": "healthy", "models": "healthy"}

    def test_create_app(self):
        """Test FastAPI app creation."""
        app = create_app()
        assert isinstance(app, FastAPI)
        assert app.title == "Synndicate AI"
        assert app.description is not None
        assert app.version is not None

        # Test CORS is configured
        assert any(middleware.cls.__name__ == "CORSMiddleware" for middleware in app.user_middleware)

    def test_process_query_endpoint(self):
        """Test query processing endpoint."""
        # Use the proven simple approach: disable authentication and mock process_query directly
        with patch('synndicate.api.server.get_settings') as mock_get_settings, \
             patch('synndicate.api.server.get_auth_manager') as mock_get_auth:

            # Mock settings to disable API key requirement (simplify test)
            mock_settings = MagicMock()
            mock_settings.api.require_api_key = False
            mock_get_settings.return_value = mock_settings

            # Mock auth manager to return None (no authentication required)
            mock_get_auth.return_value = None

            # Mock the process_query function to return a successful response
            def mock_process_query_success(http_request, request):
                from synndicate.api.server import QueryResponse
                return QueryResponse(
                    success=True,
                    response="Test response",
                    agents_used=["planner"],
                    execution_path=["analysis"],
                    confidence=0.85,
                    metadata={},
                    trace_id="test-trace-123"
                )

            with (
                patch('synndicate.api.server.process_query', side_effect=mock_process_query_success),
                TestClient(app) as client,
            ):
                response = client.post(
                    "/query",
                    json={"query": "test query"}
                )

                self.assertEqual(response.status_code, 200)

                data = response.json()
                self.assertTrue(data["success"])
                self.assertEqual(data["response"], "Test response")

    @patch('synndicate.api.server.get_metrics_registry')
    @patch('synndicate.api.server.get_auth_manager')
    def test_get_metrics_endpoint(self, mock_get_auth_manager, mock_get_metrics_registry):
        """Test metrics endpoint."""
        # Mock authentication
        mock_auth_manager = AsyncMock()
        mock_auth_manager.authenticate_request.return_value = ("admin_key", RateLimitTier.ADMIN)
        # Mock rate limiter to not be rate limited
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.is_rate_limited.return_value = (False, {"requests_remaining": 50})
        mock_auth_manager.rate_limiter = mock_rate_limiter
        mock_get_auth_manager.return_value = mock_auth_manager

        # Mock metrics registry
        mock_registry = MagicMock()
        mock_registry.generate_latest.return_value = b"# Prometheus metrics\ntest_metric 1.0\n"
        mock_get_metrics_registry.return_value = mock_registry

        with TestClient(app) as client:
            response = client.get(
                "/metrics",
                headers={"X-API-Key": "admin_key"}
            )
            self.assertEqual(response.status_code, 200)
            # Check for content-type with flexible charset handling
            content_type = response.headers["content-type"]
            self.assertTrue(content_type.startswith("text/plain"))

    @pytest.mark.asyncio
    async def test_global_exception_handler(self):
        """Test global exception handler."""
        mock_request = MagicMock()
        mock_request.url = MagicMock()
        mock_request.url.path = "/test"
        mock_request.method = "GET"
        mock_request.state = MagicMock()
        mock_request.state.trace_id = "test_trace_123"

        exception = Exception("Test error")

        # global_exception_handler is async in the actual implementation
        response = await global_exception_handler(mock_request, exception)
        self.assertEqual(response.status_code, 500)

        # Parse response body
        response_data = json.loads(response.body.decode())
        self.assertEqual(response_data["error"], "Internal server error")
        self.assertIn("trace_id", response_data)

    @patch('synndicate.api.server.get_auth_manager')
    @pytest.mark.asyncio
    async def test_get_current_user(self, mock_get_auth_manager):
        """Test get_current_user function."""
        # Mock authentication success
        mock_auth_manager = AsyncMock()
        mock_auth_manager.authenticate_request.return_value = ("test_key", RateLimitTier.USER)
        mock_get_auth_manager.return_value = mock_auth_manager

        mock_request = MagicMock()
        mock_request.state = MagicMock()

        # Should not raise exception for valid auth
        try:
            result = await get_current_user(mock_request)  # get_current_user is async
            self.assertIsNotNone(result)
        except HTTPException:
            self.fail("get_current_user raised HTTPException unexpectedly")

    def test_query_processing_errors(self):
        """Test query processing error handling."""
        # Use the proven simple approach: disable authentication and mock process_query to raise error
        with patch('synndicate.api.server.get_settings') as mock_get_settings, \
             patch('synndicate.api.server.get_auth_manager') as mock_get_auth:

            # Mock settings to disable API key requirement (simplify test)
            mock_settings = MagicMock()
            mock_settings.api.require_api_key = False
            mock_get_settings.return_value = mock_settings

            # Mock auth manager to return None (no authentication required)
            mock_get_auth.return_value = None

            # Mock the process_query function to raise an exception
            def mock_process_query_error(http_request, request):
                # Simulate orchestrator processing error
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=500,
                    detail="Processing failed"
                )

            with (
                patch('synndicate.api.server.process_query', side_effect=mock_process_query_error),
                TestClient(app) as client,
            ):
                response = client.post(
                    "/query",
                    json={"query": "test query"}
                )
                # Should get 500 due to orchestrator exception handled by global exception handler
                self.assertEqual(response.status_code, 500)


class TestAPIIntegration(TestCase):
    """Test API system integration and end-to-end functionality."""

    @patch('synndicate.api.server.get_metrics_registry')
    @patch('synndicate.api.server.get_auth_manager')
    def test_get_metrics_endpoint(self, mock_get_auth_manager, mock_get_metrics_registry):
        """Test metrics endpoint."""
        # Mock authentication
        mock_auth_manager = AsyncMock()
        mock_auth_manager.authenticate_request.return_value = ("admin_key", RateLimitTier.ADMIN)
        mock_get_auth_manager.return_value = mock_auth_manager

        # Mock metrics registry
        mock_registry = MagicMock()
        mock_registry.generate_latest.return_value = b"# Prometheus metrics\ntest_metric 1.0\n"
        mock_get_metrics_registry.return_value = mock_registry

        with TestClient(app) as client:
            response = client.get(
                "/metrics",
                headers={"X-API-Key": "admin_key"}
            )
            self.assertEqual(response.status_code, 200)
            # Check for content-type with flexible charset handling
            content_type = response.headers["content-type"]
            self.assertTrue(content_type.startswith("text/plain"))

    @patch('synndicate.api.server.get_auth_manager')
    def test_add_rate_limit_headers(self, mock_get_auth_manager):
        """Test rate limit header addition."""
        mock_response = MagicMock()
        mock_request = MagicMock()
        # Create a proper mock state object with rate_limit_tier attribute
        mock_request.state = MagicMock()
        mock_request.state.rate_limit_tier = RateLimitTier.USER

        add_rate_limit_headers(mock_response, mock_request)

        # Verify headers were added
        mock_response.headers.__setitem__.assert_called()

    def test_fastapi_test_client_integration(self):
        """Test FastAPI test client integration."""
        with (
            TestClient(app) as client,
            patch.object(app.state, 'startup_time', time.time()),
            patch.object(app.state, 'config_hash', 'test_hash_123'),
        ):
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)

            data = response.json()
            self.assertEqual(data["status"], "healthy")
            self.assertIn("uptime_seconds", data)


class TestAPIErrorHandling(TestCase):
    """Test API error handling and edge cases."""

    async def test_authentication_errors(self):
        """Test authentication error handling."""
        auth_manager = AuthManager(SecurityConfig())

        # Test invalid API key - authenticate_request returns (None, ANONYMOUS) for invalid keys
        mock_request = MagicMock()
        mock_request.headers = {"X-API-Key": "invalid_key"}
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"

        api_key, tier = await auth_manager.authenticate_request(mock_request)
        self.assertIsNone(api_key)
        self.assertEqual(tier, RateLimitTier.ANONYMOUS)

    def test_query_processing_errors(self):
        """Test query processing error handling."""
        # Use the proven simple approach: disable authentication and mock process_query to raise error
        with patch('synndicate.api.server.get_settings') as mock_get_settings, \
             patch('synndicate.api.server.get_auth_manager') as mock_get_auth:

            # Mock settings to disable API key requirement (simplify test)
            mock_settings = MagicMock()
            mock_settings.api.require_api_key = False
            mock_get_settings.return_value = mock_settings

            # Mock auth manager to return None (no authentication required)
            mock_get_auth.return_value = None

            # Mock the process_query function to raise an exception
            def mock_process_query_error(http_request, request):
                # Simulate orchestrator processing error
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=500,
                    detail="Processing failed"
                )

            with (
                patch('synndicate.api.server.process_query', side_effect=mock_process_query_error),
                TestClient(app) as client,
            ):
                response = client.post(
                    "/query",
                    json={"query": "test query"}
                )
                # Should get 500 due to orchestrator exception handled by global exception handler
                self.assertEqual(response.status_code, 500)

    def test_rate_limit_config_edge_cases(self):
        """Test rate limit configuration edge cases."""
        # Test minimum values
        config = RateLimitConfig(
            requests_per_minute=1,
            requests_per_hour=1,
            burst_size=1,
            window_size_seconds=1
        )

        assert config.requests_per_minute == 1
        assert config.requests_per_hour == 1
        assert config.burst_size == 1
        assert config.window_size_seconds == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
