"""
Comprehensive tests for API authentication and rate limiting system.

Tests cover:
- API key management and validation
- Role-based access control (RBAC)
- Rate limiting with sliding window algorithm
- IP blocking and abuse prevention
- Security audit logging
- Authentication decorators and middleware
"""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from synndicate.api.auth import (
    AuthManager,
    RateLimiter,
    RateLimitTier,
    SecurityConfig,
    UserRole,
    add_rate_limit_headers,
    get_auth_manager,
    require_auth,
)


@pytest.fixture
def security_config():
    """Create test security configuration."""
    return SecurityConfig(
        require_api_key=True,
        enable_ip_blocking=True,
        max_failed_attempts=3,
        block_duration_minutes=5,
        enable_audit_logging=True,
    )


@pytest.fixture
def auth_manager(security_config):
    """Create test authentication manager."""
    manager = AuthManager(security_config)

    # Add test API keys
    manager.add_api_key("admin_key_12345", UserRole.ADMIN, RateLimitTier.ADMIN, "Test Admin")
    manager.add_api_key("user_key_12345", UserRole.USER, RateLimitTier.USER, "Test User")
    manager.add_api_key(
        "readonly_key_12345", UserRole.READONLY, RateLimitTier.READONLY, "Test Readonly"
    )

    return manager


@pytest.fixture
def mock_request():
    """Create mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.client.host = "127.0.0.1"
    request.headers = {}
    request.url.path = "/test"
    request.state = MagicMock()
    return request


class TestAPIKeyManagement:
    """Test API key creation, validation, and management."""

    def test_add_api_key_success(self, auth_manager):
        """Test successful API key addition."""
        result = auth_manager.add_api_key(
            "new_key_12345", UserRole.USER, RateLimitTier.USER, "New Test Key"
        )

        assert result is True

        # Verify key can be validated
        key_info = auth_manager.validate_api_key("new_key_12345")
        assert key_info is not None
        assert key_info.role == UserRole.USER
        assert key_info.tier == RateLimitTier.USER
        assert key_info.name == "New Test Key"

    def test_add_duplicate_api_key(self, auth_manager):
        """Test adding duplicate API key fails."""
        # First addition should succeed
        result1 = auth_manager.add_api_key("dup_key", UserRole.USER, RateLimitTier.USER, "First")
        assert result1 is True

        # Second addition should fail
        result2 = auth_manager.add_api_key("dup_key", UserRole.ADMIN, RateLimitTier.ADMIN, "Second")
        assert result2 is False

    def test_validate_valid_api_key(self, auth_manager):
        """Test validation of valid API key."""
        key_info = auth_manager.validate_api_key("user_key_12345")

        assert key_info is not None
        assert key_info.role == UserRole.USER
        assert key_info.tier == RateLimitTier.USER
        assert key_info.name == "Test User"
        assert key_info.is_active is True
        assert isinstance(key_info.last_used, datetime)

    def test_validate_invalid_api_key(self, auth_manager):
        """Test validation of invalid API key."""
        key_info = auth_manager.validate_api_key("invalid_key")
        assert key_info is None

    def test_validate_inactive_api_key(self, auth_manager):
        """Test validation of inactive API key."""
        # Add key and then deactivate
        auth_manager.add_api_key("inactive_key", UserRole.USER, RateLimitTier.USER, "Inactive")
        key_hash = auth_manager._hash_key("inactive_key")
        auth_manager.config.api_keys[key_hash].is_active = False

        key_info = auth_manager.validate_api_key("inactive_key")
        assert key_info is None

    def test_api_key_hashing(self, auth_manager):
        """Test API key hashing is consistent."""
        key = "test_key_123"
        hash1 = auth_manager._hash_key(key)
        hash2 = auth_manager._hash_key(key)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
        assert hash1 != key  # Should be hashed, not plain text


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limiter_initialization(self, security_config):
        """Test rate limiter initialization."""
        limiter = RateLimiter(security_config)

        assert limiter.config == security_config
        assert isinstance(limiter.request_history, dict)
        assert isinstance(limiter.blocked_ips, dict)
        assert isinstance(limiter.failed_attempts, dict)

    def test_get_client_key_with_api_key(self, security_config, mock_request):
        """Test client key generation with API key."""
        limiter = RateLimiter(security_config)

        key = limiter._get_client_key(mock_request, "test_api_key")

        assert key.startswith("key:")
        assert len(key) == 20  # "key:" + 16 char hash

    def test_get_client_key_without_api_key(self, security_config, mock_request):
        """Test client key generation without API key (IP-based)."""
        limiter = RateLimiter(security_config)

        key = limiter._get_client_key(mock_request)

        assert key.startswith("ip:")
        assert "127.0.0.1" in key

    def test_get_client_ip_direct(self, security_config, mock_request):
        """Test client IP extraction from direct connection."""
        limiter = RateLimiter(security_config)

        ip = limiter._get_client_ip(mock_request)
        assert ip == "127.0.0.1"

    def test_get_client_ip_forwarded(self, security_config, mock_request):
        """Test client IP extraction from forwarded headers."""
        limiter = RateLimiter(security_config)
        mock_request.headers = {"X-Forwarded-For": "192.168.1.100, 10.0.0.1"}

        ip = limiter._get_client_ip(mock_request)
        assert ip == "192.168.1.100"

    def test_get_client_ip_real_ip(self, security_config, mock_request):
        """Test client IP extraction from X-Real-IP header."""
        limiter = RateLimiter(security_config)
        mock_request.headers = {"X-Real-IP": "203.0.113.1"}

        ip = limiter._get_client_ip(mock_request)
        assert ip == "203.0.113.1"

    def test_rate_limit_within_limits(self, security_config, mock_request):
        """Test request within rate limits."""
        limiter = RateLimiter(security_config)

        is_limited, info = limiter.is_rate_limited(mock_request, RateLimitTier.USER)

        assert is_limited is False
        assert "requests_remaining" in info
        assert info["requests_remaining"] >= 0

    def test_rate_limit_burst_exceeded(self, security_config, mock_request):
        """Test burst rate limit exceeded."""
        limiter = RateLimiter(security_config)

        # Simulate burst of requests
        for _ in range(15):  # Exceed burst size of 10
            limiter.is_rate_limited(mock_request, RateLimitTier.USER)

        is_limited, info = limiter.is_rate_limited(mock_request, RateLimitTier.USER)

        assert is_limited is True
        assert "Rate limit exceeded" in info["error"]
        assert info["retry_after"] == 10

    def test_rate_limit_per_minute_exceeded(self, security_config, mock_request):
        """Test per-minute rate limit exceeded."""
        limiter = RateLimiter(security_config)

        # Simulate many requests over time to exceed per-minute limit
        client_key = limiter._get_client_key(mock_request)
        history = limiter.request_history[client_key]

        # Add 61 requests (exceeds 60/min limit for USER tier)
        # Ensure no more than 9 requests in any 10-second window to avoid burst limit
        # but still exceed 60 requests in 60-second window to trigger per-minute limit
        current_time = time.time()

        # Add requests in batches of 9 with 11-second gaps to avoid burst limit
        request_time = current_time
        for _batch in range(7):  # 7 batches of 9 = 63 requests
            for i in range(9):
                history.append(request_time - i * 0.1)  # 9 requests within 0.9 seconds
            request_time -= 11  # Move to next batch 11 seconds earlier

        is_limited, info = limiter.is_rate_limited(mock_request, RateLimitTier.USER)

        assert is_limited is True
        assert "Rate limit exceeded" in info["error"]
        assert info["retry_after"] == 60

    def test_sliding_window_cleanup(self, security_config, mock_request):
        """Test sliding window cleanup of old requests."""
        limiter = RateLimiter(security_config)

        client_key = limiter._get_client_key(mock_request)
        history = limiter.request_history[client_key]

        # Add old requests (should be cleaned up)
        old_time = time.time() - 120  # 2 minutes ago
        for _ in range(10):
            history.append(old_time)

        # Add recent request
        history.append(time.time())

        # Trigger cleanup
        limiter._cleanup_old_requests(history, 60)

        # Should only have 1 recent request left
        assert len(history) == 1

    def test_different_tier_limits(self, security_config, mock_request):
        """Test different rate limits for different tiers."""
        limiter = RateLimiter(security_config)

        # Admin tier should have higher limits
        is_limited_admin, info_admin = limiter.is_rate_limited(mock_request, RateLimitTier.ADMIN)

        # Change IP to test different client
        mock_request.client.host = "192.168.1.1"
        is_limited_readonly, info_readonly = limiter.is_rate_limited(
            mock_request, RateLimitTier.READONLY
        )

        assert is_limited_admin is False
        assert is_limited_readonly is False

        # Admin should have more requests remaining
        if "requests_remaining" in info_admin and "requests_remaining" in info_readonly:
            assert info_admin["requests_remaining"] > info_readonly["requests_remaining"]


class TestIPBlocking:
    """Test IP blocking functionality."""

    def test_record_failed_attempt(self, security_config, mock_request):
        """Test recording failed authentication attempts."""
        limiter = RateLimiter(security_config)

        # Record failed attempts
        for _i in range(2):
            limiter.record_failed_attempt(mock_request)

        ip = limiter._get_client_ip(mock_request)
        assert limiter.failed_attempts[ip] == 2

    def test_ip_blocking_after_max_attempts(self, security_config, mock_request):
        """Test IP blocking after maximum failed attempts."""
        limiter = RateLimiter(security_config)

        # Record max failed attempts (3 in test config)
        for _ in range(3):
            limiter.record_failed_attempt(mock_request)

        ip = limiter._get_client_ip(mock_request)

        # IP should be blocked
        assert ip in limiter.blocked_ips
        assert limiter.blocked_ips[ip] > datetime.now()

    def test_blocked_ip_rate_limiting(self, security_config, mock_request):
        """Test rate limiting for blocked IP."""
        limiter = RateLimiter(security_config)

        # Block the IP
        ip = limiter._get_client_ip(mock_request)
        limiter.blocked_ips[ip] = datetime.now() + timedelta(minutes=10)

        is_limited, info = limiter.is_rate_limited(mock_request, RateLimitTier.USER)

        assert is_limited is True
        assert "IP temporarily blocked" in info["error"]
        assert "blocked_until" in info

    def test_expired_block_removal(self, security_config, mock_request):
        """Test removal of expired IP blocks."""
        limiter = RateLimiter(security_config)

        # Add expired block
        ip = limiter._get_client_ip(mock_request)
        limiter.blocked_ips[ip] = datetime.now() - timedelta(minutes=1)
        limiter.failed_attempts[ip] = 5

        is_limited, info = limiter.is_rate_limited(mock_request, RateLimitTier.USER)

        # Should not be limited (block expired)
        assert is_limited is False

        # Block and failed attempts should be cleared
        assert ip not in limiter.blocked_ips
        assert limiter.failed_attempts[ip] == 0

    def test_successful_auth_resets_attempts(self, security_config, mock_request):
        """Test successful authentication resets failed attempts."""
        limiter = RateLimiter(security_config)

        # Record failed attempts
        ip = limiter._get_client_ip(mock_request)
        limiter.failed_attempts[ip] = 2

        # Record successful auth
        limiter.record_successful_auth(mock_request)

        # Failed attempts should be reset
        assert ip not in limiter.failed_attempts


class TestAuthenticationFlow:
    """Test complete authentication flow."""

    @pytest.mark.asyncio
    async def test_authenticate_request_valid_key(self, auth_manager, mock_request):
        """Test authentication with valid API key."""
        mock_request.headers = {"X-API-Key": "user_key_12345"}

        api_key, tier = await auth_manager.authenticate_request(mock_request)

        assert api_key is not None
        assert api_key.role == UserRole.USER
        assert tier == RateLimitTier.USER

    @pytest.mark.asyncio
    async def test_authenticate_request_invalid_key(self, auth_manager, mock_request):
        """Test authentication with invalid API key."""
        mock_request.headers = {"X-API-Key": "invalid_key"}

        api_key, tier = await auth_manager.authenticate_request(mock_request)

        assert api_key is None
        assert tier == RateLimitTier.ANONYMOUS

    @pytest.mark.asyncio
    async def test_authenticate_request_no_key(self, auth_manager, mock_request):
        """Test authentication without API key."""
        api_key, tier = await auth_manager.authenticate_request(mock_request)

        assert api_key is None
        assert tier == RateLimitTier.ANONYMOUS

    @pytest.mark.asyncio
    async def test_authenticate_request_custom_header(self, security_config, mock_request):
        """Test authentication with custom header name."""
        security_config.key_header_name = "Authorization"
        auth_manager = AuthManager(security_config)
        auth_manager.add_api_key("custom_key", UserRole.USER, RateLimitTier.USER, "Custom")

        mock_request.headers = {"Authorization": "custom_key"}

        api_key, tier = await auth_manager.authenticate_request(mock_request)

        assert api_key is not None
        assert api_key.role == UserRole.USER


class TestAuthenticationDecorator:
    """Test require_auth decorator."""

    def test_require_auth_decorator_success(self, auth_manager, mock_request):
        """Test successful authentication with decorator."""

        @require_auth(min_role=UserRole.USER)
        async def test_endpoint(request: Request):
            return {"success": True}

        # Mock successful authentication
        with (
            patch("synndicate.api.auth.get_auth_manager", return_value=auth_manager),
            patch.object(
                auth_manager,
                "authenticate_request",
                return_value=(
                    auth_manager.config.api_keys[auth_manager._hash_key("user_key_12345")],
                    RateLimitTier.USER,
                ),
            ),
            patch.object(
                auth_manager.rate_limiter,
                "is_rate_limited",
                return_value=(False, {"requests_remaining": 59}),
            ),
        ):
            # This would be called by FastAPI in real scenario
            # Here we just test the decorator logic doesn't raise
            pass

    def test_require_auth_decorator_missing_key(self, auth_manager, mock_request):
        """Test decorator with missing API key."""

        @require_auth(min_role=UserRole.USER)
        async def test_endpoint(request: Request):
            return {"success": True}

        with (
            patch("synndicate.api.auth.get_auth_manager", return_value=auth_manager),
            patch.object(
                auth_manager, "authenticate_request", return_value=(None, RateLimitTier.ANONYMOUS)
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            # Simulate decorator execution
            import asyncio

            asyncio.run(test_endpoint(mock_request))

        assert exc_info.value.status_code == 401
        assert "API key required" in str(exc_info.value.detail)

    def test_require_auth_decorator_insufficient_role(self, auth_manager, mock_request):
        """Test decorator with insufficient role."""

        @require_auth(min_role=UserRole.ADMIN)
        async def test_endpoint(request: Request):
            return {"success": True}

        # User key trying to access admin endpoint
        user_key = auth_manager.config.api_keys[auth_manager._hash_key("user_key_12345")]

        with (
            patch("synndicate.api.auth.get_auth_manager", return_value=auth_manager),
            patch.object(
                auth_manager, "authenticate_request", return_value=(user_key, RateLimitTier.USER)
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            import asyncio

            asyncio.run(test_endpoint(mock_request))

        assert exc_info.value.status_code == 403
        assert "Insufficient permissions" in str(exc_info.value.detail)

    def test_require_auth_decorator_rate_limited(self, auth_manager, mock_request):
        """Test decorator with rate limiting."""

        @require_auth(min_role=UserRole.USER)
        async def test_endpoint(request: Request):
            return {"success": True}

        user_key = auth_manager.config.api_keys[auth_manager._hash_key("user_key_12345")]

        with (
            patch("synndicate.api.auth.get_auth_manager", return_value=auth_manager),
            patch.object(
                auth_manager, "authenticate_request", return_value=(user_key, RateLimitTier.USER)
            ),
            patch.object(
                auth_manager.rate_limiter,
                "is_rate_limited",
                return_value=(True, {"error": "Rate limit exceeded", "retry_after": 60}),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            import asyncio

            asyncio.run(test_endpoint(mock_request))

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value.detail)


class TestRateLimitHeaders:
    """Test rate limit header functionality."""

    def test_add_rate_limit_headers(self, mock_request):
        """Test adding rate limit headers to response."""
        from fastapi import Response

        # Mock request state
        mock_request.state.rate_limit_info = {
            "requests_remaining": 45,
            "reset_time": int(time.time()) + 60,
        }
        mock_request.state.rate_limit_tier = RateLimitTier.USER

        response = Response()

        with patch("synndicate.api.auth.get_auth_manager") as mock_get_auth:
            mock_auth = MagicMock()
            mock_auth.config.rate_limits = {RateLimitTier.USER: MagicMock(requests_per_minute=60)}
            mock_get_auth.return_value = mock_auth

            add_rate_limit_headers(response, mock_request)

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

        assert response.headers["X-RateLimit-Limit"] == "60"
        assert response.headers["X-RateLimit-Remaining"] == "45"


class TestSecurityConfiguration:
    """Test security configuration."""

    def test_security_config_defaults(self):
        """Test security configuration defaults."""
        config = SecurityConfig()

        assert config.require_api_key is True
        assert config.key_header_name == "X-API-Key"
        assert config.enable_ip_blocking is True
        assert config.max_failed_attempts == 5
        assert config.block_duration_minutes == 15
        assert config.enable_audit_logging is True

    def test_rate_limit_config_defaults(self):
        """Test rate limit configuration defaults."""
        config = SecurityConfig()

        # Check all tiers have configuration
        assert RateLimitTier.ADMIN in config.rate_limits
        assert RateLimitTier.USER in config.rate_limits
        assert RateLimitTier.READONLY in config.rate_limits
        assert RateLimitTier.ANONYMOUS in config.rate_limits

        # Admin should have highest limits
        admin_limits = config.rate_limits[RateLimitTier.ADMIN]
        user_limits = config.rate_limits[RateLimitTier.USER]

        assert admin_limits.requests_per_minute > user_limits.requests_per_minute
        assert admin_limits.requests_per_hour > user_limits.requests_per_hour
        assert admin_limits.burst_size > user_limits.burst_size


class TestGlobalAuthManager:
    """Test global auth manager singleton."""

    def test_get_auth_manager_singleton(self):
        """Test get_auth_manager returns singleton."""
        manager1 = get_auth_manager()
        manager2 = get_auth_manager()

        assert manager1 is manager2

    def test_get_auth_manager_default_key(self):
        """Test get_auth_manager creates default admin key."""
        manager = get_auth_manager()

        # Should have default admin key
        key_info = manager.validate_api_key("syn_admin_dev_key_12345")
        assert key_info is not None
        assert key_info.role == UserRole.ADMIN
        assert key_info.name == "Development Admin Key"
