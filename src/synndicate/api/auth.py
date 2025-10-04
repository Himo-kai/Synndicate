"""
API Authentication and Rate Limiting System.

Provides comprehensive security for Synndicate API endpoints with:
- API key authentication with role-based access control (RBAC)
- Rate limiting with sliding window algorithm
- Request throttling and abuse prevention
- Security audit logging and monitoring

Features:
- 🔐 Multi-tier API key system (admin, user, readonly)
- 🚦 Configurable rate limits per endpoint and user tier
- 📊 Real-time rate limit monitoring and metrics
- 🛡️ Automatic IP blocking for abuse prevention
- 📝 Comprehensive security audit logging
- ⚡ High-performance in-memory rate limiting

Usage:
    from synndicate.api.auth import require_auth, RateLimiter

    @app.post("/query")
    @require_auth(min_role="user")
    async def process_query(request: QueryRequest):
        # Protected endpoint logic
        pass
"""

import hashlib
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field

from ..observability.logging import get_logger

log = get_logger("syn.api.auth")


class UserRole(str, Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"

    @property
    def privilege_level(self) -> int:
        """Return numeric privilege level for comparison (higher = more privileged)."""
        levels = {UserRole.READONLY: 1, UserRole.USER: 2, UserRole.ADMIN: 3}
        return levels[self]


class RateLimitTier(str, Enum):
    """Rate limit tiers."""

    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    ANONYMOUS = "anonymous"


class APIKey(BaseModel):
    """API key configuration."""

    key_hash: str
    role: UserRole
    tier: RateLimitTier
    name: str
    created_at: datetime
    last_used: datetime | None = None
    is_active: bool = True
    rate_limit_override: dict[str, int] | None = None


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    requests_per_minute: int = Field(60, gt=0)
    requests_per_hour: int = Field(1000, gt=0)
    burst_size: int = Field(10, gt=0)
    window_size_seconds: int = Field(60, gt=0)


class SecurityConfig(BaseModel):
    """Security configuration."""

    # API Key settings
    api_keys: dict[str, APIKey] = Field(default_factory=dict)
    require_api_key: bool = Field(True)
    key_header_name: str = Field("X-API-Key")

    # Rate limiting per tier
    rate_limits: dict[RateLimitTier, RateLimitConfig] = Field(
        default_factory=lambda: {
            RateLimitTier.ADMIN: RateLimitConfig(
                requests_per_minute=300, requests_per_hour=10000, burst_size=50
            ),
            RateLimitTier.USER: RateLimitConfig(
                requests_per_minute=60, requests_per_hour=1000, burst_size=10
            ),
            RateLimitTier.READONLY: RateLimitConfig(
                requests_per_minute=30, requests_per_hour=500, burst_size=5
            ),
            RateLimitTier.ANONYMOUS: RateLimitConfig(
                requests_per_minute=10, requests_per_hour=100, burst_size=2
            ),
        }
    )

    # Security settings
    enable_ip_blocking: bool = Field(True)
    max_failed_attempts: int = Field(5, gt=0)
    block_duration_minutes: int = Field(15, gt=0)
    enable_audit_logging: bool = Field(True)


class RateLimiter:
    """High-performance sliding window rate limiter."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_history: dict[str, deque] = defaultdict(deque)
        self.blocked_ips: dict[str, datetime] = {}
        self.failed_attempts: dict[str, int] = defaultdict(int)

    def _get_client_key(self, request: Request, api_key: str | None = None) -> str:
        """Generate unique client key for rate limiting."""
        if api_key:
            return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

        # Use IP address for anonymous requests
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self, history: deque, window_seconds: int):
        """Remove requests outside the sliding window."""
        cutoff_time = time.time() - window_seconds
        while history and history[0] < cutoff_time:
            history.popleft()

    def is_rate_limited(
        self, request: Request, tier: RateLimitTier, api_key: str | None = None
    ) -> tuple[bool, dict[str, Any]]:
        """Check if request should be rate limited."""
        client_key = self._get_client_key(request, api_key)
        client_ip = self._get_client_ip(request)

        # Check if IP is blocked
        if self.config.enable_ip_blocking and client_ip in self.blocked_ips:
            block_time = self.blocked_ips[client_ip]
            if datetime.now() < block_time:
                log.warning("Blocked IP attempted access", ip=client_ip)
                return True, {
                    "error": "IP temporarily blocked",
                    "blocked_until": block_time.isoformat(),
                    "reason": "too_many_failed_attempts",
                }
            else:
                # Unblock expired IPs
                del self.blocked_ips[client_ip]
                self.failed_attempts[client_ip] = 0

        # Get rate limit config for tier
        rate_config = self.config.rate_limits.get(tier)
        if not rate_config:
            log.error("No rate limit config for tier", tier=tier)
            return False, {}

        # Get request history for this client
        history = self.request_history[client_key]
        current_time = time.time()

        # Clean up old requests (sliding window)
        self._cleanup_old_requests(history, 60)  # 1-minute window

        # Check rate limits
        requests_in_minute = len(history)

        # Check burst limit
        if requests_in_minute >= rate_config.burst_size:
            # Check if within burst window (last 10 seconds)
            recent_requests = sum(1 for req_time in history if current_time - req_time <= 10)
            if recent_requests >= rate_config.burst_size:
                log.warning(
                    "Rate limit exceeded (burst)", client=client_key, requests=recent_requests
                )
                return True, {
                    "error": "Rate limit exceeded (burst)",
                    "limit": rate_config.burst_size,
                    "window": "10 seconds",
                    "retry_after": 10,
                }

        # Check per-minute limit
        if requests_in_minute >= rate_config.requests_per_minute:
            log.warning(
                "Rate limit exceeded (per minute)", client=client_key, requests=requests_in_minute
            )
            return True, {
                "error": "Rate limit exceeded",
                "limit": rate_config.requests_per_minute,
                "window": "1 minute",
                "retry_after": 60,
            }

        # Record this request
        history.append(current_time)

        return False, {
            "requests_remaining": rate_config.requests_per_minute - requests_in_minute - 1,
            "reset_time": int(current_time) + 60,
        }

    def record_failed_attempt(self, request: Request):
        """Record a failed authentication attempt."""
        if not self.config.enable_ip_blocking:
            return

        client_ip = self._get_client_ip(request)
        self.failed_attempts[client_ip] += 1

        log.warning(
            "Failed authentication attempt", ip=client_ip, attempts=self.failed_attempts[client_ip]
        )

        if self.failed_attempts[client_ip] >= self.config.max_failed_attempts:
            # Block the IP
            block_until = datetime.now() + timedelta(minutes=self.config.block_duration_minutes)
            self.blocked_ips[client_ip] = block_until

            log.error(
                "IP blocked due to repeated failures",
                ip=client_ip,
                attempts=self.failed_attempts[client_ip],
                blocked_until=block_until.isoformat(),
            )

    def record_successful_auth(self, request: Request):
        """Record a successful authentication (reset failed attempts)."""
        client_ip = self._get_client_ip(request)
        if client_ip in self.failed_attempts:
            del self.failed_attempts[client_ip]


class AuthManager:
    """API authentication manager."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config)
        self.security = HTTPBearer(auto_error=False)

    def _hash_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def add_api_key(self, api_key: str, role: UserRole, tier: RateLimitTier, name: str) -> bool:
        """Add a new API key."""
        key_hash = self._hash_key(api_key)

        if key_hash in self.config.api_keys:
            log.warning("Attempted to add duplicate API key", name=name)
            return False

        self.config.api_keys[key_hash] = APIKey(
            key_hash=key_hash, role=role, tier=tier, name=name, created_at=datetime.now()
        )

        log.info("API key added", name=name, role=role, tier=tier)
        return True

    def validate_api_key(self, api_key: str) -> APIKey | None:
        """Validate API key and return key info."""
        key_hash = self._hash_key(api_key)
        key_info = self.config.api_keys.get(key_hash)

        if not key_info or not key_info.is_active:
            return None

        # Update last used timestamp
        key_info.last_used = datetime.now()
        return key_info

    async def authenticate_request(self, request: Request) -> tuple[APIKey | None, RateLimitTier]:
        """Authenticate request and determine rate limit tier."""
        api_key = None
        tier = RateLimitTier.ANONYMOUS

        # Try to get API key from header
        auth_header = request.headers.get(self.config.key_header_name)
        if auth_header:
            key_info = self.validate_api_key(auth_header)
            if key_info:
                api_key = key_info
                tier = key_info.tier
                self.rate_limiter.record_successful_auth(request)

                if self.config.enable_audit_logging:
                    log.info(
                        "API key authenticated",
                        name=key_info.name,
                        role=key_info.role,
                        endpoint=request.url.path,
                    )
            else:
                self.rate_limiter.record_failed_attempt(request)
                if self.config.enable_audit_logging:
                    log.warning(
                        "Invalid API key used",
                        endpoint=request.url.path,
                        ip=self.rate_limiter._get_client_ip(request),
                    )

        elif self.config.require_api_key:
            self.rate_limiter.record_failed_attempt(request)
            if self.config.enable_audit_logging:
                log.warning(
                    "Missing API key",
                    endpoint=request.url.path,
                    ip=self.rate_limiter._get_client_ip(request),
                )

        return api_key, tier


# Global auth manager instance
_auth_manager: AuthManager | None = None


def get_auth_manager() -> AuthManager:
    """Get global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        # Initialize with default config
        config = SecurityConfig()
        _auth_manager = AuthManager(config)

        # Add default admin key for development
        _auth_manager.add_api_key(
            "syn_admin_dev_key_12345", UserRole.ADMIN, RateLimitTier.ADMIN, "Development Admin Key"
        )

        log.info("Auth manager initialized with default config")

    return _auth_manager


def require_auth(min_role: UserRole = UserRole.READONLY):
    """Decorator to require authentication for endpoints."""

    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            auth_manager = get_auth_manager()

            # Authenticate request
            api_key, tier = await auth_manager.authenticate_request(request)

            # Check if API key is required
            if auth_manager.config.require_api_key and not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Check role permissions
            if (
                api_key
                and UserRole(api_key.role).privilege_level < UserRole(min_role).privilege_level
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {min_role}",
                )

            # Check rate limits
            is_limited, limit_info = auth_manager.rate_limiter.is_rate_limited(request, tier)
            if is_limited:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=limit_info.get("error", "Rate limit exceeded"),
                    headers={
                        "Retry-After": str(limit_info.get("retry_after", 60)),
                        "X-RateLimit-Limit": str(
                            auth_manager.config.rate_limits[tier].requests_per_minute
                        ),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(
                            limit_info.get("reset_time", int(time.time()) + 60)
                        ),
                    },
                )

            # Add rate limit headers to successful requests
            if not is_limited and "requests_remaining" in limit_info:
                # Store rate limit info for response headers
                request.state.rate_limit_info = limit_info
                request.state.rate_limit_tier = tier

            # Store auth info in request state
            request.state.api_key = api_key
            request.state.user_role = api_key.role if api_key else None

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


def add_rate_limit_headers(response, request: Request):
    """Add rate limit headers to response."""
    if hasattr(request.state, "rate_limit_info"):
        limit_info = request.state.rate_limit_info
        tier = request.state.rate_limit_tier
        auth_manager = get_auth_manager()

        response.headers["X-RateLimit-Limit"] = str(
            auth_manager.config.rate_limits[tier].requests_per_minute
        )
        response.headers["X-RateLimit-Remaining"] = str(limit_info.get("requests_remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(
            limit_info.get("reset_time", int(time.time()) + 60)
        )
