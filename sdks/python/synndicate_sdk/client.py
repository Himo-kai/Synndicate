"""
Main client class for the Synndicate Python SDK.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any, TypeGuard, cast

import httpx

from .exceptions import AuthenticationError, RateLimitError, SynndicateError
from .models import AgentResponse, AnalyticsReport, MultiModalInput


def _is_dict_str_any(x: Any) -> TypeGuard[dict[str, Any]]:
    """TypeGuard to help MyPy understand JSON response is a dict."""
    return isinstance(x, dict)


class SynndicateClient:
    """
    Main client for interacting with the Synndicate AI system.

    Provides high-level interfaces for multi-modal agents, real-time processing,
    and analytics capabilities.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.synndicate.ai",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "synndicate-python-sdk/1.0.0",
                "Content-Type": "application/json",
            }
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def process_multimodal(
        self,
        input_data: MultiModalInput,
        agent_type: str = "multimodal",
        **kwargs
    ) -> AgentResponse:
        """Process multi-modal input using specified agent."""
        endpoint = f"/api/v1/agents/{agent_type}/process"

        payload = {
            "input": input_data.to_dict(),
            **kwargs
        }

        response = await self._make_request("POST", endpoint, json=payload)
        return AgentResponse.from_dict(response)

    async def get_analytics_report(
        self,
        hours_back: int = 1,
        include_recommendations: bool = True
    ) -> AnalyticsReport:
        """Get analytics report for usage patterns and optimization."""
        endpoint = "/api/v1/analytics/report"

        params = {
            "hours_back": hours_back,
            "include_recommendations": include_recommendations
        }

        response = await self._make_request("GET", endpoint, params=params)
        return AnalyticsReport.from_dict(response)

    async def submit_realtime_document(
        self,
        document_id: str,
        content: bytes | str,
        content_type: str = "text/plain",
        priority: int = 0
    ) -> str:
        """Submit document for real-time processing."""
        endpoint = "/api/v1/realtime/submit"

        payload = {
            "document_id": document_id,
            "content": content,
            "content_type": content_type,
            "priority": priority
        }

        response = await self._make_request("POST", endpoint, json=payload)
        return str(response["task_id"])

    async def stream_processing_results(
        self,
        document_id: str | None = None
    ) -> AsyncIterator[dict]:
        """Stream real-time processing results."""
        endpoint = "/api/v1/realtime/stream"
        params = {"document_id": document_id} if document_id else {}

        async with self._client.stream("GET", f"{self.base_url}{endpoint}", params=params) as response:
            async for line in response.aiter_lines():
                if line.strip():
                    yield eval(line)  # In production, use proper JSON parsing

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> dict[str, Any]:
        """Make HTTP request with error handling and retries."""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(method, url, **kwargs)

                # Handle error status codes first
                if response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status_code == 429:
                    raise RateLimitError("Rate limit exceeded")
                elif response.status_code >= 400:
                    raise SynndicateError(f"API error: {response.status_code}")

                # Gold-standard TypeGuard + Cast pattern
                json_response = response.json()
                if not _is_dict_str_any(json_response):
                    raise SynndicateError(f"Expected JSON object, got {type(json_response).__name__}")
                return cast("dict[str, Any]", json_response)

            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise SynndicateError(f"Request failed: {e}") from e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # This should never be reached due to the loop structure, but MyPy requires it
        raise SynndicateError("Maximum retries exceeded")
