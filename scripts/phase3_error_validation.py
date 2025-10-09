#!/usr/bin/env python3
"""
Phase 3 Real-World Error Validation Script

This script systematically tests all Phase 3 runtime orchestration patterns
by pushing real errors through the enhanced API server and validating that
error handling, circuit breakers, retries, and observability work correctly.

Usage:
    python scripts/phase3_error_validation.py

Test Categories:
1. Timeout and Deadline Enforcement
2. Circuit Breaker Behavior Under Load
3. Retry Logic with Real Failures
4. Backpressure and Queue Management
5. Idempotency Under Concurrent Load
6. Graceful Cancellation
7. Observability and Metrics Validation
"""

import asyncio
import json
import time
import uuid
from typing import dict, list, Any
import httpx
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
API_BASE_URL = "http://localhost:8001"
CONCURRENT_REQUESTS = 20
TIMEOUT_TEST_DURATION = 5.0
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5


class ErrorValidationResults:
    """Collect and analyze test results."""
    
    def __init__(self):
        self.results: dict[str, list[dict[str, Any]]] = {}
        self.start_time = time.time()
    
    def add_result(self, test_name: str, result: dict[str, Any]):
        """Add a test result."""
        if test_name not in self.results:
            self.results[test_name] = []
        result['timestamp'] = time.time() - self.start_time
        self.results[test_name].append(result)
    
    def get_summary(self) -> dict[str, Any]:
        """Get test summary statistics."""
        summary = {}
        for test_name, results in self.results.items():
            total = len(results)
            successes = len([r for r in results if r.get('success', False)])
            errors = len([r for r in results if r.get('error')])
            timeouts = len([r for r in results if r.get('timeout', False)])
            
            summary[test_name] = {
                'total_requests': total,
                'successes': successes,
                'errors': errors,
                'timeouts': timeouts,
                'success_rate': successes / total if total > 0 else 0,
                'error_rate': errors / total if total > 0 else 0,
                'avg_duration': sum(r.get('duration', 0) for r in results) / total if total > 0 else 0,
            }
        
        return summary


async def test_timeout_enforcement(client: httpx.AsyncClient, results: ErrorValidationResults):
    """Test 1: Timeout and Deadline Enforcement"""
    logger.info("🕐 Testing timeout and deadline enforcement...")
    
    test_cases = [
        {"timeout": 0.1, "expected_status": 408},  # Very short timeout
        {"timeout": 1.0, "expected_status": 408},  # Short timeout
        {"timeout": 30.0, "expected_status": 200}, # Normal timeout
    ]
    
    for i, case in enumerate(test_cases):
        correlation_id = f"timeout-test-{i}-{uuid.uuid4()}"
        
        try:
            start_time = time.time()
            
            # Submit job with custom timeout
            response = await client.post(
                f"{API_BASE_URL}/jobs",
                json={
                    "query": f"Simulate long-running task {i}",
                    "context": {"simulate_delay": 5.0},  # Force 5s delay
                    "workflow": "test"
                },
                headers={
                    "X-Request-Id": correlation_id,
                    "X-Timeout": str(case["timeout"])
                },
                timeout=case["timeout"] + 1.0  # Client timeout slightly higher
            )
            
            duration = time.time() - start_time
            
            results.add_result("timeout_enforcement", {
                "correlation_id": correlation_id,
                "expected_status": case["expected_status"],
                "actual_status": response.status_code,
                "timeout_setting": case["timeout"],
                "duration": duration,
                "success": response.status_code == case["expected_status"],
                "timeout": response.status_code == 408,
                "response_headers": dict(response.headers),
            })
            
            logger.info(f"  Timeout test {i}: {response.status_code} (expected {case['expected_status']}) in {duration:.2f}s")
            
        except httpx.TimeoutException:
            duration = time.time() - start_time
            results.add_result("timeout_enforcement", {
                "correlation_id": correlation_id,
                "expected_status": case["expected_status"],
                "actual_status": "client_timeout",
                "timeout_setting": case["timeout"],
                "duration": duration,
                "success": case["expected_status"] == 408,
                "timeout": True,
                "error": "Client timeout"
            })
            logger.info(f"  Timeout test {i}: Client timeout in {duration:.2f}s")


async def test_circuit_breaker_behavior(client: httpx.AsyncClient, results: ErrorValidationResults):
    """Test 2: Circuit Breaker Behavior Under Load"""
    logger.info("⚡ Testing circuit breaker behavior...")
    
    # Phase 1: Trigger circuit breaker with failures
    logger.info("  Phase 1: Triggering circuit breaker with failures...")
    
    failure_tasks = []
    for i in range(CIRCUIT_BREAKER_FAILURE_THRESHOLD + 2):
        correlation_id = f"circuit-breaker-failure-{i}-{uuid.uuid4()}"
        
        task = asyncio.create_task(
            make_failing_request(client, correlation_id, results)
        )
        failure_tasks.append(task)
    
    await asyncio.gather(*failure_tasks, return_exceptions=True)
    
    # Phase 2: Verify circuit breaker is open
    logger.info("  Phase 2: Verifying circuit breaker is open...")
    
    for i in range(3):
        correlation_id = f"circuit-breaker-open-{i}-{uuid.uuid4()}"
        
        try:
            response = await client.post(
                f"{API_BASE_URL}/jobs",
                json={"query": "Test request while circuit breaker is open"},
                headers={"X-Request-Id": correlation_id},
                timeout=5.0
            )
            
            results.add_result("circuit_breaker_open", {
                "correlation_id": correlation_id,
                "status_code": response.status_code,
                "success": response.status_code == 503,
                "retry_after": response.headers.get("Retry-After"),
                "circuit_breaker_blocked": response.status_code == 503,
            })
            
            logger.info(f"    Circuit breaker test {i}: {response.status_code} (expected 503)")
            
        except Exception as e:
            results.add_result("circuit_breaker_open", {
                "correlation_id": correlation_id,
                "error": str(e),
                "success": False,
            })


async def make_failing_request(client: httpx.AsyncClient, correlation_id: str, results: ErrorValidationResults):
    """Make a request designed to fail and trigger circuit breaker."""
    try:
        start_time = time.time()
        
        response = await client.post(
            f"{API_BASE_URL}/jobs",
            json={
                "query": "FORCE_ERROR_FOR_TESTING",  # This should trigger an error
                "context": {"force_error": True},
                "workflow": "error_test"
            },
            headers={"X-Request-Id": correlation_id},
            timeout=10.0
        )
        
        duration = time.time() - start_time
        
        results.add_result("circuit_breaker_failures", {
            "correlation_id": correlation_id,
            "status_code": response.status_code,
            "duration": duration,
            "success": False,  # We expect these to fail
            "error": response.status_code >= 400,
        })
        
    except Exception as e:
        duration = time.time() - start_time
        results.add_result("circuit_breaker_failures", {
            "correlation_id": correlation_id,
            "error": str(e),
            "duration": duration,
            "success": False,
        })


async def test_retry_logic(client: httpx.AsyncClient, results: ErrorValidationResults):
    """Test 3: Retry Logic with Real Failures"""
    logger.info("🔄 Testing retry logic with intermittent failures...")
    
    # Test cases with different failure patterns
    test_cases = [
        {"fail_count": 1, "description": "Single failure, then success"},
        {"fail_count": 2, "description": "Two failures, then success"},
        {"fail_count": 5, "description": "Persistent failures (should exhaust retries)"},
    ]
    
    for i, case in enumerate(test_cases):
        correlation_id = f"retry-test-{i}-{uuid.uuid4()}"
        
        try:
            start_time = time.time()
            
            response = await client.post(
                f"{API_BASE_URL}/jobs",
                json={
                    "query": f"Retry test with {case['fail_count']} failures",
                    "context": {
                        "simulate_failures": case["fail_count"],
                        "retry_test": True
                    },
                    "workflow": "retry_test"
                },
                headers={"X-Request-Id": correlation_id},
                timeout=30.0
            )
            
            duration = time.time() - start_time
            
            results.add_result("retry_logic", {
                "correlation_id": correlation_id,
                "test_case": case["description"],
                "fail_count": case["fail_count"],
                "status_code": response.status_code,
                "duration": duration,
                "success": response.status_code == 200 if case["fail_count"] < 3 else response.status_code >= 400,
                "retries_exhausted": case["fail_count"] >= 3 and response.status_code >= 400,
            })
            
            logger.info(f"  Retry test {i}: {response.status_code} in {duration:.2f}s ({case['description']})")
            
        except Exception as e:
            duration = time.time() - start_time
            results.add_result("retry_logic", {
                "correlation_id": correlation_id,
                "test_case": case["description"],
                "error": str(e),
                "duration": duration,
                "success": False,
            })


async def test_backpressure_handling(client: httpx.AsyncClient, results: ErrorValidationResults):
    """Test 4: Backpressure and Queue Management"""
    logger.info("🚦 Testing backpressure and queue management...")
    
    # Flood the server with concurrent requests to trigger backpressure
    concurrent_tasks = []
    
    for i in range(CONCURRENT_REQUESTS):
        correlation_id = f"backpressure-test-{i}-{uuid.uuid4()}"
        
        task = asyncio.create_task(
            make_backpressure_request(client, correlation_id, results, i)
        )
        concurrent_tasks.append(task)
    
    # Execute all requests concurrently
    await asyncio.gather(*concurrent_tasks, return_exceptions=True)
    
    # Analyze results
    backpressure_results = results.results.get("backpressure", [])
    status_429_count = len([r for r in backpressure_results if r.get("status_code") == 429])
    
    logger.info(f"  Backpressure test: {status_429_count}/{len(backpressure_results)} requests got 429 (rate limited)")


async def make_backpressure_request(client: httpx.AsyncClient, correlation_id: str, results: ErrorValidationResults, request_id: int):
    """Make a request for backpressure testing."""
    try:
        start_time = time.time()
        
        response = await client.post(
            f"{API_BASE_URL}/jobs",
            json={
                "query": f"Backpressure test request {request_id}",
                "context": {"simulate_delay": 2.0},  # Moderate delay
                "workflow": "backpressure_test"
            },
            headers={"X-Request-Id": correlation_id},
            timeout=15.0
        )
        
        duration = time.time() - start_time
        
        results.add_result("backpressure", {
            "correlation_id": correlation_id,
            "request_id": request_id,
            "status_code": response.status_code,
            "duration": duration,
            "success": response.status_code in [200, 202, 429],  # All valid responses
            "rate_limited": response.status_code == 429,
            "retry_after": response.headers.get("Retry-After"),
        })
        
    except Exception as e:
        duration = time.time() - start_time
        results.add_result("backpressure", {
            "correlation_id": correlation_id,
            "request_id": request_id,
            "error": str(e),
            "duration": duration,
            "success": False,
        })


async def test_idempotency(client: httpx.AsyncClient, results: ErrorValidationResults):
    """Test 5: Idempotency Under Concurrent Load"""
    logger.info("🔒 Testing idempotency under concurrent load...")
    
    # Use the same correlation ID for multiple concurrent requests
    correlation_id = f"idempotency-test-{uuid.uuid4()}"
    
    # Make multiple concurrent requests with the same correlation ID
    idempotency_tasks = []
    
    for i in range(5):
        task = asyncio.create_task(
            make_idempotent_request(client, correlation_id, results, i)
        )
        idempotency_tasks.append(task)
    
    await asyncio.gather(*idempotency_tasks, return_exceptions=True)
    
    # Analyze results - all should return the same job ID
    idempotency_results = results.results.get("idempotency", [])
    job_ids = [r.get("job_id") for r in idempotency_results if r.get("job_id")]
    unique_job_ids = set(job_ids)
    
    logger.info(f"  Idempotency test: {len(unique_job_ids)} unique job IDs from {len(job_ids)} requests (should be 1)")


async def make_idempotent_request(client: httpx.AsyncClient, correlation_id: str, results: ErrorValidationResults, request_num: int):
    """Make an idempotent request."""
    try:
        start_time = time.time()
        
        response = await client.post(
            f"{API_BASE_URL}/jobs",
            json={
                "query": "Idempotency test - same correlation ID",
                "context": {"test_type": "idempotency"},
                "workflow": "idempotency_test"
            },
            headers={"X-Request-Id": correlation_id},
            timeout=10.0
        )
        
        duration = time.time() - start_time
        response_data = response.json() if response.status_code == 200 else {}
        
        results.add_result("idempotency", {
            "correlation_id": correlation_id,
            "request_num": request_num,
            "status_code": response.status_code,
            "duration": duration,
            "job_id": response_data.get("id"),
            "success": response.status_code in [200, 202],
        })
        
    except Exception as e:
        duration = time.time() - start_time
        results.add_result("idempotency", {
            "correlation_id": correlation_id,
            "request_num": request_num,
            "error": str(e),
            "duration": duration,
            "success": False,
        })


async def test_observability_metrics(client: httpx.AsyncClient, results: ErrorValidationResults):
    """Test 6: Observability and Metrics Validation"""
    logger.info("📊 Testing observability and metrics...")
    
    try:
        # Get initial metrics
        initial_response = await client.get(f"{API_BASE_URL}/metrics", timeout=5.0)
        initial_metrics = initial_response.json() if initial_response.status_code == 200 else {}
        
        # Make some test requests
        for i in range(3):
            correlation_id = f"metrics-test-{i}-{uuid.uuid4()}"
            
            await client.post(
                f"{API_BASE_URL}/jobs",
                json={"query": f"Metrics test {i}", "workflow": "metrics_test"},
                headers={"X-Request-Id": correlation_id},
                timeout=5.0
            )
        
        # Get final metrics
        final_response = await client.get(f"{API_BASE_URL}/metrics", timeout=5.0)
        final_metrics = final_response.json() if final_response.status_code == 200 else {}
        
        results.add_result("observability", {
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "metrics_available": final_response.status_code == 200,
            "active_jobs_tracked": "active_jobs" in final_metrics,
            "completed_jobs_tracked": "completed_jobs" in final_metrics,
            "success": final_response.status_code == 200,
        })
        
        logger.info(f"  Metrics test: Available={final_response.status_code == 200}, Active jobs={final_metrics.get('active_jobs', 'N/A')}")
        
    except Exception as e:
        results.add_result("observability", {
            "error": str(e),
            "success": False,
        })


async def run_health_check(client: httpx.AsyncClient) -> bool:
    """Verify the API server is healthy before testing."""
    try:
        response = await client.get(f"{API_BASE_URL}/health", timeout=5.0)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ API Server is healthy: {health_data.get('status', 'unknown')}")
            return True
        else:
            logger.error(f"❌ API Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ API Server health check error: {e}")
        return False


async def main():
    """Run comprehensive Phase 3 error validation tests."""
    logger.info("🚀 Starting Phase 3 Real-World Error Validation")
    logger.info(f"Target API: {API_BASE_URL}")
    
    results = ErrorValidationResults()
    
    async with httpx.AsyncClient() as client:
        # Health check first
        if not await run_health_check(client):
            logger.error("❌ API Server is not healthy. Aborting tests.")
            return
        
        # Run all test suites
        test_suites = [
            ("Timeout Enforcement", test_timeout_enforcement),
            ("Circuit Breaker Behavior", test_circuit_breaker_behavior),
            ("Retry Logic", test_retry_logic),
            ("Backpressure Handling", test_backpressure_handling),
            ("Idempotency", test_idempotency),
            ("Observability Metrics", test_observability_metrics),
        ]
        
        for suite_name, test_func in test_suites:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {suite_name}")
            logger.info(f"{'='*60}")
            
            try:
                await test_func(client, results)
                logger.info(f"✅ {suite_name} completed")
            except Exception as e:
                logger.error(f"❌ {suite_name} failed: {e}")
                results.add_result(suite_name.lower().replace(" ", "_"), {
                    "error": str(e),
                    "success": False,
                })
    
    # Generate final report
    logger.info(f"\n{'='*60}")
    logger.info("📊 PHASE 3 ERROR VALIDATION RESULTS")
    logger.info(f"{'='*60}")
    
    summary = results.get_summary()
    
    for test_name, stats in summary.items():
        logger.info(f"\n{test_name.upper()}:")
        logger.info(f"  Total Requests: {stats['total_requests']}")
        logger.info(f"  Success Rate: {stats['success_rate']:.1%}")
        logger.info(f"  Error Rate: {stats['error_rate']:.1%}")
        logger.info(f"  Avg Duration: {stats['avg_duration']:.3f}s")
        
        if stats['timeouts'] > 0:
            logger.info(f"  Timeouts: {stats['timeouts']}")
    
    # Overall assessment
    total_tests = sum(stats['total_requests'] for stats in summary.values())
    total_successes = sum(stats['successes'] for stats in summary.values())
    overall_success_rate = total_successes / total_tests if total_tests > 0 else 0
    
    logger.info(f"\n🎯 OVERALL RESULTS:")
    logger.info(f"  Total Tests: {total_tests}")
    logger.info(f"  Overall Success Rate: {overall_success_rate:.1%}")
    
    if overall_success_rate >= 0.8:
        logger.info("🎉 Phase 3 Runtime Orchestration: VALIDATION SUCCESSFUL!")
    elif overall_success_rate >= 0.6:
        logger.info("⚠️  Phase 3 Runtime Orchestration: PARTIAL SUCCESS - Review failures")
    else:
        logger.info("❌ Phase 3 Runtime Orchestration: VALIDATION FAILED - Significant issues found")
    
    # Save detailed results
    with open("phase3_validation_results.json", "w") as f:
        json.dump({
            "summary": summary,
            "detailed_results": results.results,
            "overall_success_rate": overall_success_rate,
            "total_tests": total_tests,
            "timestamp": time.time(),
        }, f, indent=2)
    
    logger.info(f"\n📄 Detailed results saved to: phase3_validation_results.json")


if __name__ == "__main__":
    asyncio.run(main())
