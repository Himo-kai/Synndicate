"""
Locust performance testing for Synndicate AI API endpoints.
"""

from locust import HttpUser, task, between
import json
import random


class SyndicateAPIUser(HttpUser):
    """Simulates users interacting with Synndicate AI API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts - check if API is healthy."""
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"API health check failed: {response.status_code}")
    
    @task(3)
    def health_check(self):
        """Test health endpoint - most frequent task."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {data.get('status')}")
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def simple_query(self):
        """Test query endpoint with simple requests."""
        queries = [
            "Hello, how are you?",
            "What is Python?",
            "Explain machine learning",
            "Write a simple function",
            "What is the weather like?"
        ]
        
        query = random.choice(queries)
        payload = {"query": query}
        
        with self.client.post(
            "/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    response.failure(f"Query failed: {data.get('error', 'Unknown error')}")
            else:
                response.failure(f"Query request failed: {response.status_code}")
    
    @task(1)
    def complex_query(self):
        """Test query endpoint with more complex requests."""
        complex_queries = [
            "Create a Python class for managing a todo list with methods to add, remove, and list items",
            "Explain the differences between supervised and unsupervised machine learning with examples",
            "Write a function that implements binary search and explain its time complexity",
            "Design a REST API for a blog system with proper HTTP methods and status codes"
        ]
        
        query = random.choice(complex_queries)
        payload = {"query": query}
        
        with self.client.post(
            "/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
            timeout=30  # Longer timeout for complex queries
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    response.failure(f"Complex query failed: {data.get('error', 'Unknown error')}")
            else:
                response.failure(f"Complex query request failed: {response.status_code}")


class HealthCheckUser(HttpUser):
    """Dedicated user for health check monitoring."""
    
    wait_time = between(0.5, 1)  # Faster health checks
    weight = 1  # Lower weight compared to main API users
    
    @task
    def continuous_health_check(self):
        """Continuous health monitoring."""
        self.client.get("/health")


class APIDocumentationUser(HttpUser):
    """User that accesses API documentation."""
    
    wait_time = between(5, 10)  # Less frequent access
    weight = 1
    
    @task(2)
    def get_docs(self):
        """Access Swagger UI documentation."""
        self.client.get("/docs")
    
    @task(1)
    def get_redoc(self):
        """Access ReDoc documentation."""
        self.client.get("/redoc")
    
    @task(1)
    def get_openapi_spec(self):
        """Access OpenAPI specification."""
        self.client.get("/openapi.json")
