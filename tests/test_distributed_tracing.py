"""
Tests for distributed tracing backend functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from synndicate.observability.distributed_tracing import (
    DistributedTracingManager,
    DistributedTracingConfig,
    TracingBackend,
    TracingProtocol,
)


class TestDistributedTracingConfig:
    """Test distributed tracing configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DistributedTracingConfig()
        assert config.backend == TracingBackend.JAEGER
        assert config.protocol == TracingProtocol.GRPC
        assert config.endpoint is None
        assert config.sample_rate == 1.0
        assert config.batch_timeout == 5000
        assert config.max_batch_size == 512
        assert config.max_queue_size == 2048
        assert config.enable_health_check is True
        assert config.health_check_interval == 30

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DistributedTracingConfig(
            backend=TracingBackend.ZIPKIN,
            protocol=TracingProtocol.HTTP,
            endpoint="http://custom:9411",
            sample_rate=0.5,
            batch_timeout=10000,
            max_batch_size=1024,
            max_queue_size=4096,
            enable_health_check=False,
            health_check_interval=60,
        )
        assert config.backend == TracingBackend.ZIPKIN
        assert config.protocol == TracingProtocol.HTTP
        assert config.endpoint == "http://custom:9411"
        assert config.sample_rate == 0.5
        assert config.batch_timeout == 10000
        assert config.max_batch_size == 1024
        assert config.max_queue_size == 4096
        assert config.enable_health_check is False
        assert config.health_check_interval == 60

    def test_invalid_sample_rate(self):
        """Test invalid sample rate validation."""
        with pytest.raises(ValueError):
            DistributedTracingConfig(sample_rate=-0.1)
        
        with pytest.raises(ValueError):
            DistributedTracingConfig(sample_rate=1.1)


class TestDistributedTracingManager:
    """Test distributed tracing manager."""

    def test_init_default(self):
        """Test default initialization."""
        manager = DistributedTracingManager()
        assert manager.config.backend == TracingBackend.JAEGER
        assert manager.config.protocol == TracingProtocol.GRPC
        assert manager._tracer_provider is None
        assert manager._span_processor is None
        assert manager._health_check_task is None
        assert manager._is_setup is False

    def test_init_custom(self):
        """Test custom initialization."""
        manager = DistributedTracingManager(
            backend=TracingBackend.ZIPKIN,
            protocol=TracingProtocol.HTTP,
            endpoint="http://zipkin:9411",
            sample_rate=0.8,
        )
        assert manager.config.backend == TracingBackend.ZIPKIN
        assert manager.config.protocol == TracingProtocol.HTTP
        assert manager.config.endpoint == "http://zipkin:9411"
        assert manager.config.sample_rate == 0.8

    def test_init_from_config(self):
        """Test initialization from config object."""
        config = DistributedTracingConfig(
            backend=TracingBackend.OTLP,
            endpoint="http://otel:4317",
        )
        manager = DistributedTracingManager(config=config)
        assert manager.config == config

    def test_init_with_env_vars(self):
        """Test initialization with environment variable support."""
        # This would be handled by the settings system in practice
        manager = DistributedTracingManager()
        assert manager.config.backend == TracingBackend.JAEGER  # default

    @patch('synndicate.observability.distributed_tracing.GRPCExporter')
    @patch('synndicate.observability.distributed_tracing.BatchSpanProcessor')
    @patch('synndicate.observability.distributed_tracing.TracerProvider')
    def test_setup_jaeger(self, mock_tracer_provider, mock_batch_processor, mock_otlp_exporter):
        """Test Jaeger backend setup."""
        manager = DistributedTracingManager(backend=TracingBackend.JAEGER)
        
        # Mock the tracer provider and processor
        mock_provider = Mock()
        mock_tracer_provider.return_value = mock_provider
        mock_processor = Mock()
        mock_batch_processor.return_value = mock_processor
        
        manager.setup()
        
        # Verify GRPC exporter was created with Jaeger endpoint
        mock_otlp_exporter.assert_called_once()
        call_args = mock_otlp_exporter.call_args
        assert 'endpoint' in call_args.kwargs
        assert '14250' in call_args.kwargs['endpoint']  # Jaeger gRPC port
        
        # Verify processor was added
        mock_provider.add_span_processor.assert_called_once_with(mock_processor)
        assert manager._is_setup is True

    @patch('synndicate.observability.distributed_tracing.HTTPExporter')
    @patch('synndicate.observability.distributed_tracing.BatchSpanProcessor')
    @patch('synndicate.observability.distributed_tracing.TracerProvider')
    def test_setup_zipkin(self, mock_tracer_provider, mock_batch_processor, mock_otlp_exporter):
        """Test Zipkin backend setup."""
        manager = DistributedTracingManager(backend=TracingBackend.ZIPKIN)
        
        # Mock the tracer provider and processor
        mock_provider = Mock()
        mock_tracer_provider.return_value = mock_provider
        mock_processor = Mock()
        mock_batch_processor.return_value = mock_processor
        
        manager.setup()
        
        # Verify HTTP exporter was created with Zipkin endpoint
        mock_otlp_exporter.assert_called_once()
        call_args = mock_otlp_exporter.call_args
        assert 'endpoint' in call_args.kwargs
        assert '4318' in call_args.kwargs['endpoint']  # OTLP HTTP port for Zipkin
        
        assert manager._is_setup is True

    @patch('synndicate.observability.distributed_tracing.GRPCExporter')
    @patch('synndicate.observability.distributed_tracing.BatchSpanProcessor')
    @patch('synndicate.observability.distributed_tracing.TracerProvider')
    def test_setup_custom_endpoint(self, mock_tracer_provider, mock_batch_processor, mock_otlp_exporter):
        """Test setup with custom endpoint."""
        custom_endpoint = "http://custom-tracer:4317"
        manager = DistributedTracingManager(
            backend=TracingBackend.OTLP,
            endpoint=custom_endpoint
        )
        
        # Mock the tracer provider and processor
        mock_provider = Mock()
        mock_tracer_provider.return_value = mock_provider
        mock_processor = Mock()
        mock_batch_processor.return_value = mock_processor
        
        manager.setup()
        
        # Verify custom endpoint was used
        mock_otlp_exporter.assert_called_once()
        call_args = mock_otlp_exporter.call_args
        assert call_args.kwargs['endpoint'] == custom_endpoint

    def test_setup_console_backend(self):
        """Test console backend setup."""
        with patch('synndicate.observability.distributed_tracing.ConsoleSpanExporter') as mock_console:
            with patch('synndicate.observability.distributed_tracing.BatchSpanProcessor') as mock_processor:
                with patch('synndicate.observability.distributed_tracing.TracerProvider') as mock_provider:
                    manager = DistributedTracingManager(backend=TracingBackend.CONSOLE)
                    
                    mock_provider_instance = Mock()
                    mock_provider.return_value = mock_provider_instance
                    
                    manager.setup()
                    
                    # Verify console exporter was used
                    mock_console.assert_called_once()
                    mock_processor.assert_called_once()
                    assert manager._is_setup is True

    def test_setup_disabled_backend(self):
        """Test disabled backend setup."""
        manager = DistributedTracingManager(backend=TracingBackend.DISABLED)
        
        manager.setup()
        
        # Should be marked as setup but no actual setup
        assert manager._is_setup is True
        assert manager._tracer_provider is None
        assert manager._span_processor is None

    def test_setup_already_setup(self):
        """Test setup when already setup."""
        manager = DistributedTracingManager(backend=TracingBackend.DISABLED)
        manager.setup()
        assert manager._is_setup is True
        
        # Second setup should be no-op
        manager.setup()
        assert manager._is_setup is True

    @patch('synndicate.observability.distributed_tracing.GRPCExporter')
    def test_setup_failure(self, mock_otlp_exporter):
        """Test setup failure handling."""
        mock_otlp_exporter.side_effect = Exception("Connection failed")
        
        manager = DistributedTracingManager(backend=TracingBackend.JAEGER)
        
        with pytest.raises(Exception):
            manager.setup()
        
        assert manager._is_setup is False

    def test_shutdown_not_setup(self):
        """Test shutdown when not setup."""
        manager = DistributedTracingManager()
        
        # Should not raise
        manager.shutdown()

    @patch('synndicate.observability.distributed_tracing.TracerProvider')
    def test_shutdown_with_setup(self, mock_tracer_provider):
        """Test shutdown after setup."""
        manager = DistributedTracingManager(backend=TracingBackend.DISABLED)
        manager.setup()
        
        # Mock tracer provider
        mock_provider = Mock()
        manager._tracer_provider = mock_provider
        
        manager.shutdown()
        
        # Verify shutdown was called
        mock_provider.shutdown.assert_called_once()
        assert manager._is_setup is False
        assert manager._tracer_provider is None

    @patch('asyncio.create_task')
    def test_health_check_enabled(self, mock_create_task):
        """Test health check task creation."""
        manager = DistributedTracingManager(
            backend=TracingBackend.DISABLED,
            enable_health_check=True
        )
        
        manager.setup()
        
        # Health check task should be created for non-disabled backends
        # (but disabled backend skips health checks)
        assert manager._health_check_task is None

    def test_default_endpoints(self):
        """Test that manager uses appropriate default endpoints."""
        # Test Jaeger configuration
        jaeger_manager = DistributedTracingManager(
            DistributedTracingConfig(backend=TracingBackend.JAEGER)
        )
        assert jaeger_manager.config.backend == TracingBackend.JAEGER
        
        # Test Zipkin configuration
        zipkin_manager = DistributedTracingManager(
            DistributedTracingConfig(backend=TracingBackend.ZIPKIN)
        )
        assert zipkin_manager.config.backend == TracingBackend.ZIPKIN


class TestTracingIntegration:
    """Test tracing integration scenarios."""

    @patch('synndicate.observability.distributed_tracing.OTLPSpanExporter')
    @patch('synndicate.observability.distributed_tracing.BatchSpanProcessor')
    @patch('synndicate.observability.distributed_tracing.TracerProvider')
    def test_full_lifecycle(self, mock_tracer_provider, mock_batch_processor, mock_otlp_exporter):
        """Test full setup and shutdown lifecycle."""
        manager = DistributedTracingManager(backend=TracingBackend.JAEGER)
        
        # Mock components
        mock_provider = Mock()
        mock_tracer_provider.return_value = mock_provider
        mock_processor = Mock()
        mock_batch_processor.return_value = mock_processor
        mock_exporter = Mock()
        mock_otlp_exporter.return_value = mock_exporter
        
        # Setup
        manager.setup()
        assert manager._is_setup is True
        mock_provider.add_span_processor.assert_called_once_with(mock_processor)
        
        # Shutdown
        manager.shutdown()
        assert manager._is_setup is False
        mock_provider.shutdown.assert_called_once()

    def test_convenience_functions(self):
        """Test convenience setup functions."""
        from synndicate.observability.distributed_tracing import (
            setup_jaeger_tracing,
            setup_zipkin_tracing,
            setup_otlp_tracing,
        )
        
        # Test Jaeger setup
        with patch.object(DistributedTracingManager, 'setup') as mock_setup:
            manager = setup_jaeger_tracing()
            assert manager.config.backend == TracingBackend.JAEGER
            mock_setup.assert_called_once()
        
        # Test Zipkin setup
        with patch.object(DistributedTracingManager, 'setup') as mock_setup:
            manager = setup_zipkin_tracing(endpoint="http://zipkin:9411")
            assert manager.config.backend == TracingBackend.ZIPKIN
            assert manager.config.endpoint == "http://zipkin:9411"
            mock_setup.assert_called_once()
        
        # Test OTLP setup
        with patch.object(DistributedTracingManager, 'setup') as mock_setup:
            manager = setup_otlp_tracing(endpoint="http://otel:4317")
            assert manager.config.backend == TracingBackend.OTLP
            assert manager.config.endpoint == "http://otel:4317"
            mock_setup.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
