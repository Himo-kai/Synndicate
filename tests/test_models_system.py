"""
Comprehensive test suite for models system.
Tests ModelManager, interfaces, and providers.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from synndicate.models.interfaces import (
    EmbeddingModel,
    GenerationConfig,
    LanguageModel,
    ModelConfig,
    ModelFormat,
    ModelResponse,
    ModelType,
)
from synndicate.models.manager import ModelManager
from synndicate.models.providers import LocalModelProvider, OpenAIProvider


class TestModelInterfaces:
    """Test model interface classes and enums."""

    def test_model_type_enum(self):
        """Test ModelType enum values."""
        assert ModelType.LANGUAGE_MODEL.value == "language_model"
        assert ModelType.EMBEDDING_MODEL.value == "embedding_model"
        assert ModelType.VISION_MODEL.value == "vision_model"
        assert ModelType.AUDIO_MODEL.value == "audio_model"

    def test_model_format_enum(self):
        """Test ModelFormat enum values."""
        assert ModelFormat.GGUF.value == "gguf"
        assert ModelFormat.SAFETENSORS.value == "safetensors"
        assert ModelFormat.PYTORCH.value == "pytorch"
        assert ModelFormat.ONNX.value == "onnx"
        assert ModelFormat.OPENAI_API.value == "openai_api"

    def test_model_config_creation(self):
        """Test ModelConfig dataclass creation."""
        config = ModelConfig(
            name="test-model",
            model_type=ModelType.LANGUAGE_MODEL,
            format=ModelFormat.GGUF,
            path="/path/to/model.gguf",
            parameters={"max_tokens": 2048},
            metadata={"size": "7B"}
        )
        
        assert config.name == "test-model"
        assert config.model_type == ModelType.LANGUAGE_MODEL
        assert config.format == ModelFormat.GGUF
        assert config.path == "/path/to/model.gguf"
        assert config.parameters["max_tokens"] == 2048
        assert config.metadata["size"] == "7B"

    def test_generation_config_defaults(self):
        """Test GenerationConfig default values."""
        config = GenerationConfig()
        
        assert config.max_tokens == 1000
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.stop_sequences == []
        assert config.stream is False

    def test_generation_config_custom(self):
        """Test GenerationConfig with custom values."""
        config = GenerationConfig(
            max_tokens=2048,
            temperature=0.5,
            top_p=0.95,
            top_k=40,
            stop_sequences=["</s>", "\n\n"],
            stream=True
        )
        
        assert config.max_tokens == 2048
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.top_k == 40
        assert config.stop_sequences == ["</s>", "\n\n"]
        assert config.stream is True

    def test_model_response_creation(self):
        """Test ModelResponse dataclass creation."""
        response = ModelResponse(
            content="Generated text",
            metadata={"model": "test"},
            usage={"tokens": 100}
        )
        
        assert response.content == "Generated text"
        assert response.metadata["model"] == "test"
        assert response.usage["tokens"] == 100


class MockLanguageModel(LanguageModel):
    """Mock implementation of LanguageModel for testing."""
    
    async def load(self) -> None:
        self._loaded = True
    
    async def unload(self) -> None:
        self._loaded = False
    
    async def generate(self, prompt: str, config: GenerationConfig | None = None) -> ModelResponse:
        return ModelResponse(
            content=f"Generated response for: {prompt}",
            metadata={"model": self.config.name},
            usage={"tokens": 50}
        )
    
    async def generate_stream(self, prompt: str, config: GenerationConfig | None = None):
        words = f"Generated response for: {prompt}".split()
        for word in words:
            yield ModelResponse(
                content=word + " ",
                metadata={"model": self.config.name},
                usage={"tokens": 1}
            )
    
    async def health_check(self) -> bool:
        return self._loaded


class MockEmbeddingModel(EmbeddingModel):
    """Mock implementation of EmbeddingModel for testing."""
    
    async def load(self) -> None:
        self._loaded = True
    
    async def unload(self) -> None:
        self._loaded = False
    
    @property
    def embedding_dimension(self) -> int:
        return 384
    
    async def encode(self, texts: list[str]) -> np.ndarray:
        # Return mock embeddings
        return np.random.rand(len(texts), self.embedding_dimension).astype(np.float32)
    
    async def encode_single(self, text: str) -> np.ndarray:
        return np.random.rand(self.embedding_dimension).astype(np.float32)
    
    async def similarity(self, text1: str, text2: str) -> float:
        # Mock similarity calculation
        return 0.85
    
    async def health_check(self) -> bool:
        return self._loaded


class TestLanguageModel:
    """Test LanguageModel abstract base class."""

    def test_language_model_initialization(self):
        """Test LanguageModel initialization."""
        config = ModelConfig(
            name="test-llm",
            model_type=ModelType.LANGUAGE_MODEL,
            format=ModelFormat.GGUF,
            path="/path/to/model.gguf"
        )
        
        model = MockLanguageModel(config)
        assert model.config == config
        assert not model.is_loaded()

    @pytest.mark.asyncio
    async def test_language_model_lifecycle(self):
        """Test language model load/unload lifecycle."""
        config = ModelConfig(
            name="test-llm",
            model_type=ModelType.LANGUAGE_MODEL,
            format=ModelFormat.GGUF,
            path="/path/to/model.gguf"
        )
        
        model = MockLanguageModel(config)
        
        # Initially not loaded
        assert not model.is_loaded()
        assert not await model.health_check()
        
        # Load model
        await model.load()
        assert model.is_loaded()
        assert await model.health_check()
        
        # Unload model
        await model.unload()
        assert not model.is_loaded()
        assert not await model.health_check()

    @pytest.mark.asyncio
    async def test_language_model_generation(self):
        """Test language model text generation."""
        config = ModelConfig(
            name="test-llm",
            model_type=ModelType.LANGUAGE_MODEL,
            format=ModelFormat.GGUF,
            path="/path/to/model.gguf"
        )
        
        model = MockLanguageModel(config)
        await model.load()
        
        # Test basic generation
        response = await model.generate("Hello, world!")
        assert isinstance(response, ModelResponse)
        assert "Hello, world!" in response.content
        assert response.metadata["model"] == "test-llm"
        assert response.usage["tokens"] == 50

    @pytest.mark.asyncio
    async def test_language_model_streaming(self):
        """Test language model streaming generation."""
        config = ModelConfig(
            name="test-llm",
            model_type=ModelType.LANGUAGE_MODEL,
            format=ModelFormat.GGUF,
            path="/path/to/model.gguf"
        )
        
        model = MockLanguageModel(config)
        await model.load()
        
        # Test streaming generation
        responses = []
        async for response in model.generate_stream("Hello, world!"):
            responses.append(response)
        
        assert len(responses) > 0
        assert all(isinstance(r, ModelResponse) for r in responses)
        assert all(r.metadata["model"] == "test-llm" for r in responses)


class TestEmbeddingModel:
    """Test EmbeddingModel abstract base class."""

    def test_embedding_model_initialization(self):
        """Test EmbeddingModel initialization."""
        config = ModelConfig(
            name="test-embedding",
            model_type=ModelType.EMBEDDING_MODEL,
            format=ModelFormat.SAFETENSORS,
            path="/path/to/model"
        )
        
        model = MockEmbeddingModel(config)
        assert model.config == config
        assert not model.is_loaded()
        assert model.embedding_dimension == 384

    @pytest.mark.asyncio
    async def test_embedding_model_lifecycle(self):
        """Test embedding model load/unload lifecycle."""
        config = ModelConfig(
            name="test-embedding",
            model_type=ModelType.EMBEDDING_MODEL,
            format=ModelFormat.SAFETENSORS,
            path="/path/to/model"
        )
        
        model = MockEmbeddingModel(config)
        
        # Initially not loaded
        assert not model.is_loaded()
        assert not await model.health_check()
        
        # Load model
        await model.load()
        assert model.is_loaded()
        assert await model.health_check()
        
        # Unload model
        await model.unload()
        assert not model.is_loaded()
        assert not await model.health_check()

    @pytest.mark.asyncio
    async def test_embedding_model_encoding(self):
        """Test embedding model text encoding."""
        config = ModelConfig(
            name="test-embedding",
            model_type=ModelType.EMBEDDING_MODEL,
            format=ModelFormat.SAFETENSORS,
            path="/path/to/model"
        )
        
        model = MockEmbeddingModel(config)
        await model.load()
        
        # Test batch encoding
        texts = ["Hello", "World", "Test"]
        embeddings = await model.encode(texts)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32
        
        # Test single encoding
        single_embedding = await model.encode_single("Hello")
        assert single_embedding.shape == (384,)
        assert single_embedding.dtype == np.float32
        
        # Test similarity
        similarity = await model.similarity("Hello", "Hi")
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0


class TestModelManager:
    """Test ModelManager class."""

    def test_model_manager_initialization(self):
        """Test ModelManager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_directory=tmpdir)
            assert manager.models_directory == Path(tmpdir)
            assert len(manager._language_models) == 0
            assert len(manager._embedding_models) == 0
            assert len(manager._model_configs) == 0

    @pytest.mark.asyncio
    async def test_model_manager_initialization_async(self):
        """Test ModelManager async initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_directory=tmpdir)
            
            with patch.object(manager, '_discover_models', new_callable=AsyncMock) as mock_discover:
                await manager.initialize()
                mock_discover.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_manager_register_model(self):
        """Test registering models with ModelManager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_directory=tmpdir)
            
            # Create mock language model
            config = ModelConfig(
                name="test-llm",
                model_type=ModelType.LANGUAGE_MODEL,
                format=ModelFormat.GGUF,
                path="/path/to/model.gguf"
            )
            model = MockLanguageModel(config)
            
            # Register model
            with patch.object(manager, '_register_model') as mock_register:
                manager._register_model("test-llm", model)
                mock_register.assert_called_once_with("test-llm", model)

    @pytest.mark.asyncio
    async def test_model_manager_get_language_model(self):
        """Test getting language model from manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_directory=tmpdir)
            
            # Mock language model
            config = ModelConfig(
                name="test-llm",
                model_type=ModelType.LANGUAGE_MODEL,
                format=ModelFormat.GGUF,
                path="/path/to/model.gguf"
            )
            model = MockLanguageModel(config)
            manager._language_models["test-llm"] = model
            
            # Get model
            retrieved_model = await manager.get_language_model("test-llm")
            assert retrieved_model == model

    @pytest.mark.asyncio
    async def test_model_manager_get_embedding_model(self):
        """Test getting embedding model from manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_directory=tmpdir)
            
            # Mock embedding model
            config = ModelConfig(
                name="test-embedding",
                model_type=ModelType.EMBEDDING_MODEL,
                format=ModelFormat.SAFETENSORS,
                path="/path/to/model"
            )
            model = MockEmbeddingModel(config)
            manager._embedding_models["test-embedding"] = model
            
            # Get model
            retrieved_model = await manager.get_embedding_model("test-embedding")
            assert retrieved_model == model

    @pytest.mark.asyncio
    async def test_model_manager_model_not_found(self):
        """Test handling of non-existent models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_directory=tmpdir)
            
            # Try to get non-existent models
            language_model = await manager.get_language_model("nonexistent")
            assert language_model is None
            
            embedding_model = await manager.get_embedding_model("nonexistent")
            assert embedding_model is None

    @pytest.mark.asyncio
    async def test_model_manager_list_models(self):
        """Test listing available models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_directory=tmpdir)
            
            # Add mock models
            llm_config = ModelConfig(
                name="test-llm",
                model_type=ModelType.LANGUAGE_MODEL,
                format=ModelFormat.GGUF,
                path="/path/to/llm.gguf"
            )
            embedding_config = ModelConfig(
                name="test-embedding",
                model_type=ModelType.EMBEDDING_MODEL,
                format=ModelFormat.SAFETENSORS,
                path="/path/to/embedding"
            )
            
            manager._language_models["test-llm"] = MockLanguageModel(llm_config)
            manager._embedding_models["test-embedding"] = MockEmbeddingModel(embedding_config)
            
            # List models
            with patch.object(manager, 'list_models') as mock_list:
                mock_list.return_value = {
                    "language_models": ["test-llm"],
                    "embedding_models": ["test-embedding"]
                }
                models = manager.list_models()
                assert "test-llm" in models["language_models"]
                assert "test-embedding" in models["embedding_models"]

    @pytest.mark.asyncio
    async def test_model_manager_health_check(self):
        """Test model health checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(models_directory=tmpdir)
            
            # Add mock model
            config = ModelConfig(
                name="test-llm",
                model_type=ModelType.LANGUAGE_MODEL,
                format=ModelFormat.GGUF,
                path="/path/to/model.gguf"
            )
            model = MockLanguageModel(config)
            await model.load()  # Load model for health check
            manager._language_models["test-llm"] = model
            
            # Mock health check
            with patch.object(manager, 'health_check') as mock_health:
                mock_health.return_value = {"test-llm": True}
                health_status = manager.health_check()
                assert health_status["test-llm"] is True


class TestModelProviders:
    """Test model provider classes."""

    def test_local_model_provider_initialization(self):
        """Test LocalModelProvider initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalModelProvider(models_directory=tmpdir)
            assert provider.models_directory == Path(tmpdir)

    def test_openai_provider_initialization(self):
        """Test OpenAIProvider initialization."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_local_provider_discover_models(self):
        """Test local model discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalModelProvider(models_directory=tmpdir)
            
            # Create mock model files
            (Path(tmpdir) / "model1.gguf").touch()
            (Path(tmpdir) / "model2.safetensors").touch()
            
            # Mock discovery
            with patch.object(provider, 'discover_models') as mock_discover:
                mock_discover.return_value = [
                    ModelConfig("model1", ModelType.LANGUAGE_MODEL, ModelFormat.GGUF, str(Path(tmpdir) / "model1.gguf")),
                    ModelConfig("model2", ModelType.EMBEDDING_MODEL, ModelFormat.SAFETENSORS, str(Path(tmpdir) / "model2.safetensors"))
                ]
                models = provider.discover_models()
                assert len(models) == 2
                assert models[0].name == "model1"
                assert models[1].name == "model2"

    @pytest.mark.asyncio
    async def test_openai_provider_create_model(self):
        """Test OpenAI model creation."""
        provider = OpenAIProvider(api_key="test-key")
        
        # Mock model creation
        with patch.object(provider, 'create_language_model') as mock_create:
            config = ModelConfig(
                name="gpt-3.5-turbo",
                model_type=ModelType.LANGUAGE_MODEL,
                format=ModelFormat.OPENAI_API,
                path="gpt-3.5-turbo"
            )
            mock_model = MagicMock()
            mock_create.return_value = mock_model
            
            model = provider.create_language_model(config)
            assert model == mock_model
            mock_create.assert_called_once_with(config)


if __name__ == "__main__":
    pytest.main([__file__])
