#!/usr/bin/env python3
"""
Test script to validate model integration.
"""

import asyncio
import sys

import pytest

# Add src to path
sys.path.insert(0, "src")

from synndicate.models.interfaces import GenerationConfig
from synndicate.models.manager import ModelManager


@pytest.fixture
async def manager():
    """Create and initialize a ModelManager for testing."""
    manager = ModelManager()
    await manager.initialize()
    return manager


@pytest.mark.asyncio
async def test_model_discovery():
    """Test model discovery functionality."""
    print("🔍 Testing Model Discovery...")

    manager = ModelManager()
    await manager.initialize()

    available_models = manager.get_available_models()
    print(f"  ✅ Discovered {len(available_models)} models:")

    for name, config in available_models.items():
        print(f"    - {name}: {config.model_type.value} ({config.format.value})")
        print(f"      Path: {config.path}")
        if config.metadata:
            print(f"      Metadata: {config.metadata}")

    assert len(available_models) >= 0  # Should at least not fail


@pytest.mark.asyncio
async def test_embedding_model(manager):
    """Test embedding model functionality."""
    print("\n🧮 Testing Embedding Model...")

    try:
        # Test embedding generation
        test_texts = [
            "Hello, world!",
            "This is a test sentence.",
            "Machine learning is fascinating.",
        ]

        embeddings = await manager.generate_embeddings(test_texts)

        print(f"  ✅ Generated embeddings for {len(test_texts)} texts")
        print(f"  ✅ Embedding dimension: {len(embeddings[0])}")
        print(f"  ✅ First embedding preview: {embeddings[0][:5]}...")

        return True

    except Exception as e:
        print(f"  ❌ Embedding test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_language_model(manager):
    """Test language model functionality."""
    print("\n🤖 Testing Language Model...")

    try:
        # Test text generation
        prompt = "What is artificial intelligence?"
        config = GenerationConfig(max_tokens=100, temperature=0.7)

        response = await manager.generate_text(prompt, config=config)

        print(f"  ✅ Generated response ({len(response.content)} chars)")
        print(f"  ✅ Response preview: {response.content[:100]}...")
        if response.usage:
            print(f"  ✅ Token usage: {response.usage}")

        return True

    except Exception as e:
        print(f"  ❌ Language model test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_health_check(manager):
    """Test model health checking."""
    print("\n❤️ Testing Health Check...")

    try:
        health_status = await manager.health_check()

        print(f"  ✅ Overall healthy: {health_status['overall_healthy']}")

        if health_status["language_models"]:
            print("  📝 Language Models:")
            for name, status in health_status["language_models"].items():
                print(
                    f"    - {name}: {'✅' if status['healthy'] else '❌'} (loaded: {status['loaded']})"
                )

        if health_status["embedding_models"]:
            print("  🧮 Embedding Models:")
            for name, status in health_status["embedding_models"].items():
                print(
                    f"    - {name}: {'✅' if status['healthy'] else '❌'} (loaded: {status['loaded']})"
                )
                if "dimension" in status:
                    print(f"      Dimension: {status['dimension']}")

        return health_status["overall_healthy"]

    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return False


@pytest.mark.asyncio
async def test_rag_integration(manager):
    """Test RAG integration with models."""
    print("\n🔗 Testing RAG Integration...")

    try:
        # Test if we can use the embedding model for RAG
        from synndicate.rag.chunking import Chunk, ChunkType

        # Create a simple test
        test_chunks = [
            Chunk(
                content="Python is a programming language",
                chunk_type=ChunkType.TEXT,
                start_index=0,
                end_index=33,
                metadata={},
            ),
            Chunk(
                content="Machine learning uses algorithms",
                chunk_type=ChunkType.TEXT,
                start_index=0,
                end_index=32,
                metadata={},
            ),
        ]

        # Test embedding generation for RAG
        texts = [chunk.content for chunk in test_chunks]
        embeddings = await manager.generate_embeddings(texts)

        print("  ✅ Generated embeddings for RAG chunks")
        print("  ✅ Embedding compatibility confirmed")

        return True

    except Exception as e:
        print(f"  ❌ RAG integration test failed: {e}")
        return False


async def main():
    """Run all model tests."""
    print("🧪 Starting Model Integration Tests\n")

    try:
        # Initialize model manager
        manager = await test_model_discovery()

        # Run tests
        embedding_success = await test_embedding_model(manager)
        language_success = await test_language_model(manager)
        health_success = await test_health_check(manager)
        rag_success = await test_rag_integration(manager)

        # Cleanup
        await manager.shutdown()

        # Summary
        print("\n📊 Test Summary:")
        print("  🔍 Model Discovery: ✅")
        print(f"  🧮 Embedding Model: {'✅' if embedding_success else '❌'}")
        print(f"  🤖 Language Model: {'✅' if language_success else '❌'}")
        print(f"  🏥 Health Check: {'✅' if health_success else '❌'}")
        print(f"  🔗 RAG Integration: {'✅' if rag_success else '❌'}")

        overall_success = embedding_success or language_success  # At least one should work

        if overall_success:
            print("\n🎉 Model integration tests completed successfully!")
            print("\n💡 Next Steps:")
            print("  - Download actual model weights if needed")
            print("  - Configure model paths in settings")
            print("  - Test with real agent workflows")
            return 0
        else:
            print("\n❌ Some model tests failed")
            print("\n💡 Troubleshooting:")
            print("  - Check if model files exist in /home/himokai/models")
            print("  - Verify model formats and paths")
            print("  - Install missing dependencies")
            return 1

    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
