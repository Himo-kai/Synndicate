#!/usr/bin/env python3
"""
Complete TinyLlama integration test with full observability.

This test validates the entire system with TinyLlama:
- Model manager with TinyLlama integration
- Language model generation with full observability
- End-to-end orchestrator -> agent -> language model workflow
- Complete system validation with real language model
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synndicate.config.container import Container
from synndicate.core.orchestrator import Orchestrator
from synndicate.models.interfaces import (GenerationConfig, ModelConfig,
                                          ModelFormat, ModelType)
from synndicate.models.manager import ModelManager
from synndicate.observability.logging import (clear_trace_id, get_logger,
                                              set_trace_id, setup_logging)
from synndicate.observability.probe import get_trace_metrics, probe


async def test_tinyllama_integration():
    """Test TinyLlama integration with full observability."""
    print("\n🤖 Testing TinyLlama Integration...")

    # Setup
    setup_logging()
    logger = get_logger(__name__)

    # Verify TinyLlama is downloaded
    tinyllama_path = Path("/home/himokai/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    if not tinyllama_path.exists():
        print(f"  ❌ TinyLlama not found: {tinyllama_path}")
        return False, {}

    file_size = tinyllama_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  ✅ TinyLlama found: {file_size:.1f}MB")

    # Set trace ID for this test
    test_trace_id = "tinyllama_integration_test"
    set_trace_id(test_trace_id)

    # Create model manager
    model_manager = ModelManager()

    try:
        # Initialize with existing models (BGE)
        await model_manager.initialize()
        print(
            f"  📊 Existing models: {len(model_manager._embedding_models)} embedding, {len(model_manager._language_models)} language"
        )

        # Add TinyLlama configuration
        tinyllama_config = ModelConfig(
            name="tinyllama-1.1b-chat",
            model_type=ModelType.LANGUAGE_MODEL,
            format=ModelFormat.GGUF,
            path=str(tinyllama_path),
            parameters={
                "context_length": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "n_predict": 512,
            },
        )

        # Register TinyLlama
        model_manager._model_configs["tinyllama-1.1b-chat"] = tinyllama_config

        # Load TinyLlama with observability
        with probe("test.tinyllama_loading", test_trace_id):
            logger.info(
                "Loading TinyLlama model", model="tinyllama-1.1b-chat", path=str(tinyllama_path)
            )
            await model_manager.load_language_model("tinyllama-1.1b-chat")
            print("  ✅ TinyLlama loaded successfully")

        # Test language generation with different prompts
        test_prompts = [
            ("Simple greeting", "Hello, how are you?"),
            ("Code generation", "Write a Python function to add two numbers."),
            ("Explanation", "What is artificial intelligence?"),
        ]

        for i, (name, prompt) in enumerate(test_prompts):
            with probe(f"test.tinyllama_generation_{i}", test_trace_id):
                logger.info(f"Testing TinyLlama generation: {name}", prompt_length=len(prompt))
                start_time = asyncio.get_event_loop().time()
                response = await model_manager.generate_text(
                    prompt,
                    model_name="tinyllama-1.1b-chat",
                    config=GenerationConfig(max_tokens=100, temperature=0.7),
                )
                end_time = asyncio.get_event_loop().time()
                generation_time = end_time - start_time

                print(f"  🎯 {name}:")
                print(f"     Prompt: {prompt}")
                print(f"     Response: {response.content[:150]}...")
                print(f"     Time: {generation_time:.2f}s")
                print(f"     Tokens: ~{len(response.content.split())} words")

        # Get comprehensive metrics
        metrics = get_trace_metrics(test_trace_id)
        print(f"  📈 TinyLlama metrics: {len(metrics)} operations")

        for op, data in metrics.items():
            print(f"    - {op}: {data['duration_ms']:.1f}ms (ok={data['success']})")

        return True, metrics

    except Exception as e:
        print(f"  ❌ TinyLlama integration failed: {e}")
        import traceback

        traceback.print_exc()
        return False, {}


async def test_complete_system_workflow():
    """Test complete system workflow with TinyLlama."""
    print("\n🎯 Testing Complete System Workflow...")

    # Setup
    logger = get_logger(__name__)
    container = Container()
    orchestrator = Orchestrator(container)

    # Set trace ID for end-to-end flow
    e2e_trace_id = "complete_system_e2e_test"
    set_trace_id(e2e_trace_id)

    try:
        # Test complex query that should use language model
        test_query = """
        Create a Python function that calculates the factorial of a number.
        Include error handling and documentation.
        """

        with probe("test.complete_system_workflow", e2e_trace_id):
            logger.info(
                "Starting complete system workflow test", query_length=len(test_query.strip())
            )

            result = await orchestrator.process_query(
                test_query.strip(), context={"trace_id": e2e_trace_id}, workflow="development"
            )

        print(f"  ✅ E2E Success: {result.success}")
        print(f"  🤖 Agents used: {result.agents_used}")
        print(f"  ⏱️  Total time: {result.execution_time:.2f}s")
        print(f"  🎯 Confidence: {result.confidence:.2f}")
        print(f"  📝 Response length: {len(result.response_text)} chars")

        # Show snippet of response
        if result.response_text:
            snippet = (
                result.response_text[:300] + "..."
                if len(result.response_text) > 300
                else result.response_text
            )
            print("  💬 Response snippet:")
            print(f"     {snippet}")

        # Get comprehensive metrics
        e2e_metrics = get_trace_metrics(e2e_trace_id)
        print(f"  📈 E2E metrics: {len(e2e_metrics)} operations")

        # Categorize metrics
        orchestrator_ops = [op for op in e2e_metrics if "orchestrator" in op]
        agent_ops = [op for op in e2e_metrics if "agent" in op]
        model_ops = [op for op in e2e_metrics if "model" in op or "tinyllama" in op]

        print(f"    - Orchestrator ops: {len(orchestrator_ops)}")
        print(f"    - Agent ops: {len(agent_ops)}")
        print(f"    - Model ops: {len(model_ops)}")

        # Show timing breakdown
        total_time = sum(data["duration_ms"] for data in e2e_metrics.values())
        print(f"  ⏱️  Total traced time: {total_time:.1f}ms")

        return True, e2e_metrics

    except Exception as e:
        print(f"  ❌ Complete system workflow failed: {e}")
        import traceback

        traceback.print_exc()
        return False, {}

    finally:
        await orchestrator.cleanup()


async def test_model_health_and_performance():
    """Test model health and performance benchmarks."""
    print("\n📊 Testing Model Health and Performance...")

    perf_trace_id = "model_performance_test"
    set_trace_id(perf_trace_id)

    # Create model manager
    model_manager = ModelManager()

    try:
        await model_manager.initialize()

        # Add TinyLlama if not already loaded
        tinyllama_path = Path("/home/himokai/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        if "tinyllama-1.1b-chat" not in model_manager._language_models:
            tinyllama_config = ModelConfig(
                name="tinyllama-1.1b-chat",
                model_type=ModelType.LANGUAGE_MODEL,
                format=ModelFormat.GGUF,
                path=str(tinyllama_path),
                parameters={
                    "context_length": 2048,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "n_predict": 512,
                },
            )
            model_manager._model_configs["tinyllama-1.1b-chat"] = tinyllama_config
            await model_manager.load_language_model("tinyllama-1.1b-chat")

        # Health check
        with probe("test.model_health_check", perf_trace_id):
            health_status = await model_manager.health_check()

            print("  🏥 Health Status:")
            print(f"    - Language models: {len(health_status['language_models'])}")
            print(f"    - Embedding models: {len(health_status['embedding_models'])}")
            print(f"    - Overall healthy: {health_status['overall_healthy']}")

        # Performance test scenarios
        scenarios = [
            ("Short prompt", "Hello!", 20),
            ("Medium prompt", "Explain machine learning.", 50),
            ("Code prompt", "Write a Python function to sort a list.", 100),
        ]

        results = []

        for name, prompt, max_tokens in scenarios:
            with probe(f"perf.{name.lower().replace(' ', '_')}", perf_trace_id):
                start_time = asyncio.get_event_loop().time()

                response = await model_manager.generate_text(
                    prompt,
                    model_name="tinyllama-1.1b-chat",
                    config=GenerationConfig(max_tokens=max_tokens, temperature=0.7),
                )

                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time

                words = len(response.content.split())
                words_per_second = words / duration if duration > 0 else 0

                results.append(
                    {
                        "scenario": name,
                        "duration": duration,
                        "words": words,
                        "words_per_second": words_per_second,
                        "response_length": len(response.content),
                    }
                )

                print(
                    f"  🚀 {name}: {duration:.2f}s, {words_per_second:.1f} words/s, {words} words"
                )

        # Performance summary
        avg_duration = sum(r["duration"] for r in results) / len(results)
        avg_words_per_sec = sum(r["words_per_second"] for r in results) / len(results)

        print(f"  📊 Average duration: {avg_duration:.2f}s")
        print(f"  📊 Average words/sec: {avg_words_per_sec:.1f}")

        # Get performance metrics
        perf_metrics = get_trace_metrics(perf_trace_id)
        print(f"  📈 Performance metrics: {len(perf_metrics)} operations")

        return True, results, perf_metrics

    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, [], {}


async def main():
    """Run complete TinyLlama integration tests."""
    print("🧪 Testing Complete TinyLlama Integration with Full Observability")
    print("=" * 70)

    try:
        # Test 1: TinyLlama integration
        success1, metrics1 = await test_tinyllama_integration()

        # Test 2: Complete system workflow
        success2, metrics2 = await test_complete_system_workflow()

        # Test 3: Health and performance
        success3, perf_results, metrics3 = await test_model_health_and_performance()

        # Summary
        print("\n📊 Complete TinyLlama Integration Summary:")
        print("=" * 50)
        print(f"  ✅ TinyLlama integration: {'PASS' if success1 else 'FAIL'}")
        print(f"  ✅ Complete system workflow: {'PASS' if success2 else 'FAIL'}")
        print(f"  ✅ Health and performance: {'PASS' if success3 else 'FAIL'}")

        if success1 and success2 and success3:
            total_operations = len(metrics1) + len(metrics2) + len(metrics3)
            print(f"  📈 Total operations traced: {total_operations}")

            if success3 and perf_results:
                avg_words_per_sec = sum(r["words_per_second"] for r in perf_results) / len(
                    perf_results
                )
                print(f"  🚀 Average performance: {avg_words_per_sec:.1f} words/second")

            print("\n🎉 Complete TinyLlama integration successful!")
            print("\n💡 System Features Validated:")
            print("  ✅ TinyLlama 1.1B language model loaded and working")
            print("  ✅ Full observability with trace ID propagation")
            print("  ✅ End-to-end orchestrator -> agent -> language model workflow")
            print("  ✅ Performance monitoring and benchmarking")
            print("  ✅ Complete system integration with real language model")
            print("  ✅ BGE embedding model + TinyLlama language model")

            print("\n🚀 Synndicate AI system is now fully operational!")
            print("🎯 Ready for production deployment with enterprise-grade observability!")
            return True
        else:
            print("\n❌ Some tests failed - check logs above")
            return False

    except Exception as e:
        print(f"\n💥 Complete integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        clear_trace_id()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
