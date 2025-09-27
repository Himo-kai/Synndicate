#!/usr/bin/env python3
"""
Synndicate AI System Demo
Demonstrates the complete rebuilt system with working components.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from synndicate.models.manager import ModelManager
from synndicate.models.interfaces import GenerationConfig, ModelConfig, ModelType, ModelFormat
from synndicate.rag.chunking import SemanticChunker
from synndicate.rag.retriever import RAGRetriever, QueryContext, SearchMode
from synndicate.rag.context import ContextBuilder, ContextIntegrator, ContextStrategy


class SynndacateDemo:
    """Demonstrates the complete Synndicate AI system."""
    
    def __init__(self):
        self.model_manager = None
        self.rag_retriever = None
        self.knowledge_base = []
    
    async def initialize(self):
        """Initialize the Synndicate system."""
        print("🚀 Initializing Synndicate AI System...")
        
        # Initialize model manager
        self.model_manager = ModelManager()
        await self.model_manager.initialize()
        
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            await self._setup_openai_model()
        
        # Setup knowledge base
        await self._setup_knowledge_base()
        
        print("✅ Synndicate system initialized successfully!")
    
    async def _setup_openai_model(self):
        """Setup OpenAI model if API key is available."""
        try:
            config = ModelConfig(
                name="gpt-3.5-turbo",
                model_type=ModelType.LANGUAGE_MODEL,
                format=ModelFormat.OPENAI_API,
                path="gpt-3.5-turbo",
                parameters={
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "max_tokens": 500,
                    "temperature": 0.7,
                }
            )
            
            self.model_manager._model_configs["gpt-3.5-turbo"] = config
            await self.model_manager.load_language_model("gpt-3.5-turbo")
            print("✅ OpenAI language model loaded")
            
        except Exception as e:
            print(f"⚠️  OpenAI model setup failed: {e}")
    
    async def _setup_knowledge_base(self):
        """Setup the RAG knowledge base."""
        print("📚 Setting up knowledge base...")
        
        # Sample knowledge base for coding tasks
        documents = [
            "Python functions are defined using the 'def' keyword followed by the function name and parameters. Functions should include docstrings and type hints for better code quality.",
            "List comprehensions in Python provide a concise way to create lists. The syntax is [expression for item in iterable if condition].",
            "Exception handling in Python uses try-except blocks. Always catch specific exceptions rather than using bare except clauses.",
            "Object-oriented programming in Python uses classes defined with the 'class' keyword. Classes can inherit from other classes using parentheses.",
            "Python modules are imported using the 'import' statement. Use 'from module import function' for specific imports.",
            "Virtual environments in Python isolate project dependencies. Create with 'python -m venv env' and activate with 'source env/bin/activate'.",
            "Unit testing in Python uses the unittest module or pytest. Write tests that are isolated, repeatable, and fast.",
            "Code formatting in Python follows PEP 8 standards. Use tools like black and flake8 to maintain consistent code style.",
            "Async programming in Python uses async/await keywords. Async functions return coroutines that must be awaited.",
            "Data structures in Python include lists, dictionaries, sets, and tuples. Choose the right structure for your use case.",
        ]
        
        # Chunk the documents
        chunker = SemanticChunker(max_chunk_size=200, overlap=50)
        all_chunks = []
        
        for i, doc in enumerate(documents):
            chunks = chunker.chunk(doc, metadata={"doc_id": i, "topic": "python_coding"})
            all_chunks.extend(chunks)
        
        self.knowledge_base = all_chunks
        
        # Setup RAG retriever
        self.rag_retriever = RAGRetriever(
            chunks=all_chunks,
            embedding_function=lambda texts: self.model_manager.generate_embeddings(texts)
        )
        
        print(f"✅ Knowledge base ready with {len(all_chunks)} chunks")
    
    async def demonstrate_rag_retrieval(self, query: str):
        """Demonstrate RAG retrieval capabilities."""
        print(f"\n🔍 RAG Retrieval Demo: '{query}'")
        
        # Create query context
        query_context = QueryContext(
            query=query,
            conversation_history=[],
            agent_context={"agent_type": "coder", "task": "code_generation"}
        )
        
        # Retrieve relevant information
        results = await self.rag_retriever.retrieve(
            query_context, 
            SearchMode.HYBRID, 
            max_results=3
        )
        
        print(f"📖 Found {len(results)} relevant knowledge chunks:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.score:.3f}")
            print(f"     Content: {result.chunk.content[:100]}...")
        
        # Build context
        context_builder = ContextBuilder()
        context = context_builder.build_context(
            results,
            max_tokens=300,
            strategy=ContextStrategy.CONCATENATE
        )
        
        print(f"🧠 Built context ({len(context.content)} chars)")
        return context, results
    
    async def demonstrate_language_generation(self, prompt: str, context: str = ""):
        """Demonstrate language model generation."""
        print(f"\n🤖 Language Generation Demo")
        
        # Check if we have a language model
        loaded_models = self.model_manager.get_loaded_models()
        if not loaded_models["language_models"]:
            print("⚠️  No language model available. Simulating response...")
            
            # Simulate a response based on the prompt and context
            simulated_response = f"""
Based on the query and available context, here's a simulated AI response:

Query: {prompt}

Context used: {context[:100] if context else 'No context provided'}...

Simulated Response: This would be generated by a language model like GPT-3.5, Llama, or Phi-3. 
The response would incorporate the retrieved context from the RAG system to provide accurate, 
contextual answers for coding tasks.

[This is a simulation - set OPENAI_API_KEY or download a local model for real generation]
"""
            print(simulated_response)
            return simulated_response
        
        # Use actual language model
        full_prompt = f"Context: {context}\n\nQuery: {prompt}\n\nResponse:"
        
        try:
            response = await self.model_manager.generate_text(
                full_prompt,
                config=GenerationConfig(max_tokens=200, temperature=0.7)
            )
            print(f"🎯 Generated response: {response.content}")
            return response.content
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None
    
    async def demonstrate_agent_workflow(self, task: str):
        """Demonstrate a complete agent workflow."""
        print(f"\n🔄 Agent Workflow Demo: '{task}'")
        
        # Step 1: Planning (using RAG for context)
        print("📋 Step 1: Planning Phase")
        planning_context, _ = await self.demonstrate_rag_retrieval(f"How to {task}")
        planning_response = await self.demonstrate_language_generation(
            f"Create a step-by-step plan to {task}",
            planning_context.content
        )
        
        # Step 2: Implementation (using RAG for technical details)
        print("\n💻 Step 2: Implementation Phase")
        impl_context, _ = await self.demonstrate_rag_retrieval(f"Python implementation {task}")
        impl_response = await self.demonstrate_language_generation(
            f"Implement the solution for: {task}",
            impl_context.content
        )
        
        # Step 3: Review (using RAG for best practices)
        print("\n🔍 Step 3: Review Phase")
        review_context, _ = await self.demonstrate_rag_retrieval("Python best practices code review")
        review_response = await self.demonstrate_language_generation(
            f"Review the implementation and suggest improvements",
            review_context.content
        )
        
        return {
            "plan": planning_response,
            "implementation": impl_response,
            "review": review_response
        }
    
    async def run_demo(self):
        """Run the complete Synndicate demo."""
        await self.initialize()
        
        print("\n" + "="*60)
        print("🎭 SYNNDICATE AI SYSTEM DEMONSTRATION")
        print("="*60)
        
        # Demo 1: RAG Retrieval
        await self.demonstrate_rag_retrieval("How to write Python functions with type hints")
        
        # Demo 2: Language Generation
        await self.demonstrate_language_generation(
            "Explain how to create a Python function with proper documentation"
        )
        
        # Demo 3: Complete Agent Workflow
        workflow_result = await self.demonstrate_agent_workflow(
            "create a function that calculates factorial"
        )
        
        # System Status
        print("\n" + "="*60)
        print("📊 SYSTEM STATUS SUMMARY")
        print("="*60)
        
        health = await self.model_manager.health_check()
        loaded = self.model_manager.get_loaded_models()
        
        print(f"🧮 Embedding Models: {loaded['embedding_models']}")
        print(f"🤖 Language Models: {loaded['language_models']}")
        print(f"📚 Knowledge Base: {len(self.knowledge_base)} chunks")
        print(f"🏥 System Health: {'✅ Healthy' if health['overall_healthy'] else '❌ Issues'}")
        
        # Next Steps
        print("\n💡 NEXT STEPS:")
        if not loaded['language_models']:
            print("  1. Set OPENAI_API_KEY for immediate language model access")
            print("  2. Or download a local GGUF model (see docs/MODEL_SETUP.md)")
        else:
            print("  1. ✅ Language model is working!")
        
        print("  2. Build API server for external access")
        print("  3. Add more specialized agents (debugger, optimizer, etc.)")
        print("  4. Implement full orchestration workflows")
        print("  5. Add Rust executor for sandboxed code execution")
        
        await self.model_manager.shutdown()
        print("\n🎉 Demo completed successfully!")


async def main():
    """Run the Synndicate demo."""
    demo = SynndacateDemo()
    try:
        await demo.run_demo()
        return 0
    except Exception as e:
        print(f"💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
