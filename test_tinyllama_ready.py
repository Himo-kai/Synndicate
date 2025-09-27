#!/usr/bin/env python3
"""Quick test for TinyLlama integration once download completes."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synndicate.observability.logging import setup_logging, get_logger, set_trace_id
from synndicate.models.manager import ModelManager

async def main():
    """Test TinyLlama integration."""
    print("🧪 Testing TinyLlama Integration")
    
    # Check if downloaded
    tinyllama_path = Path("/home/himokai/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    
    if not tinyllama_path.exists():
        print("⏳ TinyLlama not yet downloaded")
        return False
    
    file_size = tinyllama_path.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ TinyLlama found: {file_size:.1f}MB")
    
    if file_size < 600:  # Still downloading
        print(f"⏳ Download in progress ({file_size:.1f}MB / ~638MB)")
        return False
    
    # Setup observability
    setup_logging()
    set_trace_id("tinyllama_test")
    
    # Test model manager
    model_manager = ModelManager()
    await model_manager.initialize()
    
    print("🎉 TinyLlama ready for integration!")
    print("🚀 Run full integration tests now")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
