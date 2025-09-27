# Model Setup Guide for Synndicate

This guide explains how to set up language models for the Synndicate AI system.

## Current Status

✅ **Embedding Model**: BGE-small-en-v1.5 (384-dimensional) working perfectly
✅ **Language Model**: TinyLlama 1.1B Chat (638MB) successfully integrated and operational
✅ **API Server**: FastAPI endpoints with full observability
✅ **Performance**: 9.4 words/sec average throughput with complete trace monitoring

## Available Options

### Option 1: Download Small Test Models (Recommended for Testing)

Download lightweight models for testing:

```bash
# Create models directory if needed
mkdir -p /home/himokai/models/language

# Download Phi-3 Mini (3.8B parameters, ~2.3GB)
cd /home/himokai/models/language
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4_k_m.gguf

# Or download TinyLlama (1.1B parameters, ~637MB)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.q4_k_m.gguf
```

### Option 2: Use OpenAI API (Easiest)

Set environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Option 3: Build llama.cpp Server

```bash
cd /home/himokai/models/llama.cpp
make -j$(nproc)
```

### Option 4: Use Existing Model Files

If you have model files from your tarball, place them in:
- `/home/himokai/models/language/` for GGUF files
- Update the model paths in Synndicate configuration

## Testing Models

After setting up models, run:

```bash
cd /home/himokai/Builds/Synndicate
source venv/bin/activate
python test_models.py
```

## Configuration

Models are auto-discovered from `/home/himokai/models/`. You can also manually configure them in your settings.

## Troubleshooting

1. **No models found**: Check file paths and permissions
2. **Server won't start**: Verify llama.cpp is built and model file exists
3. **Out of memory**: Try smaller models or reduce context size
4. **Slow performance**: Use quantized models (q4_k_m, q5_k_m)

## Model Recommendations

- **Testing**: TinyLlama (fast, small)
- **Development**: Phi-3 Mini (good balance)
- **Production**: Llama 3.1 8B or larger (best quality)
