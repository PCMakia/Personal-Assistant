# qwen3-tts-api

> **This repo provides:** A Dockerfile and FastAPI server that wraps [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) with OpenAI-compatible REST endpoints. The original Qwen3-TTS is a Python library - this repo containerizes it for easy deployment.

Part of the [cornball-ai](https://github.com/cornball-ai) ecosystem, designed to work with the [tts.api](https://github.com/cornball-ai/tts.api) R package for text-to-speech generation.

## Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI TTS endpoints
- **9 Built-in Voices**: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
- **Voice Cloning**: Clone any voice from a 3-second audio sample
- **Voice Design**: Generate custom voices from natural language descriptions
- **10 Languages**: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **Blackwell GPU Support**: Optimized for RTX 50xx series

## Quick Start

### One-Liner Install

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/cornball-ai/qwen3-tts-api/main/install.sh)"
```

Auto-detects your GPU, downloads models, and starts the server on port 7811. Requires: Docker, NVIDIA GPU, nvidia-container-toolkit.

### Docker (Manual)

```bash
# For older GPUs (Ampere, Ada Lovelace)
docker build -t qwen3-tts-api .
docker run -d --gpus all --network=host --name qwen3-tts-api \
  -v ~/.cache/huggingface:/cache \
  -e PORT=7811 \
  qwen3-tts-api

# For Blackwell GPUs (RTX 50xx)
docker build -f Dockerfile.blackwell -t qwen3-tts-api:blackwell .
docker run -d --gpus all --network=host --name qwen3-tts-api \
  -v ~/.cache/huggingface:/cache \
  -e PORT=7811 \
  -e USE_FLASH_ATTENTION=false \
  qwen3-tts-api:blackwell
```

**Note:** We use `--network=host` for reliable DNS resolution (HuggingFace model downloads). The `PORT=7811` env var sets the server port directly.

### Gradio UI Mode

Run the official Qwen3-TTS Gradio demo instead of the API server:

```bash
docker run -d --gpus all --network=host --name qwen3-tts-gradio \
  -v ~/.cache/huggingface:/cache \
  -e ENABLE_GRADIO=true \
  -e USE_FLASH_ATTENTION=false \
  qwen3-tts-api:blackwell
```

Then open http://localhost:7860 in your browser.

### LXC Install (Proxmox)

Deploy in a Proxmox LXC container with GPU passthrough:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/cornball-ai/qwen3-tts-api/main/install-lxc.sh)"
```

Run this on the Proxmox host. Creates a privileged LXC container with NVIDIA GPU passthrough, Docker, and the qwen3-tts-api service with auto-start. Requires: Proxmox, NVIDIA GPU with driver installed on host.

### Local Installation

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Qwen3-TTS and dependencies
pip install qwen-tts
pip install -r requirements.txt

# Optional: Flash Attention for reduced memory
pip install flash-attn --no-build-isolation

# Run server
python main.py
```

## API Endpoints

### Generate Speech (Built-in Voices)

```bash
curl -X POST http://localhost:7811/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "voice": "Vivian",
    "language": "English"
  }' --output speech.wav
```

### Voice Cloning

Two modes available:

**High-quality (ICL mode)** - requires transcript of reference audio:

```bash
curl -X POST http://localhost:7811/v1/audio/speech/upload \
  -F "input=Hello, this is my cloned voice!" \
  -F "voice_file=@reference.wav" \
  -F "ref_text=This is the transcript of my reference audio." \
  -F "language=English" \
  --output cloned.wav
```

**Fast mode (x-vector only)** - no transcript needed, lower quality:

```bash
curl -X POST http://localhost:7811/v1/audio/speech/upload \
  -F "input=Hello, this is my cloned voice!" \
  -F "voice_file=@reference.wav" \
  -F "x_vector_only=true" \
  -F "language=English" \
  --output cloned.wav
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `input` | Yes | Text to synthesize |
| `voice_file` | Yes | Reference audio file (3+ seconds recommended) |
| `language` | No | Target language (default: English) |
| `ref_text` | No* | Transcript of reference audio (for ICL mode) |
| `x_vector_only` | No | Set to "true" for fast mode without transcript |

*Either `ref_text` or `x_vector_only=true` should be provided.

### Voice Design (Generate from Description)

```bash
curl -X POST http://localhost:7811/v1/audio/speech/design \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, I am a custom designed voice!",
    "language": "English",
    "voice_description": "A warm, friendly female voice with a slight British accent"
  }' --output designed.wav
```

### List Voices

```bash
curl http://localhost:7811/v1/voices
```

### Health Check

```bash
curl http://localhost:7811/health
```

## R Usage (tts.api)

```r
library(tts.api)
set_tts_base("http://localhost:7811")

# Check if qwen3-tts is running
qwen3_available()  # TRUE

# Generate speech with built-in voice
speech("Hello world!", voice = "Vivian", file = "hello.wav", backend = "qwen3")

# Voice cloning - fast mode (no transcript needed)
speech_clone(
  input = "Hello with my cloned voice!",
  voice_file = "reference.wav",
  x_vector_only = TRUE,
  file = "cloned.wav",
  backend = "qwen3"
)

# Voice cloning - high quality (with transcript)
speech_clone(
  input = "Hello with my cloned voice!",
  voice_file = "reference.wav",
  ref_text = "This is what I said in the reference audio.",
  file = "cloned.wav",
  backend = "qwen3"
)

# Voice design (create voice from description)
speech_design(
  input = "Hello, I am your assistant!",
  voice_description = "A warm, professional female voice",
  file = "designed.wav"
)
```

See [tts.api](https://github.com/cornball-ai/tts.api) for full documentation.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 4123 | API server port |
| `HOST` | 0.0.0.0 | Server host |
| `ENABLE_GRADIO` | false | Launch Gradio UI instead of API server |
| `GRADIO_PORT` | 7860 | Gradio UI port (when ENABLE_GRADIO=true) |
| `DEVICE` | cuda:0 | PyTorch device |
| `DTYPE` | bfloat16 | Model dtype (bfloat16, float16, float32) |
| `USE_FLASH_ATTENTION` | true | Enable Flash Attention 2 (set false for Blackwell) |
| `MODEL_NAME` | Qwen/Qwen3-TTS-12Hz-1.7B-Base | Base model |
| `VOICE_DESIGN_MODEL` | Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | Voice design model |
| `MODEL_CACHE_DIR` | /cache | HuggingFace cache directory |
| `LOCAL_FILES_ONLY` | true | Prevent auto-downloading models (requires pre-downloaded weights) |

## Pre-downloading Models

By default, `LOCAL_FILES_ONLY=true` prevents automatic model downloads. You must pre-download models before running the container.

**Models required:**
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` - Built-in speakers (~7GB)
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` - Voice cloning (~7GB)
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` - Voice design (~7GB)

**Download with Python:**
```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'); \
  snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base'); \
  snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign')"
```

**Download with R:**
```r
hfhub::hub_snapshot("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
hfhub::hub_snapshot("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
hfhub::hub_snapshot("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
```

**Fixing permissions:** If Docker previously created cache directories with root ownership, you may get permission errors when downloading. Fix with:
```bash
sudo chown -R $USER:$USER ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-*
```

**Note on hfhub (R):** The R `hfhub` package creates absolute symlinks pointing to `/home/USER/.cache/huggingface`. The Dockerfile includes a workaround symlink, but it's hardcoded to `/home/troy`. If you use a different username, either rebuild with your username or use Python's `huggingface_hub` which creates relative symlinks.

To enable auto-download (not recommended), set `LOCAL_FILES_ONLY=false`.

## GPU Requirements

| Model | VRAM |
|-------|------|
| 0.6B CustomVoice | ~4GB |
| 1.7B CustomVoice | ~8GB |
| 1.7B + Voice Clone | ~12GB |
| 1.7B + Voice Design | ~16GB |

## Model Variants

The Qwen3-TTS family has different model variants for different use cases:

| Model | Use Case | Method |
|-------|----------|--------|
| **CustomVoice** (default) | 9 built-in speakers | `generate_custom_voice()` |
| **Base** | Voice cloning from audio | `generate_voice_clone()` |
| **VoiceDesign** | Create voice from description | `generate_voice_design()` |

This API uses CustomVoice by default and lazy-loads Base/VoiceDesign on first use.

## Startup Times

| Type | Time |
|------|------|
| Cold start (loading models) | ~45-60s |
| Warm request (model loaded) | ~5s per sentence |

## Tested

- RTX 5060 Ti (Blackwell, 16GB VRAM) with CUDA 12.8 / PyTorch 2.7+

## License

Apache 2.0 (following Qwen3-TTS license)
