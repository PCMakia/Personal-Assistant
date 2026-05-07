# Dockerfile for older NVIDIA GPUs (Ampere, Ada Lovelace, etc.)
# Uses CUDA 12.4 + PyTorch 2.6

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Create and activate virtual environment
ENV VIRTUAL_ENV=/app/.venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install Qwen3-TTS
RUN pip install --no-cache-dir qwen-tts

# Install Flash Attention 2 (optional but recommended)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Flash Attention not available, continuing without it"

# Install API dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY main.py ./

# Create directories for cache and voices
RUN mkdir -p /cache /voices

# Workaround for hfhub absolute symlinks: R's hfhub creates absolute symlinks
# pointing to /home/USER/.cache/huggingface which don't exist in the container.
# Create a symlink so these paths resolve to /cache.
RUN mkdir -p /home/troy/.cache && ln -sf /cache /home/troy/.cache/huggingface

# Set default environment variables
ENV PORT=4123
ENV HOST=0.0.0.0
ENV DEVICE=cuda:0
ENV DTYPE=bfloat16
ENV USE_FLASH_ATTENTION=true
ENV MODEL_CACHE_DIR=/cache
ENV VOICE_LIBRARY_DIR=/voices
ENV MODEL_NAME=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
ENV VOICE_CLONE_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
ENV VOICE_DESIGN_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign
ENV HF_HOME=/cache

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application
CMD ["python", "main.py"]
