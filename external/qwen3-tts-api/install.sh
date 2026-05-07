#!/usr/bin/env bash
set -euo pipefail

# qwen3-tts-api one-liner installer
# Usage: bash -c "$(curl -fsSL https://raw.githubusercontent.com/cornball-ai/qwen3-tts-api/main/install.sh)"

PORT="${QWEN3_PORT:-7811}"
WORK_DIR="$HOME/qwen3-tts-api"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }

# --- Banner ---
echo -e "${BOLD}"
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║            qwen3-tts-api                  ║"
echo "  ║     GPU Text-to-Speech API Server         ║"
echo "  ╚═══════════════════════════════════════════╝"
echo -e "${NC}"
echo "This will install:"
echo "  - Qwen3-TTS API server on port $PORT"
echo "  - 9 built-in voices, voice cloning, voice design"
echo "  - OpenAI-compatible REST API"
echo ""

# --- Prerequisites ---
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        error "$1 is required but not installed."
        echo "  Install: $2"
        return 1
    fi
    ok "$1 found"
}

info "Checking prerequisites..."
MISSING=0
check_cmd docker "https://docs.docker.com/engine/install/" || MISSING=1
check_cmd git "sudo apt install git" || MISSING=1
check_cmd nvidia-smi "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" || MISSING=1

if [ "$MISSING" -eq 1 ]; then
    error "Missing prerequisites. Install them and re-run."
    exit 1
fi

# Check nvidia-container-toolkit
if ! dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
    warn "nvidia-container-toolkit may not be installed."
    echo "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    read -rp "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy] ]] || exit 1
fi

# --- GPU Architecture Detection ---
detect_gpu_type() {
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    info "Detected GPU: $gpu_name"

    if echo "$gpu_name" | grep -qE "RTX.*(50[0-9]{2}|B[0-9]{3})"; then
        info "Blackwell GPU detected -> using Dockerfile.blackwell"
        echo "blackwell"
    else
        info "Pre-Blackwell GPU detected -> using default Dockerfile"
        echo "default"
    fi
}

GPU_TYPE=$(detect_gpu_type)

# --- Clone repo ---
if [ -d "$WORK_DIR" ]; then
    info "qwen3-tts-api already exists, pulling latest..."
    git -C "$WORK_DIR" pull --ff-only 2>/dev/null || warn "Could not pull (may have local changes)"
else
    info "Cloning qwen3-tts-api..."
    git clone "https://github.com/cornball-ai/qwen3-tts-api.git" "$WORK_DIR"
fi

# --- Build image ---
info "Building qwen3-tts-api image..."
if [ "$GPU_TYPE" = "blackwell" ]; then
    docker build -t qwen3-tts-api:latest -f "$WORK_DIR/Dockerfile.blackwell" "$WORK_DIR"
else
    docker build -t qwen3-tts-api:latest -f "$WORK_DIR/Dockerfile" "$WORK_DIR"
fi
ok "Image built"

# --- Model Downloads ---
echo ""
echo -e "${BOLD}Qwen3-TTS requires pre-downloaded models.${NC}"
echo ""
echo "  1) CustomVoice only  (~7GB)  - 9 built-in voices"
echo "  2) All three models  (~21GB) - voices + cloning + voice design"
echo "  3) Skip download     - I already have the models"
echo ""
read -rp "Choose [1/2/3]: " MODEL_CHOICE

download_model() {
    local model="$1"
    info "Downloading $model..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$model"
    elif command -v python3 &>/dev/null; then
        python3 -c "
try:
    from huggingface_hub import snapshot_download
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'huggingface_hub'])
    from huggingface_hub import snapshot_download
snapshot_download('$model')
"
    else
        error "No huggingface-cli or python3 found. Install manually:"
        echo "  pip install huggingface_hub"
        echo "  huggingface-cli download $model"
        return 1
    fi
    ok "Downloaded $model"
}

case "$MODEL_CHOICE" in
    1)
        download_model "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        ;;
    2)
        download_model "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        download_model "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        download_model "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign"
        ;;
    3)
        info "Skipping model download."
        ;;
    *)
        warn "Invalid choice, skipping model download."
        ;;
esac

# --- Stop existing container ---
if docker ps -a --format '{{.Names}}' | grep -q '^qwen3-tts-api$'; then
    info "Stopping existing qwen3-tts-api container..."
    docker rm -f qwen3-tts-api
fi

# --- Run ---
info "Starting qwen3-tts-api on port $PORT..."
docker run -d --gpus all --network=host --name qwen3-tts-api \
    -v "$HOME/.cache/huggingface:/cache" \
    -e "PORT=$PORT" \
    -e USE_FLASH_ATTENTION=false \
    -e LOCAL_FILES_ONLY=true \
    --restart unless-stopped \
    qwen3-tts-api:latest

echo ""
echo -e "${GREEN}${BOLD}  ╔═══════════════════════════════════════════╗"
echo "  ║             Setup complete                ║"
echo "  ╚═══════════════════════════════════════════╝${NC}"
echo ""
echo "  qwen3-tts API:   http://localhost:$PORT"
echo "  Health check:    curl http://localhost:$PORT/health"
echo "  List voices:     curl http://localhost:$PORT/v1/voices"
echo ""
echo "  Test it:"
echo "    curl -X POST http://localhost:$PORT/v1/audio/speech \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"input\": \"Hello world\", \"voice\": \"Vivian\"}' \\"
echo "      --output hello.wav"
echo ""
echo "  Manage:"
echo "    docker logs -f qwen3-tts-api    # logs"
echo "    docker restart qwen3-tts-api    # restart"
echo "    docker rm -f qwen3-tts-api      # remove"
echo ""
echo "  Note: Takes ~1-2 minutes to load models on first start."
echo ""
