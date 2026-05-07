#!/usr/bin/env bash
set -euo pipefail

# qwen3-tts-api LXC installer for Proxmox
# Usage: bash -c "$(curl -fsSL https://raw.githubusercontent.com/cornball-ai/qwen3-tts-api/main/install-lxc.sh)"
# Run this on the Proxmox host.

PORT="${QWEN3_PORT:-7811}"
REPO_URL="https://github.com/cornball-ai/qwen3-tts-api.git"
TEMPLATE_NAME="ubuntu-24.04-standard_24.04-2_amd64.tar.zst"

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
echo "  +=============================================+"
echo "  |        qwen3-tts-api  (LXC Install)        |"
echo "  |     GPU Text-to-Speech in Proxmox LXC      |"
echo "  +=============================================+"
echo -e "${NC}"
echo "This will create a Proxmox LXC container with:"
echo "  - NVIDIA GPU passthrough"
echo "  - Docker + nvidia-container-toolkit"
echo "  - Qwen3-TTS API server on port $PORT"
echo ""

# --- Check Proxmox host ---
info "Checking Proxmox environment..."

if ! command -v pveversion &>/dev/null; then
    error "pveversion not found. This script must run on a Proxmox host."
    exit 1
fi
ok "Proxmox detected: $(pveversion --verbose 2>/dev/null | head -1)"

if ! command -v pct &>/dev/null; then
    error "pct not found. Is pve-container installed?"
    exit 1
fi
ok "pct available"

if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found on host. NVIDIA driver required."
    exit 1
fi
HOST_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
HOST_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "Host GPU: $HOST_GPU (driver $HOST_DRIVER)"

# --- Prompt for container ID ---
NEXT_ID=$(pvesh get /cluster/nextid 2>/dev/null || echo "200")
read -rp "Container ID [$NEXT_ID]: " CTID
CTID="${CTID:-$NEXT_ID}"

# Validate CTID is a number
if ! [[ "$CTID" =~ ^[0-9]+$ ]]; then
    error "Container ID must be a number."
    exit 1
fi

# Check if CTID already exists
if pct status "$CTID" &>/dev/null; then
    error "Container $CTID already exists. Choose a different ID."
    exit 1
fi

# --- Prompt for storage ---
read -rp "Storage [local-lvm]: " STORAGE
STORAGE="${STORAGE:-local-lvm}"

# --- Download template if needed ---
TEMPLATE="local:vztmpl/${TEMPLATE_NAME}"
if ! pveam list local 2>/dev/null | grep -q "$TEMPLATE_NAME"; then
    info "Downloading Ubuntu 24.04 template..."
    pveam update
    pveam download local "$TEMPLATE_NAME"
    ok "Template downloaded"
else
    ok "Template already available"
fi

# --- Create container ---
info "Creating LXC container $CTID..."
pct create "$CTID" "$TEMPLATE" \
    --hostname "qwen3-tts" \
    --memory 8192 \
    --cores 4 \
    --rootfs "${STORAGE}:20" \
    --net0 "name=eth0,bridge=vmbr0,ip=dhcp" \
    --features nesting=1 \
    --unprivileged 0 \
    --ostype ubuntu \
    --start 0
ok "Container $CTID created"

# --- GPU passthrough config ---
info "Configuring GPU passthrough..."

NVIDIA_MAJOR=$(grep nvidia-frontend /proc/devices | awk '{print $1}')
if [ -z "$NVIDIA_MAJOR" ]; then
    error "Could not detect nvidia-frontend major number from /proc/devices"
    exit 1
fi

NVIDIA_UVM_MAJOR=$(grep nvidia-uvm /proc/devices | awk '{print $1}')
if [ -z "$NVIDIA_UVM_MAJOR" ]; then
    warn "nvidia-uvm not found in /proc/devices, skipping UVM entries"
fi

CONF="/etc/pve/lxc/${CTID}.conf"

{
    echo ""
    echo "# NVIDIA GPU passthrough"
    echo "lxc.cgroup2.devices.allow: c ${NVIDIA_MAJOR}:* rwm"
    if [ -n "${NVIDIA_UVM_MAJOR:-}" ]; then
        echo "lxc.cgroup2.devices.allow: c ${NVIDIA_UVM_MAJOR}:* rwm"
    fi
    echo "lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file"
    echo "lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file"
    if [ -n "${NVIDIA_UVM_MAJOR:-}" ]; then
        echo "lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file"
        echo "lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file"
    fi
    if [ -d /dev/nvidia-caps ]; then
        echo "lxc.mount.entry: /dev/nvidia-caps dev/nvidia-caps none bind,optional,create=dir"
    fi
} >> "$CONF"

# Add additional GPUs if present
for dev in /dev/nvidia[1-9]*; do
    [ -e "$dev" ] || continue
    devname=$(basename "$dev")
    echo "lxc.mount.entry: /dev/$devname dev/$devname none bind,optional,create=file" >> "$CONF"
    info "Added extra GPU device: $devname"
done

ok "GPU config written to $CONF"

# --- Start container ---
info "Starting container $CTID..."
pct start "$CTID"
sleep 3
ok "Container started"

# --- Install NVIDIA driver inside container ---
info "Installing NVIDIA driver $HOST_DRIVER inside container (this takes a few minutes)..."
pct exec "$CTID" -- bash -c "
    set -euo pipefail
    apt-get update -qq
    apt-get install -y -qq wget build-essential kmod >/dev/null 2>&1

    DRIVER_URL=\"https://us.download.nvidia.com/XFree86/Linux-x86_64/${HOST_DRIVER}/NVIDIA-Linux-x86_64-${HOST_DRIVER}.run\"
    wget -q \"\$DRIVER_URL\" -O /tmp/nvidia-driver.run
    chmod +x /tmp/nvidia-driver.run
    /tmp/nvidia-driver.run --no-kernel-modules --silent
    rm /tmp/nvidia-driver.run
"

# Verify
pct exec "$CTID" -- nvidia-smi
ok "NVIDIA driver installed and verified"

# --- Install Docker ---
info "Installing Docker..."
pct exec "$CTID" -- bash -c '
    set -euo pipefail
    apt-get install -y -qq ca-certificates curl gnupg >/dev/null 2>&1
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io >/dev/null 2>&1
'
ok "Docker installed"

# --- Install nvidia-container-toolkit ---
info "Installing nvidia-container-toolkit..."
pct exec "$CTID" -- bash -c '
    set -euo pipefail
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit >/dev/null 2>&1
    nvidia-ctk runtime configure --runtime=docker

    # Critical for LXC: disable cgroup management
    if [ -f /etc/nvidia-container-runtime/config.toml ]; then
        sed -i "s/^#no-cgroups = false/no-cgroups = true/" /etc/nvidia-container-runtime/config.toml
        grep -q "^no-cgroups = true" /etc/nvidia-container-runtime/config.toml || \
            echo "no-cgroups = true" >> /etc/nvidia-container-runtime/config.toml
    fi

    systemctl restart docker
'
ok "nvidia-container-toolkit installed"

# --- Verify Docker GPU ---
info "Verifying Docker GPU access..."
pct exec "$CTID" -- docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
ok "Docker GPU access verified"

# --- Clone and build qwen3-tts-api ---
info "Cloning qwen3-tts-api..."
pct exec "$CTID" -- bash -c "
    set -euo pipefail
    apt-get install -y -qq git >/dev/null 2>&1
    git clone '$REPO_URL' /opt/qwen3-tts-api
"
ok "Repository cloned"

# Detect GPU type for Dockerfile selection
DOCKERFILE="Dockerfile"
if echo "$HOST_GPU" | grep -qE "RTX.*(50[0-9]{2}|B[0-9]{3})"; then
    DOCKERFILE="Dockerfile.blackwell"
    info "Blackwell GPU detected -> using $DOCKERFILE"
else
    info "Pre-Blackwell GPU detected -> using default Dockerfile"
fi

info "Building Docker image (this takes several minutes)..."
pct exec "$CTID" -- docker build -t qwen3-tts-api:latest -f "/opt/qwen3-tts-api/$DOCKERFILE" /opt/qwen3-tts-api
ok "Docker image built"

# --- Model Downloads ---
echo ""
echo -e "${BOLD}Qwen3-TTS requires pre-downloaded models.${NC}"
echo ""
echo "  1) CustomVoice only  (~7GB)  - 9 built-in voices"
echo "  2) All three models  (~21GB) - voices + cloning + voice design"
echo "  3) Skip download     - I will download models later"
echo ""
read -rp "Choose [1/2/3]: " MODEL_CHOICE

download_model() {
    local model="$1"
    info "Downloading $model inside container..."
    pct exec "$CTID" -- bash -c "
        docker run --rm --gpus all --network=host \
            -v /root/.cache/huggingface:/cache \
            -e LOCAL_FILES_ONLY=false \
            -e MODEL_CACHE_DIR=/cache \
            python:3.11-slim bash -c '
                pip install -q huggingface_hub 2>/dev/null
                python -c \"from huggingface_hub import snapshot_download; snapshot_download(\\\"$model\\\", cache_dir=\\\"/cache\\\")\"
            '
    "
    ok "Downloaded $model"
}

case "${MODEL_CHOICE:-3}" in
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
        echo "  Download later inside the container with:"
        echo "    pct exec $CTID -- docker run --rm --network=host \\"
        echo "      -v /root/.cache/huggingface:/cache python:3.11-slim bash -c \\"
        echo "      'pip install -q huggingface_hub && python -c \"from huggingface_hub import snapshot_download; snapshot_download(\\\"Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice\\\", cache_dir=\\\"/cache\\\")\"'"
        ;;
    *)
        warn "Invalid choice, skipping model download."
        ;;
esac

# --- Run qwen3-tts-api ---
info "Starting qwen3-tts-api on port $PORT..."
pct exec "$CTID" -- docker run -d --gpus all --network=host --name qwen3-tts-api \
    -v /root/.cache/huggingface:/cache \
    -e "PORT=$PORT" \
    -e USE_FLASH_ATTENTION=false \
    -e LOCAL_FILES_ONLY=true \
    --restart unless-stopped \
    qwen3-tts-api:latest
ok "Container started"

# --- Create systemd service ---
info "Creating systemd service for auto-start..."
pct exec "$CTID" -- bash -c "cat > /etc/systemd/system/qwen3-tts-api.service <<'UNIT'
[Unit]
Description=Qwen3-TTS API Server
After=docker.service
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10
ExecStartPre=-/usr/bin/docker rm -f qwen3-tts-api
ExecStart=/usr/bin/docker run --rm --gpus all --network=host --name qwen3-tts-api \
    -v /root/.cache/huggingface:/cache \
    -e PORT=${PORT} \
    -e USE_FLASH_ATTENTION=false \
    -e LOCAL_FILES_ONLY=true \
    qwen3-tts-api:latest
ExecStop=/usr/bin/docker stop qwen3-tts-api

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable qwen3-tts-api
"
ok "Systemd service created and enabled"

# --- Get container IP ---
LXC_IP=$(pct exec "$CTID" -- hostname -I 2>/dev/null | awk '{print $1}')

# --- Done ---
echo ""
echo -e "${GREEN}${BOLD}  +=============================================+"
echo "  |            Setup complete                   |"
echo "  +=============================================+${NC}"
echo ""
echo "  Container ID:    $CTID"
echo "  Container IP:    ${LXC_IP:-<waiting for DHCP>}"
echo "  API port:        $PORT"
echo ""
if [ -n "${LXC_IP:-}" ]; then
    echo "  Health check:    curl http://${LXC_IP}:${PORT}/health"
    echo "  List voices:     curl http://${LXC_IP}:${PORT}/v1/voices"
else
    echo "  Health check:    curl http://<container-ip>:${PORT}/health"
fi
echo ""
echo "  Test it:"
echo "    curl -X POST http://${LXC_IP:-<container-ip>}:${PORT}/v1/audio/speech \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"input\": \"Hello world\", \"voice\": \"Vivian\"}' \\"
echo "      --output hello.wav"
echo ""
echo "  Manage:"
echo "    pct exec $CTID -- docker logs -f qwen3-tts-api    # logs"
echo "    pct exec $CTID -- systemctl restart qwen3-tts-api  # restart"
echo "    pct stop $CTID                                      # stop container"
echo "    pct start $CTID                                     # start container"
echo ""
echo "  Note: Takes ~1-2 minutes to load models on first start."
echo ""
