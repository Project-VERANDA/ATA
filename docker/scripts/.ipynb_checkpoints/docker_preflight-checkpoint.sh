#!/bin/bash

# ============================================================================
# Docker Pre-flight Validation Script
# Run before docker compose build to verify prerequisites
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}✅${NC} $*"; }
log_warn()    { echo -e "${YELLOW}⚠️  ${NC} $*"; }
log_error()   { echo -e "${RED}❌${NC} $*" >&2; }

errors=0
warnings=0

echo ""
echo "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo "${BOLD}     Dialogue Anonymizer Docker Pre-flight Checks           ${NC}"
echo "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Check working directory
if [ ! -f "$ROOT_DIR/requirements.txt" ]; then
    log_error "Not in project root: requirements.txt not found"
    echo "  Expected: $ROOT_DIR/requirements.txt"
    ((errors++))
else
    log_success "Project root verified"
fi

# Check Docker daemon
if ! command -v docker &> /dev/null; then
    log_error "Docker not installed or not in PATH"
    ((errors++))
else
    DOCKER_VERSION=$(docker --version | cut -d',' -f1)
    log_success "Docker: $DOCKER_VERSION"
fi

# Check Docker Compose plugin
if ! docker compose version &> /dev/null 2>&1; then
    log_error "Docker Compose plugin not found"
    echo "  Try: docker compose version or install 'docker-compose-plugin'"
    ((errors++))
else
    COMPOSE_VERSION=$(docker compose version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1)
    log_success "Docker Compose: $COMPOSE_VERSION"
fi

# Check Docker daemon is running
if ! docker info &> /dev/null 2>&1; then
    log_error "Docker daemon is not running"
    echo "  Start Docker Desktop or run: sudo systemctl start docker"
    ((errors++))
else
    log_success "Docker daemon: running"
fi

# Check .env file
if [ -f "$ROOT_DIR/.env" ]; then
    log_success ".env configuration found"
    
    # Warn about placeholder values
    if grep -q "your_api_key_here" "$ROOT_DIR/.env" 2>/dev/null; then
        log_warn ".env contains placeholder values - review before deploying"
    fi
else
    log_warn ".env not found - will use defaults (review after build)"
fi

# Check requirements files
if [ -f "$ROOT_DIR/requirements.txt" ]; then
    REQ_LINES=$(wc -l < "$ROOT_DIR/requirements.txt")
    log_success "requirements.txt: $REQ_LINES dependencies"
else
    log_error "requirements.txt missing"
    ((errors++))
fi

if [ -f "$ROOT_DIR/requirements-web.txt" ]; then
    log_success "requirements-web.txt found"
else
    log_warn "requirements-web.txt missing (web features unavailable)"
fi

# Check GPU availability (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    CUDA_VER=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -1)
    log_success "GPU detected: $GPU_NAME (CUDA $CUDA_VER)"
else
    log_warn "No NVIDIA GPU detected - container will run in CPU mode"
    log_info "  Note: Add nvidia runtime to docker-compose.yml if GPU needed"
fi

# Check available disk space
DISK_AVAIL_GB=$(df -BG "$ROOT_DIR" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "${DISK_AVAIL_GB:-0}" -lt 15 ] 2>/dev/null; then
    log_warn "Low disk space: ${DISK_AVAIL_GB}GB available (recommended: 20GB+ for models)"
    ((warnings++))
else
    log_success "Disk space: ${DISK_AVAIL_GB}GB available"
fi

# Check Python version compatibility
if command -v python3 &> /dev/null; then
    PY_VER=$(python3 --version)
    if [[ "$PY_VER" =~ "3.1" ]]; then
        log_success "Host Python compatible: $PY_VER"
    else
        log_warn "Host Python may differ from container Python ($PY_VER)"
        log_info "  Container uses Python 3.12 regardless"
    fi
fi

# Check for conflicting Installer.sh environment
if conda env list 2>/dev/null | grep -q "^whisperx "; then
    log_warn "Conda environment 'whisperx' exists on host"
    log_info "  This is fine - Docker uses isolated container Python"
fi

# Check for existing Docker containers/instances
EXISTING_CONTAINER=$(docker ps -a --filter "name=dialogue-anonymizer" --format "{{.Names}}" 2>/dev/null)
if [ -n "$EXISTING_CONTAINER" ]; then
    log_info "Existing container detected: $EXISTING_CONTAINER"
    log_info "  Run 'docker compose down' before rebuilding if you want fresh state"
fi

echo ""
echo "${BOLD}─────────────────────────────────────────────────────────────${NC}"

if [ $errors -gt 0 ]; then
    log_error "Pre-flight FAILED: $errors error(s), $warnings warning(s)"
    echo "  Fix errors before proceeding with docker compose build"
    exit 1
elif [ $warnings -gt 0 ]; then
    log_warn "Pre-flight PASSED with warnings: $warnings warning(s)"
    echo "  Review warnings above if issues occur during build"
    echo ""
    read -p "Continue with docker compose build? (y/N): " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        exit 0
    fi
else
    log_success "Pre-flight PASSED: Ready to build"
fi

echo ""
echo "${BOLD}═══════════════════════════════════════════════════════════${NC}"
echo ""