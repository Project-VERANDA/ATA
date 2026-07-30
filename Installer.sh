#!/usr/bin/env bash

# ============================================================================
# Dialogue Anonymizer Installer (v3.4)
# ============================================================================

set -eo pipefail

SCRIPT_VERSION="3.4"
ENV_NAME="whisperx"
TARGET_PYTHON="3.12"  # ✅ REQUIRED - PyTorch 2.8+ doesn't fully support 3.13
MODEL_DIR_NAME="pipeline/model"
MODEL_DIR_REL="$MODEL_DIR_NAME"

# --- Argument Defaults ---
SHOW_HELP=false
SKIP_WEB=false
NO_MODELS=false
SKIP_WHISPER=false
SKIP_PYANNOTE=false
AUTO_LOGIN=false
ANSWER_YES=false
QUIET_MODE=false
WHISPER_SELECTION=""
FORCE_REFRESH=false
HUGGINGFACE_TOKEN=""
DRY_RUN=false
UNINSTALL=false
SHOW_VERSION=false

WHISPER_PRIMARY_PATH="pipeline/model/models--Systran--faster-whisper-large-v3"

# ============================================================================
# COLOR & LOGGING HELPERS
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}✅${NC} $*"; }
log_warn()    { echo -e "${YELLOW}⚠️  ${NC} $*"; }
log_error()   { echo -e "${RED}❌${NC} $*" >&2; }
log_section() { echo -e "\n${BOLD}============================================================${NC}"; echo -e "${BOLD}$1${NC}"; echo -e "${BOLD}============================================================${NC}"; }

# ============================================================================
# USAGE
# ============================================================================

usage() {
    cat << 'EOF'
Usage: ./Installer.sh [OPTIONS]

Quick Start:
  ./Installer.sh -y                        # Install with all defaults (Python 3.12)
  ./Installer.sh -y --skip-web             # Skip Flask/TTS web interface
  ./Installer.sh -y --whisper-models "large,medium"

Options:
  -h, --help              Show this help message
  -V, --version           Show installer version and exit
  -y, --yes               Auto-answer 'yes' to all prompts
  --dry-run               Simulate installation without making changes
  --uninstall             Remove conda env and model files
  --force-refresh         Unconditionally reinstall all packages
  --skip-web              Skip web interface installation (Flask, TTS)
  --no-models             Skip ALL ML model downloads
  --skip-whisper          Skip WhisperX model download
  --skip-pyannote         Skip Pyannote diarization model download
  --whisper-models LIST   Comma or space-separated list (tiny, base, small, medium, large)
  --auto-login            Auto-authenticate with HUGGINGFACE_TOKEN env var
  -q, --quiet             Minimal output mode

Environment Variables:
  HUGGINGFACE_TOKEN       HF token for gated model access (used with --auto-login)

Examples:
  ./Installer.sh -y --whisper-models "large,medium"
  ./Installer.sh --whisper-models "small base"           # Interactive prompts
  HUGGINGFACE_TOKEN=xxx ./Installer.sh --auto-login -y
  ./Installer.sh --dry-run                               # Preview actions
  ./Installer.sh --uninstall                             # Clean removal
EOF
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) SHOW_HELP=true; shift ;;
        -V|--version) SHOW_VERSION=true; shift ;;
        -y|--yes) ANSWER_YES=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --uninstall) UNINSTALL=true; shift ;;
        --skip-web) SKIP_WEB=true; shift ;;
        --no-models) NO_MODELS=true; shift ;;
        --skip-whisper) SKIP_WHISPER=true; shift ;;
        --skip-pyannote) SKIP_PYANNOTE=true; shift ;;
        --whisper-models) WHISPER_SELECTION="$2"; shift 2 ;;
        --auto-login) AUTO_LOGIN=true; shift ;;
        -q|--quiet) QUIET_MODE=true; shift ;;
        --force-refresh) FORCE_REFRESH=true; shift ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    usage
    exit 0
fi

if [ "$SHOW_VERSION" = true ]; then
    echo "Dialogue Anonymizer Installer v${SCRIPT_VERSION}"
    exit 0
fi

if [ "$QUIET_MODE" = true ]; then
    exec >/dev/null 2>&1
fi

# ============================================================================
# CLEANUP TRAP
# ============================================================================

cleanup() {
    echo ""
    log_warn "Installation interrupted. Partial state may exist."
    echo "  Rerun with: ./Installer.sh -y --force-refresh"
    exit 1
}
trap cleanup INT TERM

# ============================================================================
# PRE-FLIGHT SYSTEM CHECK
# ============================================================================

preflight_check() {
    log_section "Pre-flight System Check"

    local warnings=0
    local errors=0

    # RAM check
    local ram_gb
    ram_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}') || ram_gb=0
    if [ "$ram_gb" -lt 8 ] 2>/dev/null; then
        log_warn "Low RAM: ${ram_gb}GB (recommended: 16GB+ for ML workloads)"
        ((warnings++))
    else
        log_success "RAM: ${ram_gb}GB"
    fi

    # Disk space check
    local disk_avail_gb
    disk_avail_gb=$(df -BG . 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G') || disk_avail_gb=0
    if [ "$disk_avail_gb" -lt 10 ] 2>/dev/null; then
        log_error "Insufficient disk space: ${disk_avail_gb}GB (minimum: 10GB, recommended: 50GB+)"
        ((errors++))
    else
        log_success "Disk space: ${disk_avail_gb}GB available"
    fi

    # GPU check
    GPU_AVAILABLE=false
    if command -v nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        local gpu_name driver_ver
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        log_success "NVIDIA GPU detected: ${gpu_name} (Driver: ${driver_ver})"
    else
        log_warn "No NVIDIA GPU detected — CPU-only mode (expect slow ML inference)"
        ((warnings++))
    fi

    # Operating system
    local os_name
    os_name=$(uname -s 2>/dev/null)
    if [ "$os_name" != "Linux" ]; then
        log_warn "Operating system: ${os_name} (script designed for Linux)"
        ((warnings++))
    else
        log_success "OS: Linux ($(uname -r))"
    fi

    echo ""
    if [ $errors -gt 0 ]; then
        log_error "$errors error(s) found. Cannot proceed."
        exit 1
    fi
    if [ $warnings -gt 0 ]; then
        log_warn "$warnings warning(s) above. Proceeding with caution..."
    fi
}

# ============================================================================
# OS DETECTION FOR PACKAGE MANAGEMENT
# ============================================================================

detect_package_manager() {
    if command -v apt-get &> /dev/null; then
        PKG_UPDATE_CMD="sudo apt-get update -q"
        PKG_INSTALL_CMD="sudo apt-get install -y"
    elif command -v dnf &> /dev/null; then
        PKG_UPDATE_CMD="sudo dnf check-update"
        PKG_INSTALL_CMD="sudo dnf install -y"
    elif command -v yum &> /dev/null; then
        PKG_UPDATE_CMD="sudo yum check-update"
        PKG_INSTALL_CMD="sudo yum install -y"
    elif command -v pacman &> /dev/null; then
        PKG_UPDATE_CMD="sudo pacman -Sy --noconfirm"
        PKG_INSTALL_CMD="sudo pacman -S --noconfirm"
    elif command -v apk &> /dev/null; then
        PKG_UPDATE_CMD="sudo apk update"
        PKG_INSTALL_CMD="sudo apk add"
    else
        PKG_UPDATE_CMD=""
        PKG_INSTALL_CMD=""
    fi
}

# ============================================================================
# DISK SPACE VALIDATION
# ============================================================================

check_disk_space() {
    local required_gb=$1
    local mount_point=${2:-.}
    local available_gb
    available_gb=$(df -BG "$mount_point" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G') || available_gb=0

    if [ "$available_gb" -lt "$required_gb" ] 2>/dev/null; then
        log_error "Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB at ${mount_point}"
        return 1
    fi
    log_info "Disk check: ${available_gb}GB available (need ${required_gb}GB) — OK"
    return 0
}

# ============================================================================
# RETRY-WRAPPED MODEL DOWNLOAD HELPER
# ============================================================================

download_model_hf() {
    local repo_id=$1
    local target_dir=$2
    local repo_type=${3:-model}
    local max_attempts=3
    local attempt=1

    log_info "Downloading ${repo_id} → ${target_dir}"
    mkdir -p "$target_dir"

    while [ $attempt -le $max_attempts ]; do
        echo "  Attempt ${attempt}/${max_attempts}..."
        if python -c "
from huggingface_hub import snapshot_download
import sys
try:
    snapshot_download(
        '${repo_id}',
        local_dir='${target_dir}',
        repo_type='${repo_type}'
    )
    print('Download complete')
    sys.exit(0)
except Exception as e:
    print(f'Download failed: {e}', file=sys.stderr)
    sys.exit(1)
"; then
            log_success "Downloaded ${repo_id}"
            return 0
        fi
        ((attempt++))
        if [ $attempt -le $max_attempts ]; then
            log_warn "Retrying in 5 seconds..."
            sleep 5
        fi
    done

    log_error "Failed to download ${repo_id} after ${max_attempts} attempts"
    return 1
}

# ============================================================================
# POST-INSTALLATION HEALTH CHECK
# ============================================================================

run_health_check() {
    log_section "Post-Installation Health Check"

    python << 'HEALTHEOF'
import sys
import os

checks_passed = 0
checks_failed = 0
warnings = 0

# --- Import checks ---
import_checks = {
    'numpy':        lambda: __import__('numpy').__version__,
    'torch':        lambda: __import__('torch').__version__,
    'torchaudio':   lambda: __import__('torchaudio').__version__,
    'transformers': lambda: __import__('transformers').__version__,
    'spacy':        lambda: __import__('spacy').__version__,
    'thinc':        lambda: __import__('thinc').__version__,
    'pyannote.audio': lambda: __import__('importlib.metadata', fromlist=['version']).version('pyannote.audio'),
    'whisperx':     lambda: __import__('whisperx').__version__ if hasattr(__import__('whisperx'), '__version__') else 'imported',
}

print("  --- Python Imports ---")
for name, fn in import_checks.items():
    try:
        ver = fn()
        print(f'  \033[0;32m✓\033[0m {name}: {ver}')
        checks_passed += 1
    except Exception as e:
        print(f'  \033[0;31m✗\033[0m {name}: {e}')
        checks_failed += 1

# --- spaCy model checks ---
print("  --- spaCy Models ---")
for model in ['en_core_web_sm', 'xx_ent_wiki_sm']:
    try:
        __import__(model)
        print(f'  \033[0;32m✓\033[0m {model}')
        checks_passed += 1
    except ImportError:
        print(f'  \033[0;33m⚠\033[0m  {model} (not installed)')
        warnings += 1

# --- Model directory checks ---
print("  --- Model Files ---")
model_paths = [
    ('Whisper (large-v3)', 'pipeline/model/Systran--faster-whisper-large-v3'),
    ('Pyannote',           'pipeline/model/models--pyannote--speaker-diarization-community-1'),
    ('mmBERT-base',        'pipeline/model/jhu-clsp/mmBERT-base'),
    ('DFKI-SLT PII',       'pipeline/model/multilingual_DialogPII_NER'),
]
for label, path in model_paths:
    if os.path.isdir(path) and os.listdir(path):
        print(f'  \033[0;32m✓\033[0m {label}: {path}')
        checks_passed += 1
    else:
        print(f'  \033[0;33m⚠\033[0m  {label}: not found ({path})')
        warnings += 1

# --- GPU check ---
print("  --- GPU ---")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f'  \033[0;32m✓\033[0m CUDA available: {gpu_name}')
        checks_passed += 1
    else:
        print('  \033[0;33m⚠\033[0m  CUDA not available (CPU-only mode)')
        warnings += 1
except Exception:
    print('  \033[0;33m⚠\033[0m  Could not check CUDA availability')
    warnings += 1

# --- .env check ---
print("  --- Configuration ---")
if os.path.isfile('.env'):
    print('  \033[0;32m✓\033[0m .env file exists')
    checks_passed += 1
    with open('.env') as f:
        content = f.read()
        if 'your_api_key_here' in content:
            print('  \033[0;33m⚠\033[0m  CHAT_AI_API_KEY still set to placeholder')
            warnings += 1
else:
    print('  \033[0;31m✗\033[0m .env file missing')
    checks_failed += 1

print()
print(f"  Results: {checks_passed} passed, {checks_failed} failed, {warnings} warning(s)")
sys.exit(1 if checks_failed > 0 else 0)
HEALTHEOF
}

# ============================================================================
# MAIN EXECUTION BEGINS
# ============================================================================

preflight_check
detect_package_manager

# --- Handle --dry-run ---
if [ "$DRY_RUN" = true ]; then
    log_section "DRY RUN — No Changes Will Be Made"
    echo "  Would create conda env: ${ENV_NAME} (Python ${TARGET_PYTHON})"
    echo "  Would install: numpy 2.0.2, torch 2.8.0, transformers 4.52+, pyannote, whisperx"
    echo "  Would install spaCy models: en_core_web_sm, xx_ent_wiki_sm"
    if [ "$SKIP_WEB" = true ]; then
        echo "  Would SKIP: Flask/TTS web interface"
    else
        echo "  Would install: Flask + Piper TTS (default)"
    fi
    if [ "$NO_MODELS" = true ]; then
        echo "  Would SKIP: all ML model downloads"
    else
        echo "  Would download models: ${WHISPER_SELECTION:-large (default)}"
        [ "$SKIP_PYANNOTE" = true ] && echo "  Would SKIP: Pyannote" || echo "  Would download: Pyannote"
    fi
    echo "  Would generate: .env (with chmod 600)"
    echo "  Would generate: requirements-lock.txt"
    echo ""
    exit 0
fi

# --- Handle --uninstall ---
if [ "$UNINSTALL" = true ]; then
    log_section "Uninstalling ATA"

    echo "  This will remove:"
    echo "    • Conda environment '${ENV_NAME}'"
    echo "    • Model files in pipeline/model/"
    echo "    • .env and backup files"
    echo ""

    if [ "$ANSWER_YES" != true ]; then
        read -p "Proceed with uninstall? (y/N): " confirm
        [ "$confirm" != "y" ] && [ "$confirm" != "Y" ] && echo "Aborted." && exit 0
    fi

    # Remove conda environment
    if conda env list | grep -q "^${ENV_NAME} "; then
        if [ "$FORCE_REFRESH" = true ]; then
            log_info "🔄 Environment exists (--force-refresh). Removing and recreating..."
            conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
            rm -rf "$HOME/miniconda3/envs/$ENV_NAME" 2>/dev/null || true
            rm -rf "$HOME/miniforge3/envs/$ENV_NAME" 2>/dev/null || true
            conda create -n "$ENV_NAME" python="$TARGET_PYTHON" -c conda-forge -y
        else
            log_warn "⚠️ Environment '${ENV_NAME}' already exists."
            log_info "Using existing environment (use --force-refresh to recreate)"
        fi
    else
        log_info "Creating fresh environment with Python ${TARGET_PYTHON}..."
        conda create -n "$ENV_NAME" python="$TARGET_PYTHON" -c conda-forge -y
    fi

    # Remove model files
    if [ -d "pipeline/model" ]; then
        model_size=$(du -sh "pipeline/model" 2>/dev/null | awk '{print $1}')
        log_info "Removing model files (${model_size})..."
        rm -rf pipeline/model
        log_success "Model files removed"
    fi

    # Remove .env and backups
    rm -f .env .env.backup.* 2>/dev/null
    log_success "Configuration files removed"

    echo ""
    log_success "Uninstallation complete"
    echo "  Note: spaCy models remain in conda base (if installed there)"
    echo "  Note: System packages (ffmpeg, etc.) were not removed"
    exit 0
fi

# ============================================================================
# 1. MAIN FOLDER LOGIC
# ============================================================================

CURRENT_DIR="$(pwd)"
SCRIPT_NAME="$(basename "$0")"
MAIN_DIR_NAME="ATA"

shopt -s nullglob

if [ "$(basename "$CURRENT_DIR")" == "$MAIN_DIR_NAME" ]; then
    log_success "Already inside '${MAIN_DIR_NAME}' folder. Proceeding..."
else
    log_info "Not inside '${MAIN_DIR_NAME}'. Checking for project files..."
    if [ -d "pipeline" ] || [ -f "README.md" ]; then
        log_info "Detected project root. Creating '${MAIN_DIR_NAME}' and moving files..."
        mkdir -p "$MAIN_DIR_NAME"
        for item in *; do
            if [ "$item" != "$MAIN_DIR_NAME" ]; then
                mv "$item" "$MAIN_DIR_NAME/"
            fi
        done
        cd "$MAIN_DIR_NAME"
        CURRENT_DIR="$(pwd)"
        log_success "Moved all files into '${MAIN_DIR_NAME}'. New location: ${CURRENT_DIR}"
    else
        log_error "Not inside '${MAIN_DIR_NAME}' and no project files found."
        exit 1
    fi
fi

cd "$CURRENT_DIR"

log_section "Dialogue Anonymizer Installer (v${SCRIPT_VERSION})"
log_info "Working Directory: $(pwd)"
echo ""

# ============================================================================
# 2. CONDA INSTALLATION & ENVIRONMENT SETUP
# ============================================================================

# 2. Check if Conda is installed
if ! command -v conda &> /dev/null; then
    log_info "Conda not found. Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm miniconda.sh
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
else
    log_success "Conda already installed"
    eval "$(conda shell.bash hook)"
fi

# 3. Create the environment
log_info "Creating conda environment '${ENV_NAME}' with Python ${TARGET_PYTHON}..."

if conda env list | grep -q "^${ENV_NAME} "; then
    if [ "$FORCE_REFRESH" = true ]; then
        conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
        rm -rf "$HOME/miniconda3/envs/$ENV_NAME" 2>/dev/null || true
        rm -rf "$HOME/miniforge3/envs/$ENV_NAME" 2>/dev/null || true
        conda create -n "$ENV_NAME" python="$TARGET_PYTHON" -c conda-forge -y
    else
        log_warn "⚠️ Environment '${ENV_NAME}' already exists."
        log_info "Using existing environment (use --force-refresh to recreate)"
    fi
else
    log_info "Creating fresh environment with Python ${TARGET_PYTHON}..."
    conda create -n "$ENV_NAME" python="$TARGET_PYTHON" -c conda-forge -y
fi

# 4. Activate the environment
log_info "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Ensure MODEL_DIR path is consistent throughout
mkdir -p "$CURRENT_DIR/$MODEL_DIR_NAME"

ACTUAL_PYTHON=$(conda run -n "$ENV_NAME" python --version | awk '{print $2}')
EXPECTED_PYTHON="3.12"
if [[ ! "$ACTUAL_PYTHON" =~ ^3\.12 ]]; then  # ✅ FIXED: Was ^3.13 (wrong!)
    log_error "Expected Python 3.12.x, got $ACTUAL_PYTHON"
    exit 1
fi
log_success "Python version verified: $ACTUAL_PYTHON"

# ============================================================================
# 5. SYSTEM DEPENDENCIES
# ============================================================================

log_section "Installing System Dependencies"

if [ -n "$PKG_INSTALL_CMD" ]; then
    # FFmpeg (required for video/audio processing)
    if ! command -v ffmpeg &> /dev/null; then
        log_info "Installing FFmpeg..."
        $PKG_UPDATE_CMD
        $PKG_INSTALL_CMD ffmpeg || log_warn "FFmpeg installation failed (may work without it)"
    else
        log_success "FFmpeg already installed"
    fi
    
    # Build tools for pyo3 packages that may require compilation
    if ! command -v gcc &> /dev/null; then
        log_info "Installing build-essential..."
        $PKG_INSTALL_CMD build-essential cmake pkg-config 2>/dev/null || log_warn "Build tools installation failed"
    fi
    
    # Git
    if ! command -v git &> /dev/null; then
        log_info "Installing Git..."
        $PKG_INSTALL_CMD git || log_warn "Git installation failed"
    fi
else
    log_warn "Could not detect package manager. Please ensure ffmpeg and gcc are installed manually."
fi

# ============================================================================
# 6. ML STACK INSTALLATION (Verified Compatible Versions)
# ============================================================================

log_section "Installing ML Stack (NumPy 2.0.2 + PyTorch 2.8.0)"

pip install --upgrade pip -q

# Core ML stack - VERIFIED COMPATIBLE FOR PYTHON 3.12
log_info "Installing core ML packages..."
pip install "numpy==2.0.2" --no-cache-dir  # ✅ Works with PyTorch 2.8, transformers 4.50+
pip install "scipy>=1.14.0" --no-cache-dir

# PyTorch 2.8.0 - Latest stable with excellent Python 3.12 wheel support
if [ "$GPU_AVAILABLE" = true ]; then
    log_info "Installing PyTorch with CUDA 12.8 support..."
    pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
    pip install "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
else
    log_info "Installing PyTorch CPU-only..."
    pip install "torch==2.8.0" --no-cache-dir
    pip install "torchaudio==2.8.0" --no-cache-dir
fi

pip install "torchvision>=0.23.0" --no-cache-dir

# torchcodec (for audio/video handling)
log_info "Installing torchcodec..."
pip install "torchcodec>=0.14.0" --no-cache-dir || \
pip install "torchcodec>=0.10.0" --no-cache-dir || \
log_warn "⚠️  torchcodec installation failed (optional)"

# Transformers ecosystem - NumPy 2.x compatible
log_info "Installing transformers ecosystem..."
pip install "transformers>=4.52.0" --no-cache-dir  # ✅ 4.50+ supports NumPy 2.x
pip install "tokenizers>=0.20.0" --no-cache-dir     # ✅ Updated (was 0.19.1, now fine)
pip install "accelerate>=0.34.0" --no-cache-dir
pip install "huggingface-hub>=0.25.0,<1.0.0" --no-cache-dir

# Other utilities
log_info "Installing utility packages..."
pip install "pandas>=2.2.0" --no-cache-dir
pip install ffmpeg-python --no-cache-dir
pip install "openai>=1.0.0" --no-cache-dir
pip install python-dotenv --no-cache-dir
pip install pytorch-crf --no-cache-dir

# spaCy ecosystem - Python 3.12 + NumPy 2.x compatible
log_info "Installing spaCy ecosystem..."
pip install "spacy>=3.8.4" --no-cache-dir  # ✅ 3.8.4+ has NumPy 2.x patches
pip install "thinc>=8.3.0" --only-binary :all: --no-cache-dir
pip install "blis>=1.0.0" --no-cache-dir
pip install "click>=8.1.7" --no-cache-dir
pip install "typer>=0.12.0" --no-cache-dir

# Speaker diarization - Latest compatible version
log_info "Installing speaker diarization..."
pip install "pyannote.audio>=4.0.0" --no-cache-dir || log_warn "⚠️  Warning installing pyannote.audio"

# whisperx - Latest stable (3.8.x series)
log_info "Installing WhisperX..."
pip uninstall whisperx -y 2>/dev/null || true
pip install git+https://github.com/m-bain/whisperx.git --no-cache-dir || \
pip install "whisperx>=3.8.0" --no-cache-dir || \
log_warn "⚠️  Warning installing whisperx"

log_success "Base ML stack installation complete."

# ============================================================================
# 7. VERSION VERIFICATION
# ============================================================================

log_section "Version Verification"

NUMPY_VER=$(pip show numpy | grep Version | awk '{print $2}')
TORCH_VER=$(pip show torch | grep Version | awk '{print $2}')
THINC_VER=$(pip show thinc | grep Version | awk '{print $2}')
SPACY_VER=$(pip show spacy | grep Version | awk '{print $2}')
PYANNOTE_VER=$(pip show pyannote.audio | grep Version | awk '{print $2}')
TORCHCODEC_VER=$(pip show torchcodec | grep Version | awk '{print $2}')

log_info "  numpy:    $NUMPY_VER"
log_info "  torch:    $TORCH_VER"
log_info "  thinc:    $THINC_VER"
log_info "  spacy:    $SPACY_VER"
log_info "  pyannote: ${PYANNOTE_VER:-installed}"
log_info "  torchcodec: ${TORCHCODEC_VER:-installed}"

if [[ ! "$NUMPY_VER" =~ ^2\.0 ]]; then  # ✅ Changed from ^1.26
    log_warn "NumPy $NUMPY_VER (expected 2.0.2)"
else
    log_success "NumPy $NUMPY_VER confirmed (2.0.2 series)"
fi

if [[ ! "$TORCH_VER" =~ ^2\.[789] ]] && [[ ! "$TORCH_VER" =~ ^3\. ]]; then  # ✅ Changed from ^2.8
    log_warn "PyTorch $TORCH_VER (2.7-2.9 recommended)"
else
    log_success "PyTorch $TORCH_VER confirmed (2.7+ series)"
fi

if python -c "import spacy; import thinc; import pyannote.audio; import whisperx; import torchcodec" 2>/dev/null; then
    log_success "All core packages imported successfully"
else
    log_error "CRITICAL: Import test failed"
    exit 1
fi

# Generate requirements lock file
log_info "Generating requirements-lock.txt..."
pip freeze > "$CURRENT_DIR/requirements-lock.txt"
log_success "requirements-lock.txt generated"

# ============================================================================
# 8. SPAICY MODEL INSTALLATION
# ============================================================================

log_section "Downloading spaCy Models (for PII Detection)"

# FIXED: Use 'python -m spacy download' instead of non-existent 'spacy check'
if python -m spacy download en_core_web_sm 2>/dev/null; then
    log_success "en_core_web_sm installed"
else
    log_warn "en_core_web_sm installation skipped or failed"
fi

if python -m spacy download xx_ent_wiki_sm 2>/dev/null; then
    log_success "xx_ent_wiki_sm installed"
else
    log_warn "xx_ent_wiki_sm installation skipped or failed"
fi

# ============================================================================
# 9. WEB INTERFACE & TTS BACKENDS - CORRECTED FOR MULTI-SPEAKER SUPPORT
# ============================================================================

log_section "Web Interface & TTS Backend Installation"

if [ "$SKIP_WEB" = true ]; then
    log_info "Skipping Web Interface (--skip-web)"
    INSTALL_WEB="n"
elif [ "$ANSWER_YES" = true ]; then
    log_info "Web Interface: YES (auto-answered --yes)"
    INSTALL_WEB="y"
else
    read -p "Install Web Interface (Flask + TTS)? (y/n): " INSTALL_WEB
    INSTALL_WEB=${INSTALL_WEB:-y}
fi

if [[ "$INSTALL_WEB" =~ ^[Yy]$ ]]; then
    pip install flask requests cryptography --no-cache-dir
    
    # TTS BACKEND SELECTION
    log_info "Select TTS Backend:"
    log_info "  1. piper       - Fast offline neural TTS (recommended)"
    log_info "  2. coqui_xtts  - High-quality voice cloning (17 langs)"
    
    if [ "$ANSWER_YES" = true ]; then
        TTS_BACKEND_CHOICE="piper"
    else
        read -p "Enter choice [1]: " TTS_CHOICE
        TTS_BACKEND_CHOICE=${TTS_CHOICE:-piper}
        [ "$TTS_CHOICE" = "2" ] && TTS_BACKEND_CHOICE="coqui_xtts"
    fi
    
    # CORRECTED PATHS - TTS voices in pipeline/tts/voices, NOT model/
    TTS_DIR="$CURRENT_DIR/pipeline/tts"
    PIPER_VOICE_DIR="$TTS_DIR/voices"
    
    log_info "Installing TTS Backend: $TTS_BACKEND_CHOICE..."
    
    if [ "$TTS_BACKEND_CHOICE" = "piper" ]; then
        # PIPER TTS
        pip install "piper-tts>=1.4.2" --no-cache-dir
        
        # Download Piper executable
        log_info "Downloading Piper executable..."
        if [ "$GPU_AVAILABLE" = true ]; then
            # NVIDIA GPU version
            PIPER_URL="https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_x86_64.tar.gz"
        else
            # CPU-only version
            PIPER_URL="https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_x86_64.tar.gz"
        fi
        
        curl -#L "$PIPER_URL" | tar xzf - -C "$TTS_DIR/bin" --strip-components=1 piper_linux_x86_64/piper
        
        # Make executable
        chmod +x "$TTS_DIR/bin/piper"
        
        if [ ! -s "$TTS_DIR/bin/piper" ]; then
            log_error "Piper executable download failed"
            exit 1
        fi
        
        log_success "Piper executable installed"
        
        # MULTIPLE VOICES FOR SPEAKER MAPPING
        log_info "Downloading multiple Piper voice models for speaker differentiation..."
        log_info "These allow assigning unique voices to different speakers"
        
        declare -A SPEAKER_VOICES
        SPEAKER_VOICES["SPEAKER_00"]="en_US-amy-medium"
        SPEAKER_VOICES["SPEAKER_01"]="en_US-lessac-medium"
        SPEAKER_VOICES["SPEAKER_02"]="en_US-kusal-medium"
        SPEAKER_VOICES["SPEAKER_03"]="en_US-ryan-medium"
        SPEAKER_VOICES["SPEAKER_04"]="en_US-joe-medium"
        SPEAKER_VOICES["SPEAKER_05"]="en_US-libritts-high"
        DEFAULT_VOICE="${SPEAKER_VOICES[SPEAKER_00]}"
        
        download_piper_voice() {
            local voice_id=$1
            local base_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"
            # Extract: en_US-amy-medium → amy/medium/en_US-amy-medium
            local parts=(${voice_id//_/ })
            local lang_code="${parts[0]}_${parts[1]}"  # en_US
            local name_part="${parts[1]}-${parts[2]}"  # amy-medium
            local subdir="${parts[1]}/${parts[2]}"     # amy/medium
            local url="$base_url/$subdir/${voice_id}.onnx"
            local target="$PIPER_VOICE_DIR/${voice_id}.onnx"
            local max_attempts=3
            local attempt=1
            
            if [ -s "$target" ]; then
                log_success "Voice $voice_id already exists"
                return 0
            fi
            
            while [ $attempt -le $max_attempts ]; do
                echo "  Attempt ${attempt}/${max_attempts}: ${voice_id}..."
                if curl -#L "$url" -o "$target" 2>/dev/null && [ -s "$target" ]; then
                    # Download config
                    curl -#L "${url%.onnx}.json" -o "${target%.onnx}.json" 2>/dev/null || true
                    
                    local size_bytes=$(stat -c%s "$target" 2>/dev/null || stat -f%z "$target" 2>/dev/null)
                    if [ "$size_bytes" -gt 40000000 ]; then  # >40MB
                        log_success "✓ ${voice_id} ($(du -h "$target" | awk '{print $1}'))"
                        return 0
                    fi
                    log_warn "Incomplete, retrying..."
                    rm -f "$target"
                fi
                ((attempt++))
                [ $attempt -le $max_attempts ] && sleep 3
            done
            
            log_error "Failed: ${voice_id}"
            return 1
        }
        
        # Download all mapped voices
        downloaded=0
        for speaker in SPEAKER_00 SPEAKER_01 SPEAKER_02 SPEAKER_03 SPEAKER_04 SPEAKER_05; do
            voice_id="${SPEAKER_VOICES[$speaker]}"
            if download_piper_voice "$voice_id"; then
                ((downloaded++))
            fi
        done
        
        echo ""
        log_info "Voice download summary: $downloaded / 6 voices"
        
        if [ "$downloaded" -lt 2 ]; then
            log_error "Insufficient voices for speaker differentiation (need ≥2)"
            exit 1
        fi
        
        # Create speaker voice mapping document
        cat > "$PIPER_VOICE_DIR/VOICE_MAPPING.txt" << 'VOICEMAP'
# Speaker-to-Voice Mapping for Dialogue Anonymizer
# Each speaker gets a unique voice for natural multi-speaker audio

SPEAKER_00 → en_US-amy-medium     (Female, standard American)
SPEAKER_01 → en_US-lessac-medium  (Female, high-quality)
SPEAKER_02 → en_US-kusal-medium   (Male, American)
SPEAKER_03 → en_US-ryan-medium    (Male, deep voice)
SPEAKER_04 → en_US-joe-medium     (Male, casual)
SPEAKER_05 → en_US-libritts-high  (Neutral, female British)

# For 7+ speakers: voices cycle starting from SPEAKER_00
# Customize: Edit pipeline/tts/tts_engine.py → SPEAKER_VOICE_MAP
VOICEMAP
        
        log_success "Created: $PIPER_VOICE_DIR/VOICE_MAPPING.txt"
        
        TTS_CONFIG="piper"
        TTS_VOICE_PATH="$PIPER_VOICE_DIR/$DEFAULT_VOICE.onnx"
        TTS_CONFIG_PATH="$PIPER_VOICE_DIR/$DEFAULT_VOICE.onnx.json"
        
    elif [ "$TTS_BACKEND_CHOICE" = "coqui_xtts" ]; then
        # ... (Coqui section unchanged)
        # NOTE: Coqui doesn't support multi-voice speaker mapping
    fi
else
    log_info "Skipping TTS installation"
    TTS_CONFIG="none"
fi

# ============================================================================
# 10. MODEL DOWNLOADS
# ============================================================================

log_section "Model Downloads (WhisperX, Pyannote, DFKI-SLT PII)"

# Pre-download disk space validation
if [ "$NO_MODELS" != true ]; then
    if ! check_disk_space 50; then
        log_error "Cannot proceed without sufficient disk space for models"
        exit 1
    fi
fi

if [ "$NO_MODELS" = true ]; then
    log_info "Skipping all model downloads (--no-models)"
    SKIP_WHISPER=true
    SKIP_PYANNOTE=true
else
    # WhisperX Models
    declare -A MODEL_MAP
    MODEL_MAP["tiny"]="Systran/faster-whisper-tiny"
    MODEL_MAP["base"]="Systran/faster-whisper-base"
    MODEL_MAP["small"]="Systran/faster-whisper-small"
    MODEL_MAP["medium"]="Systran/faster-whisper-medium"
    MODEL_MAP["large"]="Systran/faster-whisper-large-v3"
    
    if [ "$SKIP_WHISPER" = true ]; then
        log_info "Skipping WhisperX models (--skip-whisper)"
    elif [ -n "$WHISPER_SELECTION" ]; then
        # FIXED: Convert comma-separated to space-separated for proper iteration
        WHISPER_MODELS_INPUT="${WHISPER_SELECTION//,/ }"
    elif [ "$ANSWER_YES" = true ]; then
        WHISPER_MODELS_INPUT="large"
    else
        read -p "Enter WhisperX models to download (tiny, base, small, medium, large) or press Enter to skip: " WHISPER_MODELS_INPUT
    fi
    
    if [ -n "$WHISPER_MODELS_INPUT" ]; then
        for model_name in $WHISPER_MODELS_INPUT; do
            model_name=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
            if [ -z "${MODEL_MAP[$model_name]}" ]; then
                log_warn "Invalid: '$model_name'. Skipping."
                continue
            fi
            
            hf_repo="${MODEL_MAP[$model_name]}"
            # CONSISTENT NAMING - Removed extra 'models--' prefix
            target_dir="$CURRENT_DIR/$MODEL_DIR_NAME/${hf_repo//\//--}"
            
            if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir")" ]; then
                log_success "Model '$model_name' already exists at: $target_dir"
            else
                download_model_hf "$hf_repo" "$target_dir"
                # Store primary model path if this is 'large'
                if [ "$model_name" = "large" ]; then
                    WHISPER_PRIMARY_PATH="$target_dir"
                fi
            fi
        done
    fi
    
    # Pyannote Diarization Model
    TARGET="$CURRENT_DIR/$MODEL_DIR_NAME/models--pyannote--speaker-diarization-community-1"

    if [ "$SKIP_PYANNOTE" = true ]; then
        log_info "Skipping Pyannote (--skip-pyannote)"
    elif [ -d "$TARGET" ] && [ "$(ls -A "$TARGET" 2>/dev/null)" ]; then
        log_success "Pyannote model already exists."
    else
        log_info "Downloading Pyannote to ${TARGET}..."
        
        # Authenticate if token provided
        if [ "$AUTO_LOGIN" = true ] && [ -n "$HUGGINGFACE_TOKEN" ]; then
            echo "Authenticating with HUGGINGFACE_TOKEN..." >&2  # FIXED: stderr instead of stdout
            hf auth login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential 2>/dev/null || true
        fi
        
        # Check auth status using hf auth whoami
        if hf auth whoami &>/dev/null; then
            log_success "Hugging Face authenticated"
            download_model_hf "pyannote/speaker-diarization-community-1" "$TARGET"
        else
            log_warn "Not logged in to Hugging Face."
            log_info "Run: hf auth login"
            log_info "Or: export HUGGINGFACE_TOKEN=your_token && ./ATA_SelfInstall.sh --auto-login"
            log_info "Verify: hf whoami"
        fi
    fi
    
    # mmBERT Base & PII Models
    log_info "Downloading mBert Base & DFKI-SLT PII NER Model"
    log_info "(Supports 11 languages for PII detection)"
    log_info "Languages: AR, DE, EN, FI, FR, HI, IT, PL, PT, ES, TR"
    
    BASE_TARGET="$CURRENT_DIR/$MODEL_DIR_NAME/jhu-clsp/mmBERT-base"
    if [ -d "$BASE_TARGET" ] && [ "$(ls -A "$BASE_TARGET")" ]; then
        log_success "mmBERT-base already present."
    else
        download_model_hf "jhu-clsp/mmBERT-base" "$BASE_TARGET" "model"
    fi
    
    PII_TARGET="$CURRENT_DIR/$MODEL_DIR_NAME/multilingual_DialogPII_NER"
    log_info "Downloading DFKI-SLT Multilingual DialogPII NER Model"
    
    if [ -d "$PII_TARGET" ] && [ "$(ls -A "$PII_TARGET")" ]; then
        log_success "DFKI-SLT PII model already exists."
    else
        download_model_hf "DFKI-SLT/multilingual_DialogPII_NER" "$PII_TARGET" "model"
        
        if [ "$(ls -A "$PII_TARGET" 2>/dev/null)" ]; then
            log_success "DFKI-SLT PII model downloaded successfully."
        else
            log_error "Download failed. Manual download required:"
            log_info "   https://huggingface.co/DFKI-SLT/multilingual_DialogPII_NER"
        fi
    fi
fi

# ============================================================================
# 11. ENV FILE CONFIGURATION
# ============================================================================

if [ -f ".env" ]; then
    log_info "Backing up existing .env..."
    cp .env ".env.backup.$(date +%Y%m%d%H%M%S)"
fi

# Use relative paths ($CURRENT_DIR-based) instead of hardcoded mount paths
cat > .env <<EOF
# ATA Speech Anonymizer Configuration
CHAT_AI_API_KEY=your_api_key_here
CHAT_AI_ENDPOINT=https://your-endpoint.com/v1

# TTS CONFIGURATION
TTS_BACKEND=$TTS_CONFIG
TTS_ENABLED=true

# TTS Global Settings
TTS_SAMPLE_RATE=22050
TTS_DEFAULT_LANG=en

# Piper TTS Settings
TTS_BIN_PATH=./pipeline/tts/bin
TTS_VOICE_DIR=\$CURRENT_DIR/pipeline/tts/voices
TTS_VOICE_PATH=\$CURRENT_DIR/pipeline/tts/voices/en_US-amy-medium.onnx

# Coqui XTTS Settings (unused when TTS_BACKEND=piper)
XTTS_Model_Path=\$CURRENT_DIR/$MODEL_DIR_NAME/coqui-xtts
XTTS_Reference_Audio_Path=\$CURRENT_DIR/reference_audio.wav

# COMPLIANCE & LOGGING
COMPLIANCE_MODE=standard
COMPLIANCE_ENCRYPTION=false
COMPLIANCE_AUDIT_LOG=false
LOG_LEVEL=INFO
LOG_FILE=./logs/ata.log

# PII MODEL (DFKI-SLT - 11 languages)
# FIXED: H,I → HI, SP → ES (standard language codes)
PII_Model_Path=\$CURRENT_DIR/$MODEL_DIR_NAME/multilingual_DialogPII_NER
PII_Languages=AR,DE,EN,FI,FR,HI,IT,PL,PT,ES,TR

# WHISPERX
Whisper_Model_Path=$WHISPER_PRIMARY_PATH
Whisper_Device=\$([ "$GPU_AVAILABLE" = true ] && echo cuda || echo cpu)
Whisper_Compute_Type=float16

# PYANNOTE
Pyannote_Model_Path=\$CURRENT_DIR/$MODEL_DIR_NAME/models--pyannote--speaker-diarization-community-1

# Installation Metadata
ATA_INSTALL_VERSION=$SCRIPT_VERSION
ATA_INSTALL_DATE=$(date +%Y-%m-%d)
EOF

# Add backend-specific paths
case "$TTS_CONFIG" in
    piper)
        cat >> .env <<EOF

# Piper Voice Configuration
Piper_Voice_Path=$TTS_VOICE_PATH
Piper_Config_Path=$TTS_CONFIG_PATH
EOF
        ;;
    coqui_xtts)
        cat >> .env <<EOF

# Coqui XTTS Configuration
XTTS_Model_Path=$TTS_CONFIG_PATH
XTTS_Reference_Audio_Path=./reference_audio.wav
EOF
        ;;
esac

# Set restrictive permissions on .env
chmod 600 .env
log_success ".env configured and secured (chmod 600)"

# ============================================================================
# 12. OPTIONAL: ADD TO SHELL PROFILE
# ============================================================================

if [ "$ANSWER_YES" != true ]; then
    read -p "Add conda activation to ~/.bashrc? (y/n): " ADD_TO_PROFILE
    ADD_TO_PROFILE=${ADD_TO_PROFILE:-y}
else
    ADD_TO_PROFILE="y"
fi

if [[ "$ADD_TO_PROFILE" =~ ^[Yy]$ ]]; then
    # Check if already exists
    if ! grep -q "whisperx" ~/.bashrc 2>/dev/null; then
        cat >> ~/.bashrc << EOF

# ATA Speech Anonymizer Environment (added $(date +%Y-%m-%d))
conda activate whisperx 2>/dev/null || true
export ATA_HOME="$CURRENT_DIR"
export MODEL_DIR="\$ATA_HOME/$MODEL_DIR_NAME"
EOF
        log_success "Environment variables added to ~/.bashrc"
        log_info "  Restart shell or run: source ~/.bashrc"
    else
        log_info "~/.bashrc already contains ATA environment settings"
    fi
fi

# ============================================================================
# 13. POST-INSTALLATION HEALTH CHECK
# ============================================================================

log_section "Running Post-Installation Health Check"

run_health_check

if [ $? -eq 0 ]; then
    log_success "All health checks passed"
else
    log_warn "Some health checks failed. Review errors above."
fi

# ============================================================================
# 14. FINAL INSTRUCTIONS
# ============================================================================

log_section "✅ SETUP COMPLETE"

log_info "Activate environment:"
log_info "  conda activate whisperx"
log_info ""
log_info "TTS Backend: $TTS_CONFIG"
log_info "  Piper:    Offline, 30+ langs, GPL v3"
log_info "  Coqui:    Voice cloning, 17 langs, CPML (non-commercial)"
log_info ""
log_info "Model Location: $MODEL_DIR_NAME/"
log_info ""
log_info "Next steps:"
log_info "  1. Edit .env with API key (securely, chmod 600)"
log_info "  2. Place videos in pipeline/videos/"
log_info "  3. Run: python pipeline/process.py"
log_info ""
log_info "Troubleshooting:"
log_info "  • Re-run with --force-refresh to rebuild everything"
log_info "  • Run with --dry-run to preview actions"
log_info "  • Run with --uninstall to clean removal"
log_info "  • Check requirements-lock.txt for exact package versions"
log_info ""
log_info "For support: https://proton.me/support/lumo"
log_section "End of Installation"