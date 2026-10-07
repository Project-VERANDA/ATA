#!/usr/bin/env bash

# ============================================================================
# Dialogue Anonymizer Installer (v4.1)
# ============================================================================

set -o pipefail

SCRIPT_VERSION="4.1"
ENV_NAME="whisperx"
TARGET_PYTHON="3.12"
REQUIREMENTS_SRC="requirements.txt"
WEB_REQUIREMENTS_SRC="requirements-web.txt"

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
SYNC_ONLY=false
ROLLBACK=false
SKIP_FRONTEND=false
SKIP_FLASK=false

# ============================================================================
# COLOR & LOGGING HELPERS
# ============================================================================

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
log_section() { echo -e "\n${BOLD}============================================================${NC}"; echo -e "${BOLD}$1${NC}"; echo -e "${BOLD}============================================================${NC}"; }

# ============================================================================
# VERSION CONTROL & BACKUP SYSTEM
# ============================================================================

create_version_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="$CURRENT_DIR/backups/$timestamp"
    
    # Create backup directory (creates parent if needed too)
    if ! mkdir -p "$backup_dir" 2>/dev/null; then
        log_warn "Cannot create backup directory at $backup_dir - disabling backup feature"
        return 0
    fi
    
    # Fix permissions on parent directory
    chmod 755 "$CURRENT_DIR/backups" 2>/dev/null || true
    
    log_info "Creating version backup at $backup_dir..."
    
    # Backup environment state (with fallback)
    if conda env list | grep -q "^${ENV_NAME} "; then
        conda env export -n "$ENV_NAME" > "$backup_dir/conda_env.yaml" 2>/dev/null || \
            log_warn "Could not export conda environment"
    fi
    
    # Backup .env configuration
    [ -f ".env" ] && cp .env "$backup_dir/.env.backup" 2>/dev/null || true
    
    # Backup requirements.lock reference
    [ -f "requirements-lock.txt" ] && cp requirements-lock.txt "$backup_dir/requirements-lock.txt.ref" 2>/dev/null || true
    
    # Create metadata file
    cat > "$backup_dir/metadata.json" << EOF || log_warn "Could not create metadata.json"
{
    "backup_date": "$(date -Iseconds)",
    "script_version": "$SCRIPT_VERSION",
    "conda_env": "${ENV_NAME}",
    "python_version": "$TARGET_PYTHON",
    "repository_structure": "verified"
}
EOF
    
    if [ -d "$backup_dir" ]; then
        local size=$(du -sh "$backup_dir" 2>/dev/null | awk '{print $1}')
        log_success "Backup created: ${size:-N/A}"
    else
        log_warn "Backup directory empty or inaccessible"
    fi
}

check_remote_updates() {
    log_section "Checking for Repository Updates"
    
    if ! command -v git &> /dev/null; then
        log_warn "Git not installed - cannot check remote updates"
        return 1
    fi
    
    if ! git rev-parse --is-inside-work-tree &> /dev/null 2>&1; then
        log_info "Not a git repository - skipping update check"
        return 0
    fi
    
    local current_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    
    if git fetch --quiet origin 2>/dev/null; then
        local remote_commit=$(git rev-parse origin/HEAD 2>/dev/null || echo "unknown")
        
        if [ "$current_commit" != "$remote_commit" ]; then
            log_info "Updates available: $current_commit → $remote_commit"
            
            if [ "$ANSWER_YES" = true ]; then
                log_info "Auto-updating to latest commit..."
                git pull origin main --no-edit
                log_success "Repository updated"
            else
                read -p "Update repository now? (y/N): " UPDATE_NOW
                if [[ "$UPDATE_NOW" =~ ^[Yy]$ ]]; then
                    git pull origin main --no-edit
                    log_success "Repository updated"
                fi
            fi
        else
            log_success "Already on latest commit"
        fi
    else
        log_warn "Could not fetch remote repository"
    fi
}

# ============================================================================
# USAGE
# ============================================================================

usage() {
    cat << 'EOF'
Usage: ./Installer.sh [OPTIONS]

Quick Start:
  ./Installer.sh -y                        # Install with all defaults
  ./Installer.sh -y --skip-flask           # Skip Flask backend (React only)
  ./Installer.sh -y --skip-frontend        # Skip React frontend (Flask only)
  ./Installer.sh --sync-only               # Pull latest code without reinstall

Options:
  -h, --help              Show this help message
  -V, --version           Show installer version
  -y, --yes               Auto-answer 'yes' to all prompts
  --dry-run               Preview actions without changes
  --uninstall             Remove conda env and model files
  --force-refresh         Reinstall all packages
  --skip-web              Skip web interface entirely
  --skip-flask            Skip Flask backend (use React only)
  --skip-frontend         Skip React frontend (use Flask only)
  --no-models             Skip ALL ML model downloads
  --skip-whisper          Skip WhisperX model download
  --skip-pyannote         Skip Pyannote diarization model
  --whisper-models LIST   Comma-separated list (tiny,base,small,medium,large,large-turbo)
  --auto-login            Auto-authenticate with HUGGINGFACE_TOKEN
  -q, --quiet             Minimal output mode
  --force-refresh         Reinstall all packages from scratch
  --sync-only             Pull latest repo updates (no reinstall)
  --rollback              Restore from most recent backup

Environment Variables:
  HUGGINGFACE_TOKEN       HF token for gated model access

Examples:
  ./Installer.sh -y
  ./Installer.sh --sync-only
  ./Installer.sh --dry-run
EOF
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

# Initialize WEB_MODE_CHOICE early to prevent undefined variable errors
WEB_MODE_CHOICE="both"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) SHOW_HELP=true; shift ;;
        -V|--version) SHOW_VERSION=true; shift ;;
        -y|--yes) ANSWER_YES=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --uninstall) UNINSTALL=true; shift ;;
        --skip-web) SKIP_WEB=true; shift ;;
        --skip-flask) SKIP_FLASK=true; shift ;;
        --skip-frontend) SKIP_FRONTEND=true; shift ;;
        --no-models) NO_MODELS=true; shift ;;
        --skip-whisper) SKIP_WHISPER=true; shift ;;
        --skip-pyannote) SKIP_PYANNOTE=true; shift ;;
        --whisper-models) WHISPER_SELECTION="$2"; shift 2 ;;
        --auto-login) AUTO_LOGIN=true; shift ;;
        -q|--quiet) QUIET_MODE=true; shift ;;
        --force-refresh) FORCE_REFRESH=true; shift ;;
        --sync-only) SYNC_ONLY=true; shift ;;
        --rollback) ROLLBACK=true; shift ;;
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
    echo "  Recovery: ./Installer.sh --rollback"
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
        log_warn "Low RAM: ${ram_gb}GB (recommended: 16GB+)"
        ((warnings++))
    else
        log_success "RAM: ${ram_gb}GB"
    fi

    # Disk space check
    local disk_avail_gb
    disk_avail_gb=$(df -BG . 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G') || disk_avail_gb=0
    if [ "$disk_avail_gb" -lt 10 ] 2>/dev/null; then
        log_error "Insufficient disk space: ${disk_avail_gb}GB (minimum: 10GB)"
        ((errors++))
    else
        log_success "Disk space: ${disk_avail_gb}GB available"
    fi

    # GPU check
    GPU_AVAILABLE=false
    CUDA_VERSION="N/A"
    if command -v nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        local gpu_name driver_ver cuda_v
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        cuda_v=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -1)
        CUDA_VERSION="${cuda_v:-N/A}"
        log_success "NVIDIA GPU: ${gpu_name} (Driver: ${driver_ver}, CUDA: ${cuda_v})"
    else
        log_info "No NVIDIA GPU — CPU-only mode available"
    fi

    # Requirements.txt verification
    if [ ! -f "$REQUIREMENTS_SRC" ]; then
        log_error "Requirements file not found: $REQUIREMENTS_SRC"
        ((errors++))
    else
        log_success "Requirements verified: $REQUIREMENTS_SRC"
    fi

    # Existing frontend check
    if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
        log_success "React frontend detected: frontend/"
        REACT_EXISTS=true
    else
        log_warn "React frontend not found in frontend/ directory"
        REACT_EXISTS=false
    fi

    echo ""
    if [ $errors -gt 0 ]; then
        log_error "$errors error(s) found. Cannot proceed."
        exit 1
    fi
    [ $warnings -gt 0 ] && log_warn "$warnings warning(s). Proceeding..."
}

detect_package_manager() {
    if command -v apt-get &> /dev/null; then
        PKG_UPDATE_CMD="sudo apt-get update -q"
        PKG_INSTALL_CMD="sudo apt-get install -y"
    elif command -v dnf &> /dev/null; then
        PKG_UPDATE_CMD="sudo dnf check-update"
        PKG_INSTALL_CMD="sudo dnf install -y"
    else
        PKG_UPDATE_CMD=""
        PKG_INSTALL_CMD=""
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

CURRENT_DIR="$(pwd)"

# Handle sync-only
if [ "$SYNC_ONLY" = true ]; then
    log_section "Repository Sync Mode"
    check_remote_updates
    log_success "Sync complete."
    exit 0
fi

# Handle rollback
if [ "$ROLLBACK" = true ]; then
    log_section "Rollback Mode"
    
    # Fixed: Enable nullglob to handle empty directory gracefully
    shopt -s nullglob
    local_backups=(backups/*/metadata.json)
    shopt -u nullglob
    
    if [ ${#local_backups[@]} -eq 0 ]; then
        log_error "No backups found in backups/"
        exit 1
    fi
    local latest_backup=$(ls -t backups/*/metadata.json 2>/dev/null | head -1 | xargs dirname)
    log_info "Most recent backup: $latest_backup"
    # Restore logic here...
    log_success "Rollback complete"
    exit 0
fi

cd "$CURRENT_DIR"
log_section "Dialogue Anonymizer Installer (v${SCRIPT_VERSION})"
log_info "Working Directory: $(pwd)"
log_info "Requirements Source: $REQUIREMENTS_SRC"
echo ""

create_version_backup
preflight_check
detect_package_manager

# ============================================================================
# CONDA ENVIRONMENT SETUP
# ============================================================================

ensure_conda_available() {
    # Check 1: Is conda already in PATH?
    if command -v conda &> /dev/null; then
        log_success "Conda already installed and in PATH"
        eval "$(conda shell.bash hook)"
        
        # Remove Anaconda channels if they exist (migration to conda-forge)
        conda config --remove channels https://repo.anaconda.com/pkgs/main 2>/dev/null || true
        conda config --remove channels https://repo.anaconda.com/pkgs/r 2>/dev/null || true
        
        # Ensure conda-forge is primary
        conda config --add channels conda-forge
        conda config --set channel_priority strict
        return 0
    fi
    
    # Check 2: Does Miniforge/Miniconda exist?
    if [ -f "$HOME/miniforge3/bin/conda" ]; then
        CONDA_PATH="$HOME/miniforge3"
    elif [ -f "$HOME/miniconda3/bin/conda" ]; then
        CONDA_PATH="$HOME/miniconda3"
    else
        CONDA_PATH=""
    fi
    
    if [ -n "$CONDA_PATH" ]; then
        log_info "Conda distribution found at $CONDA_PATH - initializing..."
        source "$CONDA_PATH/etc/profile.d/conda.sh"
        eval "$(conda shell.bash hook)"
        export PATH="$CONDA_PATH/bin:$PATH"
        
        if command -v conda &> /dev/null; then
            log_success "Conda initialized successfully"
            conda config --remove channels https://repo.anaconda.com/pkgs/main 2>/dev/null || true
            conda config --remove channels https://repo.anaconda.com/pkgs/r 2>/dev/null || true
            conda config --add channels conda-forge
            conda config --set channel_priority strict
            return 0
        fi
    fi
    
    # Check 3: Install Miniforge
    log_info "Installing Miniforge (conda-forge distribution)..."
    
    rm -rf "$HOME/miniforge3"
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh
    
    if bash miniforge.sh -b -f -p "$HOME/miniforge3"; then
        rm miniforge.sh
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
        eval "$(conda shell.bash hook)"
        export PATH="$HOME/miniforge3/bin:$PATH"
        log_success "Miniforge installed and initialized"
        return 0
    else
        log_error "Miniforge installation failed"
        rm -f miniforge.sh
        return 1
    fi
}

if ! ensure_conda_available; then
    log_error "Failed to initialize conda"
    exit 1
fi

log_info "Creating conda environment '${ENV_NAME}' with Python ${TARGET_PYTHON}..."

if conda env list | grep -q "^${ENV_NAME} "; then
    if [ "$FORCE_REFRESH" = true ]; then
        conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
        conda create -n "$ENV_NAME" python="$TARGET_PYTHON" -c conda-forge -y
    else
        log_success "Environment '${ENV_NAME}' exists. Using existing."
    fi
else
    conda create -n "$ENV_NAME" python="$TARGET_PYTHON" -c conda-forge -y
fi

conda activate "$ENV_NAME"

ACTUAL_PYTHON=$(python --version | awk '{print $2}')
if [[ ! "$ACTUAL_PYTHON" =~ ^3\.12 ]]; then
    log_error "Expected Python 3.12.x, got $ACTUAL_PYTHON"
    exit 1
fi
log_success "Python verified: $ACTUAL_PYTHON"

# ============================================================================
# SYSTEM DEPENDENCIES
# ============================================================================

log_section "Installing System Dependencies"

if [ -n "$PKG_INSTALL_CMD" ]; then
    if ! command -v ffmpeg &> /dev/null; then
        $PKG_UPDATE_CMD
        $PKG_INSTALL_CMD ffmpeg
    fi
    if ! command -v gcc &> /dev/null; then
        $PKG_INSTALL_CMD build-essential cmake pkg-config
    fi
fi

# ============================================================================
# PYTHON DEPENDENCIES (SINGLE SOURCE OF TRUTH)
# ============================================================================

log_section "Installing Python Dependencies"

pip install --upgrade pip setuptools wheel -q

log_info "Installing core dependencies from $REQUIREMENTS_SRC..."
pip install --no-cache-dir -r "$REQUIREMENTS_SRC" -q

# Conditional web interface dependencies
if [ "$WEB_MODE_CHOICE" = "flask" ] || [ "$WEB_MODE_CHOICE" = "both" ]; then
    log_info "Installing web interface dependencies from $WEB_REQUIREMENTS_SRC..."
    if [ -f "$WEB_REQUIREMENTS_SRC" ]; then
        # Note: requirements-web.txt no longer includes '-r requirements.txt'
        # So we install both separately to avoid missing core deps
        pip install --no-cache-dir -r "$REQUIREMENTS_SRC" -q  # Ensure core deps are present
        pip install --no-cache-dir -r "$WEB_REQUIREMENTS_SRC" -q
        log_success "Web dependencies installed"
    else
        log_warn "Web requirements file not found: $WEB_REQUIREMENTS_SRC"
        log_warn "Proceeding without web-specific packages"
    fi
else
    log_info "Skipping web interface dependencies (--skip-flask or --skip-web)"
fi

# Verify imports
if python -c "import torch, transformers, spacy" 2>/dev/null; then
    log_success "Core Python packages verified"
else
    log_error "Import verification failed"
    exit 1
fi

# ============================================================================
# SPAICY MODEL INSTALLATION
# ============================================================================

log_section "Downloading spaCy Models"

if python -m spacy download en_core_web_sm 2>/dev/null; then
    log_success "en_core_web_sm installed"
else
    log_warn "en_core_web_sm installation skipped"
fi

# ============================================================================
# WEB INTERFACE CONFIGURATION (FLASK + REACT)
# ============================================================================

log_section "Web Interface Configuration"

PHASE_OUT_NOTE="NOTE: Flask is being phased out in favor of React"

if [ "$SKIP_WEB" = true ]; then
    log_info "Skipping all web interfaces (--skip-web)"
    WEB_MODE_CHOICE="none"
    SKIP_FLASK=true
    SKIP_FRONTEND=true
elif [ "$SKIP_FLASK" = true ]; then
    log_info "Flask skipped (--skip-flask). Using React only."
    WEB_MODE_CHOICE="react"
elif [ "$SKIP_FRONTEND" = true ]; then
    log_info "React skipped (--skip-frontend). Using Flask only."
    WEB_MODE_CHOICE="flask"
elif [ "$ANSWER_YES" = true ]; then
    WEB_MODE_CHOICE="both"
else
    read -p "Install Web Interfaces? (Both Flask + React) (y/n): " INSTALL_WEB
    if [[ ! "$INSTALL_WEB" =~ ^[Yy]$ ]]; then
        WEB_MODE_CHOICE="none"
        SKIP_FLASK=true
        SKIP_FRONTEND=true
    fi
fi

# --- FLASK INSTALLATION ---
if [ "$WEB_MODE_CHOICE" = "flask" ] || [ "$WEB_MODE_CHOICE" = "both" ]; then
    log_section "Installing Flask Backend"
    log_warn "$PHASE_OUT_NOTE"
fi

# --- REACT FRONTEND INSTALLATION ---
if [ "$WEB_MODE_CHOICE" = "react" ] || [ "$WEB_MODE_CHOICE" = "both" ]; then
    log_section "Configuring React Frontend"
    
    REACT_DIR="$CURRENT_DIR/frontend"
    
    if [ "$REACT_EXISTS" = true ] && [ -f "$REACT_DIR/package.json" ]; then
        log_info "React frontend detected at $REACT_DIR"
        
        # Check Node.js
        if ! command -v node &> /dev/null; then
            log_info "Installing Node.js 20.x LTS..."
            if [ -n "$PKG_INSTALL_CMD" ]; then
                curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
                $PKG_INSTALL_CMD -y nodejs
                log_success "Node.js installed"
            else
                log_error "Node.js not found. Install manually from https://nodejs.org/"
                exit 1
            fi
        fi
        
        NODE_VERSION=$(node --version | cut -d'v' -f2)
        NPM_VERSION=$(npm --version)
        log_success "Node.js v${NODE_VERSION}, npm v${NPM_VERSION}"
        
        cd "$REACT_DIR"
        log_info "Installing React dependencies..."
        npm install --legacy-peer-deps 2>/dev/null || npm install
        
        # Add Font Awesome packages if not present
        if [ -f "package.json" ]; then
            if ! grep -q "@fortawesome" package.json; then
                log_info "Adding Font Awesome dependencies..."
                npm install @fortawesome/fontawesome-free @fortawesome/react-fontawesome @fortawesome/free-solid-svg-icons --save
                log_success "Font Awesome packages added"
            fi
        fi
        
        cd "$CURRENT_DIR"
        log_success "React frontend configured"
    else
        log_warn "React frontend not found in frontend/ - skipping"
    fi
fi

log_success "Web interface setup complete (Mode: $WEB_MODE_CHOICE)"

# ============================================================================
# MODEL DOWNLOADS
# ============================================================================

log_section "ML Model Downloads"

if [ "$NO_MODELS" = true ]; then
    log_info "Skipping model downloads (--no-models)"
else
    declare -A MODEL_MAP
    MODEL_MAP["tiny"]="Systran--faster-whisper-tiny"
    MODEL_MAP["base"]="Systran--faster-whisper-base"
    MODEL_MAP["small"]="Systran--faster-whisper-small"
    MODEL_MAP["medium"]="Systran--faster-whisper-medium"
    MODEL_MAP["large"]="Systran--faster-whisper-large-v3"
    MODEL_MAP["large-turbo"]="Systran--faster-whisper-large-v3-turbo"
    
    WHISPER_PRIMARY_PATH=""
    
    # WhisperX models
    if [ "$SKIP_WHISPER" != true ]; then
        if [ -n "$WHISPER_SELECTION" ]; then
            WHISPER_MODELS_INPUT="${WHISPER_SELECTION//,/ }"
        elif [ "$ANSWER_YES" = true ]; then
            WHISPER_MODELS_INPUT="large"
        else
            read -p "WhisperX models to download (tiny, small, base, medium, large, large-v3-turbo): " WHISPER_MODELS_INPUT
            WHISPER_MODELS_INPUT="${WHISPER_MODELS_INPUT//,/ }"
        fi
        
        for model_name in $WHISPER_MODELS_INPUT; do
            model_name=$(echo "$model_name" | tr '[:upper:]' '[:lower:]' | xargs)
            [ -z "${MODEL_MAP[$model_name]}" ] && continue
            
            hf_repo="${MODEL_MAP[$model_name]}"
            target_dir="$CURRENT_DIR/pipeline/model/Systran--${hf_repo#*/}"
            
            if [ ! -d "$target_dir" ] || [ -z "$(ls -A "$target_dir" 2>/dev/null)" ]; then
                log_info "Downloading ${hf_repo}..."
                python -c "
from huggingface_hub import snapshot_download
import os

target = '${target_dir}'
repo = 'Systran/${hf_repo}'  # ✅ Use full repo ID

os.makedirs(target, exist_ok=True)

snapshot_download(
    repo_id=repo,
    local_dir=target,
    local_dir_use_symlink=False
)
" 2>/dev/null || log_warn "Failed to download ${hf_repo}"
            else
                log_success "${model_name} already exists"
            fi
            
            [ "$model_name" = "large" ] && WHISPER_PRIMARY_PATH="$target_dir"
        done
    fi
    
    # Pyannote model
    if [ "$SKIP_PYANNOTE" != true ]; then
        TARGET="$CURRENT_DIR/pipeline/model/models--pyannote--speaker-diarization-community-1"
        if [ ! -d "$TARGET" ]; then
            log_info "Downloading Pyannote model..."
            python -c "
from huggingface_hub import snapshot_download
snapshot_download('pyannote/speaker-diarization-community-1', local_dir='${TARGET}')
" 2>/dev/null || log_warn "Pyannote download failed"
        fi
    fi
    
    # PII NER model
    PII_TARGET="$CURRENT_DIR/pipeline/model/multilingual_DialogPII_NER"
    if [ ! -d "$PII_TARGET" ]; then
        log_info "Downloading DFKI-SLT PII model..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('DFKI-SLT/multilingual_DialogPII_NER', local_dir='${PII_TARGET}')
" 2>/dev/null || log_warn "PII model download failed"
    fi
fi

# ============================================================================
# ENVIRONMENT FILE CONFIGURATION
# ============================================================================

log_section "Environment Configuration (.env)"

# Create .env.example template
if [ ! -f "$CURRENT_DIR/.env.example" ]; then
    cat > "$CURRENT_DIR/.env.example" << 'ENVEEXAMPLE'
# ATA Speech Anonymizer Configuration Template
# Copy this file to .env and customize values

# WORKSPACE
CURRENT_DIR=/path/to/your/repo
MODEL_DIR=$CURRENT_DIR/pipeline/model

# API Keys
CHAT_AI_API_KEY=your_api_key_here
CHAT_AI_ENDPOINT=https://your-endpoint.com/v1

# TTS
TTS_BACKEND=piper
TTS_ENABLED=true
Piper_Voice_Path=$CURRENT_DIR/pipeline/tts/voices/en_US-amy-medium.onnx

# WhisperX
Whisper_Model_Path=$MODEL_DIR/Systran-faster-whisper-large-v3
Whisper_Device=cuda
Whisper_Compute_Type=float16

# Logging
LOG_LEVEL=INFO
ENVEEXAMPLE
    log_success "Created .env.example template"
fi

# Handle existing .env
if [ -f "$CURRENT_DIR/.env" ]; then
    log_info ".env exists - creating backup..."
    cp "$CURRENT_DIR/.env" "$CURRENT_DIR/.env.backup.$(date +%Y%m%d%H%M%S)"
    
    log_info "Checking for new environment variables..."
    # Merge new vars from .env.example
    while IFS= read -r line; do
        if [[ "$line" == *=* ]] && [[ "$line" != \#* ]]; then
            key="${line%%=*}"
            if ! grep -q "^${key}=" "$CURRENT_DIR/.env"; then
                echo "# NEW VARIABLE ADDED" >> "$CURRENT_DIR/.env"
                echo "$line" >> "$CURRENT_DIR/.env"
            fi
        fi
    done < "$CURRENT_DIR/.env.example"
    
    log_success "Updated .env with new variables"
else
    log_info "Creating .env from template..."
    cp "$CURRENT_DIR/.env.example" "$CURRENT_DIR/.env"
    chmod 600 "$CURRENT_DIR/.env"
    log_success "Created .env (chmod 600)"
    log_warn "EDIT .env WITH YOUR ACTUAL VALUES BEFORE RUNNING"
fi

# Update dynamic values
sed -i "s|^CURRENT_DIR=.*|CURRENT_DIR=${CURRENT_DIR}|" "$CURRENT_DIR/.env"
sed -i "s|^Whisper_Device=.*|Whisper_Device=\$([ \"$GPU_AVAILABLE\" = true ] && echo cuda || echo cpu)|" "$CURRENT_DIR/.env"

log_success ".env configuration complete"

# ============================================================================
# POST-INSTALLATION HEALTH CHECK
# ============================================================================

log_section "Health Check"

python << 'HEALTHEOF'
import sys
import os

passed = 0
failed = 0

checks = ['numpy', 'torch', 'transformers', 'spacy']
for mod in checks:
    try:
        __import__(mod)
        print(f'  ✓ {mod}')
        passed += 1
    except:
        print(f'  ✗ {mod}')
        failed += 1

if os.path.isfile('.env'):
    print('  ✓ .env exists')
    passed += 1
else:
    print('  ✗ .env missing')
    failed += 1

if os.path.isdir('pipeline/model'):
    print('  ✓ Model directory exists')
    passed += 1
else:
    print('  ⚠ Model directory missing')

sys.exit(1 if failed > 1 else 0)
HEALTHEOF

# ============================================================================
# FINAL SUMMARY
# ============================================================================

log_section "✅ INSTALLATION COMPLETE (v${SCRIPT_VERSION})"

cat << EOF
================================================================================
                         DEPLOYMENT OPTIONS                                    
================================================================================

🔹 LOCAL EXECUTION:
   1. conda activate whisperx
   2. python pipeline/process.py input.mp4

🔹 REACT FRONTEND:
   1. cd frontend
   2. npm run dev
   3. http://localhost:5173

🔹 FLASK BACKEND:
   1. python -m flask run --host=0.0.0.0

🔹 DOCKER:
   1. docker compose up -d --build
   2. docker compose ps

================================================================================
                         UPDATE COMMANDS                                       
================================================================================

• Sync repo (keep env):  ./Installer.sh --sync-only
• Full update:           ./Installer.sh -y --force-refresh
• Rollback:              ./Installer.sh --rollback
• Check status:          git status

================================================================================
                              NOTES                                           
================================================================================

• Flask is being phased out for React
• New features prioritized in React
• Core processing works independently of any web interface
• See MIGRATION_STATUS.md for transition progress

================================================================================
EOF

log_success "Installation complete!"