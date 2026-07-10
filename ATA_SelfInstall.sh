#!/usr/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ============================================================================
# 🔧 COMMAND-LINE ARGUMENT PARSING (AUTOMATION)
# ============================================================================
SHOW_HELP=false
SKIP_WEB=false
NO_COQUI=false
NO_MODELS=false
SKIP_WHISPER=false
SKIP_PYANNOTE=false
AUTO_LOGIN=false
ANSWER_YES=false
QUIET_MODE=false
WHISPER_SELECTION=""
FORCE_REFRESH=false
HUGGINGFACE_TOKEN=""

usage() {
    cat << 'EOF'
Usage: ./ATA_SelfInstall.sh [OPTIONS]

Options:
  -h, --help              Show this help message
  -y, --yes               Auto-answer 'yes' to all prompts
  --skip-web              Skip web interface installation (Flask, TTS)
  --no-coqui              Skip Coqui TTS model downloads
  --no-models             Skip ALL ML model downloads
  --skip-whisper          Skip WhisperX model download
  --skip-pyannote         Skip Pyannote diarization model download
  --whisper-models LIST   Comma-separated Whisper models (default: large-v3)
  --auto-login            Auto-authenticate with HUGGINGFACE_TOKEN env var
  -q, --quiet             Minimal output mode
  --force-refresh         Unconditionally reinstall all packages

Environment Variables:
  HUGGINGFACE_TOKEN       Used with --auto-login for Hugging Face auth

Examples:
  ./ATA_SelfInstall.sh                    # Interactive (default)
  ./ATA_SelfInstall.sh --yes              # Auto-accept all prompts
  ./ATA_SelfInstall.sh --yes --skip-web   # Auto, no web UI
  ./ATA_SelfInstall.sh --yes --no-models  # Dependencies only
  export HUGGINGFACE_TOKEN=your_token
  ./ATA_SelfInstall.sh --yes --auto-login # Auto-auth with token
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) SHOW_HELP=true; shift ;;
        -y|--yes) ANSWER_YES=true; shift ;;
        --skip-web) SKIP_WEB=true; shift ;;
        --no-coqui) NO_COQUI=true; shift ;;
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

if [ "$QUIET_MODE" = true ]; then
    exec >/dev/null 2>&1
fi

# --- 1. MAIN FOLDER LOGIC ---
CURRENT_DIR="$(pwd)"
SCRIPT_NAME="$(basename "$0")"
MAIN_DIR_NAME="ATA"

shopt -s nullglob

if [ "$(basename "$CURRENT_DIR")" == "$MAIN_DIR_NAME" ]; then
    echo "✅ Already inside '$MAIN_DIR_NAME' folder. Proceeding..."
else
    echo "📂 Not inside '$MAIN_DIR_NAME'. Checking for project files..."
    if [ -d "pipeline" ] || [ -f "README.md" ]; then
        echo "✅ Detected project root. Creating '$MAIN_DIR_NAME' and moving files..."
        mkdir -p "$MAIN_DIR_NAME"
        for item in *; do
            if [ "$item" != "$MAIN_DIR_NAME" ]; then
                mv "$item" "$MAIN_DIR_NAME/"
            fi
        done
        cd "$MAIN_DIR_NAME"
        CURRENT_DIR="$(pwd)"
        echo "✅ Moved all files into '$MAIN_DIR_NAME'. New location: $CURRENT_DIR"
    else
        echo "❌ Error: Not inside '$MAIN_DIR_NAME' and no project files found."
        exit 1
    fi
fi

cd "$CURRENT_DIR"

echo "=== ATA Speech Anonymizer Installer ==="
echo "Working Directory: $(pwd)"
echo ""

# 2. Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
else
    echo "Conda already installed."
    eval "$(conda shell.bash hook)"
fi

# 3. Create the environment
ENV_NAME="whisperx"
echo "Creating conda environment '$ENV_NAME' with Python 3.11..."
if conda env list | grep -q "^$ENV_NAME "; then
    if [ "$FORCE_REFRESH" = true ]; then
        echo "Environment exists (--force-refresh). Removing and recreating..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Environment '$ENV_NAME' already exists. Skipping recreation."
    fi
fi
conda create -n $ENV_NAME python=3.11 -y

# 4. Activate the environment
echo "Activating environment..."
conda activate $ENV_NAME
MODEL_DIR="$CURRENT_DIR/pipeline/model"
mkdir -p "$MODEL_DIR"

# ============================================================================
# 🔧 HELPER FUNCTIONS FOR PACKAGE VERSION CHECKING
# ============================================================================

# Check if package is installed with specific version
check_package_version() {
    local pkg=$1
    local desired_version=$2
    
    if ! pip show "$pkg" &> /dev/null; then
        echo "NOT_INSTALLED"
        return
    fi
    
    local current_version=$(pip show "$pkg" | grep Version | awk '{print $2}')
    
    if [ -z "$desired_version" ]; then
        echo "INSTALLED:$current_version"
        return
    fi
    
    # Compare versions (simple semantic versioning check)
    if [ "$current_version" == "$desired_version" ]; then
        echo "MATCH"
    else
        echo "MISMATCH:$current_version:$desired_version"
    fi
}

# Smart install with version checking
smart_install() {
    local pkg_spec=$1
    local pkg_name=$2
    local desired_version=$3
    local force=${4:-false}
    
    local version_check=$(check_package_version "$pkg_name" "$desired_version")
    
    case $version_check in
        "NOT_INSTALLED")
            echo "Installing $pkg_name..."
            pip install "$pkg_spec"
            ;;
        "MATCH")
            if [ "$force" = true ]; then
                echo "Reinstalling $pkg_name (matching version, --force-refresh)"
                pip install "$pkg_spec" --force-reinstall --no-deps --no-cache-dir
            else
                echo "✅ $pkg_name already installed ($desired_version). Skipping."
            fi
            ;;
        "MISMATCH:"*)
            local current=$(echo "$version_check" | cut -d: -f2)
            local desired=$(echo "$version_check" | cut -d: -f3)
            echo "⚠️  $pkg_name version mismatch: current=$current, desired=$desired"
            if [ "$force" = true ]; then
                echo "   Replacing with $desired (--force-refresh)"
            else
                echo "   Replacing with $desired (version conflict resolution)"
            fi
            pip uninstall "$pkg_name" -y
            pip install "$pkg_spec" --no-cache-dir
            ;;
    esac
}

# Download model helper (fixes --local-dir-use-symlinks issue)
download_model_hf() {
    local repo_id=$1
    local target_dir=$2
    local repo_type=${3:-model}
    
    echo "Downloading $repo_id..."
    mkdir -p "$target_dir"
    
    # Method 1: Try hf CLI (without unsupported flag)
    if command -v hf &> /dev/null; then
        echo "   Using hf CLI..."
        # Note: --local-dir-use-symlinks is NOT supported in hf download
        # We rely on Python fallback for symlink control
        hf download "$repo_id" --local-dir "$target_dir" 2>&1 | grep -v "already downloaded" || true
        
        # Verify download worked
        if [ "$(ls -A "$target_dir" 2>/dev/null)" ]; then
            echo "✅ Downloaded via hf CLI"
            return 0
        fi
    fi
    
    # Method 2: Python fallback with full symlink control
    echo "   Using Python snapshot_download (better symlink control)..."
    python -c "
from huggingface_hub import snapshot_download
import sys
try:
    snapshot_download(
        '$repo_id',
        local_dir='$target_dir',
        local_dir_use_symlinks=False,  # This works in Python, not CLI
        repo_type='$repo_type'
    )
    print('✅ Download complete')
except Exception as e:
    print(f'❌ Download failed: {e}', file=sys.stderr)
    sys.exit(1)
" || {
        echo "❌ Failed to download $repo_id"
        return 1
    }
}

# ============================================================================
# 🔧 PACKAGE INSTALLATION WITH VERSION CHECKING
# ============================================================================

echo "Upgrading pip..."
pip install --upgrade pip -q

# ============================================================================
# 🔧 CRITICAL: click + typer version enforcement
# ============================================================================
echo "Ensuring click==8.1.7 and compatible typer are installed..."

# Install click first
smart_install "click==8.1.7" "click" "8.1.7" "$FORCE_REFRESH"

# Install typer (must be <=0.12.x)
TYPER_DESIRED="0.12.5"
typer_check=$(check_package_version "typer" "$TYPER_DESIRED")
case $typer_check in
    "NOT_INSTALLED")
        echo "Installing typer $TYPER_DESIRED..."
        pip install "typer==$TYPER_DESIRED" --no-cache-dir
        ;;
    "MATCH")
        if [ "$FORCE_REFRESH" = true ]; then
            echo "Reinstalling typer ($TYPER_DESIRED, --force-refresh)"
            pip install "typer==$TYPER_DESIRED" --force-reinstall --no-deps --no-cache-dir
        else
            echo "✅ typer $TYPER_DESIRED already installed. Skipping."
        fi
        ;;
    "MISMATCH:"*)
        current=$(echo "$typer_check" | cut -d: -f2)
        echo "⚠️  typer version mismatch: current=$current, desired=$TYPER_DESIRED"
        echo "   Replacing (compatibility requirement)"
        pip uninstall typer -y
        pip install "typer==$TYPER_DESIRED" --force-reinstall --no-deps --no-cache-dir
        ;;
esac

# Verify versions explicitly
echo "Verifying installations..."
CLICK_VER=$(pip show click | grep Version | awk '{print $2}')
TYPER_VER=$(pip show typer | grep Version | awk '{print $2}')
echo "  click: $CLICK_VER"
echo "  typer: $TYPER_VER"

if [ "$CLICK_VER" != "8.1.7" ]; then
    echo "❌ CRITICAL: click version must be 8.1.7 (found $CLICK_VER)"
    exit 1
fi

if [[ ! "$TYPER_VER" =~ ^0\.1[0-2]\. ]]; then
    echo "❌ CRITICAL: typer must be 0.10-0.12.x (found $TYPER_VER)"
    exit 1
fi

echo "Checking NLP and Audio libraries..."

# WhisperX (Git installation)
if pip show whisperx &> /dev/null; then
    if [ "$FORCE_REFRESH" = true ]; then
        echo "Reinstalling whisperx (--force-refresh)..."
        pip uninstall whisperx -y
        pip install git+https://github.com/m-bain/whisperx.git --no-cache-dir
    else
        echo "✅ whisperx already installed. Skipping."
    fi
else
    echo "Installing whisperx from source..."
    pip install git+https://github.com/m-bain/whisperx.git --no-cache-dir
fi

# Standard packages with version checking
STANDARD_PKGS=(
    "pydub"
    "ffmpeg-python"
    "pyannote.audio"
    "transformers>=4.35.0"
    "accelerate>=0.20.0"
    "sentencepiece"
    "spacy>=3.7.0,<3.9.0"
    "torchcrf"
    "pandas>=2.2.0"
    "openai>=1.0.0"
    "python-dotenv"
    "scipy>=1.14.0"
    "numpy>=2.1.0"
    "huggingface-hub>=0.20.0"
)

for pkg_spec in "${STANDARD_PKGS[@]}"; do
    pkg_name=$(echo "$pkg_spec" | grep -oE '^[a-zA-Z0-9_-]+' | tr -d '>=' | tr -d '<')
    desired_version=$(echo "$pkg_spec" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
    smart_install "$pkg_spec" "$pkg_name" "$desired_version" "$FORCE_REFRESH"
done

# SpaCy Model Check & Dependency Fix
echo "Checking spaCy multilingual model..."

# Re-verify click + typer AFTER spacy installation (spacy can overwrite)
typer_check_after=$(check_package_version "typer" "$TYPER_DESIRED")
case $typer_check_after in
    "MISMATCH:"*)
        echo "⚠️  spaCy installation changed typer version. Restoring $TYPER_DESIRED..."
        pip uninstall typer -y
        pip install "typer==$TYPER_DESIRED" --force-reinstall --no-deps --no-cache-dir
        ;;
esac

if python -m spacy check xx_ent_wiki_sm &> /dev/null; then
    echo "✅ spaCy multilingual model (xx_ent_wiki_sm) is already installed."
else
    echo "Installing spaCy multilingual model (xx_ent_wiki_sm)..."
    python -m spacy download xx_ent_wiki_sm
    if [ $? -eq 0 ]; then
        echo "✅ spaCy model downloaded successfully."
    else
        echo "⚠️  spaCy model download failed. Continuing without sentence splitting..."
    fi
fi

# Torchcodec check
echo "Checking torchcodec..."
torchcodec_check=$(check_package_version "torchcodec" "0.7.0")
case $torchcodec_check in
    "NOT_INSTALLED")
        echo "Installing torchcodec 0.7.0..."
        pip install torchcodec==0.7.0 --no-cache-dir
        ;;
    "MATCH")
        if [ "$FORCE_REFRESH" = true ]; then
            echo "Reinstalling torchcodec 0.7.0 (--force-refresh)"
            pip uninstall torchcodec -y
            pip install torchcodec==0.7.0 --no-cache-dir
        else
            echo "✅ torchcodec 0.7.0 already installed. Skipping."
        fi
        ;;
    "MISMATCH:"*)
        current=$(echo "$torchcodec_check" | cut -d: -f2)
        echo "⚠️  torchcodec version mismatch: current=$current, desired=0.7.0"
        echo "   Replacing"
        pip uninstall torchcodec -y
        pip install torchcodec==0.7.0 --no-cache-dir
        ;;
esac

# TorchCRF Installation
echo "Checking CRF library..."
if python -c "from torchcrf import CRF" 2>/dev/null; then
    torchcrf_check=$(check_package_version "pytorch-crf" "")
    case $torchcrf_check in
        "NOT_INSTALLED"|"MISMATCH:"*)
            echo "Reinstalling torchcrf..."
            pip uninstall torchcrf -y
            pip install pytorch-crf --no-cache-dir
            ;;
        "MATCH")
            if [ "$FORCE_REFRESH" = true ]; then
                echo "Reinstalling torchcrf (--force-refresh)"
                pip uninstall torchcrf -y
                pip install pytorch-crf --no-cache-dir
            else
                echo "✅ torchcrf already installed and working. Skipping."
            fi
            ;;
    esac
else
    echo "Installing CRF library (pytorch-crf)..."
    pip uninstall torchcrf -y 2>/dev/null || true
    pip install pytorch-crf --no-cache-dir
    if python -c "from torchcrf import CRF" 2>/dev/null; then
        echo "✅ CRF library installed successfully."
    else
        echo "❌ CRITICAL: Failed to install CRF library."
        exit 1
    fi
fi

# ============================================================================
# 🔧 POST-INSTALLATION VERIFICATION
# ============================================================================
echo ""
echo "=== Verifying Critical Dependencies ==="

if python -c "import click; from click import Choice; c = Choice(['a','b'])" 2>/dev/null; then
    echo "✅ click is compatible with Python 3.12"
else
    echo "❌ WARNING: click may not be compatible."
fi

if python -c "import typer" 2>/dev/null; then
    echo "✅ typer imported successfully"
else
    echo "❌ WARNING: typer import failed"
fi

if python -c "import spacy" 2>/dev/null; then
    echo "✅ spacy imported successfully"
else
    echo "❌ WARNING: spacy import failed"
fi

if python -c "from huggingface_hub import snapshot_download" 2>/dev/null; then
    echo "✅ huggingface-hub imported successfully"
else
    echo "❌ WARNING: huggingface-hub import failed"
fi

# ============================================================================
# 🖥️ WEB INTERFACE OPTION
# ============================================================================
echo ""
echo "-------------------------------------------------"
if [ "$SKIP_WEB" = true ]; then
    echo "Skipping Web Interface (--skip-web)"
    INSTALL_WEB="n"
elif [ "$ANSWER_YES" = true ]; then
    echo "Web Interface: YES (auto-answered --yes)"
    INSTALL_WEB="y"
else
    read -p "Do you want to install the Web Interface (Flask, Coqui TTS, etc.)? (y/n): " INSTALL_WEB
    INSTALL_WEB=${INSTALL_WEB:-y}
fi

if [[ "$INSTALL_WEB" =~ ^[Yy]$ ]]; then
    echo "Installing Web Interface dependencies..."
    pip install --upgrade pip setuptools wheel
    
    echo "Step 1/4: Forcing NumPy >=2.1.0 and Pandas >=2.2.3..."
    smart_install "numpy>=2.1.0,<3.0.0" "numpy" "2.1.0" "$FORCE_REFRESH"
    smart_install "pandas>=2.2.3,<3.0.0" "pandas" "2.2.3" "$FORCE_REFRESH"
    smart_install "scipy>=1.14.0" "scipy" "1.14.0" "$FORCE_REFRESH"
    
    echo "Step 2/4: Installing Coqui TTS..."
    if pip show TTS &> /dev/null; then
        if [ "$FORCE_REFRESH" = true ]; then
            pip uninstall TTS -y
            pip install "TTS>=0.22.0" --no-cache-dir
        else
            echo "✅ TTS already installed. Skipping."
        fi
    else
        pip install "TTS>=0.22.0" --no-cache-dir || \
        pip install git+https://github.com/coqui-ai/TTS.git@main --no-cache-dir || {
            echo "❌ CRITICAL: Failed to install Coqui TTS."
            exit 1
        }
    fi
    
    echo "Step 3/4: Installing Flask and dependencies..."
    WEB_DEPS=("flask" "requests" "cryptography" "pydub" "ffmpeg-python")
    for dep in "${WEB_DEPS[@]}"; do
        smart_install "$dep" "$dep" "" "$FORCE_REFRESH"
    done
    
    echo "✅ Web Interface dependencies installed successfully."
else
    echo "Skipping Web Interface installation."
fi

# ============================================================================
# 🗣️ COQUI TTS MODEL DOWNLOADS
# ============================================================================
echo ""
echo "-------------------------------------------------"
echo "Coqui TTS Multi-Voice Models"
echo "Note: Coqui manages its own cache (~/.cache/tts)."

if [ "$NO_COQUI" = true ] || [ "$NO_MODELS" = true ]; then
    echo "Skipping Coqui models (--no-coqui or --no-models)"
    DOWNLOAD_INPUT="n"
elif [ "$ANSWER_YES" = true ]; then
    echo "Downloading Coqui models (auto-answered --yes)"
    DOWNLOAD_INPUT="y"
else
    read -p "Do you want to trigger Coqui model downloads now? (y/n) [y]: " DOWNLOAD_INPUT
    DOWNLOAD_INPUT=${DOWNLOAD_INPUT:-y}
fi

if [[ "$INSTALL_WEB" =~ ^[Yy]$ ]] && [[ "$DOWNLOAD_INPUT" =~ ^[Yy]$ ]]; then
    echo "Triggering Coqui model downloads..."
    python -c "
from TTS.api import TTS
print('Downloading German Thorsten...')
try:
    tts_de = TTS(model_name='tts_models/de/thorsten/vits')
    print('✅ German Thorsten downloaded.')
except Exception as e:
    print(f'⚠️  German Thorsten failed: {e}')

print('Downloading English VCTK...')
try:
    tts_en = TTS(model_name='tts_models/en/vctk/vits')
    print('✅ English VCTK downloaded.')
except Exception as e:
    print(f'⚠️  English VCTK failed: {e}')
"
else
    echo "Skipping Coqui model pre-download."
fi

# ============================================================================
# 🧩 MODEL DOWNLOADS
# ============================================================================
echo ""
echo "-------------------------------------------------"
echo "WhisperX & NLP Model Download"
echo "Target Directory: $MODEL_DIR"

if [ "$NO_MODELS" = true ]; then
    echo "Skipping all model downloads (--no-models)"
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
        echo "Skipping WhisperX models (--skip-whisper)"
    elif [ -n "$WHISPER_SELECTION" ]; then
        WHISPER_MODELS_INPUT="$WHISPER_SELECTION"
    elif [ "$ANSWER_YES" = true ]; then
        WHISPER_MODELS_INPUT="large"
    else
        read -p "Enter WhisperX models to download (tiny, base, small, medium, large) or press Enter to skip: " WHISPER_MODELS_INPUT
    fi
    
    if [ -n "$WHISPER_MODELS_INPUT" ]; then
        for model_name in $WHISPER_MODELS_INPUT; do
            model_name=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
            if [ -z "${MODEL_MAP[$model_name]}" ]; then
                echo "⚠️  Invalid: '$model_name'. Skipping."
                continue
            fi
            
            hf_repo="${MODEL_MAP[$model_name]}"
            target_dir="$MODEL_DIR/$(echo "$hf_repo" | sed 's/\//\--/g')"
            
            if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir")" ]; then
                echo "✅ Model '$model_name' already exists."
            else
                download_model_hf "$hf_repo" "$target_dir"
            fi
        done
    elif [ "$ANSWER_YES" = true ]; then
        echo "Downloading 'large-v3' by default (auto-answered --yes)..."
        download_model_hf "Systran/faster-whisper-large-v3" "$MODEL_DIR/models--Systran--faster-whisper-large-v3"
    fi
    
    # Pyannote Diarization Model
    TARGET="$MODEL_DIR/models--pyannote--speaker-diarization-community-1"
    
    if [ "$SKIP_PYANNOTE" = true ]; then
        echo "Skipping Pyannote (--skip-pyannote)"
    elif [ -d "$TARGET" ] && [ "$(ls -A "$TARGET")" ]; then
        echo "✅ Pyannote model already exists."
    else
        echo "Downloading Pyannote to $TARGET..."
        
        # Authentication check
        if [ "$AUTO_LOGIN" = true ] && [ -n "$HUGGINGFACE_TOKEN" ]; then
            echo "Auto-authenticating with HUGGINGFACE_TOKEN..."
            hf auth login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential
            AUTHENTICATED=true
        elif hf auth status &> /dev/null 2>&1; then
            AUTHENTICATED=true
        else
            AUTHENTICATED=false
        fi
        
        if [ "$AUTHENTICATED" = true ]; then
            download_model_hf "pyannote/speaker-diarization-community-1" "$TARGET" "model"
        else
            echo "⚠️  Not logged in to Hugging Face. Pyannote requires authentication."
            echo "   Options:"
            echo "   1. Run: hf auth login"
            echo "   2. Export: export HUGGINGFACE_TOKEN=your_token"
            echo "   3. Use: --auto-login with HUGGINGFACE_TOKEN set"
            
            if [ "$ANSWER_YES" = true ]; then
                echo "   Skipping (auto-answered n due to --yes)"
            else
                read -p "Download anyway (may fail)? (y/n) " TRY_DOWNLOAD
                if [[ "$TRY_DOWNLOAD" =~ ^[Yy]$ ]]; then
                    download_model_hf "pyannote/speaker-diarization-community-1" "$TARGET" "model"
                else
                    echo "Skipped Pyannote."
                fi
            fi
        fi
    fi
    
    # MM-BERT Base & PII Models
    echo ""
    echo "=========================================================="
    echo "Downloading mBert Base & NEW DFKI-SLT PII NER Model"
    echo "=========================================================="
    
    BASE_TARGET="$MODEL_DIR/jhu-clsp/mmBERT-base"
    if [ -d "$BASE_TARGET" ] && [ "$(ls -A "$BASE_TARGET")" ]; then
        echo "✅ mmBERT-base already present."
    else
        download_model_hf "jhu-clsp/mmBERT-base" "$BASE_TARGET" "model"
    fi
    
    # DFKI-SLT PII Model (PRIMARY ANONYMIZATION MODEL)
    PII_TARGET="$MODEL_DIR/multilingual_DialogPII_NER"
    echo ""
    echo "📦 Downloading NEW DFKI-SLT Multilingual DialogPII NER Model"
    echo "   Repository: https://huggingface.co/DFKI-SLT/multilingual_DialogPII_NER"
    
    if [ -d "$PII_TARGET" ] && [ "$(ls -A "$PII_TARGET")" ]; then
        echo "✅ DFKI-SLT PII model already exists."
        
        # Verify config files
        echo "   Verifying required configuration files..."
        MISSING_CONFIG=false
        [ ! -f "$PII_TARGET/crf_config.json" ] && { echo "   ⚠️  Missing: crf_config.json"; MISSING_CONFIG=true; }
        [ ! -f "$PII_TARGET/id2label.json" ] && { echo "   ⚠️  Missing: id2label.json"; MISSING_CONFIG=true; }
        
        if [ "$MISSING_CONFIG" = true ]; then
            echo "   ⚠️  Config files missing. Will attempt re-download..."
            rm -rf "$PII_TARGET"
        else
            echo "   ✅ All required config files present."
        fi
    fi
    
    if [ ! -d "$PII_TARGET" ] || [ -z "$(ls -A "$PII_TARGET" 2>/dev/null)" ]; then
        download_model_hf "DFKI-SLT/multilingual_DialogPII_NER" "$PII_TARGET" "model"
        
        if [ "$(ls -A "$PII_TARGET" 2>/dev/null)" ]; then
            echo "✅ DFKI-SLT PII model downloaded successfully."
            echo "   Key files:"
            ls -la "$PII_TARGET"/*.json 2>/dev/null | awk '{print "   - " $NF}' || echo "   (No .json files)"
        else
            echo "❌ Download failed. Manual download required:"
            echo "   https://huggingface.co/DFKI-SLT/multilingual_DialogPII_NER"
        fi
    fi
    
    # Old PII Model (Backward Compatibility)
    OLD_PII_TARGET="$MODEL_DIR/mmbert_multilingual_pii_ner"
    if [ -d "$OLD_PII_TARGET" ] && [ "$(ls -A "$OLD_PII_TARGET")" ]; then
        echo ""
        echo "⚠️  Note: Old PII model also exists."
        echo "   New DFKI-SLT model is recommended for better accuracy."
    fi
fi

# ============================================================================
# 📁 ENVIRONMENT FILE CREATION
# ============================================================================
echo ""
if [ -f ".env" ]; then
    echo "⚠️  Existing .env file found. Skipping creation."
else
    cat > .env <<EOF
# ATA Speech Anonymizer Configuration
CHAT_AI_API_KEY=your_api_key_here
CHAT_AI_ENDPOINT=https://your-endpoint.com/v1
EOF
    echo "✅ .env file created. Edit with your API key."
fi

# ============================================================================
# 📋 FINAL INSTRUCTIONS
# ============================================================================
echo ""
echo "=========================================================================="
echo "SETUP COMPLETE!"
echo "=========================================================================="
echo ""
echo "To use the environment:"
echo "  source \$HOME/miniforge3/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
echo ""
echo "Hugging Face Authentication:"
echo "  • If Pyannote downloads failed, run:"
echo "      hf auth login"
echo "  • Or use: export HUGGINGFACE_TOKEN=your_token && ./ATA_SelfInstall.sh --auto-login"
echo ""
echo "Automation Flags (re-run installer):"
echo "  --yes              Auto-accept all prompts"
echo "  --skip-web         Skip web UI dependencies"
echo "  --no-models        Skip all model downloads"
echo "  --whisper-models LIST  Specify Whisper models"
echo "  --force-refresh    Reinstall all packages"
echo ""
echo "Project Structure:"
echo "  $(pwd)/"
echo "    ├── .env"
echo "    ├── ATA_SelfInstall.sh"
echo "    └── pipeline/"
echo "        ├── model/"
echo "        ├── videos/"
echo "        ├── audios/"
echo "        ├── transcripts/"
echo "        ├── anonym/"
echo "        └── LLM-Anon/"
echo ""
echo "Next Steps:"
echo "  1. Edit '.env' with your API Key"
echo "  2. Place videos in 'pipeline/videos'"
echo "  3. Run: python pipeline/process.py"
echo ""
echo "✨ FEATURES:"
echo "  • DFKI-SLT Multilingual DialogPII NER (11 languages)"
echo "  • FLERT-style context windowing"
echo "  • Adversarial anonymization mode"
echo ""
echo "=========================================================================="