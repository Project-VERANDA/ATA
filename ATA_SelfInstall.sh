#!/usr/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

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
        echo "   Please run this script from the root of the cloned ATA repository."
        exit 1
    fi
fi

cd "$CURRENT_DIR"

echo "=== ATA Speech Anonymizer Installer ==="
echo "Working Directory: $(pwd)"
echo "This script checks for existing dependencies and skips installation if found."
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
    echo "Environment '$ENV_NAME' already exists. Removing and recreating..."
    conda env remove -n $ENV_NAME -y
fi
conda create -n $ENV_NAME python=3.11 -y

# 4. Activate the environment
echo "Activating environment..."
conda activate $ENV_NAME
MODEL_DIR="$CURRENT_DIR/pipeline/model"
mkdir -p "$MODEL_DIR"

# --- HELPER FUNCTIONS ---

check_pip_installed() {
    local pkg=$1
    if pip show "$pkg" &> /dev/null; then
        echo "✅ $pkg is already installed. Skipping."
        return 0
    else
        return 1
    fi
}

check_conda_installed() {
    local pkg=$1
    if conda list "$pkg" &> /dev/null; then
        echo "✅ Conda package $pkg is already installed. Skipping."
        return 0
    else
        return 1
    fi
}

check_apt_installed() {
    local pkg=$1
    if dpkg -l | grep -q "^ii  $pkg"; then
        echo "✅ System package $pkg is already installed. Skipping."
        return 0
    else
        return 1
    fi
}

# 5. Install System Dependencies (APT)
echo "Checking system build tools..."
NEEDS_APT=false

if ! check_apt_installed "build-essential"; then NEEDS_APT=true; fi
if ! check_apt_installed "libsndfile1"; then NEEDS_APT=true; fi
if ! check_apt_installed "ffmpeg"; then NEEDS_APT=true; fi
if ! check_apt_installed "espeak-ng"; then NEEDS_APT=true; fi

if [ "$NEEDS_APT" = true ]; then
    echo "Installing missing system tools via apt..."
    sudo apt-get update -qq && sudo apt-get install -y -qq build-essential libsndfile1 ffmpeg espeak-ng
else
    echo "All system tools are present."
fi

# Ensure ffmpeg and libsndfile are present in conda as well
echo "Checking Conda packages (ffmpeg, libsndfile)..."
if ! check_conda_installed "ffmpeg"; then
    echo "Installing ffmpeg via Conda..."
    conda install -c conda-forge ffmpeg -y
fi
if ! check_conda_installed "libsndfile"; then
    echo "Installing libsndfile via Conda..."
    conda install -c conda-forge libsndfile -y
fi

# --- HELPER FUNCTION: Normalize package name ---
normalize_pkg_name() {
    echo "$1" | tr '-' '_'
}

# --- HELPER FUNCTION: Check if pip package is installed (Robust) ---
check_pip_installed_robust() {
    local pkg=$1
    local norm_pkg=$(normalize_pkg_name "$pkg")
    
    if pip show "$pkg" &> /dev/null; then
        return 0
    fi
    if pip show "$norm_pkg" &> /dev/null; then
        return 0
    fi
    return 1
}

# --- HELPER FUNCTION: Check if Git Package is Installed ---
check_git_installed() {
    local url=$1
    local pkg_name=$2
    
    if ! check_pip_installed_robust "$pkg_name"; then
        return 1
    fi
    return 0
}

# 6. Install Python packages (Optimized)
echo "Upgrading pip..."
pip install --upgrade pip -q


echo "Ensuring click==8.1.7 and compatible typer are installed..."
if ! check_pip_installed "click"; then
    echo "Installing click==8.1.7..."
    pip install "click==8.1.7" --force-reinstall --no-deps
else
    CLICK_VERSION=$(pip show click | grep Version | awk '{print $2}' | cut -d. -f1)
    # Check if it's 8.1.x (safe) or 8.2+ (conflict) or <8.1 (error)
    if [ "$CLICK_VERSION" -lt 8 ]; then
        echo "⚠️  Detected click version < 8.0. Upgrading to 8.1.7..."
        pip install "click==8.1.7" --force-reinstall --no-deps
    elif [ "$CLICK_VERSION" -gt 8 ]; then
        echo "⚠️  Detected click version > 8.1. Downgrading to 8.1.7..."
        pip install "click==8.1.7" --force-reinstall --no-deps
    else
        MINOR=$(pip show click | grep Version | awk '{print $2}' | cut -d. -f2)
        if [ "$MINOR" -lt 1 ]; then
             echo "⚠️  Detected click 8.0.x. Upgrading to 8.1.7..."
             pip install "click==8.1.7" --force-reinstall --no-deps
        else
             echo "✅ click version is sufficient (8.1.x)."
        fi
    fi
fi

echo "Checking NLP and Audio libraries..."

# 1. WhisperX (Special handling for Git)
if check_git_installed "git+https://github.com/m-bain/whisperx.git" "whisperx"; then
    echo "✅ whisperx is already installed. Skipping."
else
    echo "Installing whisperx from source..."
    pip install git+https://github.com/m-bain/whisperx.git
fi

# Ensure typer is compatible
if ! check_pip_installed "typer"; then
    echo "Installing compatible typer (0.9.0 - 0.12.x)..."
    pip install "typer>=0.9.0,<0.13.0" --force-reinstall --no-deps
else
    TYPER_VERSION=$(pip show typer | grep Version | awk '{print $2}')
    MAJOR=$(echo $TYPER_VERSION | cut -d. -f1)
    MINOR=$(echo $TYPER_VERSION | cut -d. -f2)
    if [ "$MAJOR" -eq 0 ] && [ "$MINOR" -ge 13 ]; then
        echo "⚠️  Detected typer >= 0.13. Downgrading to 0.12.5..."
        pip install "typer==0.12.5" --force-reinstall --no-deps
    else
        echo "✅ typer version is compatible ($TYPER_VERSION)."
    fi
fi

# 2. Define list of standard packages to check
STANDARD_PKGS=(
    "pydub"
    "ffmpeg-python"
    "pyannote.audio"
    "transformers"
    "accelerate"
    "sentencepiece"
    "spacy"
    "torchcrf"
    "pandas"
    "openai"
    "python-dotenv"
    "scipy"
    "numpy"
    "huggingface-hub"
)

for pkg in "${STANDARD_PKGS[@]}"; do
    if check_pip_installed_robust "$pkg"; then
        echo "✅ $pkg is already installed. Skipping."
    else
        echo "Installing $pkg..."
        pip install "$pkg"
    fi
done

# 3. Spacy Model Check & Dependency Fix
echo "Checking spaCy multilingual model..."

# Force reinstall click + typer AFTER spacy installation
# (spacy often overwrites these with incompatible versions)
echo "Enforcing compatible click + typer versions after spacy install..."
pip uninstall click typer -y
pip install "click==8.1.7" "typer==0.12.5" --force-reinstall --no-deps --no-cache-dir

# Verify versions explicitly
echo "Verifying installations..."
CLICK_VER=$(pip show click | grep Version | awk '{print $2}')
TYPER_VER=$(pip show typer | grep Version | awk '{print $2}')
echo "  click: $CLICK_VER"
echo "  typer: $TYPER_VER"

# NOW run spaCy CLI (dependencies guaranteed compatible)
if python -m spacy check xx_ent_wiki_sm &> /dev/null; then
    echo "✅ spaCy multilingual model (xx_ent_wiki_sm) is already installed."
else
    echo "Installing spaCy multilingual model (xx_ent_wiki_sm)..."
    python -m spacy download xx_ent_wiki_sm
    
    # Post-download verification
    if [ $? -eq 0 ]; then
        echo "✅ spaCy model downloaded successfully."
    else
        echo "⚠️  spaCy model download failed. Continuing without sentence splitting..."
    fi
fi

# 4. Torchcodec check
echo "Checking torchcodec..."
if pip show torchcodec &> /dev/null; then
    CURRENT_VERSION=$(pip show torchcodec | grep Version | awk '{print $2}')
    if [ "$CURRENT_VERSION" == "0.7.0" ]; then
        echo "✅ torchcodec 0.7.0 is already installed."
    else
        echo "Updating torchcodec to 0.7.0..."
        pip uninstall torchcodec -y
        pip install torchcodec==0.7.0
    fi
else
    echo "Installing torchcodec 0.7.0..."
    pip install torchcodec==0.7.0
fi

# --- TorchCRF Installation ---
echo "Checking CRF library..."
if python -c "from torchcrf import CRF" 2>/dev/null; then
    echo "✅ torchcrf is already installed and working."
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
# 7. Web Interface Option
# ============================================================================
echo ""
echo "-------------------------------------------------"
read -p "Do you want to install the Web Interface (Flask, Coqui TTS, etc.)? (y/n): " INSTALL_WEB
INSTALL_WEB=${INSTALL_WEB:-y}

if [[ "$INSTALL_WEB" =~ ^[Yy]$ ]]; then
    WEB_PKGS=("flask" "python-dotenv" "requests" "cryptography" "TTS" "scipy")
    SKIP_WEB=true
    for pkg in "${WEB_PKGS[@]}"; do
        if ! check_pip_installed "$pkg"; then
            SKIP_WEB=false
            break
        fi
    done

    if [ "$SKIP_WEB" = false ]; then
        echo "Installing Web Interface dependencies..."
        
        # 1. Upgrade pip and build tools
        pip install --upgrade pip setuptools wheel
        
        # 2. CRITICAL FIX: Install core data libraries FIRST with forced versions
        # This prevents scipy/TTS from pinning numpy to old versions
        echo "Step 1/4: Forcing NumPy >=2.1.0 and Pandas >=2.2.3..."
        pip install --no-cache-dir --force-reinstall \
            "numpy>=2.1.0,<3.0.0" \
            "pandas>=2.2.3,<3.0.0" \
            "scipy>=1.14.0" || {
                echo "⚠️  Force install failed. Trying standard install..."
                pip install "numpy>=2.1.0" "pandas>=2.2.3" "scipy>=1.14.0" --no-cache-dir
            }

        # 3. Verify versions before proceeding
        python -c "import numpy; import pandas; assert float(numpy.__version__.split('.')[0]) >= 2, 'NumPy too low'; print(f'✅ NumPy: {numpy.__version__}, Pandas: {pandas.__version__}')"

        # 4. Now install Coqui TTS (which will respect the new numpy)
        echo "Step 2/4: Installing Coqui TTS..."
        pip install "TTS>=0.22.0" --no-cache-dir || {
            echo "⚠️  PyPI TTS failed. Trying GitHub source..."
            pip install git+https://github.com/coqui-ai/TTS.git@main --no-cache-dir || {
                echo "❌ CRITICAL: Failed to install Coqui TTS."
                exit 1
            }
        }

        # 5. Install remaining web frameworks
        echo "Step 3/4: Installing Flask and dependencies..."
        pip install flask python-dotenv requests cryptography pydub ffmpeg-python spacy --no-cache-dir
        
        echo "✅ Step 4/4: Web Interface dependencies installed successfully."
    else
        echo "Skipping Web Interface installation (all packages already present)."
    fi
else
    echo "Skipping Web Interface installation."
fi

# ============================================================================
# 🗣️ DOWNLOAD MULTI-VOICE MODELS
# ============================================================================
echo ""
echo "-------------------------------------------------"
echo "Coqui TTS Multi-Voice Models"
echo "Note: Coqui manages its own cache (~/.cache/tts). We will verify download."
echo ""

DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-y} 
read -p "Do you want to trigger Coqui model downloads now? (y/n) [y]: " DOWNLOAD_INPUT
DOWNLOAD_INPUT=${DOWNLOAD_INPUT:-y}

if [[ "$INSTALL_WEB" =~ ^[Yy]$ ]] && [[ "$DOWNLOAD_INPUT" =~ ^[Yy]$ ]]; then
    echo "Triggering Coqui model downloads via Python..."
    
    # This forces the models to download to the default TTS cache
    python -c "
import sys
try:
    from TTS.api import TTS
    print('Downloading German Thorsten (Multi-Voice)...')
    try:
        tts_de = TTS(model_name='tts_models/de/thorsten/vits') # Standard Thorsten first
        print('✅ German Thorsten downloaded.')
    except Exception as e:
        print(f'⚠️  German Thorsten failed (might be multispeaker issue): {e}')
        # Try to find available de models
        pass
        
    print('Downloading English VCTK...')
    try:
        tts_en = TTS(model_name='tts_models/en/vctk/vits')
        print('✅ English VCTK downloaded.')
    except Exception as e:
        print(f'⚠️  English VCTK failed: {e}')
except Exception as e:
    print(f'Error initializing TTS: {e}')
"
    echo "Coqui models are now cached. If 'thorsten-multispeaker' was requested in app.py, you may need to update app.py to use 'tts_models/de/thorsten/vits' as per previous advice."
else
    echo "Skipping Coqui model pre-download."
fi

# ============================================================================
# 🧩 DOWNLOAD WHISPERX & NLP MODELS TO pipeline/model/
# ============================================================================
echo ""
echo "-------------------------------------------------"
echo "WhisperX & NLP Model Download"
echo "Target Directory: $MODEL_DIR"
echo "Expected Folder Names:"
echo "  - models--Systran--faster-whisper-large-v3"
echo "  - models--pyannote--speaker-diarization-community-1"
echo "  - jhu-clsp/mmBERT-base (Base model for PII NER)"
echo "  - multilingual_DialogPII_NER (NEW: DFKI-SLT PII Detection Model)"
echo ""

read -p "Enter WhisperX models to download (tiny, base, small, medium, large) or press Enter to skip: " WHISPER_MODELS_INPUT

# Map short names to HF repos
declare -A MODEL_MAP
MODEL_MAP["tiny"]="Systran/faster-whisper-tiny"
MODEL_MAP["base"]="Systran/faster-whisper-base"
MODEL_MAP["small"]="Systran/faster-whisper-small"
MODEL_MAP["medium"]="Systran/faster-whisper-medium"
MODEL_MAP["large"]="Systran/faster-whisper-large-v3"

if [ -n "$WHISPER_MODELS_INPUT" ]; then
    for model_name in $WHISPER_MODELS_INPUT; do
        model_name=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
        
        if [ -z "${MODEL_MAP[$model_name]}" ]; then
            echo "⚠️  Warning: '$model_name' is not valid. Skipping."
            continue
        fi

        hf_repo="${MODEL_MAP[$model_name]}"
        # CRITICAL: Use the exact naming convention expected by process.py
        target_dir="$MODEL_DIR/$(echo "$hf_repo" | sed 's/\//\--/g')"
        
        if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir")" ]; then
            echo "✅ Model '$model_name' already exists at $target_dir."
        else
            echo "Downloading: $model_name ($hf_repo) -> $target_dir"
            mkdir -p "$target_dir"
            
            # Download using huggingface download
            if command -v hf &> /dev/null; then
                hf download "$hf_repo" --local-dir "$target_dir" --local-dir-use-symlinks False 2>&1 | grep -v "already downloaded" || true
            else
                echo "⚠️  hf download not found. Install with: pip install huggingface-hub"
                # Fallback to Python if CLI missing
                python -c "from huggingface_hub import snapshot_download; snapshot_download('$hf_repo', local_dir='$target_dir', local_dir_use_symlinks=False)"
            fi
            
            if [ $? -eq 0 ] && [ "$(ls -A "$target_dir")" ]; then
                echo "✅ Successfully downloaded: $model_name"
            else
                echo "❌ Failed to download: $model_name"
            fi
        fi
    done
else
    # Auto-download Large if user pressed Enter (default recommendation)
    echo "No selection made. Downloading 'large-v3' by default..."
    HF_REPO="Systran/faster-whisper-large-v3"
    TARGET_DIR="$MODEL_DIR/models--Systran--faster-whisper-large-v3"
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Downloading large-v3..."
        mkdir -p "$TARGET_DIR"
        hf download "$HF_REPO" --local-dir "$TARGET_DIR" --local-dir-use-symlinks False 2>&1 | grep -v "already downloaded" || true
    else
        echo "✅ large-v3 already present."
    fi
fi

# ============================================================================
# 🎙️ PYANNOTE DIARIZATION MODEL
# ============================================================================
echo ""
echo "Downloading Pyannote Speaker Diarization Model"
TARGET="$MODEL_DIR/models--pyannote--speaker-diarization-community-1"

if [ -d "$TARGET" ] && [ "$(ls -A "$TARGET")" ]; then
    echo "✅ Pyannote model already exists."
else
    echo "Downloading Pyannote to $TARGET..."
    mkdir -p "$TARGET"
    
    # Check if authenticated
    if hf auth status &> /dev/null 2>&1; then
        hf download pyannote/speaker-diarization-community-1 --local-dir "$TARGET" --local-dir-use-symlinks False
        echo "✅ Pyannote downloaded."
    else
        echo "⚠️  Not logged in to Hugging Face Hub."
        echo "   Pyannote requires authentication. Two options:"
        echo ""
        echo "   Option 1: Login interactively"
        echo "     hf auth login"
        echo ""
        echo "   Option 2: Use token directly for this download"
        echo "     export HUGGINGFACE_TOKEN=your_token_here"
        echo "     hf download pyannote/speaker-diarization-community-1 --local-dir \"$TARGET\" --local-dir-use-symlinks False"
        echo ""
        echo "   Getting a token: https://huggingface.co/settings/tokens"
        echo ""
        echo "   If you login, please re-run this script."
        echo ""
        
        # Try Python fallback (also needs auth for pyannote)
        python -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        'pyannote/speaker-diarization-community-1',
        local_dir='$TARGET',
        local_dir_use_symlinks=False
    )
    print('✅ Pyannote downloaded (Python fallback).')
except Exception as e:
    print(f'⚠️  Download requires authentication: {e}')
    print('   Run: hf auth login')
"
    fi
fi

# ============================================================================
# 🧠 MBERT BASE & PII MODELS (UPDATED FOR DFKI-SLT MODEL)
# ============================================================================
echo ""
echo "=========================================================="
echo "Downloading mBert Base & NEW DFKI-SLT PII NER Model"
echo "=========================================================="
echo ""
echo "📝 IMPORTANT: The new model (DFKI-SLT/multilingual_DialogPII_NER)"
echo "   requires these configuration files:"
echo "   - crf_config.json"
echo "   - flert_config.json"  
echo "   - id2label.json"
echo ""

# 1. Base Model (mmBERT-base - still needed as foundation)
BASE_TARGET="$MODEL_DIR/jhu-clsp/mmBERT-base"
if [ ! -d "$BASE_TARGET" ]; then
    echo "Downloading mmBERT-base to $BASE_TARGET..."
    mkdir -p "$(dirname "$BASE_TARGET")"
    if command -v hf &> /dev/null; then
        hf download jhu-clsp/mmBERT-base --local-dir "$BASE_TARGET" --local-dir-use-symlinks False
    else
        python -c "from huggingface_hub import snapshot_download; snapshot_download('jhu-clsp/mmBERT-base', local_dir='$BASE_TARGET', local_dir_use_symlinks=False)"
    fi
    echo "✅ Base model downloaded."
else
    echo "✅ mmBERT-base already present."
fi

# 2. NEW: DFKI-SLT PII Model (Primary anonymization model)
echo ""
echo "📦 Downloading NEW DFKI-SLT Multilingual DialogPII NER Model"
echo "   Repository: https://huggingface.co/DFKI-SLT/multilingual_DialogPII_NER"
PII_TARGET="$MODEL_DIR/multilingual_DialogPII_NER"

if [ -d "$PII_TARGET" ] && [ "$(ls -A "$PII_TARGET")" ]; then
    echo "✅ DFKI-SLT PII model already exists at $PII_TARGET."
    
    # Verify required config files exist
    echo "   Verifying required configuration files..."
    MISSING_CONFIG=false
    
    if [ ! -f "$PII_TARGET/crf_config.json" ]; then
        echo "   ⚠️  Missing: crf_config.json"
        MISSING_CONFIG=true
    fi
    
    if [ ! -f "$PII_TARGET/id2label.json" ]; then
        echo "   ⚠️  Missing: id2label.json"
        MISSING_CONFIG=true
    fi
    
    if [ "$MISSING_CONFIG" = true ]; then
        echo "   ⚠️  Some config files are missing. Will attempt to re-download..."
        rm -rf "$PII_TARGET"
    else
        echo "   ✅ All required config files present."
    fi
fi

if [ ! -d "$PII_TARGET" ] || [ -z "$(ls -A "$PII_TARGET" 2>/dev/null)" ]; then
    echo "Downloading DFKI-SLT PII model to $PII_TARGET..."
    mkdir -p "$PII_TARGET"
    
    if command -v hf &> /dev/null; then
        echo "   Using huggingface download..."
        hf download DFKI-SLT/multilingual_DialogPII_NER \
            --local-dir "$PII_TARGET" \
            --local-dir-use-symlinks False \
            2>&1 | grep -v "already downloaded" || true
    else
        echo "   Using Python fallback..."
        python -c "
from huggingface_hub import snapshot_download
import sys
try:
    snapshot_download(
        'DFKI-SLT/multilingual_DialogPII_NER',
        local_dir='$PII_TARGET',
        local_dir_use_symlinks=False
    )
    print('✅ Download complete')
except Exception as e:
    print(f'❌ Download failed: {e}', file=sys.stderr)
    sys.exit(1)
" || {
        echo "❌ Failed to download DFKI-SLT PII model."
        exit 1
    }
    fi
    
    # Verify the download succeeded
    if [ "$(ls -A "$PII_TARGET" 2>/dev/null)" ]; then
        echo "✅ DFKI-SLT PII model downloaded successfully."
        
        # List key files for verification
        echo "   Key files found:"
        ls -la "$PII_TARGET"/*.json 2>/dev/null | awk '{print "   - " $NF}' || echo "   (No .json files found - check manually)"
    else
        echo "❌ Download appeared to succeed but target directory is empty."
        echo "   Please manually download from: https://huggingface.co/DFKI-SLT/multilingual_DialogPII_NER"
    fi
fi

# 3. Old PII Model (Keep for backward compatibility - optional)
OLD_PII_TARGET="$MODEL_DIR/mmbert_multilingual_pii_ner"
if [ -d "$OLD_PII_TARGET" ] && [ "$(ls -A "$OLD_PII_TARGET")" ]; then
    echo ""
    echo "⚠️  Note: Old PII model (mmbert_multilingual_pii_ner) also exists."
    echo "   The new DFKI-SLT model should be used for improved accuracy."
    echo "   If you want to remove the old model, delete: $OLD_PII_TARGET"
else
    echo ""
    echo "ℹ️  Old PII model (mmbert_multilingual_pii_ner) not found."
    echo "   Only the new DFKI-SLT model is installed."
fi

echo ""
echo "=== Model Installation Complete ==="
echo "Verification: Check that folders exist in $MODEL_DIR"
echo ""
echo "Installed Models:"
echo "=================="
if [ -d "$MODEL_DIR" ]; then
    ls -la "$MODEL_DIR" | grep "^d" | awk '{print "  " $NF}'
else
    echo "  ❌ Model directory not found!"
fi

# Create Environment file
if [ -f ".env" ]; then
    echo ""
    echo "⚠️  Existing .env file found. Skipping creation to preserve your settings."
    echo "   Your current API keys and configurations are safe."
else
    cat > .env <<EOF
# ATA Speech Anonymizer Configuration
# Generated by ATA_SelfInstall.sh on $(date)

# LLM API Configuration
CHAT_AI_API_KEY=your_api_key_here
CHAT_AI_ENDPOINT=https://your-endpoint.com/v1

# Optional HTTPS toggle
# USE_HTTPS=true
EOF
    echo "✅ .env file created. Edit it with your real API key."
fi

# 14. Final Instructions
echo ""
echo "=========================================================================="
echo "===              SETUP COMPLETE!                           ==="
echo "=========================================================================="
echo ""
echo "To use the environment:"
echo "  source \$HOME/miniforge3/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
echo ""
echo "Hugging Face Authentication:"
echo "  • If you see 'not logged in' errors for Pyannote, run:"
echo "      hf auth login"
echo "  • Get your token here: https://huggingface.co/settings/tokens"
echo ""
echo ""
echo "Project Structure:"
echo "  $(pwd)/"
echo "    ├── .env"
echo "    ├── ATA_SelfInstall.sh"
echo "    └── pipeline/"
echo "        ├── model/ (Models downloaded here)"
echo "        │   ├── models--Systran--faster-whisper-*/"
echo "        │   ├── models--pyannote--speaker-diarization-*/"
echo "        │   ├── jhu-clsp/mmBERT-base"
echo "        │   └── multilingual_DialogPII_NER ← NEW Model"
echo "        ├── videos/ (Place input videos here)"
echo "        ├── audios/ (Processed audio goes here)"
echo "        ├── transcripts/ (Raw transcripts)"
echo "        ├── anonym/ (BERT anonymized)"
echo "        └── LLM-Anon/ (LLM rewritten)"
echo ""
echo "Next Steps:"
echo "  1. Edit the '.env' file with your API Key."
echo "  2. Place video files in 'pipeline/videos'."
echo "  3. Run the pipeline:"
echo "     python pipeline/process.py"
echo "  4. (Optional) Run the Web Interface (with Offline Coqui TTS):"
echo "     python interactive_app/app.py"
echo ""
echo "✨ NEW FEATURES:"
echo "  • DFKI-SLT Multilingual DialogPII NER Model installed"
echo "    - Improved accuracy for 11 languages"
echo "    - FLERT-style context windowing"
echo "    - Better dialogue-aware PII detection"
echo ""
echo "💡 IMPORTANT: You may see 'dependency conflict' warnings from pip regarding numpy/pandas."
echo "   These are safe to ignore. As long as 'python -c \"import whisperx...\"' works, your system is fine."
echo ""
echo "📚 Documentation:"
echo "  • Model details: https://huggingface.co/DFKI-SLT/multilingual_DialogPII_NER"
echo "  • Support: https://proton.me/support/lumo"
echo ""
echo "=========================================================================="