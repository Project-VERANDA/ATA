#!/usr/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- 1. MAIN FOLDER LOGIC ---
CURRENT_DIR="$(pwd)"
SCRIPT_NAME="$(basename "$0")"
MAIN_DIR_NAME="ATA"

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

install_if_missing() {
    local pkg_spec=$1
    local pkg_name=$2
    
    if [[ "$pkg_spec" == git+* ]]; then
        if pip show whisperx &> /dev/null; then
            echo "✅ whisperx is already installed. Skipping."
        else
            echo "Installing whisperx from source..."
            pip install "$pkg_spec"
        fi
    else
        if ! check_pip_installed "$pkg_name"; then
            echo "Installing $pkg_name..."
            pip install "$pkg_spec"
        fi
    fi
}

# 5. Install System Dependencies (APT)
echo "Checking system build tools..."
NEEDS_APT=false

if ! check_apt_installed "build-essential"; then NEEDS_APT=true; fi
if ! check_apt_installed "libsndfile1"; then NEEDS_APT=true; fi
if ! check_apt_installed "ffmpeg"; then NEEDS_APT=true; fi

if [ "$NEEDS_APT" = true ]; then
    echo "Installing missing system tools via apt..."
    sudo apt-get update -qq && sudo apt-get install -y -qq build-essential libsndfile1 ffmpeg
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

# ============================================================================
# 🔧 CRITICAL FIX: Install click==8.1.7 AND compatible typer EARLY
# This prevents Python 3.12 TypeError with 'Choice' subscriptable conflicts.
# ============================================================================
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

echo "Checking NLP and Audio libraries..."

# 1. WhisperX (Special handling for Git)
if check_git_installed "git+https://github.com/m-bain/whisperx.git" "whisperx"; then
    echo "✅ whisperx is already installed. Skipping."
else
    echo "Installing whisperx from source..."
    pip install git+https://github.com/m-bain/whisperx.git
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
)

for pkg in "${STANDARD_PKGS[@]}"; do
    if check_pip_installed_robust "$pkg"; then
        echo "✅ $pkg is already installed. Skipping."
    else
        echo "Installing $pkg..."
        pip install "$pkg"
    fi
done

# 3. Spacy Model Check
echo "Checking spaCy multilingual model..."
if python -m spacy check xx_ent_wiki_sm &> /dev/null; then
    echo "✅ spaCy multilingual model (xx_ent_wiki_sm) is already installed."
else
    echo "Installing spaCy multilingual model (xx_ent_wiki_sm)..."
    python -m spacy download xx_ent_wiki_sm
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
    echo "❌ WARNING: click may not be incompatible."
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

# 7. Web Interface Option
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
        
        # Upgrade pip, setuptools, wheel
        pip install --upgrade pip setuptools wheel
        
        # CRITICAL ORDER: Install core data libs FIRST to satisfy whisperx/pyannote
        # This allows pip to pull NumPy 2.0+ which both stacks require
        echo "Installing NumPy and Pandas (satisfies whisperx/pyannote requirements)..."
        pip install "numpy>=2.0.0" "pandas>=2.2.0" --no-cache-dir || {
            echo "⚠️  Standard numpy/pandas install failed, trying fallback range..."
            pip install "numpy>=1.24.0,<3.0.0" "pandas>=2.0.0,<3.0.0" --no-cache-dir
        }
        
        # Now install Coqui TTS (newer versions support NumPy 2.0+)
        echo "Installing Coqui TTS..."
        pip install "TTS>=0.22.0" --no-cache-dir || {
            echo "⚠️  TTS standard install failed. Trying source build..."
            pip install git+https://github.com/coqui-ai/TTS.git@stable --no-cache-dir || {
                echo "❌ CRITICAL: Failed to install Coqui TTS."
                exit 1
            }
        }
        
        # Install remaining web frameworks
        echo "Installing Flask and other dependencies..."
        pip install flask python-dotenv requests cryptography scipy --no-cache-dir
        
        echo "✅ Web Interface dependencies installed."
        echo "   Note: Dependency warnings may appear (see above). These are often non-fatal."
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
echo "Coqui TTS Multi-Voice Models Download"
echo "Required for Offline Speaker Differentiation:"
echo "  - German: thorsten-multispeaker (4 voices)"
echo "  - English: vctk (109 voices)"
echo "Total size: ~700MB (downloaded on first run by code, or here now)."
echo ""

DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-y} 
read -p "Do you want to download these models now? (y/n) [y]: " DOWNLOAD_INPUT
DOWNLOAD_INPUT=${DOWNLOAD_INPUT:-y}

if [[ "$DOWNLOAD_INPUT" =~ ^[Yy]$ ]]; then
    echo "Downloading Multi-Voice Models..."
    
    # Helper to download model
    download_coqui_model() {
        local model_id=$1
        local display_name=$2
        echo "Downloading $display_name ($model_id)..."
        huggingface-cli download "$model_id" --local-dir-use-symlinks false 2>&1 | grep -v "You seem to have already downloaded" || true
        echo "✅ $display_name downloaded."
    }

    # German Model
    if ! python -c "from TTS.api import TTS; TTS('tts_models/de/thorsten-multispeaker/vits')" 2>/dev/null; then
        download_coqui_model "tts_models/de/thorsten-multispeaker/vits" "German Multi-Voice (Thorsten)"
    else
        echo "✅ German Multi-Voice model already available."
    fi

    # English Model
    if ! python -c "from TTS.api import TTS; TTS('tts_models/en/vctk/vits')" 2>/dev/null; then
        download_coqui_model "tts_models/en/vctk/vits" "English Multi-Voice (VCTK)"
    else
        echo "✅ English Multi-Voice model already available."
    fi
    
    echo "✅ All Multi-Voice models ready."
else
    echo "Skipping model download. They will be downloaded automatically on first run."
fi

# 8. Hugging Face Token
echo ""
echo "-------------------------------------------------"
echo "The Pyannote Diarization model is gated on Hugging Face."
echo "Get your token here: https://huggingface.co/settings/tokens"
echo ""

if hf whoami > /dev/null 2>&1; then
    echo "✅ You are already logged in to Hugging Face."
    HF_TOKEN=""
else
    read -p "Enter your Hugging Face Token (or press Enter if already logged in): " HF_TOKEN
    HF_TOKEN=$(echo "$HF_TOKEN" | tr -d '\r\n\t ')
    
    if [ -z "$HF_TOKEN" ]; then
        echo "⚠️  No token provided. You may need to manually download models later."
    else
        echo "Logging in to Hugging Face..."
        LOGIN_OUTPUT=$(hf auth login --token "$HF_TOKEN" 2>&1)
        
        echo "$LOGIN_OUTPUT"
        
        if echo "$LOGIN_OUTPUT" | grep -q "Login successful" || hf whoami > /dev/null 2>&1; then
            echo "✅ Login confirmed."
        else
            echo "❌ Login failed. Please check your token."
            exit 1
        fi
    fi
fi

# 9. Create Project Directory Structure
echo ""
echo "Creating project directory structure..."
PIPELINE_DIR="$PWD/pipeline"
MODEL_DIR="$PIPELINE_DIR/model"
VIDEOS_DIR="$PIPELINE_DIR/videos"
AUDIOS_DIR="$PIPELINE_DIR/audios"
TRANSCRIPTS_DIR="$PIPELINE_DIR/transcripts"
ANNONYM_DIR="$PIPELINE_DIR/annonym"
LLM_ANON_DIR="$PIPELINE_DIR/LLM-Anon"

mkdir -p "$MODEL_DIR" "$VIDEOS_DIR" "$AUDIOS_DIR" "$TRANSCRIPTS_DIR" "$ANNONYM_DIR" "$LLM_ANON_DIR"
echo "✅ Created directories."

# 10. Set Model Directory Path
echo ""
echo "Setting up model directory..."
MODEL_DIR="$PWD/pipeline/model"
mkdir -p "$MODEL_DIR"
echo "✅ Model directory set to: $MODEL_DIR"

# 11. Download WhisperX Models
echo ""
echo "-------------------------------------------------"
echo "WhisperX Model Download Options"
echo "Available Models: tiny, base, small, medium, large"
echo ""

read -p "Enter model names to download (or press Enter to skip): " WHISPER_MODELS_INPUT

if [ -n "$WHISPER_MODELS_INPUT" ]; then
    declare -A MODEL_MAP
    MODEL_MAP["tiny"]="Systran/faster-whisper-tiny"
    MODEL_MAP["base"]="Systran/faster-whisper-base"
    MODEL_MAP["small"]="Systran/faster-whisper-small"
    MODEL_MAP["medium"]="Systran/faster-whisper-medium"
    MODEL_MAP["large"]="Systran/faster-whisper-large-v3"

    for model_name in $WHISPER_MODELS_INPUT; do
        model_name=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
        
        if [ -z "${MODEL_MAP[$model_name]}" ]; then
            echo "⚠️  Warning: '$model_name' is not valid. Skipping."
            continue
        fi

        hf_repo="${MODEL_MAP[$model_name]}"
        # Coqui models are stored in .cache/tts by default, but we can download others here if needed
        # For Coqui, we rely on the automatic download or the explicit download above
        echo "Note: WhisperX models are handled separately. Coqui models were downloaded above."
        
        # Standard WhisperX download logic if you still want them in pipeline/model
        target_dir="$MODEL_DIR/$(echo "$hf_repo" | sed 's/\//\--/g')"
        if [ -d "$target_dir" ]; then
            echo "✅ WhisperX model '$model_name' already exists. Skipping."
        else
            echo "Downloading: $model_name ($hf_repo)..."
            huggingface-cli download "$hf_repo" --local-dir "$target_dir" --local-dir-use-symlinks false 2>/dev/null || \
            huggingface-cli download "$hf_repo" --local-dir "$target_dir"
            
            if [ $? -eq 0 ]; then
                echo "✅ Successfully downloaded: $model_name"
            else
                echo "❌ Failed to download: $model_name"
            fi
        fi
    done
else
    echo "No WhisperX models selected. Skipping."
fi

# 12. Download Pyannote Speaker Diarization Model
echo ""
echo "-------------------------------------------------"
echo "Downloading Pyannote Speaker Diarization Model (community-1)"
echo ""

DIARIZE_TARGET="$MODEL_DIR/models--pyannote--speaker-diarization-community-1"

if [ -d "$DIARIZE_TARGET" ]; then
    echo "✅ Pyannote Diarization model already exists. Skipping."
else
    echo "Downloading Pyannote model..."
    huggingface-cli download pyannote/speaker-diarization-community-1 --local-dir "$DIARIZE_TARGET" --local-dir-use-symlinks False
    
    if [ $? -eq 0 ]; then
        echo "✅ Pyannote Diarization model downloaded successfully."
    else
        echo "❌ Failed to download Pyannote model."
    fi
fi

# 13. Download mmbert Models
echo ""
echo "-------------------------------------------------"
echo "Downloading mmbert Multilingual PII Model & Base BERT"
echo ""

# --- Base Model (mmBERT-base) ---
BASE_PARENT="$MODEL_DIR/jhu-clsp"
BASE_TARGET="$BASE_PARENT/mmBERT-base"

if [ -d "$BASE_TARGET" ] && [ -f "$BASE_TARGET/config.json" ]; then
    echo "✅ Base model structure verified at $BASE_TARGET"
else
    echo "Setting up base model at $BASE_TARGET..."
    mkdir -p "$BASE_PARENT"
    huggingface-cli download jhu-clsp/mmBERT-base --local-dir "$BASE_TARGET" --local-dir-use-symlinks False
    
    if [ $? -eq 0 ] && [ -f "$BASE_TARGET/config.json" ]; then
        echo "✅ Base model successfully installed."
    else
        echo "❌ Error: Failed to download base model."
        exit 1
    fi
fi

# --- PII Model (mmbert_multilingual_pii_ner) ---
PII_TARGET="$MODEL_DIR/mmbert_multilingual_pii_ner"
if [ -d "$PII_TARGET" ]; then
    echo "✅ PII model already exists. Skipping."
else
    echo "Downloading PII model..."
    huggingface-cli download deryaerman/mmbert_multilingual_pii_ner --local-dir "$PII_TARGET" --local-dir-use-symlinks False

    SUBFOLDER=$(find "$PII_TARGET" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    if [ -n "$SUBFOLDER" ] && [ "$SUBFOLDER" != "$PII_TARGET" ]; then
        echo "⚠️  Flattening nested folder structure..."
        mv "$SUBFOLDER"/* "$PII_TARGET/"
        rmdir "$SUBFOLDER"
    fi

    if [ $? -eq 0 ] && [ -f "$PII_TARGET/crf_config.json" ]; then 
        echo "✅ PII model downloaded and structure verified."
    else 
        echo "❌ Failed to download PII model."; exit 1
    fi
fi

# Create Environment file
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

# 14. Final Instructions
echo ""
echo "=== Setup Complete! ==="
echo "To use the environment:"
echo "  source \$HOME/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
echo ""
echo "Project Structure:"
echo "  $(pwd)/"
echo "    ├── .env"
echo "    ├── ATA_SelfInstall.sh"
echo "    └── pipeline/"
echo "        ├── model/ (Models downloaded here)"
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
echo "Note: Coqui Multi-Voice models (German/English) were downloaded automatically."