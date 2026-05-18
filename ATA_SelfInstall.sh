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
echo "Creating conda environment '$ENV_NAME' with Python 3.12..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Removing and recreating..."
    conda env remove -n $ENV_NAME -y
fi
conda create -n $ENV_NAME python=3.12 -y

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

# --- HELPER FUNCTION: Check if Git package is installed ---
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
)

for pkg in "${STANDARD_PKGS[@]}"; do
    if check_pip_installed_robust "$pkg"; then
        echo "✅ $pkg is already installed. Skipping."
    else
        echo "Installing $pkg..."
        pip install "$pkg"
    fi
done

# 3. Spacy Model Check (Multilingual 'xx' required for sentence splitting)
if python -m spacy check xx &> /dev/null; then
    echo "✅ spaCy multilingual model (xx) is already installed."
else
    echo "Installing spaCy multilingual model (xx)..."
    python -m spacy download xx
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

# 7. Web Interface Option
echo ""
echo "-------------------------------------------------"
read -p "Do you want to install the Web Interface (Flask, gTTS, etc.)? (y/n): " INSTALL_WEB
INSTALL_WEB=${INSTALL_WEB:-y}

if [[ "$INSTALL_WEB" =~ ^[Yy]$ ]]; then
    WEB_PKGS=("flask" "python-dotenv" "requests" "cryptography" "gtts")
    SKIP_WEB=true
    for pkg in "${WEB_PKGS[@]}"; do
        if ! check_pip_installed "$pkg"; then
            SKIP_WEB=false
            break
        fi
    done

    if [ "$SKIP_WEB" = false ]; then
        echo "Installing Web Interface dependencies..."
        pip install flask python-dotenv requests cryptography gtts
    else
        echo "✅ All Web Interface dependencies are already installed."
    fi
else
    echo "Skipping Web Interface installation."
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
        echo "⚠️  No token provided and not logged in. You may need to manually download models later."
        echo "   To fix later: Run 'hf auth login' manually."
    else
        echo "Logging in to Hugging Face..."
        LOGIN_OUTPUT=$(hf auth login --token "$HF_TOKEN" 2>&1)
        LOGIN_STATUS=$?
        
        echo "$LOGIN_OUTPUT"
        
        if echo "$LOGIN_OUTPUT" | grep -q "Login successful"; then
            echo "✅ Login confirmed via output message."
        else
            if hf whoami > /dev/null 2>&1; then
                echo "✅ Login confirmed via whoami."
            else
                echo "❌ Login failed. Please check your token."
                exit 1
            fi
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
TARGET_BASE="$MODEL_DIR"

# 10. Download WhisperX Models
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
        folder_name="models--$(echo "$hf_repo" | sed 's/\//\--/g')"
        target_dir="$MODEL_DIR/$folder_name"

        if [ -d "$target_dir" ]; then
            echo "✅ Model '$model_name' already exists at $target_dir. Skipping download."
        else
            echo "Downloading: $model_name ($hf_repo)..."
            huggingface-cli download "$hf_repo" --local-dir "$target_dir" --local-dir-use-symlinks False
            
            if [ $? -eq 0 ]; then
                echo "✅ Successfully downloaded: $model_name"
            else
                echo "❌ Failed to download: $model_name"
            fi
        fi
    done
else
    echo "No models selected. Skipping WhisperX downloads."
fi

# 11. Download Pyannote Speaker Diarization Model
echo ""
echo "-------------------------------------------------"
echo "Downloading Pyannote Speaker Diarization Model (community-1)"
echo ""

DIARIZE_TARGET="$MODEL_DIR/models--pyannote--speaker-diarization-community-1"

if [ -d "$DIARIZE_TARGET" ]; then
    echo "✅ Pyannote Diarization model already exists at $DIARIZE_TARGET. Skipping."
else
    echo "Downloading Pyannote model..."
    huggingface-cli download pyannote/speaker-diarization-community-1 --local-dir "$DIARIZE_TARGET" --local-dir-use-symlinks False
    
    if [ $? -eq 0 ]; then
        echo "✅ Pyannote Diarization model downloaded successfully."
    else
        echo "❌ Failed to download Pyannote model. Diarization may be disabled."
        echo "   Ensure you are logged in to Hugging Face with a valid token."
    fi
fi

# 12. Download mmbert Models
echo ""
echo "-------------------------------------------------"
echo "Downloading mmbert Multilingual PII Model & Base BERT"
echo ""

TARGET_BASE="$TARGET_BASE"

# --- Base Model (mmBERT-base) ---
BASE_TARGET="$TARGET_BASE/mmBERT-base-local"
if [ -d "$BASE_TARGET" ]; then
    echo "✅ Base model already exists at $BASE_TARGET. Skipping."
else
    echo "Downloading base model..."
    huggingface-cli download jhu-clsp/mmBERT-base --local-dir "$BASE_TARGET" --local-dir-use-symlinks False
    
    if [ -d "$BASE_TARGET/jhu-clsp-mmBERT-base" ]; then
        echo "⚠️  Detected nested folder. Flattening structure..."
        mv "$BASE_TARGET/jhu-clsp-mmBERT-base"/* "$BASE_TARGET/"
        rmdir "$BASE_TARGET/jhu-clsp-mmBERT-base"
    fi
    
    if [ $? -eq 0 ]; then echo "✅ Base model downloaded."; else echo "❌ Failed."; exit 1; fi
fi

# --- PII Model (mmbert_multilingual_pii_ner) ---
PII_TARGET="$TARGET_BASE/mmbert_multilingual_pii_ner"
if [ -d "$PII_TARGET" ]; then
    echo "✅ PII model already exists at $PII_TARGET. Skipping."
else
    echo "Downloading PII model..."
    huggingface-cli download deryaerman/mmbert_multilingual_pii_ner --local-dir "$PII_TARGET" --local-dir-use-symlinks False

    SUBFOLDER=$(find "$PII_TARGET" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    
    if [ -n "$SUBFOLDER" ] && [ "$SUBFOLDER" != "$PII_TARGET" ]; then
        echo "⚠️  Detected nested folder structure. Flattening to match code expectations..."
        mv "$SUBFOLDER"/* "$PII_TARGET/"
        rmdir "$SUBFOLDER"
        echo "✅ Structure flattened successfully."
    fi

    if [ $? -eq 0 ]; then 
        echo "✅ PII model downloaded and structure verified."
        
        if [ ! -f "$PII_TARGET/crf_config.json" ]; then
            echo "❌ CRITICAL: crf_config.json missing after flattening. Installation may fail."
            exit 1
        fi
    else 
        echo "❌ Failed to download PII model."; exit 1
    fi
fi

cat > .env <<EOF
# ATA Speech Anonymizer Configuration
# Generated by ATA_SelfInstall.sh on $(date)

# LLM API Configuration (Required for indirect identifier removal)
# Replace 'your_api_key_here' with your actual API key
CHAT_AI_API_KEY=your_api_key_here

# LLM Endpoint URL
# Replace with your specific endpoint if different from the default
CHAT_AI_ENDPOINT=https://your-endpoint.com/v1

# Optional: Specify a default model key if desired
# CHAT_AI_MODEL=gpt-oss-120b

# Optional: HTTPS toggle for Web Interface
# USE_HTTPS=true
EOF

echo "✅ .env file created. Edit it with your real API key."

# 13. Final Instructions
echo ""
echo "=== Setup Complete! ==="
echo "To use the environment:"
echo "  source $HOME/miniconda3/etc/profile.d/conda.sh"
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
echo "  2. Place your video files in the 'pipeline/videos' folder."
echo "  3. Run the pipeline:"
echo "     python pipeline/process.py"
echo "  4. (Optional) Run the Web Interface:"
echo "     python interactive_app/app.py"
echo ""
echo "Offline Status: All required models are now local."
echo ""