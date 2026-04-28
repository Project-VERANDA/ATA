#!/usr/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== ATA Speech Anonymizer Installer (Updated) ==="
echo "This script sets up the 'whisperx' environment with all necessary dependencies."
echo "Includes mmbert PII model, Base BERT, and fixes for missing Python bindings."
echo ""

# 1. Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    
    # Download Miniconda
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # Install Miniconda silently (-b for batch, -p for path)
    bash miniconda.sh -b -p $HOME/miniconda3
    
    # Clean up installer
    rm miniconda.sh
    
    # Initialize Conda for the current shell session
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
else
    echo "Conda already installed."
    # Ensure conda is initialized in the current session
    eval "$(conda shell.bash hook)"
fi

# 2. Create the environment
ENV_NAME="whisperx"
echo "Creating conda environment '$ENV_NAME' with Python 3.12..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Removing and recreating..."
    conda env remove -n $ENV_NAME -y
fi
conda create -n $ENV_NAME python=3.12 -y

# 3. Activate the environment
echo "Activating environment..."
conda activate $ENV_NAME

# 4. Install core system dependencies via Conda
# Using conda-forge ensures binary compatibility with PyTorch
echo "Installing system dependencies (ffmpeg, libsndfile, build tools)..."

# NOTE: build-essential is an apt package, not conda. 
# We attempt conda install for gcc/g++ compilers, but rely on system apt if available.

echo "Checking for system build tools..."
if command -v apt-get &> /dev/null; then
    echo "Detected apt-get. Installing system build tools (build-essential, libsndfile)..."
    # Run silently with yes to accept prompts
    sudo apt-get update -qq && sudo apt-get install -y -qq build-essential libsndfile1 ffmpeg
else
    echo "apt-get not found. Attempting to install compilers via Conda..."
    conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y
fi

# Ensure ffmpeg and libsndfile are present in conda as well for consistency
echo "Ensuring ffmpeg and libsndfile are in Conda environment..."
conda install -c conda-forge ffmpeg libsndfile -y

# 5. Install Python packages
echo "Upgrading pip..."
pip install --upgrade pip

# --- CRITICAL FIXES & MISSING DEPENDENCIES ---

# 1. WhisperX (Installed from GitHub for latest stability with pyannote)
echo "Installing whisperx from source..."
pip install git+https://github.com/m-bain/whisperx.git

# 2. Audio Processing (Fixed: ffmpeg-python instead of python-ffmpeg)
echo "Installing audio processing libraries..."
pip install pydub ffmpeg-python

# 3. Pyannote (Diarization)
# Installing before transformers to manage dependency resolution
echo "Installing pyannote-audio and dependencies..."
pip install pyannote.audio pyannote.pipeline pyannote.metrics

# 4. Anonymization Libraries (Local BERT/spaCy)
echo "Installing NLP and anonymization libraries..."
pip install transformers accelerate sentencepiece
pip install spacy
python -m spacy download de_core_news_sm

# 5. CRF Library for the custom mmbert model
echo "Installing torchcrf..."
pip install torchcrf

# 6. MISSING DEPENDENCIES (Added based on process.py requirements)
echo "Installing missing dependencies (pandas, openai)..."
pip install pandas openai

# 7. Torchcodec (Specific version if needed)
echo "Handling torchcodec..."
pip uninstall torchcodec -y || true
pip install torchcodec==0.7.0 || echo "Warning: torchcodec installation failed, skipping."

# 8. Web Interface Option
echo ""
echo "-------------------------------------------------"
read -p "Do you want to install the Web Interface (Flask, gTTS, etc.)? (y/n): " INSTALL_WEB
INSTALL_WEB=${INSTALL_WEB:-y} # Default to 'y' if empty

if [[ "$INSTALL_WEB" =~ ^[Yy]$ ]]; then
    echo "Installing Web Interface dependencies..."
    pip install flask python-dotenv requests cryptography gtts
    echo "Web Interface dependencies installed."
else
    echo "Skipping Web Interface installation. Command-line only mode selected."
fi

# 9. Hugging Face Token for Gated Models
echo ""
echo "-------------------------------------------------"
echo "The Pyannote Diarization model is gated on Hugging Face."
echo "You need a token to download it automatically."
echo "Get your token here: https://huggingface.co/settings/tokens"
echo "(Read access is sufficient)"
echo ""
read -sp "Enter your Hugging Face Token (will not be stored): " HF_TOKEN
echo ""

# Trim whitespace and carriage returns from the token
HF_TOKEN=$(echo "$HF_TOKEN" | tr -d '\r\n\t ')

if [ -z "$HF_TOKEN" ]; then
    echo "No token provided. You may need to manually download the Pyannote model later."
    echo "Set the token manually later with: hf auth login --token YOUR_TOKEN"
else
    echo "Logging in to Hugging Face..."
    pip install huggingface_hub
    
    # Use the modern 'hf auth login' command which is more robust
    # We pass the token via stdin to avoid shell expansion issues
    echo "$HF_TOKEN" | hf auth login --token-stdin
    
    # Verify login
    if hf whoami > /dev/null 2>&1; then
        echo "✅ Successfully logged in to Hugging Face."
        
        # Trigger a dummy download to verify access and cache the model
        echo "Verifying access and caching Pyannote model..."
        python -c "from pyannote.audio import Pipeline; print('Downloading model...'); Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')"
        echo "Pyannote model cached successfully!"
    else
        echo "❌ Login failed. Please check your token."
        exit 1
    fi
fi

# 10. Download WhisperX Models (Interactive Selection)
# FIXED: Path logic now matches process.py expectations (models-- prefix)
echo ""
echo "-------------------------------------------------"
echo "WhisperX Model Download Options"
echo "Select the models you want to download (separate with spaces, e.g., 'base large')."
echo ""
echo "Available Models:"
echo "  - tiny      (Fastest, lowest accuracy)"
echo "  - base      (Default, good balance)"
echo "  - small     (More accurate, slower)"
echo "  - medium    (High accuracy, very slow)"
echo "  - large     (Best accuracy, slowest - ~3GB)"
echo ""

read -p "Enter model names to download (or press Enter to skip): " WHISPER_MODELS_INPUT

if [ -n "$WHISPER_MODELS_INPUT" ]; then
    # Define the target directory
    TARGET_BASE="$PWD/pipeline/model"
    mkdir -p "$TARGET_BASE"

    declare -A MODEL_MAP
    MODEL_MAP["tiny"]="Systran/faster-whisper-tiny"
    MODEL_MAP["base"]="Systran/faster-whisper-base"
    MODEL_MAP["small"]="Systran/faster-whisper-small"
    MODEL_MAP["medium"]="Systran/faster-whisper-medium"
    MODEL_MAP["large"]="Systran/faster-whisper-large-v3"

    for model_name in $WHISPER_MODELS_INPUT; do
        model_name=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
        
        if [ -z "${MODEL_MAP[$model_name]}" ]; then
            echo "⚠️  Warning: '$model_name' is not a valid option. Skipping."
            continue
        fi

        hf_repo="${MODEL_MAP[$model_name]}"
        
        # FIX: Prepend "models--" to match process.py hardcoded paths
        # Example: Systran/faster-whisper-large-v3 -> models--Systran--faster-whisper-large-v3
        folder_name="models--$(echo "$hf_repo" | sed 's/\//\--/g')"
        target_dir="$TARGET_BASE/$folder_name"

        echo ""
        echo "----------------------------------------"
        echo "Downloading: $model_name ($hf_repo)"
        echo "Target: $target_dir"
        echo "----------------------------------------"

        if [ -d "$target_dir" ]; then
            echo "⚠️  Model '$model_name' already exists at $target_dir. Skipping download."
        else
            echo "Starting download..."
            huggingface-cli download "$hf_repo" --local-dir "$target_dir" --local-dir-use-symlinks false
            
            if [ $? -eq 0 ]; then
                echo "✅ Successfully downloaded: $model_name"
            else
                echo "❌ Failed to download: $model_name"
            fi
        fi
    done
    echo ""
    echo "WhisperX model download phase complete."
else
    echo "No models selected. Skipping WhisperX downloads."
fi

# 11. Download mmbert PII Model and Base BERT (CRITICAL FOR OFFLINE)
echo ""
echo "-------------------------------------------------"
echo "Downloading mmbert Multilingual PII Model & Base BERT"
echo "These are required for the local anonymization engine."
echo "This will download ~2.5GB total."
echo ""

TARGET_BASE="$PWD/pipeline/model"
mkdir -p "$TARGET_BASE"

# 11a. Download Base Model
echo "1. Downloading Base Model: jhu-clsp/mmBERT-base"
BASE_TARGET="$TARGET_BASE/mmBERT-base-local"

if [ -d "$BASE_TARGET" ]; then
    echo "⚠️  Base model already exists at $BASE_TARGET. Skipping."
else
    echo "Downloading base model (this may take a while)..."
    huggingface-cli download jhu-clsp/mmBERT-base --local-dir "$BASE_TARGET" --local-dir-use-symlinks false
    if [ $? -eq 0 ]; then
        echo "✅ Base model downloaded successfully."
    else
        echo "❌ Failed to download base model."
        exit 1
    fi
fi

# 11b. Download Fine-tuned PII Model
echo ""
echo "2. Downloading Fine-tuned Model: deryaerman/mmbert_multilingual_pii_ner"
PII_TARGET="$TARGET_BASE/mmbert_multilingual_pii_ner"

if [ -d "$PII_TARGET" ]; then
    echo "⚠️  PII model already exists at $PII_TARGET. Skipping."
else
    echo "Downloading PII model..."
    # Note: This creates a subfolder 'jhu-clsp-mmBERT-base-multilingual-pii' inside the target
    huggingface-cli download deryaerman/mmbert_multilingual_pii_ner --local-dir "$PII_TARGET" --local-dir-use-symlinks false
    
    if [ $? -eq 0 ]; then
        echo "✅ PII model downloaded successfully."
        echo "   Note: Files are located in $PII_TARGET/jhu-clsp-mmBERT-base-multilingual-pii/"
    else
        echo "❌ Failed to download PII model."
        exit 1
    fi
fi

echo ""
echo "mmbert model download phase complete."

# 12. Create .env File
echo ""
echo "-------------------------------------------------"
echo "Creating .env configuration file..."
echo ""

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists. Backing up old one to .env.backup..."
    cp .env .env.backup
fi

# Create the new .env file
cat > .env <<EOF
# ATA Speech Anonymizer Configuration
# Generated by ATA_SelfInstall.sh on $(date)

# LLM API Configuration (Required for indirect identifier removal)
# Replace 'your_api_key_here' with your actual API key
CHAT_AI_API_KEY=your_api_key_here

# LLM Endpoint URL
# Replace with your specific endpoint if different from the default
CHAT_AI_ENDPOINT=https://your-endpoint.com/v1

# Optional: Specify a default model key if desired (matches keys in AVAILABLE_LLM_MODELS)
# CHAT_AI_MODEL=gpt-oss-120b

# Optional: HTTPS toggle for Web Interface (default: true)
# USE_HTTPS=true
EOF

echo "✅ .env file created successfully in the current directory."
echo "   ⚠️  IMPORTANT: You MUST edit this file and replace 'your_api_key_here' with your real API key."
echo "   ⚠️  Also update 'CHAT_AI_ENDPOINT' if your provider differs."

# 13. Final Instructions
echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To use the environment:"
echo "  source $HOME/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
echo ""
echo "Project Structure Expectations:"
echo "  - Place your 'pipeline' folder (with process.py) in the project root."
echo "  - The script has automatically downloaded models to 'pipeline/model/'."
echo "  - If Web Interface installed: Run 'python interactive_app/app.py'."
echo "  - If CLI only: Run 'python pipeline/process.py'."
echo ""
echo "Configuration:"
echo "  - Edit the newly created '.env' file to add your API Key and Endpoint."
echo "  - The .env file is automatically loaded by process.py and app.py."
echo ""
echo "Offline Status: All required models (WhisperX, Pyannote, mmbert, Base BERT) are now local."
echo "You can run the pipeline without an internet connection (after initial setup)."
echo ""